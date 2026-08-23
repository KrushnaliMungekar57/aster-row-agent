"""
Retrieval-Augmented Generation over the Aster & Row knowledge base.

Design choices (documented in README):
- No embedding model / vector DB. Retrieval uses a small, dependency-free
  BM25 implementation over section-level chunks. This keeps the system
  deterministic and easy to audit for a support-policy corpus of 14 short
  documents, and avoids a hidden network/API dependency for indexing.
- Each chunk is a single "##" section of a document (with the pre-heading
  intro folded into an "Overview" section). Front-matter metadata
  (status, policy_authority, audience, supersedes/superseded_by, dates) is
  attached to every chunk so the agent can reason about precedence.
- Ranking multiplies the raw BM25 score by an authority multiplier so that
  active/official policy documents dominate over superseded, draft, or
  internal-only documents *without deleting them from the index* -- they
  can still surface (e.g. when a user references "the migration note")
  so the agent can explicitly address and reject them.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "in", "on", "for", "with", "it", "this", "that", "as",
    "at", "by", "from", "do", "does", "did", "can", "will", "would",
    "should", "i", "you", "your", "my", "we", "our", "if", "not",
}

# Authority multipliers applied on top of raw lexical score.
STATUS_MULTIPLIER = {"active": 1.0, "superseded": 0.45, "draft": 0.3}
AUTHORITY_MULTIPLIER = {"official": 1.0, "none": 0.3}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    doc_filename: str
    heading: str
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_filename}#{self.heading}"

    @property
    def source_label(self) -> str:
        return f"{self.doc_filename} — {self.heading}"


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Parse a simple flat YAML-style front matter block delimited by ---."""
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    fm_block, body = parts[1], parts[2]
    meta = {}
    for line in fm_block.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body.lstrip("\n")


def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    chunks: list[Chunk] = []

    # Split on level-2 headings ("## ..."), keep the document H1/intro as
    # an "Overview" section.
    lines = body.splitlines()
    current_heading = "Overview"
    current_lines: list[str] = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(
                Chunk(
                    doc_filename=path.name,
                    heading=current_heading,
                    text=text,
                    metadata=meta,
                )
            )

    for line in lines:
        if line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            # top-level title line; skip, not a chunk boundary
            continue
        else:
            current_lines.append(line)
    flush()
    return chunks


# Once a document is judged relevant enough to select, every section of it
# is returned (see KnowledgeBaseIndex.search below) -- so the number of
# distinct documents to fully expand is capped independently from top_k
# (which historically meant "how many chunks", and is now tuned higher,
# e.g. 6, for that older meaning). Expanding 6 full documents at once was
# observed live to dilute context with barely-relevant material and hurt
# citation precision; 4 keeps the benefit without the dilution.
MAX_DOCS_TO_EXPAND = 4


class KnowledgeBaseIndex:
    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        self.chunks: list[Chunk] = []
        self._doc_freq: dict[str, int] = {}
        self._doc_len: list[int] = []
        self._avg_len: float = 0.0
        self._tokens: list[list[str]] = []
        self._build()

    def _build(self):
        for path in sorted(self.kb_dir.glob("*.md")):
            self.chunks.extend(chunk_document(path))

        self._tokens = [tokenize(c.text + " " + c.heading) for c in self.chunks]
        self._doc_len = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

        df: dict[str, int] = {}
        for toks in self._tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        self._doc_freq = df

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._doc_freq.get(term, 0)
        # BM25 idf with +1 smoothing to keep values non-negative.
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _bm25(self, query_tokens: list[str], idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        toks = self._tokens[idx]
        if not toks:
            return 0.0
        freq: dict[str, int] = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        score = 0.0
        dl = self._doc_len[idx]
        for term in query_tokens:
            if term not in freq:
                continue
            f = freq[term]
            idf = self._idf(term)
            denom = f + k1 * (1 - b + b * dl / (self._avg_len or 1))
            score += idf * (f * (k1 + 1)) / (denom or 1)
        return score

    def _authority_multiplier(self, chunk: Chunk) -> float:
        status = chunk.metadata.get("status", "active")
        authority = chunk.metadata.get("policy_authority", "official")
        return STATUS_MULTIPLIER.get(status, 0.5) * AUTHORITY_MULTIPLIER.get(authority, 0.5)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.05) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        scored = []
        for i, chunk in enumerate(self.chunks):
            raw = self._bm25(q_tokens, i)
            weighted = raw * self._authority_multiplier(chunk) if raw > 0 else 0.0
            scored.append((weighted, raw, chunk, i))

        relevant = [s for s in scored if s[0] >= min_score]
        if not relevant:
            return []
        relevant.sort(key=lambda x: x[0], reverse=True)

        # Step 1: pick top_k distinct DOCUMENTS, ranked by each document's
        # single best-scoring section.
        #
        # Bug found via evaluation (see bug diary): a top_k cutoff applied
        # directly to ranked CHUNKS can let one document's multiple relevant
        # sections crowd out a second, equally-relevant document. Fixed by
        # selecting distinct documents first.
        doc_order: list[str] = []
        for weighted, raw, chunk, i in relevant:
            if chunk.doc_filename not in doc_order:
                doc_order.append(chunk.doc_filename)
        top_docs = set(doc_order[: min(top_k, MAX_DOCS_TO_EXPAND)])

        # Step 2: for each selected document, return EVERY section that has
        # any lexical overlap with the query (raw BM25 > 0) -- not only the
        # single section that happened to score highest.
        #
        # Second bug found via evaluation: even after step 1, a document
        # can still be cited while missing a procedural requirement that
        # lives in a SIBLING section of the same page -- e.g. a query about
        # a damaged final-sale item consistently matched
        # 04-damaged-or-wrong-items.md's "Final-sale items" section (which
        # shares vocabulary with the query) while its "Reporting window"
        # section ("...within 7 calendar days...") and "Reports after
        # seven days" section ("...human review...") scored lower on pure
        # keyword overlap and were never once retrieved across 16 live
        # runs, even though they're the two procedural details the
        # customer actually needs.
        #
        # This corpus is a small set of short customer-support policy
        # pages (14 files, 3-6 short sections each). Once a document is
        # judged the right place to look, returning the complete policy
        # rather than one isolated paragraph mirrors how a human agent
        # would work (pull up the whole page, not a fragment) and is a
        # deliberate corpus-size tradeoff -- see README limitations for
        # when this stops being appropriate (a large/heterogeneous corpus
        # would need real semantic chunk-level retrieval instead).
        results = []
        for doc in doc_order:
            if doc not in top_docs:
                continue
            # Every section of this document, in original document order --
            # not filtered by raw>0. A sibling section can be the exact
            # procedural detail the customer needs (e.g. "Reporting window")
            # while sharing zero keywords with the query itself ("report
            # ... within 7 calendar days of delivery" has no lexical
            # overlap with "final sale bag broken zipper"), so no keyword
            # threshold, however low, would ever surface it. Once BM25 has
            # done its job of picking the right DOCUMENT, full-document
            # inclusion is what actually closes the gap.
            doc_chunks = sorted(
                (s for s in scored if s[2].doc_filename == doc),
                key=lambda s: s[3],  # original document order
            )
            for weighted, raw, chunk, i in doc_chunks:
                results.append(
                    {
                        "source_file": chunk.doc_filename,
                        "heading": chunk.heading,
                        "text": chunk.text,
                        "status": chunk.metadata.get("status", "active"),
                        "policy_authority": chunk.metadata.get("policy_authority", "official"),
                        "audience": chunk.metadata.get("audience", "customer"),
                        "document_id": chunk.metadata.get("document_id", ""),
                        "supersedes": chunk.metadata.get("supersedes", ""),
                        "superseded_by": chunk.metadata.get("superseded_by", ""),
                        "score": round(weighted, 4),
                        "raw_score": round(raw, 4),
                    }
                )
        return results