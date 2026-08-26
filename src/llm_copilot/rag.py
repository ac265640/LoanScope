"""
RAG over Data Dictionary and Validation Rules — Advanced Feature #7
====================================================================
Builds a TF-IDF/BM25-style retriever that chunks data_dictionary.md and
validation_rules.json, then retrieves only the relevant chunks per query
before injecting into the LLM copilot prompt.

Retrieved chunks are logged alongside prompts in logs/llm_prompt_log.jsonl.

Run: PYTHONPATH=. python src/llm_copilot/rag.py
"""

import json
import re
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_markdown(text: str, max_chars: int = 500) -> List[str]:
    """Split markdown by headers and paragraph blocks."""
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section.strip())
        else:
            # Split long sections into paragraphs
            paras = section.split("\n\n")
            buf = ""
            for p in paras:
                if len(buf) + len(p) < max_chars:
                    buf += p + "\n\n"
                else:
                    if buf.strip():
                        chunks.append(buf.strip())
                    buf = p + "\n\n"
            if buf.strip():
                chunks.append(buf.strip())
    return [c for c in chunks if len(c) > 20]


def _chunk_json_rules(rules_data) -> List[str]:
    """Convert validation_rules.json entries into text chunks."""
    chunks = []
    if isinstance(rules_data, list):
        for rule in rules_data:
            text = f"Rule: {rule.get('rule_id', '')}\n"
            text += f"Field: {rule.get('field', '')}\n"
            text += f"Condition: {rule.get('condition', '')}\n"
            text += f"Severity: {rule.get('severity', '')}\n"
            text += f"Description: {rule.get('description', '')}"
            chunks.append(text.strip())
    elif isinstance(rules_data, dict):
        for k, v in rules_data.items():
            text = f"Rule key: {k}\n"
            if isinstance(v, dict):
                for fk, fv in v.items():
                    text += f"  {fk}: {fv}\n"
            else:
                text += f"  Value: {v}\n"
            chunks.append(text.strip())
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF retriever
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """Lightweight TF-IDF retriever with BM25-style term frequency saturation."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self._tfidf_matrix, self.vocab = self._build_tfidf(chunks)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z_][a-z0-9_]*", text.lower())

    def _build_tfidf(self, docs: List[str]):
        from collections import Counter
        n_docs = len(docs)
        tokenized = [self._tokenize(d) for d in docs]
        # Build vocab
        all_terms = set(t for doc in tokenized for t in doc)
        vocab = {t: i for i, t in enumerate(sorted(all_terms))}

        # TF-IDF matrix (docs × terms)
        matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
        df = np.zeros(len(vocab), dtype=np.float32)

        for i, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            for term, cnt in counts.items():
                if term in vocab:
                    j = vocab[term]
                    matrix[i, j] = cnt / max(len(tokens), 1)
                    df[j] += 1

        # IDF
        idf = np.log((n_docs + 1) / (df + 1)) + 1.0
        matrix = matrix * idf

        return matrix, vocab

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, str]]:
        """Return top_k most relevant chunks for the query."""
        tokens = self._tokenize(query)
        from collections import Counter
        counts = Counter(tokens)
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for term, cnt in counts.items():
            if term in self.vocab:
                q_vec[self.vocab[term]] = cnt

        # Cosine similarity
        norms = np.linalg.norm(self._tfidf_matrix, axis=1) + 1e-9
        q_norm = np.linalg.norm(q_vec) + 1e-9
        scores = self._tfidf_matrix @ q_vec / (norms * q_norm)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.chunks[i]) for i in top_idx if scores[i] > 0]


# ---------------------------------------------------------------------------
# Main RAG interface
# ---------------------------------------------------------------------------

class LoanScopeRAG:
    """RAG retriever over data dictionary and validation rules."""

    def __init__(self):
        self._retriever: TFIDFRetriever = None
        self._chunks: List[str] = []
        self._build_index()

    def _build_index(self):
        chunks = []

        # Chunk data_dictionary.md
        dd_path = DATA_DIR / "data_dictionary.md"
        if not dd_path.exists():
            dd_path = DATA_DIR / "raw" / "data_dictionary.md" if (DATA_DIR / "raw").exists() else None
        if dd_path and dd_path.exists():
            text = dd_path.read_text()
            dd_chunks = _chunk_markdown(text)
            chunks.extend(dd_chunks)
            print(f"[rag] Data dictionary: {len(dd_chunks)} chunks")

        # Chunk validation_rules.json
        vr_path = DATA_DIR / "validation_rules.json"
        if not vr_path.exists():
            vr_path = DATA_DIR / "raw" / "validation_rules.json" if (DATA_DIR / "raw").exists() else None
        if vr_path and vr_path.exists():
            rules = json.loads(vr_path.read_text())
            vr_chunks = _chunk_json_rules(rules)
            chunks.extend(vr_chunks)
            print(f"[rag] Validation rules: {len(vr_chunks)} chunks")

        if not chunks:
            # Fallback minimal chunks
            chunks = [
                "Loan fields: loan_id, reporting_month, current_status, credit_score_band, days_past_due",
                "Status values: Current, 30-DPD, 60-DPD, 90-DPD, Default, Prepaid, Charged-Off",
                "Validation: current_balance must be > 0 for active loans",
                "Target: next_12m_default_flag (1=default within 12 months)",
            ]

        self._chunks = chunks
        self._retriever = TFIDFRetriever(chunks)
        print(f"[rag] Index built: {len(chunks)} total chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        results = self._retriever.retrieve(query, top_k=top_k)
        return [chunk for _, chunk in results]

    def retrieve_and_log(self, query: str, top_k: int = 5, log_entry: dict = None) -> str:
        """
        Retrieve relevant chunks, format as context string, and log to
        logs/llm_prompt_log.jsonl alongside the query metadata.
        """
        results = self._retriever.retrieve(query, top_k=top_k)
        chunks = [c for _, c in results]
        scores = [round(s, 4) for s, _ in results]

        context = "\n\n---\n\n".join(chunks)

        # Log retrieved chunks
        log_record = {
            "type": "rag_retrieval",
            "query": query,
            "top_k": top_k,
            "n_retrieved": len(chunks),
            "retrieval_scores": scores,
            "retrieved_chunks": chunks,
        }
        if log_entry:
            log_record.update(log_entry)

        log_path = LOGS_DIR / "llm_prompt_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_record) + "\n")

        return context


# Singleton
_rag_instance = None


def get_rag() -> LoanScopeRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LoanScopeRAG()
    return _rag_instance


if __name__ == "__main__":
    rag = LoanScopeRAG()

    test_queries = [
        "What does credit_score_band mean?",
        "When is exception_required triggered?",
        "How is default_flag defined?",
        "What validation rules apply to current_balance?",
    ]

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)
    for query in test_queries:
        print(f"\nQuery: {query}")
        context = rag.retrieve_and_log(query, top_k=3)
        print(f"Context (first 200 chars): {context[:200]}...")
    print("\nDone. Retrieved chunks logged to logs/llm_prompt_log.jsonl")
