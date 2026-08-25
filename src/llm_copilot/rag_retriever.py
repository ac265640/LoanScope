"""
RAG Knowledge Retriever over Data Dictionary & Validation Rules
================================================================
Provides semantic / BM25 keyword retrieval over data dictionary field definitions,
validation rules, and underwriting guidelines to ground LLM queries and reviewer notes.

Supports queries such as:
  - "What is the formula for DPD and what does VR003 enforce?"
  - "Which validation rules apply to loan modifications?"
  - "Explain the loss severity band definition."

Run: PYTHONPATH=. python src/llm_copilot/rag_retriever.py
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


class RAGKnowledgeRetriever:
    """Retrieval-Augmented Generation context retriever indexing data dictionary and validation rules."""

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Index markdown data dictionary and JSON validation rules into searchable chunks."""
        # 1. Data Dictionary
        dd_path = DATA_DIR / "data_dictionary.md"
        if dd_path.exists():
            content = dd_path.read_text()
            # Split by section headers
            sections = re.split(r"\n#{2,3}\s+", content)
            for s in sections:
                if not s.strip():
                    continue
                lines = s.strip().split("\n")
                title = lines[0]
                body = "\n".join(lines[1:])
                self.documents.append({
                    "doc_id": f"DD_{title[:30].strip().replace(' ', '_')}",
                    "source": "data_dictionary.md",
                    "title": title,
                    "content": body,
                    "tokens": set(re.findall(r"\w+", s.lower())),
                })

        # 2. Validation Rules
        vr_path = DATA_DIR / "validation_rules.json"
        if vr_path.exists():
            with open(vr_path) as f:
                rules = json.load(f)
            rule_list = rules.get("rules", rules) if isinstance(rules, dict) else rules
            for r in rule_list:
                text = f"{r.get('rule_id')} {r.get('rule_name')} {r.get('condition')} {r.get('description', '')}"
                self.documents.append({
                    "doc_id": r.get("rule_id", "VR_UNKNOWN"),
                    "source": "validation_rules.json",
                    "title": f"Rule {r.get('rule_id')}: {r.get('rule_name')}",
                    "content": json.dumps(r, indent=2),
                    "tokens": set(re.findall(r"\w+", text.lower())),
                })

        log.info(f"RAG Knowledge Base indexed {len(self.documents)} reference chunks.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-K most relevant chunks using BM25-style token overlap scoring."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []

        for doc in self.documents:
            overlap = len(q_tokens & doc["tokens"])
            if overlap > 0:
                score = overlap / (len(q_tokens) + 0.1)
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            {
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "title": doc["title"],
                "content": doc["content"],
                "relevance_score": round(score, 4),
            }
            for score, doc in scored[:top_k]
        ]
        return results


def main():
    retriever = RAGKnowledgeRetriever()
    queries = [
        "What are the rules regarding modification and missing documents?",
        "How is default_flag defined and what is VR003?",
        "Explain current_balance vs original_balance consistency checks",
    ]

    print("\n" + "=" * 60)
    print("RAG KNOWLEDGE RETRIEVAL BENCHMARK")
    print("=" * 60)
    for q in queries:
        print(f"\nQuery: '{q}'")
        hits = retriever.search(q, top_k=2)
        for h in hits:
            print(f"  → [{h['source']}] {h['title']} (Score: {h['relevance_score']:.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
