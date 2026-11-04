import json
import unittest
from unittest.mock import patch

from rag import PolicyRag


class PolicyRagTests(unittest.TestCase):
    def setUp(self):
        self.rag = PolicyRag()
        self.question = "What are the terms for sewer backup damage?"

    def test_graph_adds_a_clause_missed_by_vector_retrieval(self):
        baseline = self.rag.retrieve(self.question, use_graph=False, limit=2)
        expanded = self.rag.retrieve(self.question, use_graph=True, limit=2)
        self.assertEqual([item["id"] for item in baseline], ["water-backup"])
        self.assertEqual([item["id"] for item in expanded], ["water-backup", "backup-limit"])

    def test_offline_mode_returns_evidence_without_generated_claims(self):
        self.assertEqual(self.rag.answer(self.question)["status"], "evidence_only")
        self.assertEqual(
            self.rag.answer("How do I reset my password?")["status"], "insufficient_evidence"
        )

    @patch("rag.ollama")
    def test_rejects_a_citation_outside_the_retrieved_context(self, ollama):
        self.rag.chat_model = "test-model"
        ollama.return_value = {"message": {"content": json.dumps({
            "answer": "Example response", "citations": ["flood"],
        })}}
        self.assertEqual(self.rag.answer(self.question)["status"], "insufficient_evidence")

    @patch("rag.ollama")
    def test_passes_retrieved_clauses_to_model_and_returns_cited_sources(self, ollama):
        self.rag.chat_model = "test-model"
        ollama.return_value = {"message": {"content": json.dumps({
            "answer": "The deductible is $500.", "citations": ["backup-limit"],
        })}}
        result = self.rag.answer(self.question)
        self.assertEqual(result["status"], "answered")
        self.assertEqual([item["id"] for item in result["sources"]], ["backup-limit"])
        _, payload = ollama.call_args.args
        context = json.loads(payload["messages"][1]["content"])
        self.assertIn("$500", context["clauses"][1]["text"])


if __name__ == "__main__":
    unittest.main()
