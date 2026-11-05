"""Offline checks for the shared RAGAS evaluation adapter."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from tests.evaluation import ragas_support


class RagasSupportTests(unittest.TestCase):
    def test_missing_index_is_rejected_before_initializing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Build it first"):
                ragas_support.create_catalog_retriever(Path(directory) / "missing")

    def test_evaluate_data_passes_contexts_and_metrics_to_ragas(self):
        data = {"user_input": ["question"], "reference": ["answer"],
                "retrieved_contexts": [["course"]]}
        dataset = object()
        llm = object()
        precision = object()
        recall = object()
        result = object()

        datasets = types.ModuleType("datasets")
        datasets.Dataset = types.SimpleNamespace(from_dict=Mock(return_value=dataset))
        openai = types.ModuleType("langchain_openai")
        openai.ChatOpenAI = Mock(return_value=llm)
        ragas = types.ModuleType("ragas")
        ragas.evaluate = Mock(return_value=result)
        metrics = types.ModuleType("ragas.metrics")
        metrics.LLMContextPrecisionWithReference = Mock(return_value=precision)
        metrics.LLMContextRecall = Mock(return_value=recall)

        with patch.dict(sys.modules, {"datasets": datasets, "langchain_openai": openai,
                                      "ragas": ragas, "ragas.metrics": metrics}), \
             patch.object(ragas_support.importlib.util, "find_spec", return_value=object()):
            self.assertIs(ragas_support.evaluate_data(data), result)

        datasets.Dataset.from_dict.assert_called_once_with(data)
        openai.ChatOpenAI.assert_called_once_with(model="gpt-4o-mini", temperature=0)
        ragas.evaluate.assert_called_once_with(
            dataset=dataset, metrics=[precision, recall], llm=llm, raise_exceptions=True
        )


if __name__ == "__main__":
    unittest.main()
