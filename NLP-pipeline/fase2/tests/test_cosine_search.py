import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
from similarity.cosine_search import MotorBuscaCosseno


class TestMotorBuscaCosseno:
    def test_fit_and_search(self):
        docs = ["ola mundo", "python programacao", "linguagem natural"]
        vectors = np.array([
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ])
        engine = MotorBuscaCosseno("test", docs)
        engine.fit(vectors)

        results = engine.search(np.array([1, 0, 0, 0]), top_k=2)
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]
        assert "ola mundo" in results[0]["preview"]

    def test_search_all_top_k(self):
        docs = [f"doc{i}" for i in range(5)]
        vectors = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
        engine = MotorBuscaCosseno("test", docs)
        engine.fit(vectors)

        results = engine.search(np.array([1, 1, 0, 0, 0]), top_k=5)
        assert len(results) == 5

    def test_search_empty_engine(self):
        engine = MotorBuscaCosseno("test", ["doc1"])
        results = engine.search(np.array([1, 0]), top_k=5)
        assert results == []

    def test_preview_truncation(self):
        long_doc = "a" * 300
        docs = [long_doc]
        vectors = np.array([[1, 0]])
        engine = MotorBuscaCosseno("test", docs)
        engine.fit(vectors)

        results = engine.search(np.array([1, 0]), top_k=1)
        assert len(results[0]["preview"]) <= 203  # 200 + "..."
