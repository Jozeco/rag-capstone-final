import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

import main


class FakeEmbeddings(Embeddings):
    """Small deterministic embedding model for offline Chroma tests."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("mount") + lowered.count("rainier")),
            float(lowered.count("bond") + lowered.count("film")),
            float(len(lowered) % 101) / 100.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HybridRetrievalTests(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(main._normalize([]), [])
        self.assertEqual(main._normalize([2.0, 2.0]), [0.5, 0.5])
        self.assertEqual(
            main._normalize([1.0, 2.0, 3.0], invert=True),
            [1.0, 0.5, 0.0],
        )

    def test_fusion_rewards_articles_found_by_both_retrievers(self) -> None:
        retriever = object.__new__(main.HybridRetriever)
        retriever.paths = [Path("a.html"), Path("b.html"), Path("c.html")]
        retriever.path_by_name = {path.name: path for path in retriever.paths}
        retriever.top_k = 2
        retriever._bm25_topk = lambda query, k: [
            (retriever.paths[0], 10.0),
            (retriever.paths[1], 8.0),
            (retriever.paths[2], 1.0),
        ]
        retriever._vector_topk = lambda query, k: [
            (retriever.paths[1], 0.1),
            (retriever.paths[2], 0.5),
            (retriever.paths[0], 0.9),
        ]

        results = retriever.getTopK("test query")

        self.assertEqual([path.name for path, _ in results], ["b.html", "a.html"])

    def test_vector_index_builds_and_loads_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            article_dir = root / "articles"
            article_dir.mkdir()
            (article_dir / "Mount_Rainier.html").write_text(
                "<main id='content'><p>Mount Rainier is in Washington.</p></main>",
                encoding="utf-8",
            )
            (article_dir / "James_Bond.html").write_text(
                "<main id='content'><p>James Bond appears in many films.</p></main>",
                encoding="utf-8",
            )
            paths = sorted(article_dir.glob("*.html"))

            with (
                patch.object(main, "VECTOR_CACHE_PATH", root / "chroma"),
                patch.object(main, "_get_embeddings", return_value=FakeEmbeddings()),
            ):
                built = main._load_or_build_vector_index(paths)
                first = built.similarity_search("volcano mountain", k=1)
                self.assertEqual(first[0].metadata["source"], "Mount_Rainier.html")

                loaded = main._load_or_build_vector_index(paths)
                second = loaded.similarity_search("spy film", k=1)
                self.assertEqual(second[0].metadata["source"], "James_Bond.html")


if __name__ == "__main__":
    unittest.main()
