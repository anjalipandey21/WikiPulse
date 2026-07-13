"""Focused mocked tests for local semantic candidate grouping."""

from datetime import date
import unittest
from unittest.mock import Mock, patch

from app.clustering.semantic_clustering import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    CandidateTopicCluster,
    MiniLMArticleEncoder,
    SemanticClusteringResult,
    group_candidate_topics,
)
from app.models import Article


def make_article(title: str, summary: str | None = None) -> Article:
    observed_date = date(2026, 7, 12)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=100,
        daily_views={observed_date: 100},
        summary=summary,
        analysis_start_date=date(2026, 7, 6),
        analysis_end_date=observed_date,
    )


class FakeEncoder:
    def __init__(self, embeddings: object) -> None:
        self.embeddings = embeddings
        self.encoded_texts: list[str] | None = None

    def encode(self, texts: object) -> object:
        self.encoded_texts = list(texts)  # type: ignore[arg-type]
        return self.embeddings


class SemanticClusteringTests(unittest.TestCase):
    def test_default_encoder_wraps_minilm_with_normalized_local_batches(self) -> None:
        encoded = [[1.0, 0.0], [0.0, 1.0]]

        with patch(
            "app.clustering.semantic_clustering.SentenceTransformer"
        ) as model_class:
            model = model_class.return_value
            model.encode.return_value = encoded
            encoder = MiniLMArticleEncoder()
            result = encoder.encode(["First", "Second"])

        model_class.assert_called_once_with(DEFAULT_EMBEDDING_MODEL, device="cpu")
        model.encode.assert_called_once_with(
            ["First", "Second"],
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.assertIs(result, encoded)

    def test_embeds_title_and_optional_summary_and_preserves_identity(self) -> None:
        articles = [
            make_article("Football final", "A championship association football match."),
            make_article("Quantum mechanics", None),
            make_article("Football championship", "The tournament final."),
            make_article("Ancient history", "The study of early civilizations."),
        ]
        encoder = FakeEncoder(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.99, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = group_candidate_topics(
            articles,
            encoder=encoder,
            similarity_threshold=0.8,
        )

        self.assertEqual(
            encoder.encoded_texts,
            [
                "Football final\nA championship association football match.",
                "Quantum mechanics",
                "Football championship\nThe tournament final.",
                "Ancient history\nThe study of early civilizations.",
            ],
        )
        self.assertIsInstance(result, SemanticClusteringResult)
        self.assertEqual(len(result.clusters), 1)
        self.assertIsInstance(result.clusters[0], CandidateTopicCluster)
        self.assertIs(result.clusters[0].articles[0], articles[0])
        self.assertIs(result.clusters[0].articles[1], articles[2])
        self.assertEqual(result.unclustered_articles, (articles[1], articles[3]))
        self.assertIs(result.unclustered_articles[0], articles[1])
        self.assertIs(result.unclustered_articles[1], articles[3])

    def test_similarity_threshold_controls_candidate_grouping(self) -> None:
        articles = [make_article("Alpha"), make_article("Beta")]
        embeddings = [[1.0, 0.0], [0.7, 0.714142842854285]]

        grouped = group_candidate_topics(
            articles,
            encoder=FakeEncoder(embeddings),
            similarity_threshold=0.6,
        )
        separated = group_candidate_topics(
            articles,
            encoder=FakeEncoder(embeddings),
            similarity_threshold=0.8,
        )

        self.assertEqual(grouped.clusters[0].articles, tuple(articles))
        self.assertEqual(separated.clusters, ())
        self.assertEqual(separated.unclustered_articles, tuple(articles))

    def test_minimum_cluster_size_keeps_small_groups_unclustered(self) -> None:
        articles = [
            make_article("Alpha"),
            make_article("Beta"),
            make_article("Gamma"),
        ]
        embeddings = [[1.0, 0.0], [0.99, 0.1], [0.98, 0.2]]

        result = group_candidate_topics(
            articles,
            encoder=FakeEncoder(embeddings),
            similarity_threshold=0.8,
            min_cluster_size=4,
        )

        self.assertEqual(result.clusters, ())
        self.assertEqual(result.unclustered_articles, tuple(articles))

    def test_cluster_order_members_and_ids_are_deterministic(self) -> None:
        articles = [
            make_article("Sports A"),
            make_article("Science A"),
            make_article("Sports B"),
            make_article("Science B"),
        ]
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.1],
            [0.1, 0.99],
        ]

        first = group_candidate_topics(
            articles,
            encoder=FakeEncoder(embeddings),
            similarity_threshold=0.8,
        )
        second = group_candidate_topics(
            articles,
            encoder=FakeEncoder(embeddings),
            similarity_threshold=0.8,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first.clusters[0].articles,
            (articles[0], articles[2]),
        )
        self.assertEqual(
            first.clusters[1].articles,
            (articles[1], articles[3]),
        )
        self.assertTrue(first.clusters[0].id.startswith("candidate-"))
        self.assertNotEqual(first.clusters[0].id, first.clusters[1].id)

    def test_rejects_invalid_embedding_count_shape_and_values(self) -> None:
        articles = [make_article("Alpha"), make_article("Beta")]
        invalid_cases = [
            (
                [[1.0, 0.0]],
                "1 embeddings for 2 articles",
            ),
            (
                [1.0, 0.0],
                "finite two-dimensional embedding matrix",
            ),
            (
                [[1.0, 0.0], [float("nan"), 1.0]],
                "finite two-dimensional embedding matrix",
            ),
        ]

        for embeddings, expected_message in invalid_cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    group_candidate_topics(
                        articles,
                        encoder=FakeEncoder(embeddings),
                    )

    def test_empty_and_single_inputs_do_not_load_or_call_an_encoder(self) -> None:
        encoder = Mock()

        self.assertEqual(
            group_candidate_topics([], encoder=encoder),
            SemanticClusteringResult((), ()),
        )
        article = make_article("Standalone")
        self.assertEqual(
            group_candidate_topics([article], encoder=encoder),
            SemanticClusteringResult((), (article,)),
        )
        encoder.encode.assert_not_called()

    def test_rejects_invalid_grouping_options(self) -> None:
        article = make_article("Standalone")

        for threshold in (-0.1, 1.0, float("nan"), True):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "similarity_threshold"):
                    group_candidate_topics(
                        [article],
                        similarity_threshold=threshold,  # type: ignore[arg-type]
                    )

        for minimum in (0, 1, True, 1.5):
            with self.subTest(minimum=minimum):
                with self.assertRaisesRegex(ValueError, "min_cluster_size"):
                    group_candidate_topics(
                        [article],
                        min_cluster_size=minimum,  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    unittest.main()
