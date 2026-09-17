import unittest
import math
from threading import Lock

from pydantic import ValidationError
from fastapi.testclient import TestClient

from category_scoring import CategoryInputTooLong, SiblingScore
from lfm import LfmCategoryScorer
from main import (
    CategorizeRequest,
    ChoicesRequest,
    QwenCategoryScorer,
    categorize,
    create_app,
    make_response,
    parse_categories,
)


class ChoicesResponseTests(unittest.TestCase):
    def test_probabilities_and_relative_percentages(self):
        response = make_response(
            "query", ["best", "worse", "equal"], [(10, 0), (0, 10), (10, 0)]
        )

        self.assertEqual([score.index for score in response.choices], [0, 1, 2])
        self.assertEqual([score.choice for score in response.choices], ["best", "worse", "equal"])
        self.assertAlmostEqual(sum(score.percentage for score in response.choices), 100)
        self.assertGreater(response.choices[0].percentage, response.choices[1].percentage)
        self.assertAlmostEqual(response.choices[0].percentage, response.choices[2].percentage)
        for score in response.choices:
            self.assertAlmostEqual(score.yes_probability + score.no_probability, 1)

    def test_extreme_logits_stay_finite(self):
        response = make_response("query", ["yes", "no"], [(1000, -1000), (-1000, 1000)])

        self.assertEqual(response.choices[0].yes_probability, 1)
        self.assertEqual(response.choices[1].yes_probability, 0)
        self.assertEqual(response.choices[0].percentage, 100)
        self.assertEqual(response.choices[1].percentage, 0)

    def test_rejects_empty_and_oversized_choice_lists(self):
        for choices in ([], [""], ["valid"] * 65):
            with self.subTest(choices=choices), self.assertRaises(ValidationError):
                ChoicesRequest(query="query", choices=choices)


class FakeRanker:
    def __init__(self):
        self.batches = []

    def score(self, query, documents, instruction):
        self.batches.append(documents)
        margins = {
            "동물": 4,
            "기계": -4,
            "동물 > 포유류": 3,
            "동물 > 조류": -3,
            "동물 > 포유류 > 고양이": 5,
            "동물 > 포유류 > 개": 1,
        }
        return [
            (margins[document.splitlines()[0].removeprefix("Category: ")], 0)
            for document in documents
        ]


class CategorizationTests(unittest.TestCase):
    def test_follows_best_path_and_scores_only_its_siblings(self):
        categories = {
            "동물": [
                "살아 있는 동물",
                {
                    "포유류": ["젖을 먹이는 동물", {"고양이": ["고양잇과"], "개": ["개과"]}],
                    "조류": ["새", {"독수리": ["맹금류"]}],
                },
            ],
            "기계": ["기계 장치"],
        }
        ranker = FakeRanker()

        result = categorize("고양이 이야기", parse_categories(categories), QwenCategoryScorer(ranker))

        self.assertEqual(result.path, ["동물", "포유류", "고양이"])
        self.assertEqual([len(batch) for batch in ranker.batches], [2, 2, 2])
        self.assertIn("젖을 먹이는 동물", ranker.batches[1][0])
        self.assertEqual([level.selected for level in result.levels], result.path)
        for level in result.levels:
            self.assertAlmostEqual(sum(item.percentage for item in level.candidates), 100)
            self.assertEqual(
                max(level.candidates, key=lambda item: item.percentage).choice,
                level.selected,
            )

    def test_rejects_malformed_tree(self):
        malformed = [
            {"동물": []},
            {"동물": ["설명", "자식"]},
            {"동물": ["설명", {"고양이": ["설명"]}, {"고양이": ["중복"]}]},
            {f"분류{index}": ["설명"] for index in range(65)},
        ]
        for categories in malformed:
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                parse_categories(categories)

    def test_path_products_can_reverse_the_greedy_root_choice(self):
        class BranchRanker:
            def score(self, query, documents, instruction):
                margins = {
                    "A": math.log(1.5),
                    "B": 0,
                    "A > A1": 0,
                    "A > A2": 0,
                    "B > B1": 0,
                }
                return [
                    (margins[document.splitlines()[0].removeprefix("Category: ")], 0)
                    for document in documents
                ]

        roots = parse_categories(
            {
                "A": ["first branch", {"A1": ["leaf"], "A2": ["leaf"]}],
                "B": ["second branch", {"B1": ["leaf"]}],
            }
        )

        result = categorize("query", roots, QwenCategoryScorer(BranchRanker()), top_k=2)

        self.assertGreater(result.levels[0].candidates[0].percentage, 50)
        self.assertEqual(result.path, ["B", "B1"])
        self.assertEqual([item.path for item in result.ranked_paths], [["B", "B1"], ["A", "A1"]])
        self.assertAlmostEqual(result.ranked_paths[0].percentage, 40)
        self.assertAlmostEqual(result.ranked_paths[1].percentage, 30)

    def test_unclassified_leaf_can_be_ranked(self):
        class NoMatchRanker:
            def score(self, query, documents, instruction):
                return [(-3, 0), (3, 0)]

        roots = parse_categories({"Animals": ["Animals"], "Unclassified": ["No category fits"]})
        result = categorize("query", roots, QwenCategoryScorer(NoMatchRanker()))

        self.assertEqual(result.path, ["Unclassified"])
        self.assertEqual(result.ranked_paths[0].path, ["Unclassified"])

    def test_prefilter_scores_only_selected_candidates_and_keeps_named_leaf(self):
        class RecordingRanker:
            def __init__(self):
                self.documents = []

            def score(self, query, documents, instruction):
                self.documents.extend(documents)
                return [(1, 0) if "Unclassified" in doc else (0, 0) for doc in documents]

        class SelectiveEmbedder:
            def __init__(self):
                self.queries = []

            def encode_query(self, query):
                self.queries.append(query)
                return query

            def select(self, query_vector, documents, names, top_n, keep_names):
                self.assertion = (query_vector, top_n, keep_names)
                return [0, 3]

        roots = parse_categories(
            {"A": ["a"], "B": ["b"], "C": ["c"], "Unclassified": ["none fit"]}
        )
        ranker = RecordingRanker()
        embedder = SelectiveEmbedder()
        result = categorize(
            "query",
            roots,
            QwenCategoryScorer(ranker),
            top_k=2,
            prefilter_top_n=2,
            prefilter_min_siblings=3,
            prefilter_keep_names={"Unclassified"},
            embedder_factory=lambda: embedder,
        )

        self.assertEqual(embedder.queries, ["query"])
        self.assertEqual(embedder.assertion, ("query", 2, {"Unclassified"}))
        self.assertEqual(len(ranker.documents), 2)
        self.assertEqual(result.search_mode, "prefiltered")
        self.assertEqual(result.omitted_candidates, 2)
        self.assertEqual(result.path, ["Unclassified"])
        self.assertEqual([score.index for score in result.levels[0].candidates], [0, 3])
        self.assertAlmostEqual(sum(path.percentage for path in result.ranked_paths), 100)

    def test_prefilter_is_skipped_for_narrow_levels(self):
        roots = parse_categories({"Animals": ["Animals"], "Unclassified": ["No category fits"]})

        class NoMatchRanker:
            def score(self, query, documents, instruction):
                return [(-3, 0), (3, 0)]

        result = categorize(
            "query", roots, QwenCategoryScorer(NoMatchRanker()), prefilter_top_n=1,
            embedder_factory=lambda: self.fail("embedder should not load"),
        )
        self.assertEqual(result.search_mode, "exact")
        self.assertEqual(result.omitted_candidates, 0)

    def test_default_prefilter_starts_at_32_siblings(self):
        request = CategorizeRequest(query="query", categories={"A": ["description"]})
        self.assertEqual(request.prefilter_top_n, 8)
        self.assertEqual(request.prefilter_min_siblings, 32)
        self.assertIn("분류불가", request.prefilter_keep_names)

        class FlatRanker:
            def score(self, query, documents, instruction):
                return [(0, 0)] * len(documents)

        class FirstEightEmbedder:
            def encode_query(self, query):
                return query

            def select(self, query_vector, documents, names, top_n, keep_names):
                return list(range(top_n))

        for count, expected_mode in ((31, "exact"), (32, "prefiltered")):
            roots = parse_categories({f"C{index}": ["description"] for index in range(count)})
            result = categorize(
                "query", roots, QwenCategoryScorer(FlatRanker()), prefilter_top_n=request.prefilter_top_n,
                prefilter_min_siblings=request.prefilter_min_siblings,
                embedder_factory=lambda: FirstEightEmbedder(),
            )
            self.assertEqual(result.search_mode, expected_mode)
            self.assertEqual(result.omitted_candidates, 24 if count == 32 else 0)

    def test_top_k_is_bounded(self):
        for top_k in (0, 11):
            with self.subTest(top_k=top_k), self.assertRaises(ValidationError):
                CategorizeRequest(query="query", categories={"A": ["description"]}, top_k=top_k)

    def test_prefilter_top_n_is_bounded(self):
        for top_n in (0, 65):
            with self.subTest(top_n=top_n), self.assertRaises(ValidationError):
                CategorizeRequest(
                    query="query", categories={"A": ["description"]},
                    prefilter_top_n=top_n,
                )

        for minimum in (1, 65):
            with self.subTest(minimum=minimum), self.assertRaises(ValidationError):
                CategorizeRequest(
                    query="query", categories={"A": ["description"]},
                    prefilter_min_siblings=minimum,
                )


class CategorizeApiTests(unittest.TestCase):
    def setUp(self):
        self.body = {
            "query": "A cat needs food",
            "categories": {
                "Pets": ["Cats and dogs"],
                "Vehicles": ["Cars and trucks"],
            },
        }

    def test_lfm_uses_only_lfm_and_omits_qwen_fields(self):
        class FakeLfm:
            model_name = "LiquidAI/test-router"

            def score(self, query, documents):
                return [SiblingScore(math.log(0.8)), SiblingScore(math.log(0.2))]

        app = create_app(
            ranker_factory=lambda: self.fail("Qwen must not load"),
            embedder_factory=lambda: self.fail("Embedding must not load"),
            lfm_factory=FakeLfm,
        )
        with TestClient(app) as client:
            response = client.post("/categorize", json={**self.body, "model": "lfm"})

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["model"], "LiquidAI/test-router")
        self.assertEqual(result["path"], ["Pets"])
        self.assertEqual(result["search_mode"], "exact")
        self.assertEqual(result["omitted_candidates"], 0)
        self.assertAlmostEqual(result["levels"][0]["candidates"][0]["percentage"], 80)
        self.assertNotIn("yes_logit", result["levels"][0]["candidates"][0])

    def test_default_qwen_uses_only_qwen_and_keeps_existing_fields(self):
        class FakeQwen:
            def score(self, query, documents, instruction):
                return [(2, 0), (0, 2)]

        app = create_app(
            ranker_factory=FakeQwen,
            lfm_factory=lambda: self.fail("LFM must not load"),
        )
        with TestClient(app) as client:
            response = client.post("/categorize", json=self.body)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["model"], "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(result["path"], ["Pets"])
        self.assertIn("yes_logit", result["levels"][0]["candidates"][0])
        self.assertIn("no_probability", result["levels"][0]["candidates"][0])

    def test_lfm_rejects_qwen_prefilter_options(self):
        app = create_app(
            ranker_factory=lambda: self.fail("Qwen must not load"),
            lfm_factory=lambda: self.fail("LFM must not load for invalid input"),
        )
        with TestClient(app) as client:
            response = client.post(
                "/categorize",
                json={**self.body, "model": "lfm", "prefilter_top_n": 2},
            )
        self.assertEqual(response.status_code, 422)

    def test_lfm_input_limit_returns_422(self):
        class TooLongLfm:
            model_name = "LiquidAI/test-router"

            def score(self, query, documents):
                raise CategoryInputTooLong("LFM input exceeds the configured token limit")

        app = create_app(lfm_factory=TooLongLfm)
        with TestClient(app) as client:
            response = client.post("/categorize", json={**self.body, "model": "lfm"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("token limit", response.json()["detail"])


class LfmAdapterTests(unittest.TestCase):
    def test_sorted_router_scores_map_back_to_original_category_order(self):
        class FakeTokenizer:
            def __call__(self, text):
                return {"input_ids": list(range(len(text.split())))}

        class FakeModel:
            @staticmethod
            def _prefix(routes):
                return "Categories: " + " ".join(routes) + " Text: "

            def route(self, query, routes, tokenizer):
                return [
                    {"route": routes[1], "score": 0.8},
                    {"route": routes[0], "score": 0.2},
                ]

        scorer = LfmCategoryScorer.__new__(LfmCategoryScorer)
        scorer.tokenizer = FakeTokenizer()
        scorer.model = FakeModel()
        scorer.max_length = 100
        scorer.lock = Lock()

        scores = scorer.score("cat", ["Cats", "Dogs"])
        self.assertAlmostEqual(math.exp(scores[0].log_probability), 0.2)
        self.assertAlmostEqual(math.exp(scores[1].log_probability), 0.8)

        scorer.max_length = 1
        with self.assertRaises(CategoryInputTooLong):
            scorer.score("cat", ["Cats", "Dogs"])


if __name__ == "__main__":
    unittest.main()
