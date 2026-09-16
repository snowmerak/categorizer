import unittest
import math

from pydantic import ValidationError

from main import CategorizeRequest, ChoicesRequest, categorize, make_response, parse_categories


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

        result = categorize("고양이 이야기", parse_categories(categories), ranker)

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

        result = categorize("query", roots, BranchRanker(), top_k=2)

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
        result = categorize("query", roots, NoMatchRanker())

        self.assertEqual(result.path, ["Unclassified"])
        self.assertEqual(result.ranked_paths[0].path, ["Unclassified"])

    def test_top_k_is_bounded(self):
        for top_k in (0, 11):
            with self.subTest(top_k=top_k), self.assertRaises(ValidationError):
                CategorizeRequest(query="query", categories={"A": ["description"]}, top_k=top_k)


if __name__ == "__main__":
    unittest.main()
