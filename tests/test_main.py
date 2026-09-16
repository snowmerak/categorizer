import unittest

from pydantic import ValidationError

from main import ChoicesRequest, make_response


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


if __name__ == "__main__":
    unittest.main()
