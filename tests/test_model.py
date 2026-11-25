import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from model import MIN_LINEUP_MINUTES, Model


def _player(name, team, position):
    return {
        "name": name,
        "team": team,
        "positions": [position],
        "minutes": 900.0,
        "goals": 2.0,
        "assists": 1.0,
        "xg": 2.0,
        "xag": 1.0,
        "npxg": 1.5,
        "sca": 5.0,
        "gca": 1.0,
        "progressive_passes": 20.0,
        "progressive_carries": 10.0,
    }


class FakeRepository:
    def __init__(self, matches, players):
        self.matches = matches
        self.players = players

    def get_matches(self):
        return [match.copy() for match in self.matches]

    def get_players(self):
        return [player.copy() for player in self.players]

    def canonicalize_team_name(self, name):
        return name


class AlwaysHomeClassifier:
    classes_ = ("home win", "draw", "away win")

    def __init__(self):
        self.calls = []

    def fit(self, features, labels):
        return self

    def predict_proba(self, features):
        self.calls.append(features.copy())
        return np.tile([0.8, 0.1, 0.1], (len(features), 1))

class ModelTests(unittest.TestCase):
    def test_round_features_do_not_use_results_from_same_round(self):
        matches = [
            {
                "season": 2022,
                "round": 1,
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 0,
                "result": "home win",
            },
            {
                "season": 2022,
                "round": 1,
                "home_team": "C",
                "away_team": "D",
                "home_goals": 1,
                "away_goals": 1,
                "result": "draw",
            },
        ]

        features, _, _, _, _, _ = Model._build_match_history(matches)

        self.assertEqual(features[0].tolist(), features[1].tolist())

    def test_season_comparison_uses_only_previous_round_results(self):
        model = Model(FakeRepository([], []))
        model._loaded = True
        model._matches = []
        model._trained = True
        model._features = np.zeros((3, 4))
        model._labels = np.array(["home win", "draw", "away win"])
        model.clf = AlwaysHomeClassifier()

        with patch(
            "model._comparison_estimators",
            return_value={"random_forest": model.clf},
        ):
            comparison = model.compare_fixtures(
                [
                    {
                        "number": 1,
                        "matches": [
                            {
                                "home": "A",
                                "away": "B",
                                "status": "finished",
                                "score": {"home": 2, "away": 0},
                            },
                            {
                                "home": "C",
                                "away": "D",
                                "status": "finished",
                                "score": {"home": 0, "away": 0},
                            },
                        ],
                    },
                    {
                        "number": 2,
                        "matches": [
                            {
                                "home": "A",
                                "away": "B",
                                "status": "finished",
                                "score": {"home": 0, "away": 1},
                            },
                            {"home": "E", "away": "F", "status": "scheduled"},
                        ],
                    },
                ],
                {"random_forest": {}},
            )

        first_round_features, second_round_features = model.clf.calls
        comparison_model = comparison["models"]["random_forest"]
        self.assertTrue(np.array_equal(first_round_features[0], first_round_features[1]))
        self.assertFalse(np.array_equal(first_round_features[0], second_round_features[0]))
        self.assertEqual(comparison["summary"]["completed_matches"], 3)
        self.assertEqual(comparison["summary"]["pending_matches"], 1)
        self.assertEqual(comparison_model["summary"]["correct_predictions"], 1)
        self.assertAlmostEqual(comparison_model["summary"]["accuracy"], 1 / 3)
        self.assertIsNone(comparison_model["rounds"][1]["matches"][1]["actual"])

    def test_comparison_rejects_out_of_range_parameters(self):
        with self.assertRaisesRegex(ValueError, "entre 50 e 1000"):
            Model._normalize_comparison_configurations({
                "random_forest": {"n_estimators": 5},
            })

    def test_round_metrics_handle_predicted_classes_absent_from_actuals(self):
        metrics = Model._comparison_metrics(
            ["home win", "draw", "draw"],
            ["home win", "home win", "away win"],
        )

        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_prediction_runs_for_club_without_player_roster(self):
        model = Model(FakeRepository([], []))
        model._loaded = True
        model._trained = True
        model._rosters = {}
        model._latest_stats = {}
        model._recent_points = {}
        model._elo = {}
        model.clf = AlwaysHomeClassifier()

        result = model.predict("Mirassol", "Remo")

        self.assertEqual(result["predicted"], "home win")
        self.assertFalse(result["lineup_adjustment_applied"])
        self.assertEqual(result["lineups"], {"home": [], "away": []})
        self.assertEqual(result["lineup_strength"], {"home": None, "away": None})
        self.assertEqual(result["probabilities"], result["model_probabilities"])

    def test_recommended_lineup_has_eleven_players_in_a_433(self):
        players = []
        positions = (
            ["GK"]
            + ["CB", "LB", "RB", "CB"]
            + ["DM", "CM", "AM"]
            + ["FW", "LW", "RW"]
        )
        for team in ("Alpha", "Beta"):
            for number, position in enumerate(positions):
                players.append(_player(f"{team} player {number}", team, position))

        repository = FakeRepository([], players)
        model = Model(repository)
        model._rosters = Model._build_rosters(players)

        lineup = model._recommend_lineup("Alpha")

        self.assertEqual(len(lineup), 11)
        self.assertEqual(len({player["name"] for player in lineup}), 11)
        self.assertEqual(
            Counter(player["role"] for player in lineup),
            Counter({"Goleiro": 1, "Defesa": 4, "Meio-campo": 3, "Ataque": 3}),
        )
        selected_positions = Counter(player["position"] for player in lineup)
        self.assertEqual(selected_positions["CB"], 2)
        self.assertEqual(selected_positions["LB"], 1)
        self.assertEqual(selected_positions["RB"], 1)

    def test_short_position_appearance_does_not_make_player_eligible_for_role(self):
        players = [
            _player("Utility player", "Alpha", "RW"),
            _player("Utility player", "Alpha", "CB"),
        ]
        players[0]["minutes"] = 900.0
        players[1]["minutes"] = 45.0

        roster = Model._build_rosters(players)["Alpha"]
        utility_player = roster[0]

        self.assertNotIn("defender", utility_player["eligible_roles"])

    def test_selected_players_have_enough_minutes_for_recommended_lineup(self):
        players = []
        positions = (
            ["GK"]
            + ["CB", "LB", "RB", "CB"]
            + ["DM", "CM", "AM"]
            + ["FW", "LW", "RW"]
        )
        for number, position in enumerate(positions):
            player = _player(f"Player {number}", "Alpha", position)
            player["minutes"] = 449.0 if number == 5 else 900.0
            players.append(player)

        model = Model(FakeRepository([], players))
        model._rosters = Model._build_rosters(players)

        with self.assertRaisesRegex(ValueError, "pelo menos 450"):
            model._recommend_lineup("Alpha")


if __name__ == "__main__":
    unittest.main()
