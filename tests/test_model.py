import unittest
from collections import Counter

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
