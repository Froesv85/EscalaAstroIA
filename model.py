from bisect import bisect_right
from collections import defaultdict, deque
from math import exp, factorial, isfinite

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


LABELS = ("home win", "draw", "away win")
ROLE_GROUPS = (
    ("Goleiro", "goalkeeper", frozenset({"GK"}), 1),
    ("Defesa", "defender", frozenset({"CB", "LB", "RB", "WB"}), 4),
    ("Meio-campo", "midfielder", frozenset({"DM", "CM", "AM", "LM", "RM"}), 3),
    ("Ataque", "forward", frozenset({"FW", "RW", "LW"}), 3),
)
LINEUP_SLOTS = (
    ("Goleiro", "goalkeeper", frozenset({"GK"}), 1),
    ("Defesa", "defender", frozenset({"CB"}), 2),
    ("Defesa", "defender", frozenset({"LB", "WB"}), 1),
    ("Defesa", "defender", frozenset({"RB", "WB"}), 1),
    ("Meio-campo", "midfielder", frozenset({"DM", "CM", "AM", "LM", "RM"}), 3),
    ("Ataque", "forward", frozenset({"FW", "RW", "LW"}), 3),
)
MAX_SCORE_GOALS = 8
ELO_K = 20.0
HOME_ELO_ADVANTAGE = 55.0
SEASON_ELO_REGRESSION = 0.75
LINEUP_PROBABILITY_WEIGHT = 1.5
MIN_POSITION_SHARE = 0.25
MIN_LINEUP_MINUTES = 450.0


def _build_classifier():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=15,
        class_weight="balanced",
        n_jobs=1,
        random_state=0,
    )


COMPARISON_MODEL_OPTIONS = {
    "random_forest": {
        "label": "Random Forest",
        "description": "Ensemble de árvores, modelo usado atualmente pelo simulador.",
        "fixed_parameters": {"class_weight": "balanced", "random_state": 0},
        "parameters": {
            "n_estimators": {
                "label": "Número de árvores",
                "type": "integer",
                "min": 50,
                "max": 1000,
                "step": 50,
                "default": 300,
            },
            "max_depth": {
                "label": "Profundidade máxima",
                "type": "integer",
                "min": 2,
                "max": 24,
                "step": 1,
                "default": 8,
            },
            "min_samples_leaf": {
                "label": "Mínimo de amostras por folha",
                "type": "integer",
                "min": 1,
                "max": 50,
                "step": 1,
                "default": 15,
            },
        },
    },
    "logistic_regression": {
        "label": "Regressão Logística",
        "description": "Classificador linear multiclasse com regularização.",
        "fixed_parameters": {"class_weight": "balanced", "random_state": 0},
        "parameters": {
            "C": {
                "label": "Força inversa da regularização (C)",
                "type": "number",
                "min": 0.01,
                "max": 10,
                "step": 0.01,
                "default": 1.0,
            },
            "max_iter": {
                "label": "Máximo de iterações",
                "type": "integer",
                "min": 100,
                "max": 5000,
                "step": 100,
                "default": 1000,
            },
        },
    },
    "hist_gradient_boosting": {
        "label": "HistGradientBoosting",
        "description": "Boosting de árvores com histogramas para atributos numéricos.",
        "fixed_parameters": {"class_weight": "balanced", "random_state": 0},
        "parameters": {
            "max_iter": {
                "label": "Número de iterações",
                "type": "integer",
                "min": 50,
                "max": 500,
                "step": 25,
                "default": 150,
            },
            "learning_rate": {
                "label": "Taxa de aprendizado",
                "type": "number",
                "min": 0.01,
                "max": 0.3,
                "step": 0.01,
                "default": 0.1,
            },
            "max_leaf_nodes": {
                "label": "Máximo de folhas por árvore",
                "type": "integer",
                "min": 3,
                "max": 63,
                "step": 2,
                "default": 15,
            },
        },
    },
}


def _default_comparison_configurations():
    return {
        model_id: {
            name: specification["default"]
            for name, specification in model["parameters"].items()
        }
        for model_id, model in COMPARISON_MODEL_OPTIONS.items()
    }


def _comparison_estimators(configurations):
    estimators = {}
    for model_id, parameters in configurations.items():
        if model_id == "random_forest":
            estimators[model_id] = RandomForestClassifier(
                **parameters,
                class_weight="balanced",
                n_jobs=1,
                random_state=0,
            )
        elif model_id == "logistic_regression":
            estimators[model_id] = LogisticRegression(
                **parameters,
                class_weight="balanced",
                random_state=0,
            )
        else:
            estimators[model_id] = HistGradientBoostingClassifier(
                **parameters,
                class_weight="balanced",
                random_state=0,
            )
    return estimators


def _empty_team_stats():
    return {"played": 0, "points": 0, "goals_for": 0, "goals_against": 0}


def _rate(stats, key, default=0.0):
    if not stats or stats["played"] == 0:
        return default
    return stats[key] / stats["played"]


def _poisson_probability(goals, expected):
    return exp(-expected) * expected ** goals / factorial(goals)


class Model:
    def __init__(self, repository):
        self.repo = repository
        self.clf = _build_classifier()
        self._loaded = False
        self._trained = False
        self._players = None
        self._matches = None
        self._features = None
        self._labels = None
        self._seasons = None
        self._latest_stats = None
        self._recent_points = None
        self._elo = None
        self._rosters = None
        self._league_home_goals = 1.3
        self._league_away_goals = 1.1

    def _load(self):
        if self._loaded:
            return

        self._players = self.repo.get_players()
        self._matches = self.repo.get_matches()
        (
            self._features,
            self._labels,
            self._seasons,
            self._latest_stats,
            self._recent_points,
            self._elo,
        ) = self._build_match_history(self._matches)
        self._rosters = self._build_rosters(self._players)
        if self._matches:
            self._league_home_goals = (
                sum(match["home_goals"] for match in self._matches) / len(self._matches)
            )
            self._league_away_goals = (
                sum(match["away_goals"] for match in self._matches) / len(self._matches)
            )
        self._loaded = True

    @staticmethod
    def _build_match_history(matches):
        grouped = defaultdict(list)
        for match in matches:
            grouped[(match["season"], match["round"])].append(match)

        ratings = defaultdict(lambda: 1500.0)
        recent_points = defaultdict(lambda: deque(maxlen=5))
        latest_stats = {}
        season_stats = defaultdict(_empty_team_stats)
        features, labels, seasons = [], [], []
        previous_season = None

        for season, round_number in sorted(grouped):
            if season != previous_season:
                if previous_season is not None:
                    seasons_without_matches = season - previous_season
                    for _ in range(seasons_without_matches):
                        for team in list(ratings):
                            ratings[team] = 1500.0 + (
                                ratings[team] - 1500.0
                            ) * SEASON_ELO_REGRESSION
                    if seasons_without_matches > 1:
                        recent_points = defaultdict(lambda: deque(maxlen=5))
                season_stats = defaultdict(_empty_team_stats)
                previous_season = season

            round_matches = grouped[(season, round_number)]
            pending_updates = defaultdict(float)
            round_results = []

            for match in round_matches:
                home, away = match["home_team"], match["away_team"]
                home_stats = season_stats[home]
                away_stats = season_stats[away]
                home_recent = (
                    sum(recent_points[home]) / (3.0 * len(recent_points[home]))
                    if recent_points[home]
                    else 0.5
                )
                away_recent = (
                    sum(recent_points[away]) / (3.0 * len(recent_points[away]))
                    if recent_points[away]
                    else 0.5
                )
                elo_difference = (
                    ratings[home] + HOME_ELO_ADVANTAGE - ratings[away]
                ) / 400.0

                features.append([
                    elo_difference,
                    _rate(home_stats, "points", 1.5) / 3.0
                    - _rate(away_stats, "points", 1.5) / 3.0,
                    home_recent - away_recent,
                    _rate(home_stats, "goals_for")
                    - _rate(home_stats, "goals_against")
                    - _rate(away_stats, "goals_for")
                    + _rate(away_stats, "goals_against"),
                ])
                labels.append(match["result"])
                seasons.append(season)

                expected_home = 1.0 / (
                    1.0 + 10.0 ** (
                        (ratings[away] - (ratings[home] + HOME_ELO_ADVANTAGE)) / 400.0
                    )
                )
                if match["result"] == "home win":
                    actual_home = 1.0
                    home_points, away_points = 3, 0
                elif match["result"] == "away win":
                    actual_home = 0.0
                    home_points, away_points = 0, 3
                else:
                    actual_home = 0.5
                    home_points, away_points = 1, 1
                rating_change = ELO_K * (actual_home - expected_home)
                pending_updates[home] += rating_change
                pending_updates[away] -= rating_change
                round_results.append((match, home_points, away_points))

            for match, home_points, away_points in round_results:
                home, away = match["home_team"], match["away_team"]
                home_stats = season_stats[home]
                away_stats = season_stats[away]
                home_stats["played"] += 1
                home_stats["points"] += home_points
                home_stats["goals_for"] += match["home_goals"]
                home_stats["goals_against"] += match["away_goals"]
                away_stats["played"] += 1
                away_stats["points"] += away_points
                away_stats["goals_for"] += match["away_goals"]
                away_stats["goals_against"] += match["home_goals"]
                recent_points[home].append(home_points)
                recent_points[away].append(away_points)
                latest_stats[home] = home_stats.copy()
                latest_stats[away] = away_stats.copy()

            for team, change in pending_updates.items():
                ratings[team] += change

        return (
            np.asarray(features, dtype=float),
            np.asarray(labels),
            np.asarray(seasons, dtype=int),
            latest_stats,
            {team: list(points) for team, points in recent_points.items()},
            dict(ratings),
        )

    @staticmethod
    def _role_rating(player, role):
        minutes = player["minutes"]
        if role == "goalkeeper":
            return minutes / 90.0

        if minutes <= 0:
            return 0.0
        per90 = {
            key: player[key] * 90.0 / minutes
            for key in (
                "goals",
                "assists",
                "xg",
                "xag",
                "npxg",
                "sca",
                "gca",
                "progressive_passes",
                "progressive_carries",
            )
        }
        if role == "forward":
            weights = {
                "goals": 1.0,
                "assists": 0.7,
                "xg": 0.3,
                "xag": 0.3,
                "sca": 0.02,
                "gca": 0.04,
            }
        elif role == "midfielder":
            weights = {
                "goals": 0.5,
                "assists": 0.8,
                "xg": 0.2,
                "xag": 0.45,
                "sca": 0.02,
                "progressive_passes": 0.015,
                "progressive_carries": 0.01,
            }
        else:
            weights = {
                "goals": 0.25,
                "assists": 0.35,
                "sca": 0.01,
                "progressive_passes": 0.015,
                "progressive_carries": 0.015,
            }
        return sum(per90[key] * weight for key, weight in weights.items())

    @classmethod
    def _build_rosters(cls, player_rows):
        grouped = {}
        for row in player_rows:
            key = (row["team"], row["name"].casefold())
            player = grouped.setdefault(key, {
                "name": row["name"],
                "team": row["team"],
                "positions": set(),
                "minutes": 0.0,
                "position_minutes": defaultdict(float),
                "goals": 0.0,
                "assists": 0.0,
                "xg": 0.0,
                "xag": 0.0,
                "npxg": 0.0,
                "sca": 0.0,
                "gca": 0.0,
                "progressive_passes": 0.0,
                "progressive_carries": 0.0,
            })
            minutes = row["minutes"]
            player["minutes"] += minutes
            for position in row["positions"]:
                player["positions"].add(position)
                player["position_minutes"][position] += minutes / len(row["positions"])
            for field in (
                "goals",
                "assists",
                "xg",
                "xag",
                "npxg",
                "sca",
                "gca",
                "progressive_passes",
                "progressive_carries",
            ):
                player[field] += row[field]

        players_by_team = defaultdict(list)
        for player in grouped.values():
            if not player["positions"]:
                continue
            player["position"] = max(
                player["position_minutes"],
                key=player["position_minutes"].get,
                default=sorted(player["positions"])[0],
            )
            player["role_ratings"] = {}
            player["eligible_roles"] = set()
            for _, role, positions, _ in ROLE_GROUPS:
                role_minutes = sum(
                    player["position_minutes"].get(position, 0.0)
                    for position in positions
                )
                if (
                    player["minutes"] > 0
                    and role_minutes / player["minutes"] >= MIN_POSITION_SHARE
                ):
                    player["eligible_roles"].add(role)
                    player["role_ratings"][role] = cls._role_rating(player, role)
            players_by_team[player["team"]].append(player)

        for _, role, positions, _ in ROLE_GROUPS:
            pool = [
                player["role_ratings"][role]
                for players in players_by_team.values()
                for player in players
                if role in player["role_ratings"]
            ]
            pool.sort()
            for players in players_by_team.values():
                for player in players:
                    if role in player["role_ratings"]:
                        player.setdefault("role_percentiles", {})[role] = (
                            bisect_right(pool, player["role_ratings"][role]) / len(pool)
                        )

        return dict(players_by_team)

    def _feature_for_teams(self, home, away):
        home_stats = self._latest_stats.get(home)
        away_stats = self._latest_stats.get(away)
        home_recent = self._recent_points.get(home, [])
        away_recent = self._recent_points.get(away, [])
        home_recent_rate = sum(home_recent) / (3.0 * len(home_recent)) if home_recent else 0.5
        away_recent_rate = sum(away_recent) / (3.0 * len(away_recent)) if away_recent else 0.5
        home_elo = self._elo.get(home, 1500.0)
        away_elo = self._elo.get(away, 1500.0)

        return [
            (home_elo + HOME_ELO_ADVANTAGE - away_elo) / 400.0,
            _rate(home_stats, "points", 1.5) / 3.0
            - _rate(away_stats, "points", 1.5) / 3.0,
            home_recent_rate - away_recent_rate,
            _rate(home_stats, "goals_for")
            - _rate(home_stats, "goals_against")
            - _rate(away_stats, "goals_for")
            + _rate(away_stats, "goals_against"),
        ]

    def train(self):
        self._load()
        missing_labels = set(LABELS).difference(self._labels)
        if missing_labels:
            raise ValueError(
                "O treino precisa conter as três classes de resultado; ausentes: "
                + ", ".join(sorted(missing_labels))
            )
        self.clf.fit(self._features, self._labels)
        self._trained = True
        return self

    def evaluate(self):
        self._load()
        seasons = sorted(set(self._seasons.tolist()))
        if len(seasons) < 3:
            raise ValueError("A avaliação temporal precisa de pelo menos três temporadas.")

        test_seasons = seasons[-2:]
        train_mask = ~np.isin(self._seasons, test_seasons)
        test_mask = np.isin(self._seasons, test_seasons)
        missing_labels = set(LABELS).difference(self._labels[train_mask])
        if missing_labels:
            raise ValueError(
                "O conjunto temporal de treino não contém todas as classes: "
                + ", ".join(sorted(missing_labels))
            )

        estimator = _build_classifier()
        estimator.fit(self._features[train_mask], self._labels[train_mask])
        predicted = estimator.predict(self._features[test_mask])
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(self._features[train_mask], self._labels[train_mask])

        return {
            "train_seasons": [int(season) for season in seasons[:-2]],
            "test_seasons": [int(season) for season in test_seasons],
            "samples": int(test_mask.sum()),
            "accuracy": float(accuracy_score(self._labels[test_mask], predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(self._labels[test_mask], predicted)
            ),
            "macro_f1": float(f1_score(
                self._labels[test_mask],
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )),
            "baseline_accuracy": float(
                accuracy_score(self._labels[test_mask], baseline.predict(self._features[test_mask]))
            ),
            "confusion_matrix": confusion_matrix(
                self._labels[test_mask],
                predicted,
                labels=LABELS,
            ).tolist(),
            "labels": list(LABELS),
        }

    @staticmethod
    def _result_from_score(home_goals, away_goals):
        if home_goals > away_goals:
            return "home win"
        if away_goals > home_goals:
            return "away win"
        return "draw"

    @staticmethod
    def comparison_options():
        return {
            model_id: {
                "label": model["label"],
                "description": model["description"],
                "fixed_parameters": model["fixed_parameters"].copy(),
                "parameters": {
                    name: specification.copy()
                    for name, specification in model["parameters"].items()
                },
            }
            for model_id, model in COMPARISON_MODEL_OPTIONS.items()
        }

    @staticmethod
    def _normalize_comparison_configurations(configurations):
        defaults = _default_comparison_configurations()
        if configurations is None:
            return defaults
        if not isinstance(configurations, dict) or not configurations:
            raise ValueError("Selecione ao menos um modelo para comparar.")

        unknown_models = set(configurations).difference(COMPARISON_MODEL_OPTIONS)
        if unknown_models:
            raise ValueError(
                "Modelo de comparação desconhecido: "
                + ", ".join(sorted(unknown_models))
            )

        normalized = {}
        for model_id, values in configurations.items():
            if not isinstance(values, dict):
                raise ValueError(f"Parâmetros inválidos para {model_id}.")
            model_options = COMPARISON_MODEL_OPTIONS[model_id]
            definitions = model_options["parameters"]
            unknown_parameters = set(values).difference(definitions)
            if unknown_parameters:
                raise ValueError(
                    f"Parâmetro desconhecido para {model_options['label']}: "
                    + ", ".join(sorted(unknown_parameters))
                )

            model_values = defaults[model_id].copy()
            for name, value in values.items():
                specification = definitions[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"O parâmetro {name} deve ser numérico.")
                if specification["type"] == "integer":
                    if not isinstance(value, int):
                        raise ValueError(f"O parâmetro {name} deve ser um inteiro.")
                    normalized_value = value
                else:
                    normalized_value = float(value)
                    if not isfinite(normalized_value):
                        raise ValueError(f"O parâmetro {name} deve ser finito.")
                if not specification["min"] <= normalized_value <= specification["max"]:
                    raise ValueError(
                        f"O parâmetro {name} deve estar entre "
                        f"{specification['min']} e {specification['max']}."
                    )
                model_values[name] = normalized_value
            normalized[model_id] = model_values
        return normalized

    @staticmethod
    def _comparison_metrics(actual, predicted):
        if not actual:
            return {
                "samples": 0,
                "correct_predictions": 0,
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "confusion_matrix": [[0, 0, 0] for _ in LABELS],
            }
        matrix = confusion_matrix(actual, predicted, labels=LABELS)
        class_support = matrix.sum(axis=1)
        represented_classes = class_support > 0
        return {
            "samples": len(actual),
            "correct_predictions": sum(
                expected == prediction
                for expected, prediction in zip(actual, predicted)
            ),
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(
                np.mean(
                    np.diag(matrix)[represented_classes]
                    / class_support[represented_classes]
                )
            ),
            "macro_f1": float(f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )),
            "confusion_matrix": matrix.tolist(),
        }

    def compare_fixtures(self, rounds, configurations=None):
        self._load()
        if not self._trained:
            self.train()
        if not isinstance(rounds, list) or not rounds:
            raise ValueError("O comparativo precisa conter ao menos uma rodada.")

        configurations = self._normalize_comparison_configurations(configurations)
        missing_labels = set(LABELS).difference(self._labels)
        if missing_labels:
            raise ValueError(
                "O treino para comparação precisa conter as três classes: "
                + ", ".join(sorted(missing_labels))
            )
        estimators = _comparison_estimators(configurations)
        for estimator in estimators.values():
            estimator.fit(self._features, self._labels)

        seen_rounds = set()
        completed_history = []
        model_rounds = {model_id: [] for model_id in estimators}
        for round_data in sorted(rounds, key=lambda item: item["number"]):
            round_number = int(round_data["number"])
            if round_number < 1 or round_number in seen_rounds:
                raise ValueError(f"Número de rodada inválido ou duplicado: {round_number}.")
            seen_rounds.add(round_number)

            fixtures = round_data.get("matches")
            if not isinstance(fixtures, list) or not fixtures:
                raise ValueError(f"A rodada {round_number} não contém jogos.")

            round_matches = []
            display_matches = []
            for fixture in fixtures:
                home_display = fixture.get("home")
                away_display = fixture.get("away")
                status = fixture.get("status")
                if (
                    not isinstance(home_display, str)
                    or not home_display.strip()
                    or not isinstance(away_display, str)
                    or not away_display.strip()
                    or home_display.strip() == away_display.strip()
                    or status not in ("finished", "scheduled")
                ):
                    raise ValueError(f"Jogo inválido na rodada {round_number}.")

                home = self.repo.canonicalize_team_name(home_display.strip())
                away = self.repo.canonicalize_team_name(away_display.strip())
                score = fixture.get("score")
                if status == "finished":
                    if (
                        not isinstance(score, dict)
                        or not isinstance(score.get("home"), int)
                        or isinstance(score.get("home"), bool)
                        or not isinstance(score.get("away"), int)
                        or isinstance(score.get("away"), bool)
                        or score["home"] < 0
                        or score["away"] < 0
                    ):
                        raise ValueError(
                            f"Placar inválido no jogo {home_display} x {away_display}."
                        )
                    actual = self._result_from_score(score["home"], score["away"])
                    round_matches.append({
                        "season": 2026,
                        "round": round_number,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": score["home"],
                        "away_goals": score["away"],
                        "result": actual,
                    })
                else:
                    actual = None
                    round_matches.append({
                        "season": 2026,
                        "round": round_number,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": 0,
                        "away_goals": 0,
                        "result": "draw",
                    })
                display_matches.append((fixture, home_display.strip(), away_display.strip(), actual))

            history_size = len(self._matches) + len(completed_history)
            history_features, *_ = self._build_match_history(
                self._matches + completed_history + round_matches
            )
            round_features = history_features[history_size:history_size + len(round_matches)]
            for model_id, estimator in estimators.items():
                probability_rows = estimator.predict_proba(round_features)
                evaluated_matches = []
                for index, (fixture, home, away, actual) in enumerate(display_matches):
                    probabilities = {
                        str(label): float(value)
                        for label, value in zip(
                            estimator.classes_,
                            probability_rows[index],
                        )
                    }
                    predicted = max(LABELS, key=probabilities.get)
                    evaluated = {
                        "home": home,
                        "away": away,
                        "status": fixture["status"],
                        "predicted": predicted,
                        "probabilities": probabilities,
                        "confidence": probabilities[predicted],
                        "score": fixture.get("score") if actual is not None else None,
                        "actual": actual,
                        "correct": predicted == actual if actual is not None else None,
                    }
                    if fixture.get("kickoff"):
                        evaluated["kickoff"] = fixture["kickoff"]
                    evaluated_matches.append(evaluated)

                completed_count = sum(
                    match["status"] == "finished" for match in evaluated_matches
                )
                actual = [
                    match["actual"]
                    for match in evaluated_matches
                    if match["status"] == "finished"
                ]
                predicted = [
                    match["predicted"]
                    for match in evaluated_matches
                    if match["status"] == "finished"
                ]
                model_rounds[model_id].append({
                    "number": round_number,
                    "total_matches": len(evaluated_matches),
                    "completed_matches": completed_count,
                    "matches": evaluated_matches,
                    "metrics": self._comparison_metrics(actual, predicted),
                })

            completed_history.extend(
                match
                for fixture, match in zip(fixtures, round_matches)
                if fixture["status"] == "finished"
            )

        total_count = sum(
            round_data["total_matches"]
            for round_data in next(iter(model_rounds.values()))
        )
        completed_count = sum(
            round_data["completed_matches"]
            for round_data in next(iter(model_rounds.values()))
        )
        comparison_models = {}
        for model_id, estimator in estimators.items():
            evaluated_matches = [
                match
                for round_data in model_rounds[model_id]
                for match in round_data["matches"]
            ]
            actual = [
                match["actual"]
                for match in evaluated_matches
                if match["status"] == "finished"
            ]
            predicted = [
                match["predicted"]
                for match in evaluated_matches
                if match["status"] == "finished"
            ]
            summary = self._comparison_metrics(actual, predicted)
            comparison_models[model_id] = {
                "label": COMPARISON_MODEL_OPTIONS[model_id]["label"],
                "parameters": {
                    **configurations[model_id],
                    **COMPARISON_MODEL_OPTIONS[model_id]["fixed_parameters"],
                },
                "summary": {
                    **summary,
                    "total_matches": total_count,
                    "pending_matches": total_count - completed_count,
                },
                "rounds": model_rounds[model_id],
            }

        return {
            "model_options": self.comparison_options(),
            "models": comparison_models,
            "summary": {
                "total_matches": total_count,
                "completed_matches": completed_count,
                "pending_matches": total_count - completed_count,
            },
        }

    def _recommend_lineup(self, team):
        players = self._rosters.get(team, [])
        selected = []
        selected_ids = set()

        slot_candidates = {}
        for index, (_, role, positions, _) in enumerate(LINEUP_SLOTS):
            slot_candidates[index] = [
                player for player in players
                if role in player["eligible_roles"]
                and player["minutes"] >= MIN_LINEUP_MINUTES
                and sum(
                    player["position_minutes"].get(position, 0.0)
                    for position in positions
                ) / player["minutes"] >= MIN_POSITION_SHARE
            ]
            slot_candidates[index].sort(
                key=lambda player: (
                    player["role_percentiles"][role],
                    player["minutes"],
                ),
                reverse=True,
            )

        role_specs = sorted(
            enumerate(LINEUP_SLOTS),
            key=lambda item: (
                len(slot_candidates[item[0]]) - item[1][3],
                item[0],
            ),
        )
        for index, (display_role, role, positions, needed) in role_specs:
            candidates = [
                player for player in slot_candidates[index]
                if player["name"].casefold() not in selected_ids
            ]
            if len(candidates) < needed:
                raise ValueError(
                    f"Não há jogadores suficientes com pelo menos "
                    f"{MIN_LINEUP_MINUTES:.0f} minutos para a função "
                    f"{display_role} do clube {team}."
                )
            candidates.sort(
                key=lambda player: (
                    player["role_percentiles"][role],
                    player["minutes"],
                ),
                reverse=True,
            )
            for player in candidates[:needed]:
                selected_ids.add(player["name"].casefold())
                selected.append(
                    self._lineup_player(player, display_role, role, positions)
                )

        if len(selected) < 11:
            raise ValueError(
                f"O elenco de {team} tem menos de 11 jogadores elegíveis para a escalação."
            )

        return selected[:11]

    @staticmethod
    def _lineup_player(player, display_role, role, positions):
        role_position = max(
            (
                position
                for position in player["position_minutes"]
                if position in positions
            ),
            key=player["position_minutes"].get,
        )
        return {
            "name": player["name"],
            "position": role_position,
            "role": display_role,
            "minutes": round(player["minutes"]),
            "rating": round(player["role_ratings"][role], 2),
            "percentile": round(player["role_percentiles"][role], 3),
        }

    @staticmethod
    def _lineup_quality(lineup):
        return sum(player["percentile"] for player in lineup) / len(lineup)

    @staticmethod
    def _predicted_score(home_expected, away_expected, outcome):
        candidates = []
        for home_goals in range(MAX_SCORE_GOALS + 1):
            for away_goals in range(MAX_SCORE_GOALS + 1):
                matches = (
                    (outcome == "home win" and home_goals > away_goals)
                    or (outcome == "draw" and home_goals == away_goals)
                    or (outcome == "away win" and home_goals < away_goals)
                )
                if matches:
                    likelihood = (
                        _poisson_probability(home_goals, home_expected)
                        * _poisson_probability(away_goals, away_expected)
                    )
                    candidates.append((likelihood, home_goals, away_goals))
        _, home_goals, away_goals = max(candidates)
        return {
            "home_goals": home_goals,
            "away_goals": away_goals,
            "expected": {
                "home": round(home_expected, 2),
                "away": round(away_expected, 2),
            },
        }

    def _simulate_score(self, home, away, home_quality, away_quality, outcome):
        home_stats = self._latest_stats.get(home)
        away_stats = self._latest_stats.get(away)
        prior_matches = 5.0
        home_attack = (
            home_stats["goals_for"] + prior_matches * self._league_home_goals
        ) / (home_stats["played"] + prior_matches) if home_stats else self._league_home_goals
        away_attack = (
            away_stats["goals_for"] + prior_matches * self._league_away_goals
        ) / (away_stats["played"] + prior_matches) if away_stats else self._league_away_goals
        home_defense = (
            home_stats["goals_against"] + prior_matches * self._league_away_goals
        ) / (home_stats["played"] + prior_matches) if home_stats else self._league_away_goals
        away_defense = (
            away_stats["goals_against"] + prior_matches * self._league_home_goals
        ) / (away_stats["played"] + prior_matches) if away_stats else self._league_home_goals

        home_expected = max(
            0.05,
            (0.55 * home_attack + 0.45 * away_defense)
            * 1.08
            * exp(0.25 * (home_quality - 0.5))
            * exp(-0.25 * (away_quality - 0.5)),
        )
        away_expected = max(
            0.05,
            (0.55 * away_attack + 0.45 * home_defense)
            * exp(0.25 * (away_quality - 0.5))
            * exp(-0.25 * (home_quality - 0.5)),
        )
        return self._predicted_score(home_expected, away_expected, outcome)

    def predict(self, home, away):
        self._load()
        if not self._trained:
            raise RuntimeError("Treine o modelo antes de solicitar uma previsão.")

        home_lineup = self._recommend_lineup(home) if home in self._rosters else []
        away_lineup = self._recommend_lineup(away) if away in self._rosters else []
        has_lineups = bool(home_lineup and away_lineup)
        home_quality = self._lineup_quality(home_lineup) if home_lineup else 0.5
        away_quality = self._lineup_quality(away_lineup) if away_lineup else 0.5

        raw_probabilities = self.clf.predict_proba(
            np.asarray([self._feature_for_teams(home, away)], dtype=float)
        )[0]
        base = {
            label: float(raw_probabilities[index])
            for index, label in enumerate(self.clf.classes_)
        }
        if has_lineups:
            quality_difference = home_quality - away_quality
            adjusted = {
                "home win": base["home win"] * exp(
                    LINEUP_PROBABILITY_WEIGHT * quality_difference
                ),
                "draw": base["draw"],
                "away win": base["away win"] * exp(
                    -LINEUP_PROBABILITY_WEIGHT * quality_difference
                ),
            }
            total = sum(adjusted.values())
            probabilities = {
                label: value / total for label, value in adjusted.items()
            }
        else:
            probabilities = base.copy()
        predicted = max(LABELS, key=probabilities.get)

        return {
            "predicted": predicted,
            "probabilities": probabilities,
            "model_probabilities": base,
            "lineup_adjustment_applied": has_lineups,
            "lineups": {"home": home_lineup, "away": away_lineup},
            "lineup_strength": {
                "home": round(home_quality, 3) if home_lineup else None,
                "away": round(away_quality, 3) if away_lineup else None,
            },
            "score": self._simulate_score(
                home,
                away,
                home_quality,
                away_quality,
                predicted,
            ),
        }
