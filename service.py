from difflib import get_close_matches

from model import Model
from repository import Repository, normalize_name


class Service:
    """Coordinates the Kaggle repository, model, and interactive app."""

    def __init__(self, repository=None, refresh_data=False, season_teams=None):
        self.repo = repository or Repository(refresh=refresh_data)
        self.model = Model(self.repo)
        self._trained = False
        self._names = None
        self._season_teams = sorted(set(season_teams or []), key=str.casefold)
        self._season_teams_by_key = {
            normalize_name(team): team for team in self._season_teams
        }

    def _resolve_team(self, name):
        if not isinstance(name, str):
            return None
        canonical = self.repo.resolve_team_name(name)
        if canonical:
            return canonical
        season_team = self._season_teams_by_key.get(normalize_name(name))
        if season_team:
            return self.repo.canonicalize_team_name(season_team)
        return None

    def is_known(self, name):
        return self._resolve_team(name) is not None

    def suggest(self, name, limit=3):
        if self._names is None:
            self._names = self.list_team_names()
        normalized = name.casefold().strip()
        teams_by_key = {team.casefold(): team for team in self._names}
        matches = get_close_matches(
            normalized,
            list(teams_by_key),
            n=limit,
            cutoff=0.55,
        )
        return [teams_by_key[match] for match in matches]

    def setup(self):
        self.model.train()
        self._trained = True
        return self

    def list_team_names(self):
        return self._season_teams.copy() or self.repo.get_team_names()

    def evaluate(self):
        return self.model.evaluate()

    def compare_season(self, rounds, configurations=None):
        if not self._trained:
            self.setup()
        return self.model.compare_fixtures(rounds, configurations)

    def data_quality(self):
        return self.repo.get_data_quality()

    def predict(self, home_name, away_name):
        home = self._resolve_team(home_name)
        away = self._resolve_team(away_name)

        unknown = [
            name
            for name, team in ((home_name, home), (away_name, away))
            if team is None
        ]
        if unknown:
            return {
                "ok": False,
                "error": "unknown_team",
                "unknown": unknown,
                "suggestions": {name: self.suggest(name) for name in unknown},
            }
        if home == away:
            return {"ok": False, "error": "same_team"}

        if not self._trained:
            self.setup()

        result = self.model.predict(home, away)
        return {
            "ok": True,
            "home": home,
            "away": away,
            **result,
        }
