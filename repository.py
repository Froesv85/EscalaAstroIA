import csv
import re
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "raw"

DATASETS = {
    "players": ("eduardopalmieri/brasileiro-player-stats-2024", "database.csv"),
    "matches": ("gustavomartino/brasileirao", "CampBras.csv"),
}

MATCH_TEAM_ALIASES = {
    "athleticopr": "athleticoparanaense",
    "atleticogo": "atleticogoianiense",
    "atleticomg": "atleticomineiro",
    "botafogo": "botafogorj",
    "bragantino": "redbullbragantino",
    "vasco": "vascodagama",
}

PLAYER_COLUMNS = {
    "name": "Jogador",
    "team": "Time",
    "position": "Pos.",
    "minutes": "Min.",
    "goals": "Gols",
    "assists": "Assis.",
    "xg": "xG",
    "xag": "xAG",
    "npxg": "npxG",
    "sca": "SCA",
    "gca": "GCA",
    "progressive_passes": "PrgP",
    "progressive_carries": "PrgC",
}


def normalize_name(value):
    normalized = unicodedata.normalize("NFKD", value.strip().casefold())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_name)


def _download_dataset(dataset_ref, archive_member, destination, refresh):
    if destination.exists() and not refresh:
        return destination

    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
    request = Request(url, headers={"User-Agent": "EscalaAstroIA/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            csv_bytes = archive.read(archive_member)
    except (URLError, zipfile.BadZipFile, KeyError) as error:
        raise RuntimeError(
            f"Não foi possível baixar {dataset_ref} do Kaggle "
            f"ou encontrar o arquivo {archive_member}: {error}"
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(csv_bytes)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames or [], list(reader)


def _require_columns(columns, required, file_name):
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            f"O arquivo {file_name} não contém as colunas obrigatórias: "
            + ", ".join(missing)
        )


def _parse_number(value, field, line_number):
    if value is None or not value.strip():
        return 0.0
    try:
        return float(value.strip().replace(",", "."))
    except ValueError as error:
        raise ValueError(
            f"Valor numérico inválido na linha {line_number}, coluna {field}: {value!r}"
        ) from error


class Repository:
    """Downloads and normalizes the public Kaggle datasets used by the app."""

    def __init__(self, data_dir=DEFAULT_DATA_DIR, refresh=False):
        self.data_dir = Path(data_dir)
        self.players_path = _download_dataset(
            *DATASETS["players"],
            self.data_dir / "database.csv",
            refresh,
        )
        self.matches_path = _download_dataset(
            *DATASETS["matches"],
            self.data_dir / "CampBras.csv",
            refresh,
        )
        self._players = self._load_players()
        self._teams_by_key = {
            normalize_name(team): team for team in {player["team"] for player in self._players}
        }
        self._add_team_aliases()
        self._matches, self._data_quality = self._load_matches()

        match_teams = {
            team
            for match in self._matches
            for team in (match["home_team"], match["away_team"])
        }
        self._teams = sorted(
            {player["team"] for player in self._players}.intersection(match_teams),
            key=str.casefold,
        )
        if not self._teams:
            raise ValueError(
                "Os conjuntos do Kaggle não têm clubes em comum; "
                "confira os nomes e as versões dos arquivos."
            )

    def _add_team_aliases(self):
        canonical_by_key = dict(self._teams_by_key)
        for alias, canonical_key in MATCH_TEAM_ALIASES.items():
            canonical = canonical_by_key.get(canonical_key)
            if canonical:
                self._teams_by_key[alias] = canonical

    def _canonical_team(self, name):
        key = normalize_name(name)
        key = MATCH_TEAM_ALIASES.get(key, key)
        return self._teams_by_key.get(key, name.strip())

    def _load_players(self):
        columns, rows = _read_csv(self.players_path)
        required = list(PLAYER_COLUMNS.values())
        _require_columns(columns, required, self.players_path.name)

        players = []
        for line_number, row in enumerate(rows, start=2):
            name = (row[PLAYER_COLUMNS["name"]] or "").strip()
            team = (row[PLAYER_COLUMNS["team"]] or "").strip()
            position = (row[PLAYER_COLUMNS["position"]] or "").strip()
            if not name or not team or not position:
                continue

            player = {
                "name": name,
                "team": team,
                "positions": [code.strip() for code in position.split(",") if code.strip()],
            }
            for field, column in PLAYER_COLUMNS.items():
                if field not in ("name", "team", "position"):
                    player[field] = _parse_number(row[column], column, line_number)
            players.append(player)

        if not players:
            raise ValueError(f"O arquivo {self.players_path.name} não contém jogadores válidos.")
        return players

    def _load_matches(self):
        columns, rows = _read_csv(self.matches_path)
        required = ("season", "round", "hometeam", "goalsht", "goalsvt", "visitingteam")
        _require_columns(columns, required, self.matches_path.name)

        matches = []
        missing_scores = 0
        for line_number, row in enumerate(rows, start=2):
            if not (row["goalsht"] or "").strip() or not (row["goalsvt"] or "").strip():
                missing_scores += 1
                continue

            home_team = self._canonical_team(row["hometeam"])
            away_team = self._canonical_team(row["visitingteam"])
            if not home_team or not away_team or home_team == away_team:
                raise ValueError(f"Clubes inválidos na linha {line_number} de {self.matches_path.name}.")

            try:
                season = int(row["season"])
                round_number = int(row["round"])
                home_goals = int(row["goalsht"])
                away_goals = int(row["goalsvt"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Temporada, rodada ou placar inválido na linha {line_number} "
                    f"de {self.matches_path.name}."
                ) from error

            if home_goals < 0 or away_goals < 0:
                raise ValueError(f"Placar negativo na linha {line_number} de {self.matches_path.name}.")

            if home_goals > away_goals:
                result = "home win"
            elif away_goals > home_goals:
                result = "away win"
            else:
                result = "draw"

            matches.append({
                "season": season,
                "round": round_number,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
            })

        if not matches:
            raise ValueError(f"O arquivo {self.matches_path.name} não contém partidas válidas.")
        matches.sort(key=lambda match: (match["season"], match["round"]))
        return matches, {
            "matches_total": len(rows),
            "matches_used": len(matches),
            "matches_missing_score": missing_scores,
            "players_used": len(self._players),
        }

    def get_players(self):
        return [player.copy() for player in self._players]

    def get_matches(self):
        return [match.copy() for match in self._matches]

    def get_team_names(self):
        return list(self._teams)

    def resolve_team_name(self, name):
        canonical = self._teams_by_key.get(normalize_name(name))
        if canonical in self._teams:
            return canonical
        return None

    def canonicalize_team_name(self, name):
        return self._canonical_team(name)

    def get_data_quality(self):
        return self._data_quality.copy()
