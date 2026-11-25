import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from service import Service


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MAX_REQUEST_BYTES = 4096
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/api/fixtures/2026": ("fixtures-2026.json", "application/json; charset=utf-8"),
}


def load_2026_fixtures():
    return json.loads(
        (WEB_DIR / "fixtures-2026.json").read_text(encoding="utf-8")
    )


def get_2026_team_names():
    fixtures = load_2026_fixtures()
    teams = {
        match[key].strip()
        for round_data in fixtures["rounds"]
        for match in round_data["matches"]
        for key in ("home", "away")
    }
    if len(teams) != 20:
        raise ValueError(
            f"A base de jogos de 2026 deveria conter 20 clubes, mas contém {len(teams)}."
        )
    return sorted(teams, key=str.casefold)


def create_handler(service):
    class RequestHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self' data:; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, status, payload):
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, content, "application/json; charset=utf-8")

        def _send_error_json(self, status, message, code):
            self._send_json(status, {"ok": False, "error": code, "message": message})

        def _send_comparison(self, configurations=None):
            try:
                fixtures = load_2026_fixtures()
                comparison = service.compare_season(
                    fixtures["rounds"],
                    configurations,
                )
            except (OSError, UnicodeDecodeError, KeyError, TypeError, ValueError, RuntimeError) as error:
                invalid_parameters = configurations is not None and isinstance(error, ValueError)
                self._send_error_json(
                    400 if invalid_parameters else 500,
                    f"Não foi possível gerar o comparativo de 2026: {error}",
                    "invalid_model_configuration" if invalid_parameters else "comparison_error",
                )
                return
            comparison.update({
                "updatedAt": fixtures["updatedAt"],
                "note": fixtures["note"],
                "sources": fixtures["sources"],
            })
            self._send_json(200, comparison)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/teams":
                self._send_json(200, {
                    "teams": service.list_team_names(),
                    "match_seasons": "2007–2023",
                    "player_season": "2024",
                })
                return

            if path == "/api/comparison/2026":
                self._send_comparison()
                return

            static_file = STATIC_FILES.get(path)
            if static_file:
                file_name, content_type = static_file
                try:
                    content = (WEB_DIR / file_name).read_bytes()
                except OSError as error:
                    self._send_error_json(
                        500,
                        f"Não foi possível carregar a interface web: {error}",
                        "static_file_error",
                    )
                    return
                self._send_bytes(200, content, content_type)
                return

            self._send_error_json(404, "Página não encontrada.", "not_found")

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/simulate", "/api/comparison/2026"):
                self._send_error_json(404, "Rota não encontrada.", "not_found")
                return

            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type != "application/json":
                self._send_error_json(
                    415,
                    "Envie os dados da simulação como application/json.",
                    "unsupported_media_type",
                )
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_error_json(400, "Content-Length inválido.", "invalid_request")
                return

            if content_length <= 0:
                self._send_error_json(400, "O corpo da requisição está vazio.", "invalid_request")
                return
            if content_length > MAX_REQUEST_BYTES:
                self._send_error_json(413, "A requisição excede o limite permitido.", "request_too_large")
                return

            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_error_json(400, "O corpo deve conter um JSON válido.", "invalid_json")
                return

            if not isinstance(payload, dict):
                self._send_error_json(400, "O JSON deve ser um objeto.", "invalid_request")
                return

            if path == "/api/comparison/2026":
                configurations = payload.get("models")
                if not isinstance(configurations, dict):
                    self._send_error_json(
                        400,
                        "Envie os modelos e parâmetros em um objeto 'models'.",
                        "invalid_request",
                    )
                    return
                self._send_comparison(configurations)
                return

            home = payload.get("home")
            away = payload.get("away")
            if not isinstance(home, str) or not home.strip():
                self._send_error_json(
                    400,
                    "Informe o clube mandante.",
                    "invalid_home_team",
                )
                return
            if not isinstance(away, str) or not away.strip():
                self._send_error_json(
                    400,
                    "Informe o clube visitante.",
                    "invalid_away_team",
                )
                return

            result = service.predict(home.strip(), away.strip())
            if not result["ok"]:
                if result["error"] == "same_team":
                    self._send_error_json(
                        400,
                        "Escolha dois clubes diferentes.",
                        "same_team",
                    )
                    return
                self._send_json(422, result)
                return

            self._send_json(200, result)

        def log_message(self, format_string, *args):
            super().log_message(format_string, *args)

    return RequestHandler


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inicia a interface web de simulação do Brasileirão."
    )
    parser.add_argument("--host", default="127.0.0.1", help="endereço de escuta")
    parser.add_argument("--port", type=int, default=8000, help="porta HTTP")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="baixa novamente os arquivos públicos do Kaggle",
    )
    args = parser.parse_args(argv)

    try:
        service = Service(
            refresh_data=args.refresh_data,
            season_teams=get_2026_team_names(),
        )
        service.setup()
        server = ThreadingHTTPServer((args.host, args.port), create_handler(service))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Não foi possível iniciar o frontend: {error}", file=sys.stderr)
        return 1

    print(f"Frontend disponível em http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
