import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from web import create_handler


class FakeService:
    def list_team_names(self):
        return ["Flamengo", "Palmeiras"]

    def predict(self, home, away):
        if home == away:
            return {"ok": False, "error": "same_team"}
        if home not in self.list_team_names() or away not in self.list_team_names():
            return {
                "ok": False,
                "error": "unknown_team",
                "unknown": [home],
                "suggestions": {home: []},
            }
        return {
            "ok": True,
            "home": home,
            "away": away,
            "predicted": "draw",
            "probabilities": {"home win": 0.3, "draw": 0.4, "away win": 0.3},
            "lineups": {"home": [], "away": []},
            "lineup_strength": {"home": 0.8, "away": 0.7},
            "score": {
                "home_goals": 1,
                "away_goals": 1,
                "expected": {"home": 1.2, "away": 1.1},
            },
        }


class WebHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            create_handler(FakeService()),
        )
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=5)

    def _post_json(self, path, payload, content_type="application/json"):
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_homepage_serves_web_frontend(self):
        with urlopen(self.base_url, timeout=5) as response:
            page = response.read().decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("Escala Astro IA", page)
        self.assertIn('id="simulation-form"', page)

    def test_teams_endpoint_returns_available_clubs(self):
        with urlopen(f"{self.base_url}/api/teams", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["teams"], ["Flamengo", "Palmeiras"])
        self.assertEqual(payload["match_seasons"], "2007–2023")

    def test_simulation_endpoint_returns_prediction(self):
        status, payload = self._post_json(
            "/api/simulate",
            {"home": "Flamengo", "away": "Palmeiras"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["predicted"], "draw")
        self.assertEqual(payload["score"]["home_goals"], 1)

    def test_simulation_endpoint_rejects_the_same_club(self):
        status, payload = self._post_json(
            "/api/simulate",
            {"home": "Flamengo", "away": "Flamengo"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "same_team")

    def test_simulation_endpoint_rejects_non_json_content(self):
        status, payload = self._post_json(
            "/api/simulate",
            {"home": "Flamengo", "away": "Palmeiras"},
            content_type="text/plain",
        )

        self.assertEqual(status, 415)
        self.assertEqual(payload["error"], "unsupported_media_type")


if __name__ == "__main__":
    unittest.main()
