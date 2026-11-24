"""Perlindungan endpoint tulis dan ketahanan inferensi."""
import os
import unittest
from unittest import mock

from fastapi import HTTPException

import support  # noqa: F401  (menyiapkan path dan lingkungan)
import app
import hybrid_service
import lstm_service


class AdminTokenTest(unittest.TestCase):
    def status(self, configured, sent):
        with mock.patch.dict(os.environ):
            os.environ.pop("ADMIN_TOKEN", None)
            if configured is not None:
                os.environ["ADMIN_TOKEN"] = configured
            try:
                app.require_admin(sent)
                return 200
            except HTTPException as error:
                return error.status_code

    def test_token_rules(self):
        self.assertEqual(self.status("rahasia", "rahasia"), 200)
        self.assertEqual(self.status("rahasia", "salah"), 401)
        self.assertEqual(self.status("rahasia", "sálah"), 401)
        self.assertEqual(self.status("rahasia", None), 401)
        self.assertEqual(self.status(None, "apa saja"), 503)
        self.assertEqual(self.status("", ""), 503)

    def test_write_routes_require_token(self):
        protected = {
            route.path for route in app.app.routes
            if any(d.call is app.require_admin
                   for d in getattr(getattr(route, "dependant", None), "dependencies", []))
        }
        self.assertEqual(protected, {"/api/forecast-7days/snapshot", "/api/forecast-7days/verify"})


class InferenceTest(unittest.TestCase):
    def test_lstm_skips_incomplete_observations(self):
        records = [
            {"slot_30min": f"2026-11-24 {h:02d}:{m:02d}:00", "wind_speed": 5, "temperature": 28,
             "dew_point": 25, "pressure": 1010}
            for h in range(8) for m in (0, 30)
        ]
        records[10] = {"slot_30min": "2026-11-24 05:00:00", "wind_speed": None, "temperature": None,
                       "dew_point": None, "pressure": None}
        result = lstm_service.predict_lstm_from_gsheet(records)
        self.assertEqual(result["observed_at_utc"], "2026-11-24 07:30 UTC")

    def test_hybrid_survives_one_unavailable_horizon(self):
        xgb = {"predictions": [{"horizon": "1h", "available": False, "message": "x"}]}
        lstm = {"predictions": [{
            "horizon": "1h", "available": True, "probability": 0.65,
            "status": {"level": 1, "label": "WASPADA", "color": "amber"}, "target_utc": "u", "target_wib": "w",
        }]}
        item = hybrid_service.compute_hybrid_predictions(xgb, lstm)["predictions"][0]
        self.assertEqual((item["status"]["level"], item["xgb"]["status"], item["target_utc"]), (1, "N/A", "u"))


if __name__ == "__main__":
    unittest.main()
