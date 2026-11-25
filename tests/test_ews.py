"""Layanan peringatan METAR + ECMWF."""
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import support  # noqa: F401
import app
import ews_service as ews

UTC = timezone.utc
NOW = datetime(2026, 11, 24, 16, 0, tzinfo=UTC)


def records(n=100, bad=(), end=NOW):
    rows = []
    for i in range(n):
        t = end - timedelta(minutes=30 * (n - 1 - i))
        rows.append({"slot_30min": t.strftime("%Y-%m-%d %H:%M:%S"), "wind_speed": 5, "temperature": 28, "dew_point": 25,
                     "pressure": 1010, "bad_weather": 1 if (n - 1 - i) in bad else 0})
    return rows


def ecmwf_payload(start=datetime(2026, 11, 23, 0, 0), hours=96):
    times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
    hourly = {"time": times}
    for day in (1, 2):
        for v in ews.EC_VARS:
            hourly[f"{v}_previous_day{day}"] = [float(i + 100 * day) if v == "precipitation" else 10.0 for i in range(hours)]
    return {"hourly": hourly}


class ParityTest(unittest.TestCase):
    """Penghitungan Python murni harus sama dengan XGBoost asli."""

    def test_probabilities_match_training(self):
        data = json.loads((Path(__file__).parent / "ecmwf_parity.json").read_text())
        self.assertEqual(len(data), 15)
        for key, d in data.items():
            kind, horizon = key.split("|")
            for row, expected in zip(d["X"], d["p"]):
                with self.subTest(key):
                    values = [math.nan if v is None else v for v in row]
                    self.assertAlmostEqual(ews.probability(kind, horizon, values), expected, places=5)


class AvailabilityTest(unittest.TestCase):
    def test_no_horizon_uses_a_run_that_is_not_published_yet(self):
        # Run terbaru yang sudah terbit bisa berumur sampai 13,6 jam.
        for horizon, day in ews.LEAD_DAY.items():
            with self.subTest(horizon):
                self.assertGreaterEqual(day * 24, int(horizon[:-1]) + 13.6)


class FeatureTest(unittest.TestCase):
    def test_storm_history(self):
        obs = ews.prepare(records(bad=(1, 5, 60)))
        self.assertEqual(ews.storm_features(obs), [0, 1, 2, 2, 2, 1])

    def test_storm_history_without_storm(self):
        self.assertEqual(ews.storm_features(ews.prepare(records())), [0, 0, 0, 0, 0, 96])

    def test_season_uses_target_time(self):
        a, b, c, d = ews.season_features(datetime(2026, 1, 1, 6, 0, tzinfo=UTC))
        self.assertAlmostEqual(a, math.sin(2 * math.pi / 365.25))
        self.assertAlmostEqual(c, 1.0)
        self.assertAlmostEqual(d, 0.0)

    def test_ecmwf_features_use_the_requested_lead_and_interpolate_half_hours(self):
        table = ews.EcmwfTable(ecmwf_payload())
        at = lambda day, h, m: dict(zip(ews.EC_NAMES, table.features(day, datetime(2026, 11, 24, h, m, tzinfo=UTC))))
        self.assertEqual(at(1, 6, 0)["ec_precipitation"], 130)
        self.assertEqual(at(2, 6, 0)["ec_precipitation"], 230)
        self.assertEqual(at(1, 6, 30)["ec_precipitation"], 130.5)
        self.assertEqual(at(1, 6, 0)["ec_precipitation_max7"], 133)

    def test_wind_components(self):
        payload = ecmwf_payload()
        n = len(payload["hourly"]["time"])
        payload["hourly"]["wind_direction_10m_previous_day1"] = [90.0] * n
        f = dict(zip(ews.EC_NAMES, ews.EcmwfTable(payload).features(1, datetime(2026, 11, 24, 6, tzinfo=UTC))))
        self.assertAlmostEqual(f["ec_u10"], -10)
        self.assertAlmostEqual(f["ec_v10"], 0)

    def test_missing_humidity_is_nan_but_missing_rain_is_unavailable(self):
        payload = ecmwf_payload()
        n = len(payload["hourly"]["time"])
        payload["hourly"]["relative_humidity_2m_previous_day1"] = [None] * n
        f = dict(zip(ews.EC_NAMES, ews.EcmwfTable(payload).features(1, datetime(2026, 11, 24, 6, tzinfo=UTC))))
        self.assertTrue(math.isnan(f["ec_relative_humidity_2m"]))
        payload["hourly"]["precipitation_previous_day1"] = [None] * n
        with self.assertRaises(LookupError):
            ews.EcmwfTable(payload).features(1, datetime(2026, 11, 24, 6, tzinfo=UTC))


class PredictTest(unittest.TestCase):
    def predict(self, recs, fetch):
        with mock.patch.object(ews, "fetch_ecmwf", **fetch):
            return ews.predict(recs)

    def test_uses_ecmwf_when_available(self):
        result = self.predict(records(), {"return_value": ecmwf_payload()})
        self.assertEqual([p["horizon"] for p in result["predictions"]], ["1h", "3h", "9h", "18h", "24h"])
        self.assertTrue(all(p["model"] == "LSTM, XGBoost, ECMWF IFS" for p in result["predictions"]))
        self.assertTrue(result["ecmwf_available"])
        self.assertEqual(result["predictions"][0]["target_utc"], "2026-11-24 17:00 UTC")

    def test_falls_back_to_metar_only(self):
        result = self.predict(records(), {"side_effect": OSError("offline")})
        self.assertTrue(all(p["model"] == "XGBoost, METAR saja" for p in result["predictions"]))
        self.assertFalse(result["ecmwf_available"])

    def test_status_follows_thresholds(self):
        result = self.predict(records(bad=range(12)), {"side_effect": OSError("offline")})
        for p in result["predictions"]:
            t = p["thresholds"]
            level = 2 if p["probability"] >= t["th_bahaya"] else 1 if p["probability"] >= t["th_waspada"] else 0
            self.assertEqual(p["status"]["level"], level)
        self.assertEqual(result["overall_status"]["level"], max(p["status"]["level"] for p in result["predictions"]))
        self.assertGreaterEqual(result["predictions"][0]["status"]["level"], 1)

    def test_incomplete_rows_are_skipped_and_dew_point_is_capped(self):
        recs = records()
        recs[40].update(temperature=None)
        recs[-1].update(dew_point=29)
        self.assertEqual(len(self.predict(recs, {"side_effect": OSError()})["predictions"]), 5)

    def test_needs_a_day_of_history(self):
        with self.assertRaises(ValueError):
            self.predict(records(n=20), {"side_effect": OSError()})

    def test_advice_is_not_about_airport_operations(self):
        for text in ews.ADVICE.values():
            self.assertNotRegex(text.lower(), r"penerbangan|runway|landasan|bandara")


class CacheTest(unittest.TestCase):
    def test_old_cache_is_not_used_when_source_is_down(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(ews, "CACHE_FILE", Path(folder) / "c.json"), \
                mock.patch.object(ews.urllib.request, "urlopen", side_effect=OSError("offline")):
            ews.CACHE_FILE.write_text(json.dumps({"_cached_at": 0, "data": ecmwf_payload()}))
            with self.assertRaises(OSError):
                ews.fetch_ecmwf()

    def test_request_uses_the_trained_model(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(ews, "CACHE_FILE", Path(folder) / "c.json"), \
                mock.patch.object(ews.urllib.request, "urlopen", side_effect=OSError("offline")) as call:
            with self.assertRaises(OSError):
                ews.fetch_ecmwf()
        self.assertIn("models=" + ews.BUNDLE["ecmwf_model"], call.call_args.args[0].full_url)


class EndpointTest(unittest.TestCase):
    def test_dashboard_payload(self):
        recs = records()
        with mock.patch.object(app, "get_sheet"), mock.patch.object(app, "get_recent_rows_from_sheet", return_value=(recs, 101)), \
                mock.patch.object(ews, "fetch_ecmwf", return_value=ecmwf_payload()):
            result = app.ews_status()
        p = result["predictions"][0]
        self.assertEqual(result["source"], "google_sheets")
        self.assertEqual(p["hybrid_probability"], round(p["probability"], 4))
        self.assertEqual(p["xgb"]["status"], p["status"]["label"])
        self.assertIsNotNone(p["lstm"]["probability"])

    def test_lstm_failure_does_not_hide_the_warning(self):
        with mock.patch.object(app, "get_sheet"), mock.patch.object(app, "get_recent_rows_from_sheet", return_value=(records(), 101)), \
                mock.patch.object(ews, "fetch_ecmwf", side_effect=OSError()), \
                mock.patch.object(app, "predict_lstm_from_gsheet", side_effect=ValueError("x")):
            result = app.ews_status()
        self.assertEqual(result["predictions"][0]["lstm"]["status"], "N/A")


if __name__ == "__main__":
    unittest.main()
