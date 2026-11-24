"""Cache dan kejujuran label prospek 7 hari."""
import importlib.util
import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from support import frozen
import weekly_nwp_service as weekly


def nwp(days):
    hours = [f"{d}T{h:02d}:00" for d in days for h in range(24)]
    values = {
        "temperature_2m": 28, "dew_point_2m": 22, "relative_humidity_2m": 70, "surface_pressure": 1010,
        "wind_speed_10m": 5, "wind_direction_10m": 90, "wind_gusts_10m": 8, "precipitation": 0,
        "cape": 100, "weather_code": 1,
    }
    hourly = {name: [value] * len(hours) for name, value in values.items()}
    return {
        "hourly": {"time": hours, **hourly},
        "daily": {"time": days, "weather_code": [1] * len(days),
                  "precipitation_probability_max": [10 + i for i in range(len(days))]},
    }


class WeeklyCacheTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = Path(folder.name) / "c.json"
        patch = mock.patch.object(weekly, "CACHE_FILE", self.cache)
        patch.start()
        self.addCleanup(patch.stop)

    def store(self, age_seconds, days):
        self.cache.write_text(json.dumps({"_cached_at": time.time() - age_seconds, "data": nwp(days)}))

    def compute(self):
        with mock.patch.object(weekly, "datetime", frozen(2026, 11, 24, 18, 0)), \
                mock.patch.object(weekly.urllib.request, "urlopen", side_effect=OSError("offline")):
            return weekly.compute_ai_nwp_weekly_forecast(force_refresh=True)

    def test_old_cache_is_refused_when_source_is_down(self):
        self.store(40 * 86400, ["2026-11-25", "2026-11-26"])
        with self.assertRaises(RuntimeError):
            self.compute()

    def test_recent_cache_is_used_and_marked_stale(self):
        self.store(3 * 3600, ["2026-11-25", "2026-11-26"])
        result = self.compute()
        self.assertTrue(result["stale"])
        self.assertEqual(result["total_days"], 2)

    def test_past_days_in_wib_are_dropped(self):
        self.store(3 * 3600, ["2026-11-24", "2026-11-25"])  # sekarang 25 Nov 01:00 WIB
        days = self.compute()["daily_summaries"]
        self.assertEqual(
            [(d["date"], d["day_index"], d["is_today"], d["precip_prob_max"]) for d in days],
            [("2026-11-25", 0, True, 11)],
        )

    def test_source_does_not_claim_machine_learning(self):
        self.store(3 * 3600, ["2026-11-25"])
        self.assertIsNone(re.search(r"\b(ML|Hybrid|Machine Learning)\b", self.compute()["source"], re.I))

    def test_default_cache_is_outside_the_published_folder(self):
        spec = importlib.util.spec_from_file_location("weekly_fresh", weekly.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertNotIn("public", Path(module.CACHE_FILE).parts)


if __name__ == "__main__":
    unittest.main()
