"""LSTM peramal METAR dan rantai cadangan peringatan."""
import json
import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import support  # noqa: F401
from test_ews import NOW, ecmwf_payload, records
import ews_service as ews
import lstm_forecaster as lf

UTC = timezone.utc


def observations(times, values):
    return [{"timestamp": t, "temperature": v[0], "dew_point": v[1], "pressure": v[2], "wind_speed": v[3], "bad": 0}
            for t, v in zip(times, values)]


class ParityTest(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads((Path(__file__).parent / "lstm_parity.json").read_text())

    def each(self):
        for c in self.cases:
            times = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in c["t"]]
            yield c, times, lf.grid(observations(times, c["v"]))

    def test_forecast_matches_training(self):
        self.assertGreaterEqual(len(self.cases), 5)
        for c, times, grid in self.each():
            out = lf.forecast(grid, times[-1])
            self.assertEqual(len(out), 48)
            for k, expected in c["forecast"].items():
                for a, b in zip(out[int(k) - 1], expected):
                    self.assertAlmostEqual(a, b, delta=0.01)

    def test_classifier_features_match_training(self):
        for c, times, grid in self.each():
            out = lf.forecast(grid, times[-1])
            for h, expected in c["features"].items():
                got = lf.features(grid, out, times[-1], ews.BUNDLE["models"]["berurutan"][h]["lead_steps"])
                for a, b in zip(got, expected):
                    self.assertAlmostEqual(a, b, delta=max(0.02, abs(b) * 0.005))


class GridTest(unittest.TestCase):
    def test_short_gap_is_filled_from_the_past_only(self):
        times = [NOW - timedelta(minutes=30 * k) for k in range(39, -1, -1)]
        obs = observations(times, [[28, 25, 1000 + i, 5] for i in range(40)])
        del obs[-3]
        self.assertEqual([row[2] for row in lf.grid(obs)[-4:]], [1036, 1036, 1038, 1039])

    def test_long_gap_is_refused(self):
        times = [NOW - timedelta(minutes=30 * k) for k in range(39, -1, -1)]
        obs = observations(times, [[28, 25, 1010, 5]] * 40)
        del obs[-10:-4]
        with self.assertRaises(LookupError):
            lf.grid(obs)

    def test_forecast_is_physical(self):
        times = [NOW - timedelta(minutes=30 * k) for k in range(29, -1, -1)]
        out = lf.forecast(lf.grid(observations(times, [[24, 24, 1010, 0]] * 30)), NOW)
        self.assertTrue(all(td <= t and w >= 0 and not math.isnan(p) for t, td, p, w in out))


class ChainTest(unittest.TestCase):
    def models(self, recs, ecmwf):
        with mock.patch.object(ews, "fetch_ecmwf", **ecmwf):
            return ews.predict(recs)

    def test_sequential_model_is_primary(self):
        r = self.models(records(), {"return_value": ecmwf_payload()})
        self.assertEqual({p["model"] for p in r["predictions"]}, {ews.MODEL_NAMES["berurutan"]})
        self.assertEqual(len(r["lstm_forecast"]["temperature"]), 48)

    def test_broken_history_uses_direct_model(self):
        recs = [x for i, x in enumerate(records()) if not 85 <= i <= 92]
        r = self.models(recs, {"return_value": ecmwf_payload()})
        self.assertEqual({p["model"] for p in r["predictions"]}, {ews.MODEL_NAMES["utama"]})
        self.assertIsNone(r["lstm_forecast"])

    def test_without_ecmwf_uses_metar_only(self):
        r = self.models(records(), {"side_effect": OSError()})
        self.assertEqual({p["model"] for p in r["predictions"]}, {ews.MODEL_NAMES["cadangan"]})


if __name__ == "__main__":
    unittest.main()
