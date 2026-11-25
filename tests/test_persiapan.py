"""Catatan prediksi, data peta, ringkasan model, dan penanda data terlambat."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from support import FakeSheet, frozen, metar
from test_ews import NOW, ecmwf_payload, records
import app
import ews_log
import ews_service as ews
import map_service
import predict_service

UTC = timezone.utc


def ews_result(**kwargs):
    with mock.patch.object(ews, "fetch_ecmwf", return_value=ecmwf_payload()):
        return app.build_ews(records(**kwargs))


class FakeDoc:
    def __init__(self, titles=()):
        self.tabs = {t: mock.Mock(title=t) for t in titles}

    def worksheets(self):
        return list(self.tabs.values())

    def worksheet(self, title):
        return self.tabs[title]

    def add_worksheet(self, title, rows, cols):
        self.tabs[title] = mock.Mock(title=title)
        return self.tabs[title]


class LogTest(unittest.TestCase):
    def test_row_matches_headers(self):
        row = ews_log.build_row(ews_result(), NOW + timedelta(minutes=7))
        d = dict(zip(ews_log.HEADERS, row))
        self.assertEqual(len(row), len(ews_log.HEADERS))
        self.assertEqual((d["observed_utc"], d["delay_min"]), ("2026-11-24 16:00", 7))
        self.assertEqual(d["model_version"], ews.MODEL_VERSION)
        self.assertEqual(d["model_1h"], "LSTM, XGBoost, ECMWF IFS")
        self.assertIsInstance(d["p_24h"], float)
        self.assertIn(d["level_24h"], (0, 1, 2))
        self.assertIsInstance(d["lstm_1h"], float)

    def test_creates_tab_with_headers_once(self):
        doc = FakeDoc(["Sheet1"])
        ews_log.append(doc, ews_result(), NOW + timedelta(minutes=7))
        tab = doc.tabs["EWS_Log"]
        self.assertEqual(tab.append_row.call_args_list[0].args[0], ews_log.HEADERS)
        self.assertEqual(tab.append_row.call_count, 2)

    def test_existing_tab_is_not_given_headers_again(self):
        doc = FakeDoc(["EWS_Log"])
        doc.tabs["EWS_Log"].row_values.return_value = ews_log.HEADERS
        ews_log.append(doc, ews_result(), NOW + timedelta(minutes=7))
        self.assertEqual(doc.tabs["EWS_Log"].append_row.call_count, 1)

    def test_tab_with_other_columns_is_left_alone(self):
        doc = FakeDoc(["EWS_Log"])
        doc.tabs["EWS_Log"].row_values.return_value = ["catatan saya"]
        with self.assertRaises(RuntimeError):
            ews_log.append(doc, ews_result(), NOW)
        doc.tabs["EWS_Log"].append_row.assert_not_called()

    def test_late_prediction_is_not_logged(self):
        # Prediksi yang dibuat setelah kejadian tidak boleh masuk data uji.
        doc = FakeDoc(["Sheet1"])
        self.assertFalse(ews_log.append(doc, ews_result(), NOW + timedelta(minutes=50)))
        self.assertNotIn("EWS_Log", doc.tabs)


class CollectorLogTest(unittest.TestCase):
    def collect(self, sheet, log):
        clock = frozen(2026, 11, 24, 16, 5)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(app, "LOCAL_HISTORY_FILE", str(Path(folder) / "h.json")), \
                mock.patch.object(app, "_tail_hint", None), mock.patch.object(app, "_raw_column_ok", None), \
                mock.patch.object(app, "_last_collect", None), mock.patch.object(app, "datetime", clock), \
                mock.patch.object(predict_service, "datetime", clock), mock.patch.object(app, "get_sheet", return_value=sheet), \
                mock.patch.object(app.requests, "get", return_value=mock.Mock(text=metar("241600Z"))), \
                mock.patch.object(ews, "fetch_ecmwf", side_effect=OSError()), mock.patch.object(ews_log, "append", log):
            return app.fetch_and_append_metar()

    def sheet(self):
        slots = [(NOW - timedelta(minutes=30 * i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(100, 0, -1)]
        s = FakeSheet(slots)
        s.spreadsheet = object()
        return s

    def test_new_observation_is_logged_once(self):
        log, sheet = mock.Mock(), self.sheet()
        self.collect(sheet, log)
        self.collect(sheet, log)
        self.assertEqual(log.call_count, 1)
        self.assertEqual(log.call_args.args[1]["observed_at_utc"], "2026-11-24 16:00 UTC")

    def test_log_failure_does_not_stop_collection(self):
        sheet = self.sheet()
        result = self.collect(sheet, mock.Mock(side_effect=RuntimeError("quota")))
        self.assertEqual(result["status"], "success")
        self.assertEqual(sheet.slots[-1], "2026-11-24 16:00:00")


class FreshnessTest(unittest.TestCase):
    def test_age_and_stale_flag(self):
        for minutes, stale in ((10, False), (75, False), (76, True)):
            with self.subTest(minutes):
                clock = frozen(2026, 11, 24, 16, 0)
                moment = NOW + timedelta(minutes=minutes)
                clock = frozen(*moment.timetuple()[:5])
                with mock.patch.object(app, "datetime", clock):
                    result = ews_result()
                self.assertEqual((result["data_age_minutes"], result["stale"]), (minutes, stale))

    def test_history_length_is_bounded(self):
        with mock.patch.object(app, "_latest_raw_metar", "x"), mock.patch.object(app, "get_sheet"), \
                mock.patch.object(app, "get_recent_rows_from_sheet", return_value=([{"slot_30min": "x"}], 2)) as read:
            for asked, used in ((None, 20), (31, 31), (5000, 100), (0, 1)):
                app.get_history() if asked is None else app.get_history(count=asked)
                self.assertEqual(read.call_args.kwargs["count"], used)

    def test_history_does_not_overwrite_a_stored_raw_metar(self):
        rows = [{"slot_30min": "2026-11-24 15:30:00", "raw_metar": "asli"}]
        with mock.patch.object(app, "_latest_raw_metar", "lebih baru"), mock.patch.object(app, "get_sheet"), \
                mock.patch.object(app, "get_recent_rows_from_sheet", return_value=(rows, 2)):
            self.assertEqual(app.get_history()["data"][-1]["raw_metar"], "asli")


class MapTest(unittest.TestCase):
    def payload(self):
        hourly = {"time": ["2026-11-24T01:00", "2026-11-24T02:00"], "temperature_2m": [28.04, None], "relative_humidity_2m": [70, 71],
                  "wind_speed_10m": [5.0, 6.0], "wind_direction_10m": [90, 100], "precipitation": [0.0, 1.0], "cape": [1000.0, 1200.0]}
        return [{"hourly": hourly}] * (map_service.NX * map_service.NY + 1)

    def test_shape(self):
        data = map_service.shape(self.payload())
        self.assertEqual(len(data["pts"]), map_service.NX * map_service.NY)
        self.assertEqual((data["t0"], data["n"]), ("2026-11-24T01:00", 2))
        self.assertEqual(data["pts"][0]["t"], [28.0, None])
        self.assertIn("c", data["site"])
        self.assertEqual(data["model"], "ECMWF IFS")

    def test_incomplete_response_is_rejected(self):
        with self.assertRaises(ValueError):
            map_service.shape(self.payload()[:-1])

    def test_cache_rules(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(map_service, "CACHE_FILE", Path(folder) / "m.json"), \
                mock.patch.object(map_service.urllib.request, "urlopen", side_effect=OSError("offline")) as call:
            for age, ok in ((60, True), (3 * 3600, True), (7 * 3600, False)):
                map_service.CACHE_FILE.write_text(json.dumps({"_cached_at": map_service.time.time() - age, "data": {"n": 1}}))
                with self.subTest(age):
                    if ok:
                        self.assertEqual(map_service.get_map()["n"], 1)
                    else:
                        with self.assertRaises(OSError):
                            map_service.get_map()
            self.assertIn("models=ecmwf_ifs", call.call_args.args[0].full_url)

    def test_geography_file_is_published(self):
        geo = json.loads((Path(app.BASE_DIR) / "public" / "assets" / "geo.json").read_text(encoding="utf-8"))
        self.assertEqual(geo["bbox"], map_service.BBOX)
        self.assertEqual(len(geo["elev"]["v"]), geo["elev"]["nx"] * geo["elev"]["ny"])
        self.assertTrue(geo["land"])


class ModelInfoTest(unittest.TestCase):
    def test_summary_is_complete_and_matches_the_served_model(self):
        info = app.ews_model()
        self.assertEqual(info["version"], ews.MODEL_VERSION)
        self.assertEqual(info["evaluation"]["model_version"], ews.MODEL_VERSION)
        for h in ews.LEAD_DAY:
            e = info["evaluation"]["horizons"][h]
            self.assertEqual(set(e["prauc"]), {"klimatologi", "situs_lama", "cadangan", "utama", "berurutan"})
            for kind in ("utama", "berurutan"):
                self.assertAlmostEqual(e["prauc"][kind], ews.BUNDLE["models"][kind][h]["metrics_2026"]["prauc"], places=4)
                self.assertEqual(e[kind]["waspada"]["ambang"], info["horizons"][h][kind]["th_waspada"])
            self.assertEqual(len(info["evaluation"]["lstm_mae"][h]["temperature"]), 3)
            c = info["evaluation"]["confirmation"]["horizons"][h]
            self.assertEqual(c["events"], c["berurutan"]["waspada"]["TP"] + c["berurutan"]["waspada"]["FN"])
        self.assertRegex(info["evaluation"]["confirmation"]["model_version"], r"^[0-9a-f]{8}$")
        self.assertEqual(info["primary"], "berurutan")
        for h in ews.LEAD_DAY:
            e = info["evaluation"]["horizons"][h]
            self.assertIn("better_than_climatology", e)
            # Tanda 'terbukti lebih baik' harus mengikuti selang 95%, bukan ditulis tangan.
            self.assertEqual(e["better_than_climatology"], e["gain_over_climatology"][1] > 0)
        self.assertEqual([info["evaluation"]["horizons"][h]["better_than_climatology"] for h in ("1h", "3h")], [True, True])


if __name__ == "__main__":
    unittest.main()
