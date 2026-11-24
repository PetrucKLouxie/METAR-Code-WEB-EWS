"""Alur pencatatan METAR ke Google Sheet."""
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from support import FakeSheet, frozen, metar
import app
import predict_service


class CollectorTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for target, value in (
            ("LOCAL_HISTORY_FILE", os.path.join(folder.name, "h.json")),
            ("_tail_hint", None),
            ("_raw_column_ok", None),
            ("_last_collect", None),
        ):
            patch = mock.patch.object(app, target, value, create=True)
            patch.start()
            self.addCleanup(patch.stop)

    def collect(self, sheet, now, lines):
        lines = [line if line.startswith("METAR") else metar(line) for line in lines]
        clock = frozen(*now)
        app._last_collect = None
        get_sheet = {"side_effect": sheet} if isinstance(sheet, Exception) else {"return_value": sheet}
        with mock.patch.object(app, "datetime", clock), mock.patch.object(predict_service, "datetime", clock), \
                mock.patch.object(app, "get_sheet", **get_sheet), \
                mock.patch.object(app.requests, "get", return_value=mock.Mock(text="\n".join(lines))):
            return app.fetch_and_append_metar()

    def test_december_first_appends_in_order_and_keeps_collecting(self):
        sheet = FakeSheet(["2026-11-30 22:30:00", "2026-11-30 23:00:00"])
        self.collect(sheet, (2026, 12, 1, 0, 10), ["010000Z", "302330Z", "302300Z", "302230Z"])
        self.collect(sheet, (2026, 12, 1, 0, 40), ["010030Z", "010000Z", "302330Z", "302300Z"])
        self.assertEqual(sheet.slots[2:], ["2026-11-30 23:30:00", "2026-12-01 00:00:00", "2026-12-01 00:30:00"])

    def test_november_first_is_not_blocked_by_day_31(self):
        sheet = FakeSheet(["2026-10-31 23:00:00"])
        self.collect(sheet, (2026, 11, 1, 0, 10), ["010000Z", "312330Z", "312300Z"])
        self.assertEqual(sheet.slots[1:], ["2026-10-31 23:30:00", "2026-11-01 00:00:00"])

    def test_nil_report_is_not_recorded(self):
        sheet = FakeSheet(["2026-11-24 15:00:00"])
        self.collect(sheet, (2026, 11, 24, 16, 5), ["241600Z", "METAR WARR 241530Z NIL"])
        self.assertEqual(sheet.slots[1:], ["2026-11-24 16:00:00"])

    def test_nil_as_newest_report_shows_last_valid_one(self):
        sheet = FakeSheet(["2026-11-24 15:30:00"])
        result = self.collect(sheet, (2026, 11, 24, 16, 35), ["METAR WARR 241630Z NIL", "241600Z"])
        self.assertEqual(result["data"]["slot_30min"], "2026-11-24 16:00:00")
        self.assertEqual(sheet.slots[1:], ["2026-11-24 16:00:00"])

    def test_raw_metar_is_stored_in_column_g(self):
        sheet = FakeSheet(["2026-11-24 15:30:00"])
        self.collect(sheet, (2026, 11, 24, 16, 5), ["241600Z"])
        self.assertEqual(sheet.rows[0][6], "raw_metar")
        self.assertEqual(sheet.rows[-1][6:], [metar("241600Z")])

    def test_column_g_used_for_something_else_is_left_alone(self):
        sheet = FakeSheet(["2026-11-24 15:30:00"], header_g="catatan")
        self.collect(sheet, (2026, 11, 24, 16, 5), ["241600Z"])
        self.assertEqual(sheet.rows[0][6], "catatan")
        self.assertEqual(len(sheet.rows[-1]), 6)

    def test_six_column_sheet_gets_six_values(self):
        sheet = FakeSheet(["2026-11-24 15:30:00"], col_count=6)
        self.collect(sheet, (2026, 11, 24, 16, 5), ["241600Z"])
        self.assertEqual(len(sheet.rows[-1]), 6)

    def test_unreachable_sheet_is_not_reported_as_recorded(self):
        result = self.collect(RuntimeError("offline"), (2026, 11, 24, 16, 5), ["241600Z"])
        self.assertFalse(result["sheet_synced"])
        self.assertEqual(result["status"], "warning")

    def test_failed_append_is_not_reported_as_recorded(self):
        sheet = FakeSheet(["2026-11-24 15:30:00"])
        sheet.append_rows = mock.Mock(side_effect=RuntimeError("quota"))
        result = self.collect(sheet, (2026, 11, 24, 16, 5), ["241600Z"])
        self.assertFalse(result["sheet_synced"])
        self.assertEqual(result["status"], "warning")

    def test_long_running_process_survives_row_count_drift(self):
        sheet = FakeSheet(["2026-11-24 15:30:00"], blank_rows=10)
        self.collect(sheet, (2026, 11, 24, 16, 5), ["241600Z"])
        sheet.row_count += 5000
        result = self.collect(sheet, (2026, 11, 24, 16, 35), ["241630Z", "241600Z"])
        self.assertIsNone(result["sheet_error"])
        self.assertEqual(sheet.slots[1:], ["2026-11-24 16:00:00", "2026-11-24 16:30:00"])

    def test_simultaneous_calls_do_not_duplicate_rows(self):
        sheet = FakeSheet(["2026-11-24 15:30:00"])
        original = sheet.append_rows

        def slow_append(*args, **kwargs):
            time.sleep(0.2)
            original(*args, **kwargs)

        sheet.append_rows = slow_append
        clock = frozen(2026, 11, 24, 16, 5)
        with mock.patch.object(app, "datetime", clock), mock.patch.object(predict_service, "datetime", clock), \
                mock.patch.object(app, "get_sheet", return_value=sheet), \
                mock.patch.object(app.requests, "get", return_value=mock.Mock(text=metar("241600Z"))):
            threads = [threading.Thread(target=app.fetch_and_append_metar) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(sheet.slots[1:], ["2026-11-24 16:00:00"])

    def test_tail_is_found_when_grid_ends_with_many_blank_rows(self):
        sheet = FakeSheet(["2026-11-24 15:00:00", "2026-11-24 15:30:00"], blank_rows=4000)
        records, last_row = app.get_recent_rows_from_sheet(sheet, count=20)
        self.assertEqual((len(records), last_row), (2, 3))

    def test_training_label_rule_is_unchanged(self):
        # Kolom bad_weather adalah label training; definisinya harus tetap.
        for rest, expected in (
            ("27010KT 4000 TSRA FEW018CB 26/24 Q1008", 1),
            ("27010KT 9999 FEW018 30/25 Q1005 TEMPO TL0900 3000 TSRA", 1),
            ("27010KT 1000 +RA BKN018CB 26/25 Q1007 TEMPO TL2130 1000 +RA", 0),
            ("27010KT 8000 -RA FEW018 26/25 Q1007 RETS NOSIG", 0),
        ):
            with self.subTest(rest):
                self.assertEqual(app.parse_metar(metar("241600Z", rest))["bad_weather"], expected)


if __name__ == "__main__":
    unittest.main()
