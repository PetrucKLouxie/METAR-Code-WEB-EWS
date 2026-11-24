"""Verifikasi snapshot prakiraan terhadap observasi."""
import unittest
from datetime import datetime, timedelta
from unittest import mock

from support import FakeSheet, frozen, metar
import snapshot_service as snap

TS = "27010KT 4000 TSRA FEW018CB 26/24 Q1008"
RA = "27010KT 8000 -RA FEW018 26/24 Q1008"
CLEAR = "09005KT 9999 FEW020 28/25 Q1010 NOSIG"

# 24 Nov WIB = 23 Nov 17:00 UTC s.d. 24 Nov 16:30 UTC
START = "2026-11-23 17:00:00"


def day_rows(count=48, special=None, with_raw=True, bad=()):
    """Baris Sheet1 tiap 30 menit mulai START; special = {indeks: isi METAR}."""
    rows, start = [], datetime.fromisoformat(START)
    for i in range(count):
        t = start + timedelta(minutes=30 * i)
        row = [f"{t:%Y-%m-%d} {t.hour}:{t:%M:%S}", "5", "25", "1010", "28", "1" if i in bad else "0"]
        if with_raw:
            row.append(metar(t.strftime("%d%H%MZ"), (special or {}).get(i, CLEAR)))
        rows.append(row)
    return rows


def verdict(level, rows, today="2026-11-26"):
    day = snap.summarize_observations(rows)["2026-11-24"]
    return snap.decide_verification(level, day, is_today=(today == "2026-11-24"))


class DailySummaryTest(unittest.TestCase):
    def test_days_are_grouped_by_wib_date(self):
        days = snap.summarize_observations(day_rows())
        self.assertEqual({k: v["slots_count"] for k, v in days.items()}, {"2026-11-24": 48})

    def test_evening_utc_storm_belongs_to_next_wib_day(self):
        rows = [["2026-11-24 18:00:00", "5", "25", "1010", "28", "1", metar("241800Z", TS)]]
        self.assertEqual(snap.summarize_observations(rows)["2026-11-25"]["max_severity"], 2)

    def test_duplicate_slots_are_counted_once(self):
        rows = day_rows(count=2)
        self.assertEqual(snap.summarize_observations(rows + rows)["2026-11-24"]["slots_count"], 2)


class DecisionTest(unittest.TestCase):
    def test_storm_warning_with_thunderstorm_is_hit(self):
        self.assertEqual(verdict(2, day_rows(special={20: TS})), "HIT")

    def test_storm_warning_with_thunderstorm_flag_only_is_hit(self):
        self.assertEqual(verdict(2, day_rows(with_raw=False, bad=(20,))), "HIT")

    def test_storm_warning_with_light_rain_is_over_warning(self):
        self.assertEqual(verdict(2, day_rows(special={20: RA})), "HIT (OVER-WARNING)")

    def test_storm_warning_on_clear_day_is_false_alarm(self):
        self.assertEqual(verdict(2, day_rows()), "FALSE ALARM")

    def test_watch_with_light_rain_is_hit(self):
        self.assertEqual(verdict(1, day_rows(special={20: RA})), "HIT")

    def test_safe_on_clear_day_is_correct_negative(self):
        self.assertEqual(verdict(0, day_rows()), "CORRECT NEGATIVE")

    def test_safe_with_storm_is_miss(self):
        self.assertEqual(verdict(0, day_rows(special={20: TS})), "MISS")

    def test_clear_verdict_needs_enough_raw_reports(self):
        self.assertEqual(verdict(0, day_rows(count=10)), "UNVERIFIABLE")
        self.assertEqual(verdict(1, day_rows(with_raw=False)), "UNVERIFIABLE")

    def test_unfinished_day_is_not_judged_clear(self):
        for level in (0, 1, 2):
            with self.subTest(level):
                self.assertIsNone(verdict(level, day_rows(), today="2026-11-24"))

    def test_unfinished_day_waits_for_storm_when_only_rain_so_far(self):
        self.assertIsNone(verdict(2, day_rows(special={2: RA}), today="2026-11-24"))

    def test_unfinished_day_with_storm_is_final(self):
        rows = day_rows(special={2: TS})
        self.assertEqual(verdict(2, rows, today="2026-11-24"), "HIT")
        self.assertEqual(verdict(0, rows, today="2026-11-24"), "MISS")


class VerifyWriteTest(unittest.TestCase):
    def test_results_are_written_in_one_batch(self):
        def row(target, level):
            return ["id", "t", target, "", "1", level, "", "", "", "", "", "", "FALSE", "-", "PENDING"]

        ws = mock.Mock()
        ws.get_all_values.return_value = [
            snap.HEADERS, row("2026-11-24", "2"), row("2026-11-24", "0"), row("2026-12-01", "0"),
        ]
        sheet = FakeSheet()
        sheet.rows += day_rows(special={20: TS})
        sheet.grid_rows = sheet.row_count = len(sheet.rows)
        doc = mock.Mock()
        doc.worksheet.return_value = sheet
        with mock.patch.object(snap, "datetime", frozen(2026, 11, 26, 3, 0)):
            result = snap.verify_snapshots_with_metar_records(doc=doc, ws_snap=ws)
        self.assertEqual(result["verified_count"], 2)
        ws.update_cell.assert_not_called()
        ws.batch_update.assert_called_once()
        updates = {u["range"]: u["values"][0] for u in ws.batch_update.call_args.args[0]}
        self.assertEqual(sorted(updates), ["M2:O2", "M3:O3"])
        # "+TSRA" akan menjadi rumus bila dikirim sebagai USER_ENTERED.
        self.assertNotEqual(ws.batch_update.call_args.kwargs.get("value_input_option"), "USER_ENTERED")
        self.assertEqual((updates["M2:O2"][2], updates["M3:O3"][2]), ("HIT", "MISS"))

    def test_unverifiable_is_reported_but_not_scored(self):
        ws = mock.Mock()
        ws.get_all_records.return_value = [
            {"verification_result": v} for v in ("HIT", "HIT (OVER-WARNING)", "MISS", "UNVERIFIABLE", "PENDING")
        ]
        with mock.patch.object(snap, "_get_spreadsheet_doc"), \
                mock.patch.object(snap, "_ensure_snapshot_worksheet", return_value=ws):
            data = snap.get_all_snapshots(auto_verify=False)
        self.assertEqual((data["total_verified"], data["metrics"]["unverifiable"]), (3, 1))

    def test_repeated_reads_do_not_verify_every_time(self):
        ws = mock.Mock()
        ws.get_all_records.return_value = []
        with mock.patch.object(snap, "_get_spreadsheet_doc"), \
                mock.patch.object(snap, "_ensure_snapshot_worksheet", return_value=ws), \
                mock.patch.object(snap, "_last_auto_verify", None, create=True), \
                mock.patch.object(snap, "verify_snapshots_with_metar_records") as verify:
            snap.get_all_snapshots()
            snap.get_all_snapshots()
        self.assertEqual(verify.call_count, 1)


if __name__ == "__main__":
    unittest.main()
