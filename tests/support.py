"""Alat bantu tes: jam yang dibekukan dan Google Sheet tiruan di memori."""
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

os.environ["VERCEL"] = "1"  # cegah scheduler latar belakang berjalan saat tes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def frozen(*args):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(*args, tzinfo=timezone.utc)
            return value.astimezone(tz) if tz else value

    return FrozenDatetime


def metar(code, rest="09005KT 9999 FEW020 28/25 Q1010 NOSIG"):
    return f"METAR WARR {code} {rest}"


class FakeSheet:
    """Sheet di memori dengan perilaku gspread yang dipakai kode produksi.

    Seperti gspread asli, row_count adalah salinan yang bertambah pada tiap
    append walaupun Google hanya mengisi baris kosong di dalam grid.
    """

    def __init__(self, slots=(), blank_rows=0, col_count=26, header_g=""):
        self.rows = [["slot_30min", "", "", "", "", "bad_weather", header_g]]
        self.rows += [[s, "5", "25", "1010", "28", "0"] for s in slots]
        self.grid_rows = len(self.rows) + blank_rows
        self.row_count = self.grid_rows
        self.col_count = col_count

    def get_values(self, cell_range):
        match = re.fullmatch(r"A(\d+):([A-Z])(\d+)?", cell_range)
        first, last_col = int(match[1]), ord(match[2]) - 64
        if first > self.grid_rows:
            raise RuntimeError("Range exceeds grid limits")
        last = int(match[3]) if match[3] else self.grid_rows
        return [row[:last_col] for row in self.rows[first - 1:last]]

    def append_rows(self, rows, value_input_option=None):
        self.rows.extend([list(r) for r in rows])
        self.grid_rows = max(self.grid_rows, len(self.rows))
        self.row_count += len(rows)

    def acell(self, label):
        assert label == "G1"
        return mock.Mock(value=self.rows[0][6] or None)

    def update_acell(self, label, value):
        assert label == "G1"
        self.rows[0][6] = value

    @property
    def slots(self):
        return [row[0] for row in self.rows[1:]]
