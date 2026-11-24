"""Waktu observasi METAR di sekitar pergantian bulan."""
import unittest
from unittest import mock

from support import frozen, metar
import app
import predict_service


class ObservationTimeTest(unittest.TestCase):
    def slot(self, now, code):
        clock = frozen(*now)
        with mock.patch.object(app, "datetime", clock), mock.patch.object(predict_service, "datetime", clock):
            return app.get_metar_observation_time_utc(metar(code))

    def test_same_day(self):
        self.assertEqual(self.slot((2026, 11, 24, 16, 5), "241600Z"), ("2026-11-24 16:00:00", "241600Z"))

    def test_last_day_of_30_day_month_read_on_first_of_next(self):
        self.assertEqual(self.slot((2026, 12, 1, 0, 10), "302330Z"), ("2026-11-30 23:30:00", "302330Z"))

    def test_day_31_read_in_30_day_month(self):
        self.assertEqual(self.slot((2026, 11, 1, 0, 10), "312330Z"), ("2026-10-31 23:30:00", "312330Z"))

    def test_new_year(self):
        self.assertEqual(self.slot((2027, 1, 1, 0, 10), "312330Z"), ("2026-12-31 23:30:00", "312330Z"))

    def test_leap_day(self):
        self.assertEqual(self.slot((2028, 3, 1, 0, 10), "292330Z"), ("2028-02-29 23:30:00", "292330Z"))


if __name__ == "__main__":
    unittest.main()
