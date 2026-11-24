"""Tingkat cuaca yang benar-benar diobservasi dalam METAR."""
import unittest

from support import metar
from metar_weather import observed_severity


class ObservedSeverityTest(unittest.TestCase):
    def check(self, level, reports):
        for rest in reports:
            with self.subTest(rest):
                self.assertEqual(observed_severity(metar("241600Z", rest))[0], level)

    def test_clear(self):
        self.assertEqual(observed_severity(metar("241600Z")), (0, "NORMAL / CLEAR"))

    def test_extreme(self):
        self.check(2, (
            "27010KT 8000 TS FEW020CB 28/24 Q1011",
            "27010KT 4000 -TSRA FEW018CB 26/24 Q1008",
            "27010KT 4000 +TSRA FEW018CB 26/24 Q1008",
            "27010KT 9999 VCTS FEW018CB 26/24 Q1008",
            "27012G35KT 9999 FEW018 27/25 Q1010",
            "27025KT 9999 FEW018 27/25 Q1010",
            "27010KT 1000 +RA BKN018 26/25 Q1007",
            "27010KT 0000 FG VV/// 26/26 Q1007",
            "27010KT 240V300 1500 RA BKN018 26/25 Q1007",
        ))

    def test_moderate(self):
        self.check(1, (
            "27010KT 9000 -RA FEW018 27/25 Q1010",
            "27010KT 9000 +RA FEW018 27/25 Q1010",
            "27010KT 9000 +SHRA FEW018 27/25 Q1010",
            "27010KT 4000 HZ FEW018 27/25 Q1010",
            "27010G18KT 9999 FEW018 27/25 Q1010",
            "27010KT 9999 FEW020CB 32/26 Q1009",
            "27010KT 9999 FEW020TCU 35/25 Q1010",
        ))

    def test_forecast_and_recent_weather_do_not_count(self):
        self.check(0, (
            "27005KT 9999 FEW018 27/25 Q1010 TEMPO TL0100 2000 TSRA",
            "27005KT 9999 FEW018 27/25 Q1010 BECMG 27030G45KT",
            "27005KT 8000 FEW020 28/24 Q1012 RETS NOSIG",
            "27005KT 8000 FEW020 28/24 Q1012 NOSIG RMK CB TO S",
        ))


if __name__ == "__main__":
    unittest.main()
