"""Halaman web: berkas yang dirujuk ada, dan teks lama tidak kembali."""
import json
import re
import unittest
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "public" / "assets"


class PageTest(unittest.TestCase):
    def setUp(self):
        self.html = (ROOT / "index.html").read_text(encoding="utf-8")
        self.scripts = {p.name: p.read_text(encoding="utf-8") for p in ASSETS.glob("*.js")}

    def test_referenced_files_exist(self):
        paths = re.findall(r'(?:src|href)="/(assets/[^"]+)"', self.html)
        self.assertGreaterEqual(len(paths), 8)
        for path in paths:
            self.assertTrue((ROOT / "public" / path).is_file(), path)

    def test_every_element_id_used_by_scripts_exists(self):
        everything = self.html + "".join(self.scripts.values())
        defined = set(re.findall(r'id="([\w-]+)"', everything))
        for name, code in self.scripts.items():
            for used in re.findall(r"\$\('([\w-]+)'\)", code):
                self.assertIn(used, defined | {"p-"}, f"{name}: {used}")

    def test_not_about_airport_operations(self):
        text = (self.html + "".join(self.scripts.values())).lower()
        for word in ("runway", "landasan", "penerbangan", "holding", "ews engine", "ai-nwp"):
            self.assertNotIn(word, text)

    def test_no_unsafe_claims_about_the_seven_day_score(self):
        self.assertIn("bukan peluang", self.scripts["prospek.js"])


class DeployConfigTest(unittest.TestCase):
    def test_included_files_fit_the_limit_and_cover_the_models(self):
        config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
        pattern = config["functions"]["api/**/*.py"]["includeFiles"]
        self.assertLessEqual(len(pattern), 256)  # batas skema Vercel; lebih panjang membuat deployment ditolak
        parts = pattern.strip("{}").split(",")
        for path in ("index.html", "saved_models_ecmwf/model.json", "saved_models_ecmwf/evaluasi.json",
                     "saved_models_ecmwf/lstm_forecaster.json", "saved_models_pure_math/model_trees_pure_math.json",
                     "saved_models_lstm/lstm_config.json", "saved_models_lstm/scaler_params.json",
                     "saved_models_lstm/lstm_weights_pure.json", "public/assets/geo.json"):
            self.assertTrue((ROOT / path).is_file(), path)
            self.assertTrue(any(fnmatch(path, p.replace("**", "*")) for p in parts), path)


if __name__ == "__main__":
    unittest.main()
