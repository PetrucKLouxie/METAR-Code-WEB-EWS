import os
import sys

# Tambahkan direktori root project ke sys.path agar modul app dapat di-import oleh Vercel
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app
