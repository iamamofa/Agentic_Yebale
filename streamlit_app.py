"""Launch with: python -m streamlit run streamlit_app.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from knowing_where.ui.app import main

main()
