"""Launch the AeroShape interactive GUI dashboard.

This script starts the Streamlit-based virtual wing laboratory,
which provides an interactive interface for wing design, GVM
computation, and CAD export.

Requirements:
    pip install aeroshape[gui]

Usage:
    python examples/run_gui.py

    Or directly:
    streamlit run app.py
"""

import subprocess
import sys
import os


def main():
    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app.py"
    )
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


if __name__ == "__main__":
    main()
