"""
app.py
------
Entry point for DeliveryNet AI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from ui.controls import render_sidebar
from ui.dashboard import configure_page, render_dashboard


def main() -> None:
    configure_page()
    controls = render_sidebar()
    render_dashboard(controls)


if __name__ == "__main__":
    main()
