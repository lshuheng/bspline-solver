"""Shared figure output helpers for example scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def save_and_show(fig, filename: str) -> None:
    """Save a demo figure under figures/ and then display it."""
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
