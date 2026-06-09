"""Small waypoint datasets used by the classical variational demos."""

DATASETS = {
    "hanging_chain": {
        "vertices": [
            [0.0, 0.0],
            [0.5, 1.0],
            [2.0, 2.0],
        ],
        "metadata": {
            "description": "Three waypoints for the fixed-length hanging-chain demo.",
            "source": "manually curated",
        },
    },
    "isoperimetric": {
        "vertices": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        "metadata": {
            "description": "Square initialization for the closed isoperimetric demo.",
            "source": "manually curated",
        },
    },
    "beam_buckling": {
        "vertices": [
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 0.0],
        ],
        "metadata": {
            "description": "Three waypoints for the fixed-length beam demo.",
            "source": "manually curated",
        },
    },
}
