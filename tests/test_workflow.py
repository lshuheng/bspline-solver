import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    TrajectoryDataset,
    VariationalProblem,
    generate_ground_truth,
    load_dataset,
    save_result,
    solve_experiment,
)


class DatasetTests(unittest.TestCase):
    def test_loads_manual_kepler_vertices(self):
        dataset = load_dataset("kepler_orbit_1")

        self.assertEqual(dataset.vertices.shape, (15, 2))
        self.assertIsNone(dataset.trajectory)
        self.assertFalse(dataset.metadata["ground_truth_available"])

    def test_rejects_time_without_trajectory(self):
        with self.assertRaisesRegex(ValueError, "time requires"):
            TrajectoryDataset(
                name="invalid",
                vertices=[[0.0, 0.0], [1.0, 1.0]],
                time=[0.0, 1.0],
            )

    def test_ground_truth_generation_is_explicitly_unimplemented(self):
        with self.assertRaises(NotImplementedError):
            generate_ground_truth("kepler")


class ExperimentTests(unittest.TestCase):
    def test_solves_and_persists_solver_independent_result(self):
        dataset = TrajectoryDataset(
            name="line",
            vertices=[[0.0, 0.0], [1.0, 0.0]],
            metadata={"source": "test"},
        )
        ut, vt = sp.symbols("ut vt")
        problem = VariationalProblem(
            name="shortest_path",
            lagrangian=ut**2 + vt**2,
            title="Shortest path",
        )

        result = solve_experiment(
            dataset,
            problem,
            ExperimentConfig(n_bisections=1, n_quad=3, max_iteration=1),
        )

        self.assertEqual(result.dataset_name, "line")
        self.assertEqual(len(result.optimized_controls), 1)
        self.assertGreaterEqual(len(result.energy_history), 2)
        self.assertFalse(hasattr(result, "_solver"))
        self.assertEqual(result.diagnostics["n_segments"], 1)
        self.assertIn("final_energy", result.diagnostics)

        with tempfile.TemporaryDirectory() as directory:
            arrays_path, metadata_path = save_result(
                result,
                output_dir=directory,
                name="test result",
            )
            self.assertEqual(arrays_path.name, "test_result.npz")
            self.assertEqual(metadata_path.name, "test_result.json")

            with np.load(arrays_path) as arrays:
                np.testing.assert_allclose(arrays["vertices"], dataset.vertices)
                self.assertIn("optimized_control_000", arrays.files)

            metadata = json.loads(Path(metadata_path).read_text())
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["problem_name"], "shortest_path")
            self.assertEqual(metadata["diagnostics"]["n_segments"], 1)

    def test_global_constraint_target_is_recorded(self):
        dataset = TrajectoryDataset(
            name="two_segments",
            vertices=[[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]],
        )
        ut, vt = sp.symbols("ut vt")
        speed = sp.sqrt(ut**2 + vt**2)
        problem = VariationalProblem(
            name="fixed_length",
            lagrangian=ut**2 + vt**2,
            constraint=speed,
            constraint_target=2.0,
        )

        result = solve_experiment(
            dataset,
            problem,
            ExperimentConfig(n_bisections=1, n_quad=3, max_iteration=1),
        )

        self.assertEqual(result.problem_metadata["constraint_target"], 2.0)
        self.assertEqual(len(result.constraint_history), 1)


if __name__ == "__main__":
    unittest.main()
