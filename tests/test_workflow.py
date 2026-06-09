import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    TrajectoryDataset,
    VariationalProblem,
    ground_truth_kepler,
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
        self.assertEqual(dataset.metadata["energy"], -0.32766314)
        self.assertEqual(len(dataset.metadata["masses"]), 2)

    def test_rejects_invalid_trajectory_shape(self):
        with self.assertRaisesRegex(ValueError, "trajectory must have shape"):
            TrajectoryDataset(
                name="invalid",
                vertices=[[0.0, 0.0], [1.0, 1.0]],
                trajectory=[0.0, 1.0],
            )

    def test_kepler_ivp_solver_is_explicitly_unimplemented(self):
        with self.assertRaises(NotImplementedError):
            ground_truth_kepler(
                masses=[{"center": [0.0, 0.0], "mass": 1.0}],
                gravitational_constant=1.0,
                initial_position=[1.0, 0.0],
                initial_velocity=[0.0, 1.0],
                t_span=(0.0, 1.0),
                n_vertices=3,
                n_dense=5,
            )

    @patch("bspline_solver.ground_truth._solve_kepler_ivp")
    def test_kepler_ground_truth_subsamples_dense_trajectory(self, solve_ivp):
        trajectory = np.column_stack(
            [np.arange(9, dtype=float), -np.arange(9, dtype=float)]
        )
        solve_ivp.return_value = trajectory, -0.5

        dataset = ground_truth_kepler(
            masses=[{"center": [0.0, 0.0], "mass": 2.0}],
            gravitational_constant=1.0,
            initial_position=[1.0, 0.0],
            initial_velocity=[0.0, 1.0],
            t_span=(0.0, 8.0),
            n_vertices=3,
            n_dense=9,
            name="test_orbit",
        )

        np.testing.assert_allclose(dataset.trajectory, trajectory)
        np.testing.assert_allclose(dataset.vertices, trajectory[[0, 4, 8]])
        self.assertEqual(dataset.metadata["energy"], -0.5)
        self.assertEqual(dataset.metadata["sample_indices"], [0, 4, 8])
        self.assertTrue(dataset.metadata["ground_truth_available"])


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
