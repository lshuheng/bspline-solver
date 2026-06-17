import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

from bspline_solver import (
    ExperimentConfig,
    TrajectoryDataset,
    VariationalProblem,
    ground_truth,
    ground_truth_kepler,
    load_dataset,
    solve_experiment,
    solve_sampling_experiments,
)


class DatasetTests(unittest.TestCase):
    def test_loads_manual_basic_vertices(self):
        dataset = load_dataset("hanging_chain")

        self.assertEqual(dataset.vertices.shape, (3, 2))
        self.assertIsNone(dataset.trajectory)
        self.assertEqual(dataset.metadata["source"], "manually curated")

    def test_rejects_invalid_trajectory_shape(self):
        with self.assertRaisesRegex(ValueError, "trajectory must have shape"):
            TrajectoryDataset(
                name="invalid",
                vertices=[[0.0, 0.0], [1.0, 1.0]],
                trajectory=[0.0, 1.0],
            )

    def test_kepler_ivp_solver_generates_circular_orbit(self):
        dataset = ground_truth_kepler(
            masses=[{"center": [0.0, 0.0], "mass": 1.0}],
            gravitational_constant=1.0,
            initial_position=[1.0, 0.0],
            initial_velocity=[0.0, 1.0],
            t_span=(0.0, 2.0 * np.pi),
            n_vertices=5,
            n_dense=33,
        )

        self.assertEqual(dataset.trajectory.shape, (33, 2))
        self.assertEqual(dataset.vertices.shape, (5, 2))
        self.assertAlmostEqual(dataset.metadata["energy"], -0.5)
        np.testing.assert_allclose(dataset.trajectory[0], [1.0, 0.0])
        np.testing.assert_allclose(dataset.trajectory[-1], [1.0, 0.0], atol=1e-8)

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
            geometric_sampling=False,
        )

        np.testing.assert_allclose(dataset.trajectory, trajectory)
        np.testing.assert_allclose(dataset.vertices, trajectory[[0, 4, 8]])
        self.assertEqual(dataset.metadata["energy"], -0.5)
        self.assertEqual(dataset.metadata["sample_indices"], [0, 4, 8])
        self.assertTrue(dataset.metadata["ground_truth_available"])
        self.assertFalse(dataset.metadata["geometric_sampling"])
        self.assertEqual(dataset.metadata["sampling_method"], "index")

    @patch("bspline_solver.ground_truth._solve_kepler_ivp")
    def test_kepler_ground_truth_accepts_multiple_sampling_levels(self, solve_ivp):
        trajectory = np.column_stack(
            [np.arange(9, dtype=float), -np.arange(9, dtype=float)]
        )
        solve_ivp.return_value = trajectory, -0.5

        datasets = ground_truth_kepler(
            masses=[{"center": [0.0, 0.0], "mass": 2.0}],
            gravitational_constant=1.0,
            initial_position=[1.0, 0.0],
            initial_velocity=[0.0, 1.0],
            t_span=(0.0, 8.0),
            n_vertices=[3, 5],
            n_dense=9,
            name="test_orbit",
            geometric_sampling=False,
        )

        self.assertEqual(len(datasets), 2)
        solve_ivp.assert_called_once()
        self.assertEqual(datasets[0].name, "test_orbit_3_points")
        self.assertEqual(datasets[1].name, "test_orbit_5_points")
        np.testing.assert_allclose(datasets[0].vertices, trajectory[[0, 4, 8]])
        np.testing.assert_allclose(datasets[1].vertices, trajectory[[0, 2, 4, 6, 8]])
        self.assertEqual(datasets[0].metadata["n_vertices"], 3)
        self.assertEqual(datasets[1].metadata["n_vertices"], 5)
        self.assertEqual(datasets[0].metadata["sample_indices"], [0, 4, 8])
        self.assertEqual(datasets[1].metadata["sample_indices"], [0, 2, 4, 6, 8])

    @patch("bspline_solver.ground_truth._solve_kepler_ivp")
    def test_kepler_ground_truth_geometric_sampling_uses_arclength(self, solve_ivp):
        trajectory = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [10.0, 0.0],
                [11.0, 0.0],
            ]
        )
        solve_ivp.return_value = trajectory, -0.5

        dataset = ground_truth_kepler(
            masses=[{"center": [0.0, 0.0], "mass": 2.0}],
            gravitational_constant=1.0,
            initial_position=[1.0, 0.0],
            initial_velocity=[0.0, 1.0],
            t_span=(0.0, 8.0),
            n_vertices=3,
            n_dense=5,
            geometric_sampling=True,
        )

        np.testing.assert_allclose(
            dataset.vertices,
            [[0.0, 0.0], [5.5, 0.0], [11.0, 0.0]],
        )
        self.assertTrue(dataset.metadata["geometric_sampling"])
        self.assertEqual(dataset.metadata["sampling_method"], "arclength")
        self.assertEqual(dataset.metadata["sample_indices"], [0, 3, 4])
        self.assertEqual(dataset.metadata["sample_segment_indices"], [0, 2, 3])
        np.testing.assert_allclose(
            dataset.metadata["sample_arclengths"],
            [0.0, 5.5, 11.0],
        )

    def test_variational_ground_truth_generates_harmonic_orbit(self):
        u, v, ut, vt = sp.symbols("u v ut vt")
        potential = sp.Rational(1, 2) * (u**2 + v**2)
        problem = VariationalProblem(
            name="harmonic_oscillator",
            lagrangian=sp.Rational(1, 2) * (ut**2 + vt**2) - potential,
        )

        dataset = ground_truth(
            problem=problem,
            initial_position=[1.0, 0.0],
            initial_velocity=[0.0, 1.0],
            t_span=(0.0, 2.0 * np.pi),
            n_vertices=5,
            n_dense=33,
        )

        self.assertEqual(dataset.name, "generated_harmonic_oscillator_trajectory")
        self.assertEqual(dataset.trajectory.shape, (33, 2))
        self.assertEqual(dataset.vertices.shape, (5, 2))
        self.assertAlmostEqual(dataset.metadata["energy"], 1.0)
        self.assertEqual(dataset.metadata["problem_name"], "harmonic_oscillator")
        np.testing.assert_allclose(dataset.trajectory[0], [1.0, 0.0])
        np.testing.assert_allclose(dataset.trajectory[-1], [1.0, 0.0], atol=1e-8)

    def test_variational_ground_truth_can_stop_at_position_limit(self):
        u = sp.symbols("u")
        problem = VariationalProblem(
            name="repelling_potential",
            lagrangian=-sp.Rational(1, 2) * u**2,
        )

        dataset = ground_truth(
            problem=problem,
            initial_position=[1.0, 0.0],
            initial_velocity=[0.0, 0.0],
            t_span=(0.0, 5.0),
            n_vertices=3,
            n_dense=51,
            max_position_norm=2.0,
        )

        self.assertEqual(dataset.metadata["termination"]["reason"], "max_position_norm")
        self.assertLess(dataset.metadata["actual_t_span"][1], 5.0)
        self.assertEqual(dataset.metadata["t_span"], [0.0, 5.0])
        self.assertGreaterEqual(len(dataset.trajectory), 3)


class ExperimentTests(unittest.TestCase):
    def test_solves_solver_independent_result(self):
        dataset = TrajectoryDataset(
            name="line",
            vertices=[[0.0, 0.0], [1.0, 0.0]],
            metadata={"source": "test"},
        )
        ut, vt = sp.symbols("ut vt")
        problem = VariationalProblem(
            name="shortest_path",
            lagrangian=ut**2 + vt**2,
        )

        result = solve_experiment(
            dataset,
            problem,
            ExperimentConfig(n_bisections=1, n_quad=3, max_iteration=1),
        )

        np.testing.assert_allclose(result.vertices, dataset.vertices)
        self.assertEqual(len(result.optimized_controls), 1)
        self.assertGreaterEqual(len(result.energy_history), 2)
        self.assertFalse(hasattr(result, "_solver"))
        self.assertFalse(result.has_constraint)

    def test_solves_sampling_experiments(self):
        datasets = [
            TrajectoryDataset(name="line_a", vertices=[[0.0, 0.0], [1.0, 0.0]]),
            TrajectoryDataset(name="line_b", vertices=[[0.0, 0.0], [2.0, 0.0]]),
        ]
        ut, vt = sp.symbols("ut vt")

        def problem_factory(_dataset):
            return VariationalProblem(
                name="shortest_path",
                lagrangian=ut**2 + vt**2,
            )

        results = solve_sampling_experiments(
            datasets,
            problem_factory,
            ExperimentConfig(n_bisections=1, n_quad=3, max_iteration=1),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([len(result.optimized_controls) for result in results], [1, 1])

    def test_global_constraint_is_tracked(self):
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

        self.assertTrue(result.has_constraint)
        self.assertEqual(len(result.constraint_history), 1)


if __name__ == "__main__":
    unittest.main()
