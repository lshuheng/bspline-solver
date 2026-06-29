# Physics-based B-Spline Interpolator 

Physics-based B-spline Interpolator is a small Python library for interpolating sparsely sampled 2D trajectories governed by nonlinear ODEs. By incorporating a symbolic action functional for the underlying dynamics as the optimization objective, it can better recover the structure of the trajectory than purely geometric interpolation methods. When supplied, global integral constraints are enforced through an adaptive penalty scheme.

## Demo Gallery

Curated images can live in `figures/` and be linked here as the project settles. The current examples generate comparison figures interactively:

| Example | What it shows |
| --- | --- |
| `examples/hanging_chain.py` | Fixed-length catenary-style interpolation. |
| `examples/beam_buckling.py` | Bending-energy minimization with endpoint angle constraints. |
| `examples/isoperimetric.py` | Closed curve area maximization at fixed length. |
| `examples/kepler_1.py`, `examples/kepler_2.py`, `examples/kepler_3.py` | Reconstruction of generated fixed-mass Kepler trajectories. |
| `examples/henon_heiles.py` | Reconstruction of a generated Henon-Heiles trajectory. |
| `examples/polynomial_channel.py` | Reconstruction of a generated polynomial channel-scattering trajectory. |

## Reproducibility

Create an environment, install the package in editable mode, and run the tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest
```

Run an example from the repository root:

```bash
python examples/hanging_chain.py
python examples/kepler_1.py
python examples/henon_heiles.py
```

The examples open Matplotlib windows. On headless systems, configure a non-interactive backend before running plotting code.

## Mathematical Formulation

Given interpolation vertices \(q_i \in \mathbb{R}^2\), the solver represents each segment as a clamped B-spline curve

$$
q(t) = (u(t), v(t)) = \sum_j c_j B_j(t), \quad t \in [0, 1].
$$

For a symbolic Lagrangian $L$, the optimized curve minimizes

$$
\sum_{\text{segments}} \int_0^1 L(t, u, u', u'', v, v', v'')\,dt
$$

optionally subject to an integral constraint

$$
\sum_{\text{segments}} \int_0^1 G(t, u, u', u'', v, v', v'')\,dt = g_\star.
$$

Quadrature is precomputed on the knot intervals, symbolic derivatives are generated with SymPy, and the finite-dimensional control-point problem is solved with SciPy's L-BFGS-B optimizer inside an augmented-Lagrangian loop.

## Status

This is research-oriented code and still work in progress.
