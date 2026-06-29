# Physics-based B-Spline Interpolator 

Physics-based B-spline Interpolator is a small Python library for interpolating sparsely sampled 2D trajectories governed by nonlinear ODEs. By incorporating a symbolic action functional for the underlying dynamics as the optimization objective, it can better recover the structure of the trajectory than purely geometric interpolation methods. When supplied, global integral constraints are enforced through an adaptive penalty scheme.

## Demo Gallery

Running the scripts in `examples/` saves these figures under `figures/` and also displays the Matplotlib window:

| Example | What it shows | Figure |
| --- | --- | --- |
| `examples/hanging_chain.py` | Fixed-length catenary-style interpolation. | ![Hanging chain demo](figures/hanging_chain.png) |
| `examples/beam_buckling.py` | Bending-energy minimization with endpoint angle constraints. | ![Beam buckling demo](figures/beam_buckling.png) |
| `examples/isoperimetric.py` | Closed curve area maximization at fixed length. | ![Isoperimetric demo](figures/isoperimetric.png) |
| `examples/kepler_1.py` | Reconstruction of a generated three-center Kepler trajectory. | ![Kepler 1 demo](figures/kepler_1.png) |
| `examples/kepler_2.py` | Reconstruction of a generated three-center Kepler trajectory. | ![Kepler 2 demo](figures/kepler_2.png) |
| `examples/kepler_3.py` | Reconstruction of a generated two-center Kepler trajectory. | ![Kepler 3 demo](figures/kepler_3.png) |
| `examples/henon_heiles.py` | Reconstruction of a generated Henon-Heiles trajectory. | ![Henon-Heiles demo](figures/henon_heiles.png) |
| `examples/polynomial_channel.py` | Reconstruction of a generated polynomial channel-scattering trajectory. | ![Polynomial channel demo](figures/polynomial_channel.png) |

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

The examples save Matplotlib figures to `figures/` and open Matplotlib windows. On headless systems, configure a non-interactive backend before running plotting code.

## Mathematical Formulation

Given $N$ interpolation vertices $v_i \in \mathbb{R}^2$, the solver constructs $N-1$ cubic B-spline segments

$$
q_i(t) = (u_i(t), v_i(t)) = \sum_j c_j B_j(t), \quad t \in [0, 1]
$$

such that $q_i(1) = q_{i+1}(0) = v_i$ and $q_0(0) = v_0$, with $C^1$ continuity enforced at the junctions. 

For a symbolic Lagrangian $L$, the optimized curve minimizes

$$
\sum_{\text{segments}} \int_0^1 L(t, u, u', u'', v, v', v'')dt
$$

optionally subject to an integral constraint

$$
\sum_{\text{segments}} \int_0^1 G(t, u, u', u'', v, v', v'')dt = g_\star.
$$

Quadrature is precomputed on the knot intervals, and symbolic derivatives are generated with SymPy. Control point optimization is handled with SciPy's L-BFGS-B optimizer inside an augmented-Lagrangian loop.

## Status

This is research-oriented code and still work in progress.
