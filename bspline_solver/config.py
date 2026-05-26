"""Global configuration constants for the B-spline solver."""

import numpy as np

# B-spline degree (cubic by default).
DEGREE = 3

# Penalty stiffness for the integral constraint in the augmented Lagrangian.
CONSTRAINT_STIFFNESS = 10.0

# Stiffness for the control-point regularization term.
REGULARIZATION_STIFFNESS = 10.0

# Scaling factor used when seeding boundary tangents during initialization.
TANGENT_INIT_SCALE = 1.0

# Shared random number generator for reproducible initialization noise.
rng = np.random.default_rng()
