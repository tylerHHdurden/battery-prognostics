"""
Stage 7.3: DeepONet (Lu et al. 2021) as a real SPM/SPMe-style
electrochemical-dynamics surrogate - a genuine neural OPERATOR learning
the mapping current-profile -> voltage-response for a discharge cycle,
i.e. G: I(.) -> V(.), directly from this project's own raw cycle curves
(sequence_features.py's already-established V_t/I_t channels, no new
data-loading path).

CHOICE: DeepONet over FNO, disclosed. FNO is naturally suited to fields
on a REGULAR GRID (its core mechanism is a spectral/FFT convolution),
which fits PDE state fields (e.g. 2D/3D spatial concentration fields)
far more directly than this project's 1D, irregularly-sampled-in-time,
per-cycle current/voltage curves. DeepONet's branch/trunk formulation
naturally handles an ARBITRARY QUERY COORDINATE (a query time point can
be anything in [0,1], not tied to a fixed grid) and an arbitrary
"sensor" input function sampled at fixed points - a direct match for
this project's actual data shape (a current curve sampled at n_bins=200
uniform time points, predicting voltage at any query time), without
forcing a spectral-grid framing this project's data doesn't naturally
have.

Architecture (Eq. 1 of the source):
    G(u)(y) ~= sum_k b_k(u) * t_k(y) + bias
    branch net b: MLP(sensor readings of u, i.e. I(t) at n_bins fixed
        points) -> R^p
    trunk net t: MLP(query coordinate y, i.e. normalized time t in
        [0,1]) -> R^p
    output: dot(b, t) + bias -> predicted V(y)

INTEGRATION (task's own stated option, taken here): the trained
branch net's OWN embedding b(I) for a given cycle's current curve is a
genuine physics-surrogate summary of that cycle's electrochemical
response (an operator-derived embedding, not a hand-engineered ratio) -
extracted per cycle and used as an ADDITIONAL feature/embedding input
to the existing XGBoost-fusion pipeline (the same pattern this project
already uses for the ICA encoder's fusion_embeddings.csv), rather than
as a standalone SOH predictor - the physically meaningful quantity
DeepONet is trained to predict is V(t), not SOH; SOH prediction stays
XGBoost's job, exactly as the task's option 2 describes.
"""
import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, n_sensors: int = 200, p: int = 32, hidden: int = 64):
        super().__init__()
        self.p = p
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, p), nn.ReLU(),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def branch_embed(self, I_sensors: torch.Tensor) -> torch.Tensor:
        """I_sensors: (batch, n_sensors) -> (batch, p) - the reusable
        physics-surrogate embedding extracted for XGBoost fusion."""
        return self.branch(I_sensors)

    def forward(self, I_sensors: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
        """I_sensors: (batch, n_sensors); query_t: (batch, n_query, 1).
        Returns predicted V at each query point: (batch, n_query)."""
        b = self.branch_embed(I_sensors)  # (batch, p)
        t = self.trunk(query_t)  # (batch, n_query, p)
        out = torch.einsum("bp,bqp->bq", b, t) + self.bias
        return out
