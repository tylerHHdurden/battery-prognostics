"""
Stage 7.1: a battery-specific World Model - forecasts FUTURE SOH
TRAJECTORIES from a window of past raw cycles, a genuinely different
task from every other model in this project (point-in-time SOH/RUL
regression). Architecture per the task's own description, recovered
from arXiv 2603.10527: 1D-CNN cycle encoder -> PatchTST transformer ->
residual dynamics equation -> decoder, with iterative rollout.

DISCLOSED JUDGMENT CALLS (the paper's own code was not available to
consult directly - implemented from the architecture description given,
same standing as BatLiNet's own disclosed adaptation in Stage 5.2):

1. INPUT REPRESENTATION: raw per-cycle tensors (200, 6) - V_t, I_t, T_t,
   dQdV, dVdQ, dIdV - from sequence_features.py, this project's own
   already-established "less processed" sequence representation (same
   one BatLiNet/VLSTM/CNN-LSTM/PiFormer use), NOT the engineered HI
   features. Matches the task's explicit instruction.

2. CHANNEL FUSION BEFORE PATCHING: the paper's PatchTST stage patches a
   (possibly multivariate) time series with CHANNEL INDEPENDENCE (each
   variate patched/attended separately). Here, the "1D-CNN cycle
   encoder" stage already fuses this project's 6 raw channels into ONE
   per-cycle embedding BEFORE the sequence-level transformer ever runs -
   so what gets patched is the SEQUENCE OF PER-CYCLE EMBEDDINGS (one
   token per cycle, not one channel per token). This is a necessary,
   disclosed simplification: this project's "6 channels" are 6
   different physical signals of the SAME cycle, not 6 independent
   forecastable variates the way PatchTST's original multivariate
   time-series channels are - channel-independent patching would not
   make sense applied directly to V_t/I_t/T_t/dQdV/dVdQ/dIdV in this
   setting (matches the same reasoning BatLiNet's own docstring gives
   for reusing this project's fused-tensor convention).

3. LATENT (not raw-curve) ITERATIVE ROLLOUT: the decoder does NOT
   generate future raw cycle tensors (V_t/I_t/... curves) - that would
   be a much harder, less verifiable generative task the brief
   architecture description does not fully specify, and forecasting
   raw curves is not needed to answer the actual question asked
   ("does it produce plausible future SOH trajectories"). Instead: the
   encoder + PatchTST transformer condition ONCE on a real observed
   window (W past cycles) to produce an initial latent dynamics state
   h_0; the "residual dynamics equation" (a GRUCell residual update)
   evolves h_t -> h_{t+1} PURELY IN LATENT SPACE for the forecast
   horizon, with a decoder head reading off a per-step SOH DELTA at
   each latent state (SOH_{t+1} = SOH_t + delta_t - the "residual
   dynamics" the task asks for: predicting an increment, not an
   absolute value, matching physically-continuous degradation).
   Iterative rollout = repeatedly applying (dynamics step -> decode
   delta -> update SOH) for N future steps, needing NO future raw
   curves at inference - the physically correct behavior for a genuine
   forecaster (you should not need the future to predict the future).
   This exactly mirrors latent-space world models generally (e.g.
   PlaNet/Dreamer's recurrent state-space rollout), a standard,
   well-precedented way to instantiate "iterative rollout" for this
   kind of architecture.

4. WINDOW/HORIZON: W=10 past cycles of context, N=10 future cycles
   forecast horizon - a fixed, untuned choice (same standing as session
   4's lambda=0.1) chosen to keep CPU training tractable while still
   giving a genuinely multi-step (not single-step) trajectory forecast.

Trained with a multi-horizon teacher-forced loss: MSE between the
predicted and true SOH at EVERY one of the N rollout steps (standard
practice for autoregressive forecasters, avoids only supervising the
final horizon), backpropagated through the unrolled latent recurrence.
"""
import torch
import torch.nn as nn


class CycleCNNEncoder(nn.Module):
    """1D-CNN cycle encoder: (batch, 200, 6) -> (batch, embed_dim)."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, embed_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 200, 6) -> (batch, 6, 200)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)  # (batch, embed_dim)


class PatchTSTLiteEncoder(nn.Module):
    """
    Patches the (batch, W, embed_dim+1) sequence of [cycle_embedding,
    soh] tokens along the CYCLE dimension into non-overlapping patches
    of length patch_len, projects each patch to d_model, adds a learned
    positional embedding, runs a standard Transformer encoder, and
    mean-pools the output tokens into one context vector h_0 - the
    "initial dynamics state" the residual dynamics equation rolls
    forward from.
    """

    def __init__(self, token_dim: int, window: int, patch_len: int = 2,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 2):
        super().__init__()
        assert window % patch_len == 0, "window must be divisible by patch_len"
        self.patch_len = patch_len
        self.n_patches = window // patch_len
        self.patch_proj = nn.Linear(token_dim * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=d_model * 2,
                                            batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, W, token_dim)
        b, w, d = tokens.shape
        patches = tokens.reshape(b, self.n_patches, self.patch_len * d)
        x = self.patch_proj(patches) + self.pos_embed
        x = self.transformer(x)
        return self.out_proj(x.mean(dim=1))  # (batch, d_model)


class WorldModel(nn.Module):
    def __init__(self, embed_dim: int = 32, window: int = 10, patch_len: int = 2,
                 d_model: int = 64):
        super().__init__()
        self.window = window
        self.cnn_encoder = CycleCNNEncoder(in_channels=6, embed_dim=embed_dim)
        self.patchtst = PatchTSTLiteEncoder(token_dim=embed_dim + 1, window=window,
                                             patch_len=patch_len, d_model=d_model)
        # Residual dynamics equation: a GRUCell residual update in latent
        # space - h_{t+1} = h_t + GRUCell(h_t) reads oddly for a GRUCell
        # (which already returns the new full state); here it is used
        # literally as the "dynamics operator" D(h_t) -> h_{t+1}, and its
        # OWN internal gating already implements a residual-style update
        # (new state = interpolation between old state and a candidate),
        # matching the "residual dynamics equation" description without
        # a redundant second residual wrapper.
        self.dynamics = nn.GRUCell(input_size=1, hidden_size=d_model)  # driven by the previous delta
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode_context(self, cycle_tensors: torch.Tensor, soh_window: torch.Tensor) -> torch.Tensor:
        """cycle_tensors: (batch, W, 200, 6); soh_window: (batch, W).
        Returns h_0: (batch, d_model)."""
        b, w = soh_window.shape
        flat = cycle_tensors.reshape(b * w, cycle_tensors.shape[2], cycle_tensors.shape[3])
        z = self.cnn_encoder(flat).reshape(b, w, -1)  # (b, W, embed_dim)
        tokens = torch.cat([z, soh_window.unsqueeze(-1)], dim=-1)  # (b, W, embed_dim+1)
        return self.patchtst(tokens)

    def rollout(self, h0: torch.Tensor, soh_last: torch.Tensor, n_steps: int):
        """Iterative rollout in latent space only - no future raw curves
        needed. Returns predicted SOH trajectory (batch, n_steps)."""
        h = h0
        soh = soh_last
        prev_delta = torch.zeros_like(soh_last).unsqueeze(-1)
        preds = []
        for _ in range(n_steps):
            h = self.dynamics(prev_delta, h)
            delta = self.decoder(h).squeeze(-1)
            soh = soh + delta
            preds.append(soh)
            prev_delta = delta.unsqueeze(-1)
        return torch.stack(preds, dim=1)  # (batch, n_steps)

    def forward(self, cycle_tensors: torch.Tensor, soh_window: torch.Tensor, n_steps: int):
        h0 = self.encode_context(cycle_tensors, soh_window)
        return self.rollout(h0, soh_window[:, -1], n_steps)
