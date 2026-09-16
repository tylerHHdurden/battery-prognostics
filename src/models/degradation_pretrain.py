"""
Stage 7.2: degradation-aligned self-supervised pretraining.

Pretext task: cycle-order-ranking. Given two raw cycle tensors (c_i, c_j)
sampled from the SAME battery's own cycle sequence, predict (a) which
came first (binary: did j occur after i in real cycle-index order) and
(b) how far apart they are in degradation progress (regression on
log1p(|cycle_idx_j - cycle_idx_i|)). Needs NO SOH label - only the raw
cycle tensor and its own cycle index, which this project's SOH-per-cycle
convention normally computes as ONE combined step; the pretext task
deliberately withholds SOH and uses only ordering, so the encoder must
learn to recognize degradation PROGRESS directly from curve SHAPE, not
be handed the answer.

DISCLOSED SCOPE NOTE (this project's data does not literally have "far
more unlabeled than labeled cycles" the way e.g. unlabeled images
outnumber labeled ones in vision - every raw cycle in this project's
pool already has a computable SOH from its own capacity-fade
convention). The genuine exploitable signal here is COMBINATORIAL, not
an extra unlabeled data pool: a battery with ~700 cycles yields up to
~700 choose 2 (~244,000) ORDERING PAIRS versus only 700 individual SOH
labels - two full orders of magnitude more training signal for the
pretext task than the eventual fine-tuning task sees, which is the
actual mechanism being tested (does more self-supervised PAIRWISE
signal, from the SAME raw curves, produce a better starting
representation than random initialization) - reported honestly here
rather than the "extra unlabeled data" framing that doesn't literally
apply to this project.

Uses models.world_model.CycleCNNEncoder (the SAME small CNN cycle
encoder architecture as the World Model, item 7.1) so this stage is
testing "does pretraining help THIS class of encoder," not introducing
a fourth unrelated architecture.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_model import CycleCNNEncoder


class OrderRankingHead(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        )
        self.order_head = nn.Linear(32, 1)  # logit: P(j after i)
        self.gap_head = nn.Linear(32, 1)    # regresses log1p(|idx_j - idx_i|)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        feat = torch.cat([z_i, z_j, z_i - z_j], dim=-1)
        h = self.net(feat)
        return self.order_head(h).squeeze(-1), self.gap_head(h).squeeze(-1)


class PretrainableSOHModel(nn.Module):
    """Wraps CycleCNNEncoder with an SOH regression head - used both for
    the pretrained-then-fine-tuned model and the random-init baseline
    (same architecture, only the encoder's starting weights differ)."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.encoder = CycleCNNEncoder(in_channels=6, embed_dim=embed_dim)
        self.head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)
