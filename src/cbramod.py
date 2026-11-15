"""CBraMod, ported from https://github.com/wjq-learning/CBraMod (commit b9e9610).

Module names match the official code so the HuggingFace checkpoint loads with strict=True.
Masking (pretraining only) is removed. scripts/check_parity.py checks equality with the original.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_SIZE = 200  # 1 s at 200 Hz
D_MODEL = 200


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # ACPE: depthwise conv over the (channel x patch) grid, long range across channels (19), short in time (7)
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(D_MODEL, D_MODEL, kernel_size=(19, 7), padding=(9, 3), groups=D_MODEL),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(PATCH_SIZE), requires_grad=False)  # unused, kept for loading
        # Time branch: each patch -> 25 feature maps x 8 steps = 200
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        # Frequency branch: |rFFT| of a 200-sample patch has 101 bins
        self.spectral_proj = nn.Sequential(nn.Linear(101, D_MODEL), nn.Dropout(0.1))

    def forward(self, x):  # x: (B, C, n, 200)
        b, c, n, t = x.shape
        time_emb = self.proj_in(x.reshape(b, 1, c * n, t))  # (B, 25, C*n, 8)
        time_emb = time_emb.permute(0, 2, 1, 3).reshape(b, c, n, D_MODEL)
        spectrum = torch.fft.rfft(x, dim=-1, norm="forward").abs()  # (B, C, n, 101)
        emb = time_emb + self.spectral_proj(spectrum)
        pos = self.positional_encoding(emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return emb + pos


class CrissCrossLayer(nn.Module):
    """Pre-norm transformer layer: half of the features attend across channels, half across time."""

    def __init__(self, nhead=8, dim_feedforward=800, dropout=0.1):
        super().__init__()
        half = D_MODEL // 2
        self.self_attn_s = nn.MultiheadAttention(half, nhead // 2, dropout=dropout, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(half, nhead // 2, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(D_MODEL, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.criss_cross_attention(self.norm1(x)))
        x = x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))
        return x

    def criss_cross_attention(self, x):  # x: (B, C, n, d)
        b, c, n, d = x.shape
        xs = x[..., : d // 2].transpose(1, 2).reshape(b * n, c, d // 2)  # spatial: sequence over channels
        xt = x[..., d // 2 :].reshape(b * c, n, d // 2)  # temporal: sequence over patches
        xs = self.self_attn_s(xs, xs, xs, need_weights=False)[0].reshape(b, n, c, d // 2).transpose(1, 2)
        xt = self.self_attn_t(xt, xt, xt, need_weights=False)[0].reshape(b, c, n, d // 2)
        return torch.cat([xs, xt], dim=-1)


class Encoder(nn.Module):
    def __init__(self, n_layer=12):
        super().__init__()
        self.layers = nn.ModuleList([CrissCrossLayer() for _ in range(n_layer)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CBraMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.encoder = Encoder()
        self.proj_out = nn.Sequential(nn.Linear(D_MODEL, PATCH_SIZE))  # reconstruction head
        self.apply(_init_weights)

    def forward(self, x):
        return self.proj_out(self.encoder(self.patch_embedding(x)))


def _init_weights(m):
    if isinstance(m, nn.Linear):  # as in the official code
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class CBraModClassifier(nn.Module):
    """Backbone + binary head. head='large' is the official SHU-MI head (`all_patch_reps`),
    head='small' averages patch representations and applies one linear layer."""

    def __init__(self, n_channels, n_patches, head, dropout, weights=None):
        super().__init__()
        self.backbone = CBraMod()
        if weights:
            self.backbone.load_state_dict(torch.load(weights, map_location="cpu"))
        self.backbone.proj_out = nn.Identity()
        self.head = head
        if head == "large":
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_channels * n_patches * D_MODEL, n_patches * D_MODEL),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(n_patches * D_MODEL, D_MODEL),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(D_MODEL, 1),
            )
        elif head == "small":
            self.classifier = nn.Linear(D_MODEL, 1)
        else:
            raise ValueError(f"unknown head: {head}")

    def forward(self, x):  # x: (B, C, T) -> logits (B,)
        b, c, t = x.shape
        feats = self.backbone(x.reshape(b, c, t // PATCH_SIZE, PATCH_SIZE))  # (B, C, n, d)
        if self.head == "small":
            feats = feats.mean(dim=(1, 2))
        return self.classifier(feats).squeeze(-1)
