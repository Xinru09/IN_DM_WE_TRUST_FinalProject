from __future__ import annotations

import math
import torch
from torch import nn

class SoftmaxAttention(nn.Module):
    """Standard multi-head softmax attention with optional relative bias.
    Standard multi-head softmax attention with optional relative positional bias.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        relative_bias: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout)

        self.relative_bias = relative_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.relative_bias is not None:
            # Bias shape: (1, num_heads, seq_len, seq_len)
            # Bias shape: (1, num_heads, seq_len, seq_len)
            scores = scores + self.relative_bias(seq_len, device=x.device, dtype=x.dtype)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class CayleyRelativeBias(nn.Module):
    """Simplified Cayley-STRING-inspired 2D relative positional bias.
    Simplified Cayley-STRING-style 2D relative positional bias.

    We use a learnable skew-symmetric 2D parameter per head and map it through
    the Cayley transform to obtain an orthogonal rotation. The resulting rotated
    coordinates define a relative positional bias.

    Each head uses a learnable 2D skew-symmetric parameter, mapped through the Cayley transform to obtain
    an orthogonal rotation matrix, then uses the rotated coordinates to construct relative positional bias.
    """
    def __init__(self, grid_size: int, num_heads: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_heads = num_heads

        self.theta = nn.Parameter(torch.zeros(num_heads))
        self.scale = nn.Parameter(torch.ones(num_heads) * 0.01)

        coords = []
        denom = max(grid_size - 1, 1)
        for row in range(grid_size):
            for col in range(grid_size):
                # Normalise coordinates into approximately [-1, 1].
                # Normalize coordinates to approximately [-1, 1].
                x = 2.0 * row / denom - 1.0
                y = 2.0 * col / denom - 1.0
                coords.append((x, y))

        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        self.register_buffer("coords", coords_tensor, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        patch_tokens = seq_len - 1
        expected_patches = self.grid_size * self.grid_size

        if patch_tokens != expected_patches:
            raise ValueError(
                f"CayleyRelativeBias expected {expected_patches} patch tokens, got {patch_tokens}."
            )

        coords = self.coords.to(device=device, dtype=dtype)  # (patches, 2)
        theta = self.theta.to(device=device, dtype=dtype)
        scale = self.scale.to(device=device, dtype=dtype)

        # Cayley rotation for a 2D skew matrix.
        # Cayley rotation corresponding to a 2D skew-symmetric matrix.
        denom = 1.0 + theta.square()
        cos_term = (1.0 - theta.square()) / denom
        sin_term = (2.0 * theta) / denom

        patch_bias_heads = []
        for h in range(self.num_heads):
            rotation = torch.stack([
                torch.stack([cos_term[h], -sin_term[h]]),
                torch.stack([sin_term[h], cos_term[h]]),
            ])

            rotated = coords @ rotation.T
            diff = rotated[:, None, :] - coords[None, :, :]
            dist_sq = diff.square().sum(dim=-1)

            # Negative distance gives larger bias to closer/compatible positions.
            # Negative distance gives closer/more compatible positions a larger bias.
            patch_bias = -scale[h].abs() * dist_sq
            patch_bias_heads.append(patch_bias)

        patch_bias = torch.stack(patch_bias_heads, dim=0)

        full_bias = torch.zeros(
            1,
            self.num_heads,
            seq_len,
            seq_len,
            device=device,
            dtype=dtype,
        )
        full_bias[:, :, 1:, 1:] = patch_bias.unsqueeze(0)
        return full_bias

# Part 6: ViT Backbone Network
class PatchEmbedding(nn.Module):
    """Convert image to patch tokens.
    Convert images into patch tokens.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """Feed-forward network.
    Feed-forward network.
    """
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """One transformer encoder block.
    A Transformer encoder block.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        model_type: str,
        grid_size: int,
        performer_features: int = 64,
        performer_redraw_projection: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        if model_type == "baseline":
            self.attention = SoftmaxAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                relative_bias=None,
            )
        elif model_type == "performer":
            self.attention = PerformerAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                nb_features=performer_features,
                redraw_projection=performer_redraw_projection,
            )
        elif model_type == "circulant_string":
            self.attention = SoftmaxAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                relative_bias=CirculantRelativeBias(grid_size=grid_size, num_heads=num_heads),
            )
        elif model_type == "cayley_string":
            self.attention = SoftmaxAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                relative_bias=CayleyRelativeBias(grid_size=grid_size, num_heads=num_heads),
            )
        else:
            raise ValueError(
                "model_type must be one of: baseline, performer, circulant_string, cayley_string"
            )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Unified ViT wrapper.
    Unified ViT wrapper.
    """
    def __init__(
        self,
        model_type: str,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        performer_features: int = 64,
        performer_redraw_projection: bool = False,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                model_type=model_type,
                grid_size=grid_size,
                performer_features=performer_features,
                performer_redraw_projection=performer_redraw_projection,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        batch_size = tokens.size(0)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, tokens), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)

        for block in self.encoder:
            x = block(x)

        x = self.norm(x[:, 0])
        logits = self.head(x)
        return logits


def build_cayley_string_vit(num_classes: int, config) -> VisionTransformer:
    return VisionTransformer(
        model_type="cayley_string",
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_classes=num_classes,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        performer_features=config.performer_features,
        performer_redraw_projection=config.performer_redraw_projection,
    )
