"""Hyena 모델 정의 - Sliding Window용"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]  # (seq_len, dim)


class ImplicitFilter(nn.Module):
    """작은 MLP로 긴 필터 생성 (Hyena의 핵심)"""

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        # 위치 [0, 1, ..., seq_len-1] 생성
        positions = torch.linspace(0, 1, seq_len, device=next(self.parameters()).device)
        positions = positions.unsqueeze(-1)  # (seq_len, 1)
        filter_weights = self.mlp(positions)  # (seq_len, dim)
        return filter_weights


class HyenaOperator(nn.Module):
    """진짜 Hyena: Implicit filter + Short conv + FFT long conv + Multiple gates"""

    def __init__(self, dim: int, order: int = 2):
        super().__init__()
        self.dim = dim
        self.order = order  # gating paths 개수

        # Implicit long filter
        self.implicit_filter = ImplicitFilter(dim)

        # Short convolution (data-controlled)
        self.short_conv = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1, groups=dim  # depthwise conv
        )

        # Projections for multiple paths (v, u, z, ...)
        self.in_proj = nn.Linear(dim, dim * (order + 1))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # Multiple paths projection
        proj = self.in_proj(x)  # (batch, seq_len, dim * (order+1))
        paths = proj.chunk(self.order + 1, dim=-1)  # List of (batch, seq_len, dim)

        v = paths[0]  # 값 경로

        # Implicit filter 생성
        filt = self.implicit_filter(seq_len)  # (seq_len, dim)

        # Short convolution (data-controlled)
        u_input = paths[1].transpose(1, 2)  # (batch, dim, seq_len)
        u_short = self.short_conv(u_input).transpose(1, 2)  # (batch, seq_len, dim)

        # FFT long convolution
        # filt와 u_short를 element-wise 곱한 뒤 FFT conv
        # cuFFT는 half precision에서 2의 거듭제곱 크기만 지원하므로 float32로 변환
        orig_dtype = u_short.dtype
        u_short_f32 = u_short.float()
        filt_f32 = filt.float()

        U = torch.fft.rfft(u_short_f32, dim=1)  # (batch, freq, dim)
        Filt = torch.fft.rfft(filt_f32.unsqueeze(0), n=seq_len, dim=1)  # (1, freq, dim)
        filtered = torch.fft.irfft(U * Filt, n=seq_len, dim=1)  # (batch, seq_len, dim)

        # 원래 dtype으로 복원
        filtered = filtered.to(orig_dtype)

        # Multiple gating: v * filtered * z (if order >= 2)
        output = v * filtered

        if self.order >= 2:
            z = paths[2]
            output = output * torch.sigmoid(z)

        return self.out_proj(output)


class HyenaBlock(nn.Module):
    """Hyena Block with normalization and residual"""

    def __init__(self, dim: int, order: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.hyena = HyenaOperator(dim, order=order)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        """
        h = self.norm(x)
        out = self.hyena(h)
        out = self.dropout(out)
        return out + x  # residual connection


class HyenaPositioning(nn.Module):
    """Hyena 기반 실내 측위 모델 - Sliding Window용"""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 2,  # 항상 2 (x, y)
        depth: int = 8,
        order: int = 2,
        dropout: float = 0.1,
        num_edge_types: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 입력 projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Edge path embedding (방향성 인코딩)
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim)

        # Hyena blocks
        self.blocks = nn.ModuleList(
            [HyenaBlock(hidden_dim, order=order, dropout=dropout) for _ in range(depth)]
        )

        # Output head (각 타임스텝마다 좌표 예측)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),  # (x, y)
        )

    def forward(
        self, x: torch.Tensor, edge_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        edge_ids: (batch,) - edge path의 ID (방향성 구분)
        returns: (batch, seq_len, output_dim)
        """
        batch, seq_len, _ = x.shape

        # Input projection
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        pos = self.pos_encoding(seq_len)  # (seq_len, hidden_dim)
        h = h + pos.unsqueeze(0)  # broadcast

        # Add edge path embedding (방향성 정보)
        if edge_ids is not None:
            edge_emb = self.edge_embedding(edge_ids)  # (batch, hidden_dim)
            h = h + edge_emb.unsqueeze(1)  # broadcast to all timesteps

        # Hyena blocks
        for block in self.blocks:
            h = block(h)

        # Output head (각 타임스텝마다 좌표 예측)
        h = self.norm(h)
        coords = self.head(h)  # (batch, seq_len, output_dim)

        return coords