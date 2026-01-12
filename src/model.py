# =========================
# LSTM Positioning Model
# =========================
# [논문 반영 포인트]
# - LSTM hidden layers: 3
# - hidden nodes: 600
# - 출력: (x_norm, y_norm) 2차원 회귀
#
# [현재 Hyena 프로젝트 데이터와의 정합]
# - 입력: JSONL의 features -> Tensor shape [B, T, F]
#   (T=window_size=250, F=n_features=meta["n_features"])
# - target: [B, 2]
#
# [주의]
# - PyTorch LSTM 내부 활성함수는 기본적으로 tanh/sigmoid를 사용하므로,
#   논문 표의 "tanh"와 자연스럽게 정합됨.
# - 논문 표에 ReLU도 언급되어 있으나 LSTM cell 내부를 ReLU로 바꾸는 건 일반적이지 않음.
#   필요하면 FC 앞/뒤에 ReLU를 붙이는 ablation으로만 실험 권장.
import math
import torch
import torch.nn as nn



class LSTMPositioning(nn.Module):
    """
    Many-to-one LSTM regressor.

    Input:
      x: [B, T, F]
    Output:
      y: [B, 2]  (x_norm, y_norm)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 600,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_fc_relu: bool = False,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.use_fc_relu = use_fc_relu
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, 2)

        # FC 초기화(선택): 학습 안정성
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)      # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        if self.use_fc_relu:
            last = self.relu(last)
        return self.fc(last)       # [B, 2]