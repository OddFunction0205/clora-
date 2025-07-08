import torch
import torch.nn as nn
import torch.nn.functional as F

class CLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=32, k=128, alpha=32, lambda_orth=1.0, device=None, bias=True):
        super().__init__()
        self.r = r
        self.k = k
        self.alpha = alpha
        self.lambda_orth = lambda_orth

        device = device or torch.device('cpu')

        # === 原始 Linear 权重（冻结，不训练） ===
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device), requires_grad=False) if bias else None

        # === 可训练低秩参数 ===
        self.A = nn.Parameter(torch.randn(out_features, r, device=device) * 0.01)
        self.B = nn.Parameter(torch.zeros(in_features, r, device=device))

        # === 正交缓冲区 ===
        from clora.utils import generate_orthogonal_matrix
        P_A_init = generate_orthogonal_matrix(out_features, k, device=device, dtype=torch.float16)
        P_B_init = generate_orthogonal_matrix(in_features, k, device=device, dtype=torch.float16)

        self.register_buffer('P_A', P_A_init.to(dtype=torch.float16))
        self.register_buffer('P_B', P_B_init.to(dtype=torch.float16))

    def forward(self, x):
        # LoRA 增量部分
        delta_w = (self.A @ self.B.T) * (self.alpha / self.r)
        delta_w = delta_w.to(x.dtype)

        # 总输出 = 原始权重 + LoRA 权重增量
        return F.linear(x, self.weight + delta_w, self.bias)

    def orthogonal_loss(self):
        loss_A = torch.norm(self.A.T @ self.P_A, p='fro') ** 2
        loss_B = torch.norm(self.B.T @ self.P_B, p='fro') ** 2
        return (loss_A + loss_B) * self.lambda_orth
