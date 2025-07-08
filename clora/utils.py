import torch

def generate_orthogonal_matrix(dim, k, device='cuda', dtype=torch.float16):
    # Step 1: 在 GPU 上用 float32 做 QR
    rand = torch.randn(dim, k, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(rand)

    # Step 2: cast 回目标 dtype（如 float16）
    return q[:, :k].to(dtype=dtype)



