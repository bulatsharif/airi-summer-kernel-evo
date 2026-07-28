import torch


if __name__ == "__main__":
    torch.manual_seed(0)
    left = torch.randn((1024, 1024), device="cuda")
    right = torch.randn((1024, 1024), device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    print(f"result={result[0, 0].item():.6f}")
