import os, time
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "cuda:", torch.version.cuda)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def fn(x):
    # 放点融合机会：pointwise + reduction
    y = torch.sin(x) * torch.cos(x) + 1.0
    return (y * y).sum()

opt_fn = torch.compile(fn, backend="inductor", dynamic=False)

x = torch.randn(4096, 4096, device=device, dtype=dtype)

# 第一次：通常会触发编译
t0 = time.time()
out = opt_fn(x)
if device == "cuda":
    torch.cuda.synchronize()
print("first run:", out.item(), "time(s)=", round(time.time()-t0, 4))

# 第二次：应主要走缓存/复用（明显更快、日志里有 reuse/skip compile）
t0 = time.time()
out = opt_fn(x)
if device == "cuda":
    torch.cuda.synchronize()
print("second run:", out.item(), "time(s)=", round(time.time()-t0, 4))

print("cache dir:", os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
print("triton cache:", os.environ.get("TRITON_CACHE_DIR"))