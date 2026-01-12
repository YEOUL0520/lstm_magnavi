import importlib

pkgs = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
}

print("🔍 Package check\n" + "-"*30)
for name, mod in pkgs.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "unknown")
        print(f"✅ {name:12s} {ver}")
    except Exception as e:
        print(f"❌ {name:12s} NOT INSTALLED")

print("\n🔥 Torch CUDA check")
try:
    import torch
    print(" torch version :", torch.__version__)
    print(" cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(" cuda device   :", torch.cuda.get_device_name(0))
except Exception as e:
    print(" torch error:", e)