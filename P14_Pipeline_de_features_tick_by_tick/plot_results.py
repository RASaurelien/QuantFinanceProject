"""Trace le microprix, l'imbalance et la vol réalisée à partir du CSV produit par le pipeline Rust."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "tick_features_sample.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV introuvable: {csv_path}")

df = pd.read_csv(csv_path)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes[0].plot(df["index"], df["microprice"], color="steelblue", lw=0.8)
axes[0].plot(df["index"], df["ema_microprice"], color="firebrick", lw=1.0, label="EMA")
axes[0].set_title("Microprix (pondéré par la taille du côté opposé)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(df["index"], df["imbalance"], color="darkorange", lw=0.7)
axes[1].axhline(0, color="black", lw=0.6)
axes[1].set_title("Order flow imbalance ((bid_size - ask_size) / total)")
axes[1].grid(alpha=0.3)

axes[2].plot(df["index"], df["realized_vol"], color="seagreen", lw=1.0)
axes[2].set_title("Volatilité réalisée en ligne (Welford, cumulative)")
axes[2].set_xlabel("Index du tick")
axes[2].grid(alpha=0.3)

fig.suptitle("Pipeline de features tick-by-tick (5M ticks, échantillonné 1/5000)", fontsize=12)
fig.tight_layout()

output_dir = script_dir / "outputs"
output_dir.mkdir(exist_ok=True)

output_paths = [
    script_dir / "tick_features.png",
    script_dir / "tick_features.jpg",
    output_dir / "tick_features.png",
    output_dir / "tick_features.jpg",
]

for path in output_paths:
    if path.suffix.lower() == ".jpg":
        fig.savefig(path, dpi=140, format="jpeg", pil_kwargs={"quality": 95})
    else:
        fig.savefig(path, dpi=140)

plt.close(fig)
print("Graphiques exportés -> " + ", ".join(str(p) for p in output_paths))
