import re, sys, pandas as pd, matplotlib.pyplot as plt


pat = re.compile(
    r"Epoch\s+(\d+):.*?"
    r"loss=([0-9.+-eE]+).*?"                       # actual loss near the start of the line
    r"train/loss_simple_step=([0-9.+-eE]+).*?"
    r"train/loss_vlb_step=([0-9.+-eE]+).*?"
    r"train/loss_step=([0-9.+-eE]+)",
    re.IGNORECASE
)

#!/usr/bin/env python3
import re, sys, os
import pandas as pd
import matplotlib.pyplot as plt

# --- Inputs ---
log_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/YOURNAME/project/plantShade/logs/controlnet_out.txt"
# Where to save CSV/PNGs (fallback = alongside the log file)
out_dir = "/scratch/YOURNAME/project/plantShade/ControlNet/0out/ControlNet_vanilla_Tempe/2025-09-30_02-57-04/curves"

os.makedirs(out_dir, exist_ok=True)

# A robust floating number pattern (handles 0.1, .1, 1., 1e-3, 3.47e+7, etc.)
NUM = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
# Use look-ahead (?=,|\s|]) to avoid swallowing trailing commas or tokens.
pat = re.compile(
    rf"Epoch\s+(\d+):.*?"
    rf"loss={NUM}(?=,|\s|\])"                        # actual loss
    rf".*?train/loss_simple_step={NUM}(?=,|\s|\])"
    rf".*?train/loss_vlb_step={NUM}(?=,|\s|\])"
    rf".*?train/loss_step={NUM}(?=,|\s|\])",
    re.IGNORECASE
)

rows = []
with open(log_path, "r", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            epoch = int(m.group(1))
            loss_actual = float(m.group(2))
            loss_simple = float(m.group(3))
            loss_vlb = float(m.group(4))
            loss_step = float(m.group(5))
            rows.append((epoch, loss_actual, loss_simple, loss_vlb, loss_step))

if not rows:
    raise SystemExit("No matching lines found. Check the log format or regex.")

# Build DataFrame in found order
df = pd.DataFrame(
    rows,
    columns=["Epoch", "loss_actual", "loss_simple_step", "loss_vlb_step", "loss_step"]
)
print(df.to_string(index=False))

# Save CSV next to the log (or your chosen out_dir)
base = os.path.splitext(os.path.basename(log_path))[0]
out_csv = os.path.join(out_dir, f"{base}_parsed.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved CSV: {out_csv}")

# Helper to plot a single series (no subplots)
def plot_series(series_name, ylabel="Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[series_name], label=series_name)
    plt.xlabel("Occurrence index (order in log)")
    plt.ylabel(ylabel)
    plt.title(f"{series_name} (parsed in appearance order)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    png_path = os.path.join(out_dir, f"{base}_{series_name}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    print(f"Saved plot: {png_path}")

# Make one figure per series (single-by-single)
plot_series("loss_actual")
plot_series("loss_simple_step")
plot_series("loss_vlb_step")
plot_series("loss_step")

print("Done.")