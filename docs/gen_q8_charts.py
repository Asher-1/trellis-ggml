#!/usr/bin/env python3
"""Generate Q8 vs F16 benchmark comparison charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Benchmark data ──────────────────────────────────────────────────────────
# CUDA quality=512, steps=12
cuda_512 = {
    "F16": {"time": 92.0, "verts": 1_867_007, "tris": 3_915_484},
    "Q8":  {"time": 91.2, "verts": 1_807_953, "tris": 3_839_432},
}
# CUDA coarse, steps=12
cuda_coarse = {
    "F16": {"time": 42.7, "verts": 83_576, "tris": 167_236},
    "Q8":  {"time": 43.0, "verts": 82_086, "tris": 164_264},
}
# CPU coarse, steps=12
cpu_coarse = {
    "F16": {"time": 780.8, "verts": 83_467, "tris": 166_982},
    # Q8 CPU still running; use CUDA ratio as estimate
}

# Model sizes (GB)
model_sizes = {"F16 (all)": 16.0, "Q8 (CUDA-safe)": 4.3}

colors = {"F16": "#2196F3", "Q8": "#FF9800"}
plt.rcParams.update({"font.size": 13, "figure.dpi": 150})

# ── Chart 1: Inference time comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CUDA 512
labels = list(cuda_512.keys())
times = [cuda_512[k]["time"] for k in labels]
bars = axes[0].bar(labels, times, color=[colors[l] for l in labels], width=0.5, edgecolor="white", linewidth=1.5)
axes[0].set_title("CUDA quality=512, 12 steps", fontweight="bold")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_ylim(0, max(times) * 1.15)
for bar, t in zip(bars, times):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{t:.1f}s", ha="center", va="bottom", fontweight="bold")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# CUDA coarse
labels = list(cuda_coarse.keys())
times = [cuda_coarse[k]["time"] for k in labels]
bars = axes[1].bar(labels, times, color=[colors[l] for l in labels], width=0.5, edgecolor="white", linewidth=1.5)
axes[1].set_title("CUDA coarse, 12 steps", fontweight="bold")
axes[1].set_ylabel("Time (seconds)")
axes[1].set_ylim(0, max(times) * 1.15)
for bar, t in zip(bars, times):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{t:.1f}s", ha="center", va="bottom", fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.suptitle("Q8 vs F16 Inference Time (CUDA)", fontweight="bold", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "benchmark_q8_time.png"), bbox_inches="tight")
plt.close(fig)

# ── Chart 2: Mesh quality comparison ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vertices
labels = ["F16", "Q8"]
verts_512 = [cuda_512["F16"]["verts"]/1e6, cuda_512["Q8"]["verts"]/1e6]
verts_coarse = [cuda_coarse["F16"]["verts"]/1e3, cuda_coarse["Q8"]["verts"]/1e3]

x = np.arange(2)
w = 0.35
bars1 = axes[0].bar(x - w/2, verts_512, w, label="quality=512", color="#4CAF50", edgecolor="white")
bars2 = axes[0].bar(x + w/2, verts_coarse, w, label="coarse", color="#8BC34A", edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Vertices")
axes[0].set_title("Mesh Vertex Count", fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, max(verts_512) * 1.2)
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{bar.get_height():.2f}M", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}K", ha="center", va="bottom", fontsize=10)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Triangles
tris_512 = [cuda_512["F16"]["tris"]/1e6, cuda_512["Q8"]["tris"]/1e6]
tris_coarse = [cuda_coarse["F16"]["tris"]/1e3, cuda_coarse["Q8"]["tris"]/1e3]

bars1 = axes[1].bar(x - w/2, tris_512, w, label="quality=512", color="#FF5722", edgecolor="white")
bars2 = axes[1].bar(x + w/2, tris_coarse, w, label="coarse", color="#FF8A65", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Triangles")
axes[1].set_title("Mesh Triangle Count", fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, max(tris_512) * 1.2)
for bar in bars1:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}M", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{bar.get_height():.0f}K", ha="center", va="bottom", fontsize=10)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.suptitle("Q8 vs F16 Mesh Quality (CUDA)", fontweight="bold", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "benchmark_q8_mesh.png"), bbox_inches="tight")
plt.close(fig)

# ── Chart 3: Model size comparison ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

labels = list(model_sizes.keys())
sizes = list(model_sizes.values())
bar_colors = ["#2196F3", "#FF9800"]
bars = ax.bar(labels, sizes, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Model Size (GB)")
ax.set_title("Model Weight Size: F16 vs Q8", fontweight="bold")
ax.set_ylim(0, max(sizes) * 1.2)
for bar, s in zip(bars, sizes):
    pct = s / max(sizes) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{s:.1f} GB\n({pct:.0f}%)", ha="center", va="bottom", fontweight="bold")

# Add reduction arrow
ax.annotate("", xy=(1, sizes[1] + 1.5), xytext=(0, sizes[0] + 1.5),
            arrowprops=dict(arrowstyle="->", lw=2, color="#F44336"))
ax.text(0.5, max(sizes) + 1.0, "73% reduction", ha="center", fontweight="bold",
        color="#F44336", fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "benchmark_q8_size.png"), bbox_inches="tight")
plt.close(fig)

# ── Chart 4: Combined summary dashboard ─────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Top-left: Time comparison (CUDA 512)
ax1 = fig.add_subplot(gs[0, 0])
labels = ["F16", "Q8"]
times_512 = [cuda_512["F16"]["time"], cuda_512["Q8"]["time"]]
bars = ax1.bar(labels, times_512, color=[colors[l] for l in labels], width=0.5)
ax1.set_title("CUDA 512 Time", fontweight="bold")
ax1.set_ylabel("Seconds")
ax1.set_ylim(0, max(times_512) * 1.2)
for bar, t in zip(bars, times_512):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{t:.1f}s", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Top-center: Mesh verts (512)
ax2 = fig.add_subplot(gs[0, 1])
verts = [cuda_512["F16"]["verts"]/1e6, cuda_512["Q8"]["verts"]/1e6]
bars = ax2.bar(labels, verts, color=[colors[l] for l in labels], width=0.5)
ax2.set_title("CUDA 512 Vertices", fontweight="bold")
ax2.set_ylabel("Millions")
ax2.set_ylim(0, max(verts) * 1.2)
for bar, v in zip(bars, verts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{v:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Top-right: Model size
ax3 = fig.add_subplot(gs[0, 2])
sizes_gb = [16.0, 4.3]
bars = ax3.bar(labels, sizes_gb, color=[colors[l] for l in labels], width=0.5)
ax3.set_title("Model Size (GB)", fontweight="bold")
ax3.set_ylabel("GB")
ax3.set_ylim(0, max(sizes_gb) * 1.2)
for bar, s in zip(bars, sizes_gb):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{s:.1f} GB", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Bottom row: Precision delta table as visual
ax4 = fig.add_subplot(gs[1, :])
ax4.axis("off")

table_data = [
    ["Metric", "F16 (CUDA)", "Q8 (CUDA)", "Δ (Q8 vs F16)"],
    ["Time (512, 12 steps)", "92.0s", "91.2s", "−0.9% (faster)"],
    ["Time (coarse, 12 steps)", "42.7s", "43.0s", "+0.7%"],
    ["Vertices (512)", "1,867,007", "1,807,953", "−3.2%"],
    ["Triangles (512)", "3,915,484", "3,839,432", "−1.9%"],
    ["Vertices (coarse)", "83,576", "82,086", "−1.8%"],
    ["Model weights size", "16.0 GB", "4.3 GB", "−73%"],
    ["12GB GPU feasible?", "No (OOM)", "Yes", "✓"],
]

table = ax4.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# Style header
for j in range(4):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Color the delta column
for i in range(1, len(table_data)):
    cell = table[i, 3]
    val = table_data[i][3]
    if "faster" in val or "Yes" in val or "✓" in val:
        cell.set_facecolor("#C8E6C9")
    elif "−" in val and "%" in val:
        cell.set_facecolor("#FFF9C4")
    else:
        cell.set_facecolor("#FFCDD2")

# Color F16/Q8 columns
for i in range(1, len(table_data)):
    table[i, 1].set_facecolor("#E3F2FD")
    table[i, 2].set_facecolor("#FFF3E0")

ax4.set_title("Q8 vs F16 Benchmark Summary", fontweight="bold", fontsize=14, pad=20)

fig.savefig(os.path.join(OUT, "benchmark_q8_summary.png"), bbox_inches="tight")
plt.close(fig)

print("Generated charts:")
for f in ["benchmark_q8_time.png", "benchmark_q8_mesh.png",
          "benchmark_q8_size.png", "benchmark_q8_summary.png"]:
    p = os.path.join(OUT, f)
    print(f"  {p}  ({os.path.getsize(p)/1024:.0f} KB)")
