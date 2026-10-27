# -*- coding: utf-8 -*-
"""重新生成训练曲线图：边框黑色、线条蓝色。"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 颜色规范
BORDER = "black"   # 边框（坐标轴/刻度）
LINE = "#1f6fd6"   # 线条蓝色

with open("curve_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

epochs = [d["epoch"] for d in data]
train_loss = [d["train_loss"] for d in data]
cv_loss = [d["cv_loss"] for d in data]
cv_acc = [d["cv_acc"] for d in data]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1.2)
    ax.tick_params(colors=BORDER, which="both")
    ax.xaxis.label.set_color(BORDER)
    ax.yaxis.label.set_color(BORDER)
    ax.title.set_color(BORDER)
    ax.grid(True, linestyle="--", alpha=0.3, color=BORDER)

# 1) Train Loss
ax = axes[0]
ax.plot(epochs, train_loss, color=LINE, linewidth=1.8, label="Train Loss")
ax.set_title("训练损失 (Train Loss)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
style_ax(ax)

# 2) CV Loss
ax = axes[1]
ax.plot(epochs, cv_loss, color=LINE, linewidth=1.8, label="CV Loss")
ax.set_title("验证损失 (CV Loss)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
style_ax(ax)

# 3) CV Accuracy
ax = axes[2]
ax.plot(epochs, cv_acc, color=LINE, linewidth=1.8, label="CV Accuracy")
ax.set_title("验证准确率 (CV Accuracy)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
style_ax(ax)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150, facecolor="white", bbox_inches="tight")
print("done -> training_curve.png")
