# -*- coding: utf-8 -*-
"""把 gif_frames 里的逐帧 PNG 合成为 demo.gif"""
import os
from PIL import Image

FRAME_DIR = "gif_frames"
OUT = "demo.gif"
FPS = 12
DURATION = int(1000 / FPS)  # 每帧毫秒
TARGET_WIDTH = 900  # 缩小到 900px 宽，控制体积

frames = sorted(f for f in os.listdir(FRAME_DIR) if f.endswith(".png"))
imgs = []
for f in frames:
    img = Image.open(os.path.join(FRAME_DIR, f)).convert("RGB")
    if img.width > TARGET_WIDTH:
        h = int(img.height * TARGET_WIDTH / img.width)
        img = img.resize((TARGET_WIDTH, h), Image.LANCZOS)
    # 量化到 256 色以减小体积
    img = img.quantize(colors=256, method=Image.MEDIANCUT)
    imgs.append(img)

# 最后一帧多停留一会儿
durations = [DURATION] * len(imgs)
durations[-1] = 2000

imgs[0].save(
    OUT,
    save_all=True,
    append_images=imgs[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
size = os.path.getsize(OUT) / 1024
print(f"完成 -> {OUT} ({len(imgs)} 帧, {size:.0f} KB)")
