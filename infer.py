# -*- coding: utf-8 -*-
"""本地语音识别推理脚本：加载 epoch_35 模型，识别单条音频"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# 让 wenet 包可被导入
WENET_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wenet_local')
sys.path.insert(0, WENET_LOCAL)

import torch
from wenet.cli.model import load_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asr_model_epoch35')


def recognize(wav_path, mode='ctc_greedy_search'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[infer] device = {device}')

    model = load_model(MODEL_DIR, device=device)
    model.eval()

    # 计算特征
    feat = model.compute_feature(wav_path)
    feat = feat.unsqueeze(0).to(device)
    feat_lengths = torch.tensor([feat.shape[1]], dtype=torch.long).to(device)

    with torch.no_grad():
        results = model.decode(
            [mode],
            feat,
            feat_lengths,
            beam_size=10,
            ctc_weight=0.3,
            reverse_weight=0.3,
        )

    hyps = results[mode]
    tokens = hyps[0].tokens
    text = model.tokenizer.detokenize(tokens)[0]
    return text


if __name__ == '__main__':
    wav = sys.argv[1] if len(sys.argv) > 1 else 'test_audio.wav'
    text = recognize(wav)
    print('=== 识别结果 ===')
    print(text)
