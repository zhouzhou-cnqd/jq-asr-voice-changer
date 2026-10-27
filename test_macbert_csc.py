"""测试 macbert4csc 中文拼写纠错模型"""
import sys
import time

from transformers import BertForMaskedLM, BertTokenizerFast
import torch

MODEL_PATH = 'models/macbert4csc'

print('[1] 加载 tokenizer ...', flush=True)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

print('[2] 加载模型 ...', flush=True)
t0 = time.time()
model = BertForMaskedLM.from_pretrained(MODEL_PATH)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f'[3] 模型加载完成，device={device}，耗时 {time.time()-t0:.1f}s', flush=True)


def correct(text):
    """对中文句子做拼写纠错，返回纠正后的句子"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        # 取每个位置 argmax 作为纠错结果
        pred_ids = torch.argmax(logits, dim=-1)[0]
        tokens = tokenizer.convert_ids_to_tokens(pred_ids)
        # 还原文本
        corrected = ''.join(t.replace('##', '') for t in tokens)
        corrected = corrected.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        corrected = corrected.replace(' ', '')
        return corrected


tests = [
    '少先队员因该为老人让坐',
    '今天新情很好',
    '你找到你最喜欢的工作，我也很高心。',
    '甚至出现交易几乎停滞的情况',  # 之前测试音频的正确文本
]

print('\n' + '=' * 60)
for t in tests:
    t0 = time.time()
    r = correct(t)
    print(f'原始: {t}')
    print(f'纠错: {r}  ({time.time()-t0:.2f}s)')
    print('-' * 60)
