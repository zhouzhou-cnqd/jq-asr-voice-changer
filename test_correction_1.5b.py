"""测试 chinese-text-correction-1.5b 中文纠错模型（Qwen2.5 基座，生成式）"""
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = 'models/chinese-text-correction-1.5b'

print('[1] 加载 tokenizer ...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print('[2] 加载模型 ...', flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map='auto'
)
model.eval()
device = next(model.parameters()).device
print(f'[3] 模型加载完成，device={device}，耗时 {time.time()-t0:.1f}s', flush=True)
if torch.cuda.is_available():
    print(f'    显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB', flush=True)


def correct(text):
    """用 1.5B 模型对中文句子纠错，返回纠正后文本"""
    input_content = '文本纠错:\n' + text
    messages = [{'role': 'user', 'content': input_content}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.08,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉输入部分，只保留生成结果
    result = decoded[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
    return result


tests = [
    '少先队员因该为老人让坐',
    '今天新情很好',
    '你找到你最喜欢的工作，我也很高心。',
    '你们痛知都为了吗',  # 之前 ASR 识别乱码的句子
    '机七学习是人工智能领遇最能体现智能的一个分知',
]

print('\n' + '=' * 60)
for t in tests:
    t0 = time.time()
    r = correct(t)
    print(f'原始: {t}')
    print(f'纠错: {r}  ({time.time()-t0:.2f}s)')
    print('-' * 60)
