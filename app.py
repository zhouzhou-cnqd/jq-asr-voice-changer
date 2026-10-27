# -*- coding: utf-8 -*-
"""语音识别 Web Demo 后端
- 启动时加载 epoch_35 模型（只加载一次）
- POST /recognize 接收音频文件，返回识别文字
- GET / 返回前端页面
零依赖（仅用 Python 标准库 + torch + wenet）
"""
import sys
import os
import io
import json
import time
import wave
import struct
import logging
import tempfile
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(message)s')

BASE = os.path.dirname(os.path.abspath(__file__))
WENET_LOCAL = os.path.join(BASE, 'wenet_local')
sys.path.insert(0, WENET_LOCAL)

import torch
from wenet.cli.model import load_model

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft

MODEL_DIR = os.path.join(BASE, 'asr_model_epoch99')

# 本地中文纠错模型（MacBERT CSC）
CSC_MODEL_PATH = os.path.join(BASE, 'models', 'macbert4csc')
_csc_tokenizer = None
_csc_model = None


def get_csc_model():
    """懒加载本地 MacBERT 中文拼写纠错模型"""
    global _csc_tokenizer, _csc_model
    if _csc_model is None:
        from transformers import BertForMaskedLM, BertTokenizerFast
        print('[server] loading local CSC model (macbert4csc) ...', flush=True)
        _csc_tokenizer = BertTokenizerFast.from_pretrained(CSC_MODEL_PATH)
        _csc_model = BertForMaskedLM.from_pretrained(CSC_MODEL_PATH)
        _csc_model.eval()
        _csc_model.to(_device)
        print('[server] local CSC model loaded.', flush=True)
    return _csc_tokenizer, _csc_model


# 本地 1.5B 中文纠错模型（Qwen2.5 基座，生成式，懂语义）
CORRECTION_15B_PATH = os.path.join(BASE, 'models', 'chinese-text-correction-1.5b')
_correction_tokenizer = None
_correction_model = None


def get_correction_15b_model():
    """懒加载本地 1.5B 中文纠错模型"""
    global _correction_tokenizer, _correction_model
    if _correction_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print('[server] loading 1.5B correction model ...', flush=True)
        _correction_tokenizer = AutoTokenizer.from_pretrained(CORRECTION_15B_PATH)
        _correction_model = AutoModelForCausalLM.from_pretrained(
            CORRECTION_15B_PATH, dtype=torch.float16, device_map='auto'
        )
        _correction_model.eval()
        print('[server] 1.5B correction model loaded.', flush=True)
    return _correction_tokenizer, _correction_model

# 加载配置（API key 等）
CONFIG_PATH = os.path.join(BASE, 'config.json')
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            CONFIG = json.load(f)
    except Exception as e:
        print(f'[server] failed to load config: {e}', flush=True)

# 全局模型（惰性加载一次）
_model = None
_device = None


def get_model():
    global _model, _device
    if _model is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[server] loading model on {_device} ...', flush=True)
        _model = load_model(MODEL_DIR, device=_device)
        print('[server] load_model returned', flush=True)
        _model.eval()
        print('[server] model loaded.', flush=True)
    return _model, _device


def diagnose_wav(audio_bytes):
    """分析 wav 音频：时长、采样率、声道、最大/平均振幅，用于排查录音问题"""
    try:
        buf = io.BytesIO(audio_bytes)
        wf = wave.open(buf, 'rb')
        nch = wf.getnchannels()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        sw = wf.getsampwidth()
        duration = nframes / float(sr) if sr else 0
        raw = wf.readframes(nframes)
        wf.close()
        # 计算振幅（16bit）
        if sw == 2:
            samples = struct.unpack('<%dh' % (len(raw) // 2), raw)
        elif sw == 1:
            samples = [b * 256 for b in raw]
        else:
            samples = []
        info = {
            'channels': nch,
            'sample_rate': sr,
            'sampwidth': sw,
            'frames': nframes,
            'duration_sec': round(duration, 3),
            'bytes': len(audio_bytes),
        }
        if samples:
            abs_samples = [abs(s) for s in samples]
            peak = max(abs_samples)
            avg = sum(abs_samples) / len(abs_samples)
            info['peak'] = peak
            info['avg_abs'] = round(avg, 1)
            info['peak_db'] = round(20 * __import__('math').log10(max(peak, 1) / 32768.0), 1)
        return info
    except Exception as e:
        return {'error': str(e)}


def spectral_subtraction(audio_bytes, target_sr=16000, noise_est_ratio=0.1,
                         alpha=1.5, gain_floor=0.01):
    """谱减法去噪 + 能量门控。

    流程：
      1. 解码 wav 到 float 单声道，重采样到 target_sr
      2. 用前 noise_est_ratio 比例的帧估计噪声谱
      3. STFT -> 减去 alpha*噪声谱 -> 能量门控 -> ISTFT 还原
    返回 (去噪后 wav 字节, 信息字典)
    """
    try:
        buf = io.BytesIO(audio_bytes)
        sr, data = wavfile.read(buf)
        if data.dtype != np.int16:
            data = (data / np.max(np.abs(data)) * 32767).astype(np.int16)
        # 转单声道 float [-1, 1]
        if data.ndim == 2:
            x = data.mean(axis=1)
        else:
            x = data
        x = x.astype(np.float32) / 32768.0

        # 重采样到 16k（如果不同）
        if sr != target_sr:
            from scipy.signal import resample_poly
            import fractions
            g = fractions.gcd(sr, target_sr) if hasattr(fractions, 'gcd') else np.gcd(sr, target_sr)
            up = target_sr // g
            down = sr // g
            x = resample_poly(x, up, down).astype(np.float32)

        # 短时傅里叶变换
        n_fft = 512
        hop = 160  # 10ms @ 16k
        win = np.hanning(n_fft).astype(np.float32)
        f, t, Zxx = stft(x, fs=target_sr, window=win, nperseg=n_fft,
                         noverlap=n_fft - hop, boundary='zeros', padded=True)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # 噪声估计：取前 noise_est_ratio 比例的帧的平均幅度谱
        n_frames = mag.shape[1]
        n_noise = max(1, int(n_frames * noise_est_ratio))
        noise_mag = np.mean(mag[:, :n_noise], axis=1, keepdims=True)

        # 谱减法
        sub = mag - alpha * noise_mag
        # 能量门控：低于 gain_floor*噪声谱的成分压到底
        sub = np.maximum(sub, gain_floor * noise_mag)

        # 还原
        _, x_denoised = istft(sub * np.exp(1j * phase), fs=target_sr, window=win,
                              nperseg=n_fft, noverlap=n_fft - hop, boundary='zeros')
        # 裁掉首尾 padding（各 n_fft//2 点）
        pad = n_fft // 2
        if len(x_denoised) > 2 * pad:
            x_denoised = x_denoised[pad:-pad]
        # 归一化到 int16
        peak = np.max(np.abs(x_denoised)) + 1e-8
        x_denoised = x_denoised / peak * 0.95
        x_denoised = (x_denoised * 32767).astype(np.int16)

        # 写回 wav 字节
        out_buf = io.BytesIO()
        wavfile.write(out_buf, target_sr, x_denoised)
        info = {
            'orig_sr': sr,
            'target_sr': target_sr,
            'n_frames': int(n_frames),
            'noise_frames': int(n_noise),
            'peak_db': round(20 * np.log10(max(float(np.max(np.abs(x_denoised))), 1e-6) / 32767.0), 1),
        }
        return out_buf.getvalue(), info
    except Exception as e:
        return None, {'error': str(e)}


def infer_from_bytes(audio_bytes, tag='raw'):
    """从 wav 字节做识别，返回 (text, elapsed)。tag 仅用于日志。

    同时把原始音频落盘到 recordings/，便于离线调参评估。
    """
    model, device = get_model()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_bytes)
        wav_path = f.name
    try:
        t0 = time.time()
        feat = model.compute_feature(wav_path)
        if feat.shape[0] < 10:
            print(f'[server] {tag}: audio too short ({feat.shape[0]} frames)', flush=True)
            return '', 0.0
        feat = feat.unsqueeze(0).to(device)
        feat_lengths = torch.tensor([feat.shape[1]], dtype=torch.long).to(device)
        with torch.no_grad():
            # attention_rescoring：CTC 前缀束搜索 + attention 解码器重打分，
            # 对同音字/连续语音识别明显优于 ctc_greedy_search
            results = model.decode(
                ['attention_rescoring'],
                feat,
                feat_lengths,
                beam_size=10,
                ctc_weight=0.3,
                reverse_weight=0.3,
            )
        tokens = results['attention_rescoring'][0].tokens
        text = model.tokenizer.detokenize(tokens)[0]
        elapsed = time.time() - t0
        print(f'[server] {tag}: {elapsed:.2f}s -> {text}', flush=True)

        # —— 落盘保存原始音频，便于离线调参 ——
        try:
            save_dir = os.path.join(BASE, 'recordings')
            os.makedirs(save_dir, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(save_dir, f'{tag}_{ts}.wav')
            with open(save_path, 'wb') as wf:
                wf.write(audio_bytes)
            # 记录识别结果到元数据文件
            meta_path = os.path.join(save_dir, 'recognition_log.tsv')
            with open(meta_path, 'a', encoding='utf-8') as mf:
                mf.write(f'{ts}\t{tag}\t{text}\t{os.path.basename(save_path)}\n')
        except Exception as e:
            print(f'[server] save audio failed: {e}', flush=True)

        return text, elapsed
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def recognize_bytes(audio_bytes):
    """接收 wav 音频字节，返回识别文字（保留兼容）"""
    text, _ = infer_from_bytes(audio_bytes, 'raw')
    return text


def llm_correct(raw_text, api_key=None, base_url=None, model=None):
    """用 DeepSeek/OpenAI 兼容 API 对 ASR 结果做语义纠错，返回 (纠正后文本, 错误信息或None, usage字典或None)

    参数优先级：显式传入 > config.json > 默认值。
    支持任意 OpenAI 兼容的 chat/completions 接口（DeepSeek、OpenAI、通义、Moonshot 等）。
    """
    api_key = api_key or CONFIG.get('deepseek_api_key', '')
    if not api_key:
        return raw_text, '未配置 API key（请在界面「AI 辅助配置」中填写）', None
    base_url = (base_url or CONFIG.get('deepseek_base_url', 'https://api.deepseek.com')).rstrip('/')
    model = model or CONFIG.get('deepseek_model', 'deepseek-chat')
    url = base_url + '/chat/completions'

    system_prompt = (
        '你是一个中文语音识别结果整理助手。'
        '用户会给出一段语音识别（ASR）得到的文字，可能包含同音字错误、错别字、缺少标点、'
        '口语化的填词（如"那个""就是说"）以及不通顺的表达。'
        '请你做「语义级整理」：修正错别字、补全标点、把一长串话切成通顺的句子、去掉无意义的口语填词。'
        '规则：'
        '1. 修正明显的同音/近音错别字和不通顺表达；'
        '2. 补全缺失的标点，把长句切分为通顺的短句；'
        '3. 可以删除无意义的口语填词（那个、就是说、嗯、啊），但不要改写原意、不要扩写、不要添加新信息；'
        '4. 如果原文已经是通顺正确的，就原样返回；'
        '5. 只输出整理后的文字本身，不要任何解释或前后缀。'
    )
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': raw_text},
        ],
        'temperature': 0.1,
        'max_tokens': 256,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + api_key,
        },
        method='POST',
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode('utf-8')
        result = json.loads(body)
        corrected = result['choices'][0]['message']['content'].strip()
        usage = result.get('usage', None)
        model_used = result.get('model', model)
        print(f'[server] llm_correct {time.time()-t0:.2f}s [{model_used}] tokens={usage}: "{raw_text}" -> "{corrected}"', flush=True)
        return corrected, None, {'usage': usage, 'model': model_used, 'mode': 'semantic'}
    except urllib.error.HTTPError as e:
        err = f'LLM HTTP {e.code}: {e.read().decode("utf-8", "ignore")[:200]}'
        print(f'[server] llm_correct error: {err}', flush=True)
        return raw_text, err, None
    except Exception as e:
        err = f'LLM error: {str(e)}'
        print(f'[server] llm_correct error: {err}', flush=True)
        return raw_text, err, None


def local_correct(raw_text):
    """用本地 MacBERT CSC 模型做中文拼写纠错，返回 (纠正后文本, 错误信息或None, meta字典或None)"""
    if not raw_text:
        return raw_text, None, None
    try:
        tokenizer, model = get_csc_model()
        t0 = time.time()
        with torch.no_grad():
            inputs = tokenizer(raw_text, return_tensors='pt')
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
            tokens = tokenizer.convert_ids_to_tokens(pred_ids)
        corrected = ''.join(t.replace('##', '') for t in tokens)
        corrected = (corrected.replace('[CLS]', '')
                             .replace('[SEP]', '')
                             .replace('[PAD]', '')
                             .replace('[UNK]', '')
                             .replace(' ', ''))
        elapsed = time.time() - t0
        print(f'[server] local_correct {elapsed:.2f}s: "{raw_text}" -> "{corrected}"', flush=True)
        return corrected, None, {'model': 'macbert4csc-base-chinese', 'elapsed': round(elapsed, 2)}
    except Exception as e:
        err = f'local CSC error: {str(e)}'
        print(f'[server] local_correct error: {err}', flush=True)
        return raw_text, err, None


def correction_15b(raw_text):
    """用本地 1.5B Qwen 模型做「字词级纠错」（错别字/同音字），返回 (纠正后文本, 错误信息或None, meta字典或None)

    能力边界（实测确认）：
      - 强项：错别字、同音字替换（因该→应该、痛知→知道）
      - 弱项：断句、加标点、口语整理、语义重组（这些交给 DeepSeek）
    兜底策略：模型吐空 / 输出与原句长度差异过大时，回退原文本，避免把对的改错。
    """
    if not raw_text:
        return raw_text, None, None
    try:
        tokenizer, model = get_correction_15b_model()
        t0 = time.time()
        input_content = '文本纠错:\n' + raw_text
        messages = [{'role': 'user', 'content': input_content}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0,          # 官方参数：确定性贪心解码，避免瞎编
                repetition_penalty=1.08,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去掉输入 prompt 部分，只保留生成结果
        prompt_decoded = tokenizer.decode(inputs[0], skip_special_tokens=True)
        corrected = decoded[len(prompt_decoded):].strip()

        # —— 兜底：防止模型吐空 / 改得面目全非 ——
        fallback = False
        if not corrected:
            # 模型没把握，直接吐空了 -> 回退原文本
            corrected = raw_text
            fallback = True
        else:
            # 长度差异过大（增删超过 50% 或绝对差异 > 8 字），大概率是乱编 -> 回退
            len_ratio = len(corrected) / max(len(raw_text), 1)
            abs_diff = abs(len(corrected) - len(raw_text))
            if len_ratio < 0.5 or len_ratio > 1.5 or abs_diff > 8:
                corrected = raw_text
                fallback = True

        elapsed = time.time() - t0
        print(f'[server] correction_1.5b {elapsed:.2f}s (fallback={fallback}): "{raw_text}" -> "{corrected}"', flush=True)
        return corrected, None, {
            'model': 'chinese-text-correction-1.5b',
            'elapsed': round(elapsed, 2),
            'fallback': fallback,
            'mode': 'char-level',  # 字词级纠错，非语义整理
        }
    except Exception as e:
        err = f'1.5B correction error: {str(e)}'
        print(f'[server] correction_15b error: {err}', flush=True)
        return raw_text, err, None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默访问日志

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            html_path = os.path.join(BASE, 'index.html')
            with open(html_path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == '/health':
            self._send_json({'status': 'ok', 'device': _device})
        else:
            self._send_json({'error': 'not found'}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/diagnose':
            length = int(self.headers.get('Content-Length', 0))
            audio_bytes = self.rfile.read(length)
            info = diagnose_wav(audio_bytes)
            self._send_json(info)
            return
        if path == '/recognize_llm':
            length = int(self.headers.get('Content-Length', 0))
            audio_bytes = self.rfile.read(length)
            if length == 0:
                self._send_json({'error': 'empty audio'}, 400)
                return
            try:
                # 从请求头读取用户填写的 AI 配置（可为空，则回退 config.json）
                api_key = self.headers.get('X-Api-Key', '') or None
                base_url = self.headers.get('X-Base-Url', '') or None
                model = self.headers.get('X-Model', '') or None
                # ASR 识别
                raw_text, raw_time = infer_from_bytes(audio_bytes, 'raw')
                # LLM 纠错
                if raw_text:
                    corrected, llm_err, llm_meta = llm_correct(raw_text, api_key=api_key, base_url=base_url, model=model)
                else:
                    corrected, llm_err, llm_meta = '', None, None
                self._send_json({
                    'raw_text': raw_text,
                    'raw_time': round(raw_time, 2),
                    'corrected_text': corrected,
                    'llm_error': llm_err,
                    'llm_meta': llm_meta,
                })
            except Exception as e:
                logging.exception('recognize_llm failed')
                self._send_json({'error': str(e)}, 500)
            return
        if path == '/recognize_local':
            length = int(self.headers.get('Content-Length', 0))
            audio_bytes = self.rfile.read(length)
            if length == 0:
                self._send_json({'error': 'empty audio'}, 400)
                return
            try:
                # ASR 识别
                raw_text, raw_time = infer_from_bytes(audio_bytes, 'raw')
                # 本地 MacBERT 纠错
                if raw_text:
                    corrected, csc_err, csc_meta = local_correct(raw_text)
                else:
                    corrected, csc_err, csc_meta = '', None, None
                self._send_json({
                    'raw_text': raw_text,
                    'raw_time': round(raw_time, 2),
                    'corrected_text': corrected,
                    'csc_error': csc_err,
                    'csc_meta': csc_meta,
                })
            except Exception as e:
                logging.exception('recognize_local failed')
                self._send_json({'error': str(e)}, 500)
            return
        if path == '/recognize_15b':
            length = int(self.headers.get('Content-Length', 0))
            audio_bytes = self.rfile.read(length)
            if length == 0:
                self._send_json({'error': 'empty audio'}, 400)
                return
            try:
                # ASR 识别
                raw_text, raw_time = infer_from_bytes(audio_bytes, 'raw')
                # 本地 1.5B 纠错
                if raw_text:
                    corrected, c_err, c_meta = correction_15b(raw_text)
                else:
                    corrected, c_err, c_meta = '', None, None
                self._send_json({
                    'raw_text': raw_text,
                    'raw_time': round(raw_time, 2),
                    'corrected_text': corrected,
                    'correction_error': c_err,
                    'correction_meta': c_meta,
                })
            except Exception as e:
                logging.exception('recognize_15b failed')
                self._send_json({'error': str(e)}, 500)
            return
        if path == '/recognize_denoise':
            length = int(self.headers.get('Content-Length', 0))
            audio_bytes = self.rfile.read(length)
            if length == 0:
                self._send_json({'error': 'empty audio'}, 400)
                return
            try:
                # 落盘保存原始录音，便于离线分析
                import base64
                save_dir = os.path.join(BASE, 'recordings')
                os.makedirs(save_dir, exist_ok=True)
                ts = time.strftime('%Y%m%d_%H%M%S')
                raw_path = os.path.join(save_dir, f'raw_{ts}.wav')
                with open(raw_path, 'wb') as f:
                    f.write(audio_bytes)
                print(f'[server] saved raw audio to {raw_path}', flush=True)

                # 1. 原始识别
                raw_text, raw_time = infer_from_bytes(audio_bytes, 'raw')
                # 2. 去噪 + 识别
                denoised_bytes, diag = spectral_subtraction(audio_bytes)
                if denoised_bytes is None:
                    self._send_json({
                        'raw_text': raw_text,
                        'raw_time': raw_time,
                        'denoise_error': diag.get('error', 'denoise failed'),
                    })
                    return
                denoised_text, denoised_time = infer_from_bytes(denoised_bytes, 'denoised')
                # 保存去噪后音频
                denoised_path = os.path.join(save_dir, f'denoised_{ts}.wav')
                with open(denoised_path, 'wb') as f:
                    f.write(denoised_bytes)
                # 3. 返回去噪后的音频（base64），供前端回放对比
                denoised_b64 = base64.b64encode(denoised_bytes).decode('ascii')
                self._send_json({
                    'raw_text': raw_text,
                    'raw_time': round(raw_time, 2),
                    'denoised_text': denoised_text,
                    'denoised_time': round(denoised_time, 2),
                    'denoised_wav_b64': denoised_b64,
                    'diag': diag,
                })
            except Exception as e:
                logging.exception('recognize_denoise failed')
                self._send_json({'error': str(e)}, 500)
            return
        if path != '/recognize':
            self._send_json({'error': 'not found'}, 404)
            return
        length = int(self.headers.get('Content-Length', 0))
        audio_bytes = self.rfile.read(length)
        if length == 0:
            self._send_json({'error': 'empty audio'}, 400)
            return
        try:
            text = recognize_bytes(audio_bytes)
            self._send_json({'text': text})
        except Exception as e:
            logging.exception('recognize failed')
            self._send_json({'error': str(e)}, 500)


def main():
    port = int(os.environ.get('PORT', 8000))
    # 启动时预加载模型，避免第一次请求长时间无响应
    print(f'[server] torch cuda available: {torch.cuda.is_available()}')
    get_model()
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'[server] listening on http://127.0.0.1:{port}', flush=True)
    print('[server] open this URL in your browser', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
