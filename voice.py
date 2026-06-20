"""
语音识别 — Faster-Whisper Medium (CTranslate2 int8)
CPU 推理，~1.5GB 模型，int8 量化后 ~800MB，带知识库热词表注入
"""
import os
import tempfile
import threading

_model = None
_load_started = False
_lock = threading.Lock()
_hotwords_str = ""
# 简体中文引导提示词，防止 Whisper 输出繁体
_SIMPLIFIED_PROMPT = "以下是普通话的简体中文句子。"


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8",
            num_workers=2,
        )
    return _model


def _load_hotwords():
    """加载知识库高频实体作为热词表"""
    global _hotwords_str
    hw_path = os.path.join(os.path.dirname(__file__), "hotwords.txt")
    if not os.path.exists(hw_path):
        return
    with open(hw_path, encoding="utf-8") as f:
        words = [l.strip() for l in f if l.strip()]
    # 限制数量避免影响推理速度，取前 100 个高频词，过滤太长/太短的
    words = [w for w in words[:100] if 2 <= len(w) <= 20]
    _hotwords_str = ",".join(words)


def preload_model():
    """后台预加载模型 + 热词表"""
    global _load_started
    if _load_started:
        return
    _load_started = True
    _load_hotwords()

    def _load():
        _get_model()
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    return t


def is_model_ready() -> bool:
    return _model is not None


def transcribe(audio_bytes: bytes, language: str = "zh") -> str:
    """将音频 bytes 转为文字。返回空字符串表示识别失败。"""
    if not audio_bytes or len(audio_bytes) < 100:
        return ""

    suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        model = _get_model()
        kwargs = dict(
            language=language,
            beam_size=3,
            vad_filter=True,
            initial_prompt=_SIMPLIFIED_PROMPT,
        )
        if _hotwords_str:
            kwargs["hotwords"] = _hotwords_str
        segments, _ = model.transcribe(tmp_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    finally:
        os.unlink(tmp_path)

    return text
