from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from models.model_load_status import ModelLoadState, ModelLoadStatus
from models.result_schema import RecognitionResult
from utils.atc_hotwords import load_hotwords
from utils.config import (
    build_default_temp_dir,
    resolve_asr_model_dir,
    resolve_hotword_vocab_freq_path,
    resolve_hotword_wordlist_path,
    resolve_punc_model_dir,
    resolve_text_rules_path,
    resolve_vad_model_dir,
)
from utils.device import normalize_device_value
from utils.funasr_bootstrap import bootstrap_funasr_runtime

if TYPE_CHECKING:
    from engines.recognition_service import RecognitionService


ModelCacheKey = tuple[str, str, str, str, str, str, str, bool, bool, bool, bool, bool, int, int, int]


@dataclass(slots=True)
class RecognitionRequest:
    audio_path: str
    device: str
    enable_hotwords: bool = True
    hotword_wordlist_path: str = ""
    hotword_vocab_freq_path: str = ""
    enable_text_rules: bool = True
    text_rules_path: str = ""
    enable_asr: bool = True
    enable_vad: bool = True
    enable_punc: bool = True
    max_single_segment_time: int = 30000
    batch_size_s: int = 20
    batch_size_threshold_s: int = 10


class RecognitionController:
    def __init__(self) -> None:
        self._service_cache: dict[ModelCacheKey, RecognitionService] = {}
        self._service_states: dict[ModelCacheKey, ModelLoadStatus] = {}

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        return self.recognize_with_callbacks(request)

    def is_model_loaded(self, request: RecognitionRequest) -> bool:
        return self.get_model_load_status(request).state is ModelLoadState.READY

    def get_model_load_status(self, request: RecognitionRequest) -> ModelLoadStatus:
        try:
            _, _, _, _, _, _, _, cache_key = self._resolve_service_config(request)
        except FileNotFoundError as exc:
            return ModelLoadStatus(state=ModelLoadState.MISSING, detail=str(exc))
        return self._service_states.get(cache_key, ModelLoadStatus(state=ModelLoadState.UNLOADED))

    def preload_models(self, request: RecognitionRequest) -> None:
        self.preload_models_with_callbacks(request)

    def preload_models_with_callbacks(
        self,
        request: RecognitionRequest,
        on_status: Callable[[str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> None:
        self._preload_service(
            request=request,
            on_status=on_status,
            on_log=on_log,
            on_progress=on_progress,
        )

    def recognize_with_callbacks(
        self,
        request: RecognitionRequest,
        on_status: Callable[[str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> RecognitionResult:
        if not request.enable_asr:
            raise ValueError("ASR 是识别必需项，请先启用 ASR 模型。")

        audio_path = Path(request.audio_path).expanduser().resolve()
        if on_status is not None:
            on_status("检查模型状态")
        if on_log is not None:
            on_log(f"[INFO] 输入音频: {audio_path}")

        service = self._get_ready_service(request)
        if on_log is not None:
            on_log("[INFO] 模型已就绪，直接开始识别")

        return service.recognize(
            audio_path=audio_path,
            temp_dir=build_default_temp_dir(audio_path),
            keep_temp_segments=False,
            on_status=on_status,
            on_log=on_log,
            on_progress=on_progress,
        )

    def _preload_service(
        self,
        request: RecognitionRequest,
        on_status: Callable[[str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> RecognitionService:
        bootstrap_funasr_runtime()
        from engines.recognition_service import RecognitionService

        (
            asr_model_dir,
            vad_model_dir,
            punc_model_dir,
            hotword_wordlist_path,
            hotword_vocab_freq_path,
            text_rules_path,
            device,
            cache_key,
        ) = self._resolve_service_config(request)
        enabled_models = [
            model_name
            for model_name, enabled in (
                ("ASR", request.enable_asr),
                ("VAD", request.enable_vad),
                ("PUNC", request.enable_punc),
            )
            if enabled
        ]
        if not enabled_models:
            raise ValueError("请至少启用一个模型后再加载。")

        if on_status is not None:
            on_status("读取模型配置")
        if on_log is not None:
            hotword_count = len(
                load_hotwords(
                    wordlist_path=hotword_wordlist_path,
                    vocab_freq_path=hotword_vocab_freq_path,
                )
            )
            on_log("[INFO] 开始加载模型")
            on_log("[INFO] 正在读取配置")
            on_log(f"[INFO] 当前启用模型: {', '.join(enabled_models)}")
            on_log(f"[INFO] ASR 模型目录: {asr_model_dir if asr_model_dir is not None else '未启用'}")
            on_log(f"[INFO] VAD 模型目录: {vad_model_dir if vad_model_dir is not None else '未启用'}")
            on_log(f"[INFO] PUNC 模型目录: {punc_model_dir if punc_model_dir is not None else '未启用'}")
            on_log(f"[INFO] 热词增强: {'开启' if request.enable_hotwords else '关闭'}")
            on_log(f"[INFO] 热词词表: {hotword_wordlist_path if hotword_wordlist_path is not None else '未启用'}")
            on_log(f"[INFO] 热词词频表: {hotword_vocab_freq_path if hotword_vocab_freq_path is not None else '未启用'}")
            on_log(f"[INFO] 热词数量: {hotword_count}")
            on_log(f"[INFO] 文本规则: {'开启' if request.enable_text_rules else '关闭'}")
            on_log(f"[INFO] 规则文件: {text_rules_path if text_rules_path is not None else '未启用'}")
            on_log(f"[INFO] 当前设备: {device}")

        service = self._service_cache.get(cache_key)
        status = self._service_states.get(cache_key)
        if service is not None and status is not None and status.state is ModelLoadState.READY:
            if on_status is not None:
                on_status("READY")
            if on_log is not None:
                on_log("[INFO] 当前配置模型已就绪，跳过重复加载")
            if on_progress is not None:
                on_progress(100, "模型已就绪")
            return service

        self._service_states[cache_key] = ModelLoadStatus(state=ModelLoadState.LOADING)
        if on_status is not None:
            on_status("LOADING")
        if on_log is not None:
            on_log("[INFO] 正在初始化推理后端")
            on_log("[INFO] 正在加载权重")
        if on_progress is not None:
            on_progress(15, "正在初始化推理后端")

        try:
            service = RecognitionService(
                asr_model_dir=asr_model_dir,
                vad_model_dir=vad_model_dir,
                punc_model_dir=punc_model_dir,
                device=device,
                enable_asr=request.enable_asr,
                enable_vad=request.enable_vad,
                enable_punc=request.enable_punc,
                max_single_segment_time=request.max_single_segment_time,
                batch_size_s=request.batch_size_s,
                batch_size_threshold_s=request.batch_size_threshold_s,
                hotword_wordlist_path=hotword_wordlist_path,
                hotword_vocab_freq_path=hotword_vocab_freq_path,
                text_rules_path=text_rules_path,
            )
            self._service_cache[cache_key] = service

            self._service_states[cache_key] = ModelLoadStatus(state=ModelLoadState.WARMING_UP)
            if on_status is not None:
                on_status("WARMING_UP")
            if on_log is not None:
                on_log("[INFO] 正在执行预热")
            if on_progress is not None:
                on_progress(80, "正在执行预热")
            service.warmup(on_log=on_log)

            self._service_states[cache_key] = ModelLoadStatus(state=ModelLoadState.READY)
        except Exception as exc:
            self._service_cache.pop(cache_key, None)
            self._service_states[cache_key] = ModelLoadStatus(
                state=ModelLoadState.ERROR,
                detail=str(exc),
            )
            raise

        if on_status is not None:
            on_status("READY")
        if on_log is not None:
            on_log("[INFO] 模型已就绪")
        if on_progress is not None:
            on_progress(100, "模型已就绪")
        return service

    def _get_ready_service(self, request: RecognitionRequest) -> RecognitionService:
        _, _, _, _, _, _, _, cache_key = self._resolve_service_config(request)
        status = self._service_states.get(cache_key)
        service = self._service_cache.get(cache_key)
        if service is None or status is None or status.state is not ModelLoadState.READY:
            raise RuntimeError("当前配置模型未加载完成，请先点击“加载模型”。")
        return service

    def _resolve_service_config(
        self,
        request: RecognitionRequest,
    ) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, Path | None, str, ModelCacheKey]:
        asr_model_dir = resolve_asr_model_dir("") if request.enable_asr else None
        vad_model_dir = resolve_vad_model_dir("") if request.enable_vad else None
        punc_model_dir = resolve_punc_model_dir("") if request.enable_punc else None
        hotword_wordlist_path = (
            resolve_hotword_wordlist_path(request.hotword_wordlist_path) if request.enable_hotwords else None
        )
        hotword_vocab_freq_path = (
            resolve_hotword_vocab_freq_path(request.hotword_vocab_freq_path) if request.enable_hotwords else None
        )
        text_rules_path = resolve_text_rules_path(request.text_rules_path) if request.enable_text_rules else None
        device = normalize_device_value(request.device)
        cache_key = (
            str(asr_model_dir or ""),
            str(vad_model_dir or ""),
            str(punc_model_dir or ""),
            str(hotword_wordlist_path or ""),
            str(hotword_vocab_freq_path or ""),
            str(text_rules_path or ""),
            device,
            request.enable_hotwords,
            request.enable_text_rules,
            request.enable_asr,
            request.enable_vad,
            request.enable_punc,
            request.max_single_segment_time,
            request.batch_size_s,
            request.batch_size_threshold_s,
        )
        return (
            asr_model_dir,
            vad_model_dir,
            punc_model_dir,
            hotword_wordlist_path,
            hotword_vocab_freq_path,
            text_rules_path,
            device,
            cache_key,
        )
