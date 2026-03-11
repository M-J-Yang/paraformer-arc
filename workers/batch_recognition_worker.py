from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QObject, Signal, Slot

from controllers.recognition_controller import RecognitionController, RecognitionRequest
from engines.audio_loader import collect_audio_files
from exporters import export_result
from models.history_entry import HistoryEntry
from utils.config import build_export_output_dir, normalize_output_root_dir
from utils.history_store import append_history_entry


class BatchRecognitionWorker(QObject):
    started = Signal()
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()
    status_changed = Signal(str)
    log_message = Signal(str)
    progress_changed = Signal(int, str)

    def __init__(
        self,
        controller: RecognitionController,
        folder_path: str,
        request_template: RecognitionRequest,
        output_root_dir: str,
        export_formats: list[str],
    ) -> None:
        super().__init__()
        self._controller = controller
        self._folder_path = folder_path
        self._request_template = request_template
        self._output_root_dir = output_root_dir
        self._export_formats = export_formats

    @Slot()
    def run(self) -> None:
        self.started.emit()
        try:
            audio_files = collect_audio_files(self._folder_path)
            if not audio_files:
                raise RuntimeError("所选文件夹中没有找到支持的音频文件。")

            output_root_dir = normalize_output_root_dir(self._output_root_dir)
            total = len(audio_files)
            self.log_message.emit(f"批量文件数: {total}")

            for index, audio_path in enumerate(audio_files, start=1):
                base_percent = int((index - 1) / total * 100)
                percent_span = max(1, int(100 / total))
                self.status_changed.emit(f"批量处理中 {index}/{total}")
                self.log_message.emit(f"[{index}/{total}] 开始处理: {audio_path}")

                request = RecognitionRequest(
                    audio_path=str(audio_path),
                    device=self._request_template.device,
                    enable_hotwords=self._request_template.enable_hotwords,
                    hotword_wordlist_path=self._request_template.hotword_wordlist_path,
                    hotword_vocab_freq_path=self._request_template.hotword_vocab_freq_path,
                    enable_text_rules=self._request_template.enable_text_rules,
                    text_rules_path=self._request_template.text_rules_path,
                    enable_asr=self._request_template.enable_asr,
                    enable_vad=self._request_template.enable_vad,
                    enable_punc=self._request_template.enable_punc,
                    max_single_segment_time=self._request_template.max_single_segment_time,
                    batch_size_s=self._request_template.batch_size_s,
                    batch_size_threshold_s=self._request_template.batch_size_threshold_s,
                )
                result = self._controller.recognize_with_callbacks(
                    request,
                    on_status=lambda message, i=index, t=total: self.status_changed.emit(f"[{i}/{t}] {message}"),
                    on_log=lambda message, i=index, t=total: self.log_message.emit(f"[{i}/{t}] {message}"),
                    on_progress=lambda percent, message, start=base_percent, width=percent_span, i=index, t=total: self.progress_changed.emit(
                        min(100, start + int(width * percent / 100)),
                        f"[{i}/{t}] {message}",
                    ),
                )

                export_dir = build_export_output_dir(output_root_dir, result.audio.stem)
                export_result(result, export_dir, self._export_formats)
                append_history_entry(
                    HistoryEntry(
                        created_at=datetime.now(timezone.utc).isoformat(),
                        action="batch-recognize-export",
                        audio_path=result.audio.path,
                        status="success",
                        segment_count=len(result.segments),
                        output_dir=str(export_dir),
                        exported_formats=list(self._export_formats),
                    )
                )
                self.log_message.emit(f"[{index}/{total}] 已导出: {export_dir}")
                self.succeeded.emit(result)

            self.progress_changed.emit(100, "批量识别完成")
            self.status_changed.emit("批量识别完成")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
