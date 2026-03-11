from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from controllers.recognition_controller import RecognitionController, RecognitionRequest


class RecognitionWorker(QObject):
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
        request: RecognitionRequest,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._request = request

    @Slot()
    def run(self) -> None:
        self.started.emit()
        try:
            result = self._controller.recognize_with_callbacks(
                self._request,
                on_status=self.status_changed.emit,
                on_log=self.log_message.emit,
                on_progress=self.progress_changed.emit,
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
