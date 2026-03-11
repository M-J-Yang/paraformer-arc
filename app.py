from __future__ import annotations

import os
import sys
import traceback


def _append_debug_log(message: str) -> None:
    log_path = os.getenv("ATC_DEBUG_LOG", "").strip()
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(message.rstrip())
            file.write("\n")
    except Exception:
        pass


def main() -> int:
    try:
        from PySide6.QtGui import QFont
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        print(
            "PySide6 is not installed. Install it first, for example: pip install PySide6",
            file=sys.stderr,
        )
        return 1

    from ui.main_window import MainWindow

    def handle_exception(exc_type: type[BaseException], exc_value: BaseException, exc_tb: object) -> None:
        traceback_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(traceback_text, file=sys.stderr)
        _append_debug_log(traceback_text)

    sys.excepthook = handle_exception

    app = QApplication(sys.argv)
    app.setApplicationName("ATC 空管语音识别")
    app.setOrganizationName("ATC")
    app.setFont(QFont("Microsoft YaHei UI", 10))

    window = MainWindow()
    window.show()

    if os.getenv("ATC_DEBUG_PRELOAD", "").strip() == "1":
        def run_debug_preload() -> None:
            _append_debug_log("ATC_DEBUG_PRELOAD=1")
            try:
                window._preload_models()
                _append_debug_log(f"status={window.status_label.text()}")
                _append_debug_log(f"model_status={window.model_status_label.text()}")
                _append_debug_log(f"progress={window.progress_bar.value()}")
            except Exception:
                traceback_text = traceback.format_exc()
                print(traceback_text, file=sys.stderr)
                _append_debug_log(traceback_text)
            finally:
                QTimer.singleShot(0, app.quit)

        QTimer.singleShot(0, run_debug_preload)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
