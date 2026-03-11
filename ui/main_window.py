from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from controllers.recognition_controller import RecognitionController, RecognitionRequest
from exporters import export_result
from models.app_settings import AppSettings
from models.history_entry import HistoryEntry
from models.model_load_status import ModelLoadState
from models.result_schema import RecognitionResult
from utils.config import (
    build_export_output_dir,
    default_output_root_dir,
    load_app_settings,
    normalize_output_root_dir,
    resolve_hotword_vocab_freq_path,
    resolve_hotword_wordlist_path,
    resolve_text_rules_path,
    save_app_settings,
)
from utils.device import list_device_options, normalize_device_option
from utils.history_store import append_history_entry, clear_history_entries, load_history_entries
from utils.time_format import format_srt_timestamp
from workers.batch_recognition_worker import BatchRecognitionWorker
from workers.recognition_worker import RecognitionWorker


class MainWindow(QMainWindow):
    ACTION_LABELS = {
        "single-recognize": "单文件识别",
        "single-export": "单文件导出",
        "batch-recognize-export": "批量识别并导出",
    }
    STATUS_LABELS = {
        "success": "成功",
        "failed": "失败",
    }

    def __init__(self) -> None:
        super().__init__()
        self.controller = RecognitionController()
        self.worker_thread: QThread | None = None
        self.worker: object | None = None
        self.current_result: RecognitionResult | None = None
        self.settings = load_app_settings()

        self.setWindowTitle("ATC 空管语音识别")
        self.setMinimumSize(920, 640)

        self.audio_path_input = QLineEdit()
        self.audio_path_input.setPlaceholderText("请选择单个音频文件")
        self.batch_folder_input = QLineEdit()
        self.batch_folder_input.setPlaceholderText("请选择批量识别文件夹")
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("请选择导出根目录")

        self.device_combo = QComboBox()
        for device_value, device_label in list_device_options():
            self.device_combo.addItem(device_label, userData=device_value)

        self.max_segment_spin = QSpinBox()
        self.max_segment_spin.setRange(1000, 120000)
        self.max_segment_spin.setSingleStep(1000)
        self.max_segment_spin.setSuffix(" ms")

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 120)

        self.batch_threshold_spin = QSpinBox()
        self.batch_threshold_spin.setRange(1, 120)

        self.export_txt_check = QCheckBox("TXT")
        self.export_srt_check = QCheckBox("SRT")
        self.export_json_check = QCheckBox("JSON")
        self.enable_hotwords_check = QCheckBox("热词增强")
        self.enable_text_rules_check = QCheckBox("文本规则")
        self.enable_asr_check = QCheckBox("ASR（识别必选）")
        self.enable_vad_check = QCheckBox("VAD 分段")
        self.enable_punc_check = QCheckBox("PUNC 标点")
        self.hotword_wordlist_input = QLineEdit()
        self.hotword_wordlist_input.setPlaceholderText("热词词表文件")
        self.hotword_vocab_input = QLineEdit()
        self.hotword_vocab_input.setPlaceholderText("热词词频 JSON")
        self.text_rules_input = QLineEdit()
        self.text_rules_input.setPlaceholderText("文本规则 JSON")

        self.model_status_label = QLabel("模型状态：未加载")
        self.model_status_label.setObjectName("modelStatusLabel")
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self.full_text_edit = QTextEdit()
        self.full_text_edit.setReadOnly(True)
        self.full_text_edit.setPlaceholderText("识别完成后，全文会显示在这里。")

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("任务日志会显示在这里。")

        self.segment_table = QTableWidget(0, 3)
        self.segment_table.setHorizontalHeaderLabels(["开始时间", "结束时间", "识别文本"])
        self.segment_table.horizontalHeader().setStretchLastSection(True)
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.setAlternatingRowColors(True)
        self.segment_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.segment_table.setSelectionBehavior(QTableWidget.SelectRows)

        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(
            ["时间", "操作", "音频", "状态", "分段数", "导出格式", "输出目录"]
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)

        self._build_ui()
        self._apply_style()
        self._apply_window_geometry()
        self._load_settings_into_form()
        self._reload_history_table()
        self._update_export_buttons()

    def _build_ui(self) -> None:
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(16)

        header_label = QLabel("ATC 空管语音识别")
        header_label.setObjectName("headerLabel")
        subtitle_label = QLabel("支持单文件识别、批量识别、结果导出与本地历史记录。")
        subtitle_label.setObjectName("subtitleLabel")

        input_group = QGroupBox("输入")
        input_layout = QGridLayout(input_group)
        input_layout.setHorizontalSpacing(12)
        input_layout.setVerticalSpacing(12)

        self.browse_button = QPushButton("选择音频")
        self.browse_button.clicked.connect(self._select_audio_file)
        self.batch_folder_button = QPushButton("选择文件夹")
        self.batch_folder_button.clicked.connect(self._select_batch_folder)
        self.output_browse_button = QPushButton("选择导出目录")
        self.output_browse_button.clicked.connect(self._select_output_dir)
        self.hotword_wordlist_button = QPushButton("选择词表")
        self.hotword_wordlist_button.clicked.connect(self._select_hotword_wordlist_file)
        self.hotword_vocab_button = QPushButton("选择词频表")
        self.hotword_vocab_button.clicked.connect(self._select_hotword_vocab_file)
        self.text_rules_button = QPushButton("选择规则文件")
        self.text_rules_button.clicked.connect(self._select_text_rules_file)

        self.recognize_button = QPushButton("开始单文件识别")
        self.recognize_button.setObjectName("primaryButton")
        self.recognize_button.clicked.connect(self._run_single_recognition)
        self.batch_recognize_button = QPushButton("开始批量识别")
        self.batch_recognize_button.clicked.connect(self._run_batch_recognition)
        self.clear_button = QPushButton("清空结果")
        self.clear_button.clicked.connect(self._clear_result_view)
        self.save_settings_button = QPushButton("保存配置")
        self.save_settings_button.clicked.connect(self._save_settings_from_form)
        self.preload_button = QPushButton("加载模型")
        self.preload_button.clicked.connect(self._preload_models)

        input_layout.addWidget(QLabel("单个音频"), 0, 0)
        input_layout.addWidget(self.audio_path_input, 0, 1)
        input_layout.addWidget(self.browse_button, 0, 2)
        input_layout.addWidget(QLabel("批量文件夹"), 1, 0)
        input_layout.addWidget(self.batch_folder_input, 1, 1)
        input_layout.addWidget(self.batch_folder_button, 1, 2)
        input_layout.addWidget(QLabel("导出根目录"), 2, 0)
        input_layout.addWidget(self.output_dir_input, 2, 1)
        input_layout.addWidget(self.output_browse_button, 2, 2)

        config_group = QGroupBox("识别配置")
        config_layout = QGridLayout(config_group)
        config_layout.setHorizontalSpacing(12)
        config_layout.setVerticalSpacing(10)
        config_layout.addWidget(QLabel("设备"), 0, 0)
        config_layout.addWidget(self.device_combo, 0, 1)
        config_layout.addWidget(QLabel("单段最长时长"), 0, 2)
        config_layout.addWidget(self.max_segment_spin, 0, 3)
        config_layout.addWidget(QLabel("批处理时长"), 1, 0)
        config_layout.addWidget(self.batch_size_spin, 1, 1)
        config_layout.addWidget(QLabel("批处理阈值"), 1, 2)
        config_layout.addWidget(self.batch_threshold_spin, 1, 3)
        config_layout.addWidget(QLabel("启用模型"), 2, 0)
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(self.enable_hotwords_check)
        model_select_layout.addWidget(self.enable_text_rules_check)
        model_select_layout.addWidget(self.enable_asr_check)
        model_select_layout.addWidget(self.enable_vad_check)
        model_select_layout.addWidget(self.enable_punc_check)
        model_select_layout.addStretch(1)
        config_layout.addLayout(model_select_layout, 2, 1, 1, 3)
        config_layout.addWidget(QLabel("热词词表"), 3, 0)
        config_layout.addWidget(self.hotword_wordlist_input, 3, 1, 1, 2)
        config_layout.addWidget(self.hotword_wordlist_button, 3, 3)
        config_layout.addWidget(QLabel("热词词频"), 4, 0)
        config_layout.addWidget(self.hotword_vocab_input, 4, 1, 1, 2)
        config_layout.addWidget(self.hotword_vocab_button, 4, 3)
        config_layout.addWidget(QLabel("规则文件"), 5, 0)
        config_layout.addWidget(self.text_rules_input, 5, 1, 1, 2)
        config_layout.addWidget(self.text_rules_button, 5, 3)

        export_group = QGroupBox("导出格式")
        export_layout = QHBoxLayout(export_group)
        export_layout.addWidget(self.export_txt_check)
        export_layout.addWidget(self.export_srt_check)
        export_layout.addWidget(self.export_json_check)
        export_layout.addStretch(1)
        export_layout.addWidget(self.save_settings_button)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.preload_button)
        action_layout.addWidget(self.recognize_button)
        action_layout.addWidget(self.batch_recognize_button)
        action_layout.addWidget(self.clear_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.model_status_label)
        action_layout.addWidget(self.status_label)

        input_layout.addWidget(config_group, 3, 0, 1, 3)
        input_layout.addWidget(export_group, 4, 0, 1, 3)
        input_layout.addLayout(action_layout, 5, 0, 1, 3)
        input_layout.addWidget(self.progress_bar, 6, 0, 1, 3)

        result_group = QGroupBox("当前结果")
        result_layout = QGridLayout(result_group)
        result_layout.setHorizontalSpacing(14)
        result_layout.setVerticalSpacing(12)

        export_button_layout = QHBoxLayout()
        self.export_selected_button = QPushButton("导出已选格式")
        self.export_selected_button.clicked.connect(self._export_selected_formats)
        self.export_txt_button = QPushButton("导出 TXT")
        self.export_txt_button.clicked.connect(lambda: self._export_single_format("txt"))
        self.export_srt_button = QPushButton("导出 SRT")
        self.export_srt_button.clicked.connect(lambda: self._export_single_format("srt"))
        self.export_json_button = QPushButton("导出 JSON")
        self.export_json_button.clicked.connect(lambda: self._export_single_format("json"))
        export_button_layout.addWidget(self.export_selected_button)
        export_button_layout.addWidget(self.export_txt_button)
        export_button_layout.addWidget(self.export_srt_button)
        export_button_layout.addWidget(self.export_json_button)
        export_button_layout.addStretch(1)

        self.full_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.segment_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        result_top_splitter = QSplitter(Qt.Horizontal)
        result_top_splitter.setChildrenCollapsible(False)
        result_top_splitter.addWidget(self.full_text_edit)
        result_top_splitter.addWidget(self.log_edit)
        result_top_splitter.setStretchFactor(0, 3)
        result_top_splitter.setStretchFactor(1, 2)

        result_layout.addWidget(QLabel("识别全文"), 0, 0)
        result_layout.addWidget(QLabel("运行日志"), 0, 1)
        result_layout.addWidget(result_top_splitter, 1, 0, 1, 2)
        result_layout.addLayout(export_button_layout, 2, 0, 1, 2)
        result_layout.addWidget(QLabel("分段结果"), 3, 0, 1, 2)
        result_layout.addWidget(self.segment_table, 4, 0, 1, 2)
        result_layout.setRowStretch(1, 3)
        result_layout.setRowStretch(4, 4)
        result_layout.setColumnStretch(0, 3)
        result_layout.setColumnStretch(1, 2)

        history_group = QGroupBox("历史记录")
        history_layout = QVBoxLayout(history_group)
        history_button_layout = QHBoxLayout()
        self.refresh_history_button = QPushButton("刷新历史")
        self.refresh_history_button.clicked.connect(self._reload_history_table)
        self.clear_history_button = QPushButton("清空历史")
        self.clear_history_button.clicked.connect(self._clear_history)
        history_button_layout.addWidget(self.refresh_history_button)
        history_button_layout.addWidget(self.clear_history_button)
        history_button_layout.addStretch(1)
        history_layout.addLayout(history_button_layout)
        history_layout.addWidget(self.history_table)

        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)
        self.main_tabs.setTabPosition(QTabWidget.North)

        input_page = QWidget()
        input_page_layout = QVBoxLayout(input_page)
        input_page_layout.setContentsMargins(0, 0, 0, 0)
        input_page_layout.addWidget(input_group)
        input_page_layout.addStretch(1)

        result_page = QWidget()
        result_page_layout = QVBoxLayout(result_page)
        result_page_layout.setContentsMargins(0, 0, 0, 0)
        result_page_layout.addWidget(result_group, stretch=1)

        history_page = QWidget()
        history_page_layout = QVBoxLayout(history_page)
        history_page_layout.setContentsMargins(0, 0, 0, 0)
        history_page_layout.addWidget(history_group, stretch=1)

        self.main_tabs.addTab(input_page, "输入与配置")
        self.main_tabs.addTab(result_page, "当前结果")
        self.main_tabs.addTab(history_page, "历史记录")

        root_layout.addWidget(header_label)
        root_layout.addWidget(subtitle_label)
        root_layout.addWidget(self.main_tabs, stretch=1)

        self.setCentralWidget(central_widget)

    def _apply_window_geometry(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 900)
            return

        available = screen.availableGeometry()
        width = int(available.width() * 0.88)
        height = int(available.height() * 0.86)
        width = max(960, min(width, available.width() - 40))
        height = max(680, min(height, available.height() - 40))
        self.resize(width, height)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f0e8;
                color: #1e2a2f;
                font-family: "Microsoft YaHei UI";
            }
            QGroupBox {
                background: #fbf8f1;
                border: 1px solid #d5cab8;
                border-radius: 14px;
                margin-top: 12px;
                padding: 14px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLabel#headerLabel {
                font-size: 28px;
                font-weight: 700;
                color: #172127;
            }
            QLabel#subtitleLabel {
                color: #5e6b71;
                margin-bottom: 4px;
            }
            QLabel#statusLabel {
                color: #4d5b61;
                font-weight: 600;
            }
            QLineEdit, QComboBox, QTextEdit, QTableWidget, QSpinBox {
                background: #fffdf8;
                border: 1px solid #d7ccbb;
                border-radius: 10px;
                padding: 8px 10px;
            }
            QProgressBar {
                background: #eee6da;
                border: 1px solid #d7ccbb;
                border-radius: 9px;
                text-align: center;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background: #1f6a5b;
                border-radius: 8px;
            }
            QHeaderView::section {
                background: #e7ded0;
                color: #2a363c;
                border: none;
                padding: 8px;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #d5cab8;
                border-radius: 14px;
                background: #fbf8f1;
                top: -1px;
            }
            QTabBar::tab {
                background: #e7ded0;
                color: #44555c;
                border: 1px solid #d5cab8;
                border-bottom: none;
                padding: 10px 18px;
                margin-right: 6px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                min-width: 120px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #fbf8f1;
                color: #172127;
            }
            QTabBar::tab:hover:!selected {
                background: #ddd2c2;
            }
            QPushButton {
                background: #d8cfbf;
                border: none;
                border-radius: 10px;
                padding: 10px 16px;
                color: #1e2a2f;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #c8bda9;
            }
            QPushButton#primaryButton {
                background: #1f6a5b;
                color: #f7f4ee;
            }
            QPushButton#primaryButton:hover {
                background: #185648;
            }
            """
        )

    def _load_settings_into_form(self) -> None:
        self.audio_path_input.setText(self.settings.audio_path)
        self.batch_folder_input.setText(self.settings.batch_folder)
        self.output_dir_input.setText(self.settings.output_dir or str(default_output_root_dir()))
        self.hotword_wordlist_input.setText(
            self.settings.hotword_wordlist_path or str(resolve_hotword_wordlist_path("") or "")
        )
        self.hotword_vocab_input.setText(
            self.settings.hotword_vocab_freq_path or str(resolve_hotword_vocab_freq_path("") or "")
        )
        self.text_rules_input.setText(
            self.settings.text_rules_path or str(resolve_text_rules_path("") or "")
        )
        self._set_current_device(normalize_device_option(self.settings.device))
        self.max_segment_spin.setValue(self.settings.max_single_segment_time)
        self.batch_size_spin.setValue(self.settings.batch_size_s)
        self.batch_threshold_spin.setValue(self.settings.batch_size_threshold_s)
        self.enable_hotwords_check.setChecked(self.settings.enable_hotwords)
        self.enable_text_rules_check.setChecked(self.settings.enable_text_rules)
        self.enable_asr_check.setChecked(self.settings.enable_asr)
        self.enable_vad_check.setChecked(self.settings.enable_vad)
        self.enable_punc_check.setChecked(self.settings.enable_punc)
        self.export_txt_check.setChecked(self.settings.export_txt)
        self.export_srt_check.setChecked(self.settings.export_srt)
        self.export_json_check.setChecked(self.settings.export_json)
        self._bind_form_change_signals()
        self._refresh_model_state_label()

    def _bind_form_change_signals(self) -> None:
        self.device_combo.currentIndexChanged.connect(self._refresh_model_state_label)
        self.max_segment_spin.valueChanged.connect(self._refresh_model_state_label)
        self.batch_size_spin.valueChanged.connect(self._refresh_model_state_label)
        self.batch_threshold_spin.valueChanged.connect(self._refresh_model_state_label)
        self.enable_hotwords_check.stateChanged.connect(self._refresh_model_state_label)
        self.enable_text_rules_check.stateChanged.connect(self._refresh_model_state_label)
        self.enable_asr_check.stateChanged.connect(self._on_model_selection_changed)
        self.enable_vad_check.stateChanged.connect(self._refresh_model_state_label)
        self.enable_punc_check.stateChanged.connect(self._refresh_model_state_label)
        self.hotword_wordlist_input.textChanged.connect(self._refresh_model_state_label)
        self.hotword_vocab_input.textChanged.connect(self._refresh_model_state_label)
        self.text_rules_input.textChanged.connect(self._refresh_model_state_label)

    def _collect_settings_from_form(self) -> AppSettings:
        return AppSettings(
            audio_path=self.audio_path_input.text().strip(),
            batch_folder=self.batch_folder_input.text().strip(),
            device=str(self.device_combo.currentData()),
            output_dir=self.output_dir_input.text().strip() or str(default_output_root_dir()),
            enable_hotwords=self.enable_hotwords_check.isChecked(),
            hotword_wordlist_path=self.hotword_wordlist_input.text().strip(),
            hotword_vocab_freq_path=self.hotword_vocab_input.text().strip(),
            enable_text_rules=self.enable_text_rules_check.isChecked(),
            text_rules_path=self.text_rules_input.text().strip(),
            enable_asr=self.enable_asr_check.isChecked(),
            enable_vad=self.enable_vad_check.isChecked(),
            enable_punc=self.enable_punc_check.isChecked(),
            export_txt=self.export_txt_check.isChecked(),
            export_srt=self.export_srt_check.isChecked(),
            export_json=self.export_json_check.isChecked(),
            max_single_segment_time=int(self.max_segment_spin.value()),
            batch_size_s=int(self.batch_size_spin.value()),
            batch_size_threshold_s=int(self.batch_threshold_spin.value()),
        )

    def _save_settings_from_form(self) -> None:
        self.settings = self._collect_settings_from_form()
        settings_path = save_app_settings(self.settings)
        self._append_log(f"配置已保存: {settings_path}")
        self.status_label.setText("配置已保存")

    def _set_current_device(self, device_value: str) -> None:
        for index in range(self.device_combo.count()):
            if self.device_combo.itemData(index) == device_value:
                self.device_combo.setCurrentIndex(index)
                return
        self.device_combo.setCurrentIndex(0)

    def _selected_export_formats(self) -> list[str]:
        formats: list[str] = []
        if self.export_txt_check.isChecked():
            formats.append("txt")
        if self.export_srt_check.isChecked():
            formats.append("srt")
        if self.export_json_check.isChecked():
            formats.append("json")
        return formats

    def _build_request_from_form(self, audio_path: str = "") -> RecognitionRequest:
        return RecognitionRequest(
            audio_path=audio_path,
            device=str(self.device_combo.currentData()),
            enable_hotwords=self.enable_hotwords_check.isChecked(),
            hotword_wordlist_path=self.hotword_wordlist_input.text().strip(),
            hotword_vocab_freq_path=self.hotword_vocab_input.text().strip(),
            enable_text_rules=self.enable_text_rules_check.isChecked(),
            text_rules_path=self.text_rules_input.text().strip(),
            enable_asr=self.enable_asr_check.isChecked(),
            enable_vad=self.enable_vad_check.isChecked(),
            enable_punc=self.enable_punc_check.isChecked(),
            max_single_segment_time=int(self.max_segment_spin.value()),
            batch_size_s=int(self.batch_size_spin.value()),
            batch_size_threshold_s=int(self.batch_threshold_spin.value()),
        )

    def _on_model_selection_changed(self) -> None:
        if not self.enable_asr_check.isChecked():
            self.model_status_label.setText("模型状态：ASR 已关闭")
        self._refresh_model_state_label()

    def _refresh_model_state_label(self) -> None:
        request = self._build_request_from_form()
        if not request.enable_asr:
            self.model_status_label.setText("模型状态：ASR 已关闭")
            return
        status = self.controller.get_model_load_status(request)
        state_to_label = {
            ModelLoadState.UNLOADED: "模型状态：未加载",
            ModelLoadState.LOADING: "模型状态：加载中",
            ModelLoadState.WARMING_UP: "模型状态：预热中",
            ModelLoadState.READY: "模型状态：已就绪",
            ModelLoadState.ERROR: "模型状态：加载失败",
            ModelLoadState.MISSING: "模型状态：未找到模型",
        }
        self.model_status_label.setText(state_to_label.get(status.state, "模型状态：未知"))

    def _select_audio_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.m4a)",
        )
        if file_path:
            self.audio_path_input.setText(file_path)

    def _select_batch_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择批量识别文件夹",
            self.batch_folder_input.text().strip() or str(Path.cwd()),
        )
        if directory:
            self.batch_folder_input.setText(directory)

    def _select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择导出根目录",
            self.output_dir_input.text().strip() or str(default_output_root_dir()),
        )
        if directory:
            self.output_dir_input.setText(directory)

    def _select_hotword_wordlist_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择热词词表",
            self.hotword_wordlist_input.text().strip() or str(Path.cwd()),
            "文本文件 (*.txt);;所有文件 (*.*)",
        )
        if file_path:
            self.hotword_wordlist_input.setText(file_path)

    def _select_hotword_vocab_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择热词词频表",
            self.hotword_vocab_input.text().strip() or str(Path.cwd()),
            "JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if file_path:
            self.hotword_vocab_input.setText(file_path)

    def _select_text_rules_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择文本规则文件",
            self.text_rules_input.text().strip() or str(Path.cwd()),
            "JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if file_path:
            self.text_rules_input.setText(file_path)

    def _preload_models(self) -> None:
        if self.worker_thread is not None:
            return

        self._save_settings_from_form()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.log_edit.clear()
        self._on_model_preload_started()

        request = self._build_request_from_form()
        if not request.enable_asr:
            QMessageBox.warning(self, "无法加载模型", "请至少启用 ASR 后再加载模型。")
            self._on_task_finished()
            self._refresh_model_state_label()
            return
        try:
            self.controller.preload_models_with_callbacks(
                request,
                on_status=self._handle_preload_status,
                on_log=self._handle_preload_log,
                on_progress=self._handle_preload_progress,
            )
        except Exception as exc:
            if os.getenv("ATC_RERAISE_EXCEPTIONS", "").strip() == "1":
                raise
            self._on_task_failed(str(exc))
        else:
            self._on_model_preload_succeeded()
        finally:
            self._on_task_finished()

    def _run_single_recognition(self) -> None:
        audio_path = self.audio_path_input.text().strip()
        if not audio_path:
            QMessageBox.warning(self, "缺少音频", "请先选择单个音频文件。")
            return
        if self.worker_thread is not None:
            return

        self._save_settings_from_form()
        self.current_result = None
        self.full_text_edit.clear()
        self.segment_table.setRowCount(0)
        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self._update_export_buttons()

        request = self._build_request_from_form(audio_path=audio_path)
        if not self.controller.is_model_loaded(request):
            QMessageBox.information(self, "请先加载模型", "当前配置模型尚未就绪，请先点击“加载模型”。")
            self._refresh_model_state_label()
            return
        self.worker_thread = QThread(self)
        self.worker = RecognitionWorker(self.controller, request)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.started.connect(self._on_recognition_started)
        self.worker.succeeded.connect(self._on_single_recognition_succeeded)
        self.worker.failed.connect(self._on_task_failed)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.log_message.connect(self._append_log)
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _run_batch_recognition(self) -> None:
        folder_path = self.batch_folder_input.text().strip()
        if not folder_path:
            QMessageBox.warning(self, "缺少文件夹", "请先选择批量识别文件夹。")
            return
        export_formats = self._selected_export_formats()
        if not export_formats:
            QMessageBox.warning(self, "缺少导出格式", "批量模式至少选择一种导出格式。")
            return
        if self.worker_thread is not None:
            return

        self._save_settings_from_form()
        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        request_template = self._build_request_from_form()
        if not self.controller.is_model_loaded(request_template):
            QMessageBox.information(self, "请先加载模型", "批量识别前请先完成当前配置的模型加载。")
            self._refresh_model_state_label()
            return
        self.worker_thread = QThread(self)
        self.worker = BatchRecognitionWorker(
            self.controller,
            folder_path=folder_path,
            request_template=request_template,
            output_root_dir=self.output_dir_input.text().strip(),
            export_formats=export_formats,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.started.connect(self._on_batch_started)
        self.worker.succeeded.connect(self._on_batch_item_succeeded)
        self.worker.failed.connect(self._on_task_failed)
        self.worker.finished.connect(self._on_batch_finished)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.log_message.connect(self._append_log)
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _apply_result(self, result: RecognitionResult) -> None:
        self.full_text_edit.setPlainText(result.full_text)
        self.segment_table.setRowCount(len(result.segments))
        for row_index, segment in enumerate(result.segments):
            start_item = QTableWidgetItem(format_srt_timestamp(segment.start_ms))
            end_item = QTableWidgetItem(format_srt_timestamp(segment.end_ms))
            text_item = QTableWidgetItem(segment.text)
            start_item.setTextAlignment(Qt.AlignCenter)
            end_item.setTextAlignment(Qt.AlignCenter)
            self.segment_table.setItem(row_index, 0, start_item)
            self.segment_table.setItem(row_index, 1, end_item)
            self.segment_table.setItem(row_index, 2, text_item)
        self.segment_table.resizeColumnsToContents()

    def _clear_result_view(self) -> None:
        self.current_result = None
        self.full_text_edit.clear()
        self.log_edit.clear()
        self.segment_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.status_label.setText("就绪")
        self._update_export_buttons()

    def _set_busy(self, is_busy: bool) -> None:
        self.audio_path_input.setEnabled(not is_busy)
        self.batch_folder_input.setEnabled(not is_busy)
        self.output_dir_input.setEnabled(not is_busy)
        self.hotword_wordlist_input.setEnabled(not is_busy)
        self.hotword_vocab_input.setEnabled(not is_busy)
        self.text_rules_input.setEnabled(not is_busy)
        self.device_combo.setEnabled(not is_busy)
        self.max_segment_spin.setEnabled(not is_busy)
        self.batch_size_spin.setEnabled(not is_busy)
        self.batch_threshold_spin.setEnabled(not is_busy)
        self.enable_hotwords_check.setEnabled(not is_busy)
        self.enable_text_rules_check.setEnabled(not is_busy)
        self.enable_asr_check.setEnabled(not is_busy)
        self.enable_vad_check.setEnabled(not is_busy)
        self.enable_punc_check.setEnabled(not is_busy)
        self.export_txt_check.setEnabled(not is_busy)
        self.export_srt_check.setEnabled(not is_busy)
        self.export_json_check.setEnabled(not is_busy)
        self.browse_button.setEnabled(not is_busy)
        self.batch_folder_button.setEnabled(not is_busy)
        self.output_browse_button.setEnabled(not is_busy)
        self.hotword_wordlist_button.setEnabled(not is_busy)
        self.hotword_vocab_button.setEnabled(not is_busy)
        self.text_rules_button.setEnabled(not is_busy)
        self.preload_button.setEnabled(not is_busy)
        self.recognize_button.setEnabled(not is_busy)
        self.batch_recognize_button.setEnabled(not is_busy)
        self.clear_button.setEnabled(not is_busy)
        self.save_settings_button.setEnabled(not is_busy)
        self.refresh_history_button.setEnabled(not is_busy)
        self.clear_history_button.setEnabled(not is_busy)
        if is_busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
        self._update_export_buttons()

    def _append_log(self, message: str) -> None:
        self.log_edit.append(message)

    def _process_ui_events(self) -> None:
        QApplication.processEvents()

    def _handle_preload_status(self, message: str) -> None:
        self._on_status_changed(message)
        self._refresh_model_state_label()
        self._process_ui_events()

    def _handle_preload_log(self, message: str) -> None:
        self._append_log(message)
        self._process_ui_events()

    def _handle_preload_progress(self, percent: int, message: str) -> None:
        self._on_progress_changed(percent, message)
        self._process_ui_events()

    def _export_selected_formats(self) -> None:
        export_formats = self._selected_export_formats()
        if not export_formats:
            QMessageBox.warning(self, "缺少导出格式", "请至少选择一种导出格式。")
            return
        self._export_formats(export_formats)

    def _export_single_format(self, export_format: str) -> None:
        self._export_formats([export_format])

    def _export_formats(self, export_formats: list[str]) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "没有结果", "请先执行一次单文件识别。")
            return

        output_root = normalize_output_root_dir(self.output_dir_input.text().strip())
        output_dir = build_export_output_dir(output_root, self.current_result.audio.stem)
        exported_files = export_result(self.current_result, output_dir, export_formats)
        self._append_log(f"导出完成: {output_dir}")
        for exported_file in exported_files:
            self._append_log(f"  - {exported_file}")
        self.status_label.setText("导出完成")
        append_history_entry(
            HistoryEntry(
                created_at=datetime.utcnow().isoformat(),
                action="single-export",
                audio_path=self.current_result.audio.path,
                status="success",
                segment_count=len(self.current_result.segments),
                output_dir=str(output_dir),
                exported_formats=list(export_formats),
            )
        )
        self._reload_history_table()

    def _update_export_buttons(self) -> None:
        enabled = self.current_result is not None and self.worker_thread is None
        self.export_selected_button.setEnabled(enabled)
        self.export_txt_button.setEnabled(enabled)
        self.export_srt_button.setEnabled(enabled)
        self.export_json_button.setEnabled(enabled)

    def _on_recognition_started(self) -> None:
        self.main_tabs.setCurrentIndex(1)
        self.status_label.setText("单文件识别已开始")
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("1%")
        self._append_log("开始单文件识别")
        self._set_busy(True)

    def _on_model_preload_started(self) -> None:
        self.main_tabs.setCurrentIndex(1)
        self.status_label.setText("开始加载模型")
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("1%")
        self._append_log("[INFO] 开始加载模型")
        self._set_busy(True)

    def _on_model_preload_succeeded(self) -> None:
        self.status_label.setText("模型已就绪，可开始识别")
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100%")
        self._append_log("[INFO] 模型已就绪")
        self._refresh_model_state_label()

    def _on_batch_started(self) -> None:
        self.main_tabs.setCurrentIndex(1)
        self.status_label.setText("批量识别已开始")
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat("1%")
        self._append_log("开始批量识别")
        self._set_busy(True)

    def _on_single_recognition_succeeded(self, result: object) -> None:
        if isinstance(result, RecognitionResult):
            self.current_result = result
            self._apply_result(result)
            self.status_label.setText(f"单文件识别完成，共 {len(result.segments)} 段")
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("100%")
            self._append_log("单文件识别成功")
            self._refresh_model_state_label()
            append_history_entry(
                HistoryEntry(
                    created_at=datetime.utcnow().isoformat(),
                    action="single-recognize",
                    audio_path=result.audio.path,
                    status="success",
                    segment_count=len(result.segments),
                    output_dir="",
                    exported_formats=[],
                )
            )
            self._reload_history_table()

    def _on_batch_item_succeeded(self, result: object) -> None:
        if isinstance(result, RecognitionResult):
            self.current_result = result
            self._apply_result(result)
            self._append_log(f"批量任务完成: {result.audio.name}")

    def _on_task_failed(self, error_message: str) -> None:
        self.status_label.setText("任务失败")
        self._append_log(f"错误: {error_message}")
        QMessageBox.critical(self, "任务失败", error_message)

    def _on_task_finished(self) -> None:
        self._set_busy(False)
        self._update_export_buttons()

    def _on_batch_finished(self) -> None:
        self._set_busy(False)
        self._refresh_model_state_label()
        self._reload_history_table()
        self._update_export_buttons()

    def _on_status_changed(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_progress_changed(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{percent}%")
        if message:
            self.status_label.setText(message)

    def _reload_history_table(self) -> None:
        entries = load_history_entries()
        self.history_table.setRowCount(len(entries))
        for row_index, entry in enumerate(entries):
            created_at_label = entry.created_at
            try:
                created_at_label = datetime.fromisoformat(entry.created_at.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                pass
            values = [
                created_at_label,
                self.ACTION_LABELS.get(entry.action, entry.action),
                Path(entry.audio_path).name if entry.audio_path else "",
                self.STATUS_LABELS.get(entry.status, entry.status),
                str(entry.segment_count),
                ",".join(entry.exported_formats),
                entry.output_dir,
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column_index in {0, 4}:
                    item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row_index, column_index, item)
        self.history_table.resizeColumnsToContents()

    def _clear_history(self) -> None:
        if QMessageBox.question(self, "清空历史", "确认删除全部本地历史记录吗？") != QMessageBox.Yes:
            return
        clear_history_entries()
        self._reload_history_table()
        self._append_log("历史记录已清空")
        self.status_label.setText("历史记录已清空")

    def _cleanup_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._refresh_model_state_label()
