from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from huggingface_hub import HfApi, hf_hub_url
from PySide6.QtCore import QProcess, QRunnable, QThreadPool, Qt, QObject, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "DoLLAMACPP Frontend"
CONFIG_PATH = Path("frontend_config.json")
MODELS_DIR = Path("models")


@dataclass
class ModelSearchResult:
    repo_id: str
    likes: int
    downloads: int
    last_modified: str


@dataclass
class RepoFile:
    name: str
    size_text: str
    size_bytes: int | None = None


@dataclass
class RepoDetails:
    repo_id: str
    author: str
    downloads: int
    likes: int
    last_modified: str
    library_name: str
    pipeline_tag: str
    license_name: str
    gated: bool
    private: bool
    tags: list[str]
    summary: str


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(object)


class Worker(QRunnable):
    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        use_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.use_progress = use_progress
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            call_kwargs = dict(self.kwargs)
            if self.use_progress:
                call_kwargs["progress_callback"] = self.signals.progress.emit
            result = self.fn(*self.args, **call_kwargs)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)


class HuggingFaceClient:
    def __init__(self) -> None:
        self.api = HfApi()

    def search_models(self, query: str, token: str | None = None) -> list[ModelSearchResult]:
        query = query.strip()
        if not query:
            return []

        models = self.api.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=25,
            token=token or None,
        )

        results: list[ModelSearchResult] = []
        for model in models:
            model_id = getattr(model, "id", "")
            if not model_id:
                continue
            results.append(
                ModelSearchResult(
                    repo_id=model_id,
                    likes=int(getattr(model, "likes", 0) or 0),
                    downloads=int(getattr(model, "downloads", 0) or 0),
                    last_modified=self._format_date(getattr(model, "last_modified", None)),
                )
            )
        return results

    def get_repo_details(self, repo_id: str, token: str | None = None) -> tuple[RepoDetails, list[RepoFile]]:
        info = self.api.model_info(repo_id=repo_id, token=token or None, files_metadata=True)
        card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None) or {}
        summary = self._coalesce_summary(card_data)
        license_name = str(card_data.get("license") or "") if isinstance(card_data, dict) else ""

        details = RepoDetails(
            repo_id=repo_id,
            author=str(getattr(info, "author", "") or ""),
            downloads=int(getattr(info, "downloads", 0) or 0),
            likes=int(getattr(info, "likes", 0) or 0),
            last_modified=self._format_date(getattr(info, "last_modified", None)),
            library_name=str(getattr(info, "library_name", "") or ""),
            pipeline_tag=str(getattr(info, "pipeline_tag", "") or ""),
            license_name=license_name,
            gated=bool(getattr(info, "gated", False)),
            private=bool(getattr(info, "private", False)),
            tags=list(getattr(info, "tags", []) or []),
            summary=summary,
        )

        files: list[RepoFile] = []
        for sibling in info.siblings or []:
            name = getattr(sibling, "rfilename", "")
            if not name.lower().endswith(".gguf"):
                continue
            size_value = getattr(sibling, "size", None)
            files.append(
                RepoFile(
                    name=name,
                    size_text=self._format_size(size_value),
                    size_bytes=size_value,
                )
            )

        return details, sorted(files, key=lambda item: item.name.lower())

    def download_file(
        self,
        repo_id: str,
        filename: str,
        destination_dir: Path,
        token: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        destination = destination_dir / safe_name
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        downloaded = 0
        started = time.monotonic()

        with requests.get(url, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()
            total_header = response.headers.get("Content-Length")
            total_bytes = int(total_header) if total_header and total_header.isdigit() else None

            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        elapsed = max(time.monotonic() - started, 0.001)
                        progress_callback(
                            {
                                "downloaded": downloaded,
                                "total": total_bytes,
                                "speed": downloaded / elapsed,
                                "filename": safe_name,
                            }
                        )

        return destination

    @staticmethod
    def _coalesce_summary(card_data: Any) -> str:
        if not isinstance(card_data, dict):
            return ""
        for key in ("summary", "description", "model_summary"):
            value = card_data.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _format_date(value: Any) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        return str(value or "")

    @staticmethod
    def _format_size(size_value: int | None) -> str:
        if not size_value:
            return ""
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_value)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024
        return ""


class LlamaServerClient:
    def chat(
        self,
        base_url: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": "local-model",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from llama-server.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
        return str(content).strip()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1380, 900)

        self.thread_pool = QThreadPool()
        self.hf_client = HuggingFaceClient()
        self.server_client = LlamaServerClient()
        self.search_results: list[ModelSearchResult] = []
        self.repo_files: list[RepoFile] = []
        self.repo_details: RepoDetails | None = None
        self.chat_history: list[dict[str, str]] = []
        self.pending_user_message = ""

        self.server_process = QProcess(self)
        self.server_process.setProcessChannelMode(QProcess.MergedChannels)
        self.server_process.readyReadStandardOutput.connect(self._append_server_output)
        self.server_process.errorOccurred.connect(self._handle_process_error)
        self.server_process.started.connect(lambda: self._set_server_state(True))
        self.server_process.finished.connect(lambda *_: self._set_server_state(False))

        self._build_ui()
        self._load_config()
        self._set_server_state(False)

    def _build_ui(self) -> None:
        root = QWidget()
        main_layout = QVBoxLayout(root)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self._build_search_panel())
        top_splitter.addWidget(self._build_right_panel())
        top_splitter.setSizes([660, 680])

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self._build_log_panel())
        bottom_splitter.addWidget(self._build_chat_panel())
        bottom_splitter.setSizes([700, 640])

        main_layout.addWidget(top_splitter, stretch=3)
        main_layout.addWidget(bottom_splitter, stretch=2)
        self.setCentralWidget(root)

    def _build_search_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        search_box = QGroupBox("Hugging Face Search")
        search_layout = QVBoxLayout(search_box)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search GGUF repos, e.g. qwen gguf, gemma gguf, tinyllama gguf")
        self.search_input.returnPressed.connect(self.search_models)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_models)
        search_row.addWidget(self.search_input)
        search_row.addWidget(self.search_button)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Repository", "Downloads", "Likes", "Updated"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self.load_selected_repo)

        self.search_status = QLabel("Enter a query to search Hugging Face.")

        search_layout.addLayout(search_row)
        search_layout.addWidget(self.results_table)
        search_layout.addWidget(self.search_status)

        files_box = QGroupBox("GGUF Files")
        files_layout = QVBoxLayout(files_box)
        self.files_list = QListWidget()
        self.files_list.itemSelectionChanged.connect(self._sync_selected_file_path)

        download_row = QHBoxLayout()
        self.download_button = QPushButton("Download Selected File")
        self.download_button.clicked.connect(self.download_selected_file)
        self.download_button.setEnabled(False)
        self.browse_models_button = QPushButton("Browse Models Folder")
        self.browse_models_button.clicked.connect(self.pick_model_file)
        download_row.addWidget(self.download_button)
        download_row.addWidget(self.browse_models_button)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status = QLabel("No download started.")

        files_layout.addWidget(self.files_list)
        files_layout.addLayout(download_row)
        files_layout.addWidget(self.download_progress)
        files_layout.addWidget(self.download_status)

        layout.addWidget(search_box)
        layout.addWidget(files_box)
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._build_run_panel())
        layout.addWidget(self._build_repo_details_panel())
        return container

    def _build_run_panel(self) -> QWidget:
        settings_box = QGroupBox("Server Settings")
        settings_layout = QFormLayout(settings_box)

        self.llama_path_input = QLineEdit()
        self.llama_path_input.setPlaceholderText("Path to llama-server executable")
        browse_llama = QPushButton("Browse")
        browse_llama.clicked.connect(self.pick_llama_server)
        llama_row = QHBoxLayout()
        llama_row.addWidget(self.llama_path_input)
        llama_row.addWidget(browse_llama)
        settings_layout.addRow("llama-server", self._wrap_layout(llama_row))

        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to downloaded .gguf file")
        browse_model = QPushButton("Browse")
        browse_model.clicked.connect(self.pick_model_file)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_input)
        model_row.addWidget(browse_model)
        settings_layout.addRow("Model File", self._wrap_layout(model_row))

        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("Optional Hugging Face token for gated/private repos")
        self.hf_token_input.setEchoMode(QLineEdit.Password)
        settings_layout.addRow("HF Token", self.hf_token_input)

        network_layout = QGridLayout()
        self.host_input = QLineEdit("127.0.0.1")
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(8080)
        network_layout.addWidget(QLabel("Host"), 0, 0)
        network_layout.addWidget(self.host_input, 0, 1)
        network_layout.addWidget(QLabel("Port"), 0, 2)
        network_layout.addWidget(self.port_input, 0, 3)
        settings_layout.addRow("Network", self._wrap_layout(network_layout))

        generation_layout = QGridLayout()
        self.ctx_size_input = QSpinBox()
        self.ctx_size_input.setRange(256, 1048576)
        self.ctx_size_input.setSingleStep(256)
        self.ctx_size_input.setValue(4096)
        self.temperature_input = QSpinBox()
        self.temperature_input.setRange(0, 200)
        self.temperature_input.setValue(70)
        self.chat_max_tokens_input = QSpinBox()
        self.chat_max_tokens_input.setRange(1, 32768)
        self.chat_max_tokens_input.setValue(512)
        generation_layout.addWidget(QLabel("Context"), 0, 0)
        generation_layout.addWidget(self.ctx_size_input, 0, 1)
        generation_layout.addWidget(QLabel("Temp x100"), 0, 2)
        generation_layout.addWidget(self.temperature_input, 0, 3)
        generation_layout.addWidget(QLabel("Reply Tokens"), 1, 0)
        generation_layout.addWidget(self.chat_max_tokens_input, 1, 1)
        settings_layout.addRow("Generation", self._wrap_layout(generation_layout))

        self.extra_args_input = QLineEdit()
        self.extra_args_input.setPlaceholderText("Optional extra args, e.g. --n-gpu-layers 999")
        settings_layout.addRow("Extra Args", self.extra_args_input)

        buttons_row = QHBoxLayout()
        self.start_button = QPushButton("Start Server")
        self.start_button.clicked.connect(self.start_server)
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        buttons_row.addWidget(self.start_button)
        buttons_row.addWidget(self.stop_button)
        settings_layout.addRow("Controls", self._wrap_layout(buttons_row))

        self.server_status = QLabel("Server is stopped.")
        settings_layout.addRow("Status", self.server_status)
        return settings_box
    def _build_repo_details_panel(self) -> QWidget:
        box = QGroupBox("Repository Details")
        layout = QVBoxLayout(box)

        self.repo_title_label = QLabel("Select a Hugging Face repo to inspect it.")
        self.repo_title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.repo_meta_label = QLabel("")
        self.repo_meta_label.setWordWrap(True)
        self.repo_meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.repo_summary_text = QPlainTextEdit()
        self.repo_summary_text.setReadOnly(True)
        self.repo_summary_text.setPlaceholderText("Repo summary, tags, and metadata will appear here.")

        layout.addWidget(self.repo_title_label)
        layout.addWidget(self.repo_meta_label)
        layout.addWidget(self.repo_summary_text)
        return box

    def _build_log_panel(self) -> QWidget:
        box = QGroupBox("Server Log")
        layout = QVBoxLayout(box)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        return box

    def _build_chat_panel(self) -> QWidget:
        box = QGroupBox("Chat Tester")
        layout = QVBoxLayout(box)

        self.chat_history_output = QPlainTextEdit()
        self.chat_history_output.setReadOnly(True)
        self.chat_history_output.setPlaceholderText("Start the server, then send a prompt here.")

        self.system_prompt_input = QLineEdit()
        self.system_prompt_input.setPlaceholderText("Optional system prompt")

        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Write a test prompt for the running model...")
        self.chat_input.setFixedHeight(120)

        buttons_row = QHBoxLayout()
        self.send_chat_button = QPushButton("Send Prompt")
        self.send_chat_button.clicked.connect(self.send_chat_message)
        self.clear_chat_button = QPushButton("Clear Chat")
        self.clear_chat_button.clicked.connect(self.clear_chat)
        buttons_row.addWidget(self.send_chat_button)
        buttons_row.addWidget(self.clear_chat_button)

        self.chat_status = QLabel("Chat is idle.")

        layout.addWidget(self.chat_history_output)
        layout.addWidget(self.system_prompt_input)
        layout.addWidget(self.chat_input)
        layout.addLayout(buttons_row)
        layout.addWidget(self.chat_status)
        return box

    def _wrap_layout(self, inner_layout: QGridLayout | QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(inner_layout)
        return widget

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return

        self.llama_path_input.setText(data.get("llama_server_path", ""))
        self.model_path_input.setText(data.get("model_path", ""))
        self.hf_token_input.setText(data.get("hf_token", ""))
        self.host_input.setText(data.get("host", "127.0.0.1"))
        self.port_input.setValue(int(data.get("port", 8080)))
        self.ctx_size_input.setValue(int(data.get("ctx_size", 4096)))
        self.extra_args_input.setText(data.get("extra_args", ""))
        self.system_prompt_input.setText(data.get("system_prompt", ""))
        self.temperature_input.setValue(int(data.get("temperature_x100", 70)))
        self.chat_max_tokens_input.setValue(int(data.get("chat_max_tokens", 512)))

    def _save_config(self) -> None:
        data = {
            "llama_server_path": self.llama_path_input.text().strip(),
            "model_path": self.model_path_input.text().strip(),
            "hf_token": self.hf_token_input.text().strip(),
            "host": self.host_input.text().strip(),
            "port": self.port_input.value(),
            "ctx_size": self.ctx_size_input.value(),
            "extra_args": self.extra_args_input.text().strip(),
            "system_prompt": self.system_prompt_input.text().strip(),
            "temperature_x100": self.temperature_input.value(),
            "chat_max_tokens": self.chat_max_tokens_input.value(),
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def pick_llama_server(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select llama-server executable")
        if path:
            self.llama_path_input.setText(path)
            self._save_config()

    def pick_model_file(self) -> None:
        MODELS_DIR.mkdir(exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF model",
            str(MODELS_DIR.resolve()),
            "GGUF files (*.gguf);;All files (*)",
        )
        if path:
            self.model_path_input.setText(path)
            self._save_config()

    def search_models(self) -> None:
        query = self.search_input.text().strip()
        token = self.hf_token_input.text().strip()

        self.search_button.setEnabled(False)
        self.search_status.setText("Searching Hugging Face...")
        self.results_table.setRowCount(0)
        self.files_list.clear()
        self.download_button.setEnabled(False)
        self._clear_repo_details("Searching...")

        worker = Worker(self.hf_client.search_models, query, token)
        worker.signals.finished.connect(self._populate_search_results)
        worker.signals.error.connect(self._show_search_error)
        self.thread_pool.start(worker)

    def _populate_search_results(self, results: list[ModelSearchResult]) -> None:
        self.search_button.setEnabled(True)
        self.search_results = results
        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(result.repo_id))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{result.downloads:,}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.likes:,}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(result.last_modified))

        if results:
            self.search_status.setText(f"Found {len(results)} repositories. Select one to inspect it.")
            self.results_table.selectRow(0)
        else:
            self.search_status.setText("No matching repositories found.")
            self._clear_repo_details("No repo selected.")

    def _show_search_error(self, message: str) -> None:
        self.search_button.setEnabled(True)
        self.search_status.setText("Search failed.")
        self._clear_repo_details("Search failed.")
        QMessageBox.critical(self, "Search Failed", message)

    def load_selected_repo(self) -> None:
        selected_row = self.results_table.currentRow()
        if selected_row < 0 or selected_row >= len(self.search_results):
            return

        repo_id = self.search_results[selected_row].repo_id
        token = self.hf_token_input.text().strip()
        self.files_list.clear()
        self.files_list.addItem("Loading GGUF files...")
        self.download_button.setEnabled(False)
        self.repo_title_label.setText(repo_id)
        self.repo_meta_label.setText("Loading metadata...")
        self.repo_summary_text.setPlainText("")

        worker = Worker(self.hf_client.get_repo_details, repo_id, token)
        worker.signals.finished.connect(self._populate_repo_details)
        worker.signals.error.connect(self._show_repo_files_error)
        self.thread_pool.start(worker)

    def _populate_repo_details(self, payload: tuple[RepoDetails, list[RepoFile]]) -> None:
        details, files = payload
        self.repo_details = details
        self.repo_files = files

        self.repo_title_label.setText(details.repo_id)
        meta_parts = [
            f"Author: {details.author or 'unknown'}",
            f"Downloads: {details.downloads:,}",
            f"Likes: {details.likes:,}",
            f"Updated: {details.last_modified or 'unknown'}",
        ]
        if details.license_name:
            meta_parts.append(f"License: {details.license_name}")
        if details.pipeline_tag:
            meta_parts.append(f"Pipeline: {details.pipeline_tag}")
        if details.library_name:
            meta_parts.append(f"Library: {details.library_name}")
        if details.gated:
            meta_parts.append("Gated repo")
        if details.private:
            meta_parts.append("Private repo")
        summary_lines = []
        if details.summary:
            summary_lines.append(details.summary)
        if details.tags:
            summary_lines.append("")
            summary_lines.append("Tags: " + ", ".join(details.tags[:20]))

        self.repo_meta_label.setText(" | ".join(meta_parts))
        self.repo_summary_text.setPlainText("\n".join(summary_lines).strip())

        self.files_list.clear()
        if not files:
            self.files_list.addItem("No GGUF files were found in this repo.")
            self.download_button.setEnabled(False)
            return

        for file_info in files:
            label = file_info.name if not file_info.size_text else f"{file_info.name} ({file_info.size_text})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, file_info.name)
            self.files_list.addItem(item)

        self.files_list.setCurrentRow(0)
        self.download_button.setEnabled(True)

    def _show_repo_files_error(self, message: str) -> None:
        self.files_list.clear()
        self.files_list.addItem("Failed to load files.")
        self._clear_repo_details("Failed to load repository details.")
        QMessageBox.critical(self, "Repository Load Failed", message)

    def _clear_repo_details(self, title: str) -> None:
        self.repo_details = None
        self.repo_title_label.setText(title)
        self.repo_meta_label.setText("")
        self.repo_summary_text.setPlainText("")

    def _selected_repo_id(self) -> str:
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.search_results):
            return ""
        return self.search_results[row].repo_id

    def _selected_filename(self) -> str:
        item = self.files_list.currentItem()
        if item is None:
            return ""
        return item.data(Qt.UserRole) or ""

    def _sync_selected_file_path(self) -> None:
        filename = self._selected_filename()
        if not filename:
            return
        self.model_path_input.setText(str((MODELS_DIR / Path(filename).name).resolve()))
        self._save_config()

    def download_selected_file(self) -> None:
        repo_id = self._selected_repo_id()
        filename = self._selected_filename()
        token = self.hf_token_input.text().strip()

        if not repo_id or not filename:
            QMessageBox.warning(self, "No File Selected", "Select a GGUF file first.")
            return

        self.download_button.setEnabled(False)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status.setText(f"Starting download for {Path(filename).name}...")

        worker = Worker(
            self.hf_client.download_file,
            repo_id,
            filename,
            MODELS_DIR,
            token,
            use_progress=True,
        )
        worker.signals.progress.connect(self._update_download_progress)
        worker.signals.finished.connect(self._download_complete)
        worker.signals.error.connect(self._download_failed)
        self.thread_pool.start(worker)

    def _update_download_progress(self, payload: dict[str, Any]) -> None:
        downloaded = int(payload.get("downloaded", 0) or 0)
        total = payload.get("total")
        speed = float(payload.get("speed", 0.0) or 0.0)
        filename = str(payload.get("filename", "model.gguf"))

        self.download_progress.setRange(0, 100)
        if total:
            percent = min(100, int(downloaded * 100 / total))
            self.download_progress.setValue(percent)
            self.download_status.setText(
                f"Downloading {filename}: {self._format_bytes(downloaded)} / {self._format_bytes(total)}"
                f" at {self._format_bytes(speed)}/s"
            )
        else:
            self.download_progress.setValue(0)
            self.download_status.setText(
                f"Downloading {filename}: {self._format_bytes(downloaded)} at {self._format_bytes(speed)}/s"
            )

    def _download_complete(self, path: Path) -> None:
        self.download_button.setEnabled(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(100)
        resolved = str(path.resolve())
        self.model_path_input.setText(resolved)
        self.download_status.setText(f"Downloaded to {resolved}")
        self._save_config()

    def _download_failed(self, message: str) -> None:
        self.download_button.setEnabled(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status.setText("Download failed.")
        QMessageBox.critical(self, "Download Failed", message)

    def start_server(self) -> None:
        if self.server_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Server Running", "The server is already running.")
            return

        llama_path = self.llama_path_input.text().strip()
        model_path = self.model_path_input.text().strip()
        host = self.host_input.text().strip() or "127.0.0.1"

        if not llama_path or not Path(llama_path).exists():
            QMessageBox.warning(self, "Missing llama-server", "Set a valid llama-server executable path.")
            return

        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Missing model", "Set a valid GGUF model path.")
            return

        arguments = [
            "-m",
            model_path,
            "--host",
            host,
            "--port",
            str(self.port_input.value()),
            "-c",
            str(self.ctx_size_input.value()),
        ]

        extra_args = self.extra_args_input.text().strip()
        if extra_args:
            arguments.extend(shlex.split(extra_args, posix=False))

        self._save_config()
        self.log_output.appendPlainText(f"> {llama_path} {' '.join(arguments)}")
        self.server_process.start(llama_path, arguments)

    def stop_server(self) -> None:
        if self.server_process.state() == QProcess.NotRunning:
            return
        self.server_process.kill()
        self.server_process.waitForFinished(3000)
    def send_chat_message(self) -> None:
        prompt = self.chat_input.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Empty Prompt", "Enter a prompt first.")
            return

        base_url = self.server_base_url
        messages: list[dict[str, str]] = []
        system_prompt = self.system_prompt_input.text().strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": prompt})

        self.pending_user_message = prompt
        self.send_chat_button.setEnabled(False)
        self.chat_status.setText(f"Sending request to {base_url}/v1/chat/completions ...")
        self._append_chat_message("user", prompt)
        self.chat_input.clear()

        worker = Worker(
            self.server_client.chat,
            base_url,
            messages,
            self.chat_max_tokens_input.value(),
            self.temperature_input.value() / 100.0,
        )
        worker.signals.finished.connect(self._chat_complete)
        worker.signals.error.connect(self._chat_failed)
        self.thread_pool.start(worker)

    def _chat_complete(self, reply: str) -> None:
        self.send_chat_button.setEnabled(True)
        self.chat_status.setText("Chat reply received.")
        if self.pending_user_message:
            self.chat_history.append({"role": "user", "content": self.pending_user_message})
        self.chat_history.append({"role": "assistant", "content": reply})
        self.pending_user_message = ""
        self._append_chat_message("assistant", reply)
        self._save_config()

    def _chat_failed(self, message: str) -> None:
        self.send_chat_button.setEnabled(True)
        self.chat_status.setText("Chat request failed.")
        self.pending_user_message = ""
        self.chat_history_output.appendPlainText(f"[error]\n{message}\n")
        QMessageBox.critical(self, "Chat Failed", message)

    def clear_chat(self) -> None:
        self.chat_history.clear()
        self.pending_user_message = ""
        self.chat_history_output.clear()
        self.chat_status.setText("Chat cleared.")

    def _append_chat_message(self, role: str, content: str) -> None:
        self.chat_history_output.appendPlainText(f"[{role}]\n{content}\n")

    @property
    def server_base_url(self) -> str:
        host = self.host_input.text().strip() or "127.0.0.1"
        return f"http://{host}:{self.port_input.value()}"

    def _append_server_output(self) -> None:
        text = bytes(self.server_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log_output.appendPlainText(text.rstrip())

    def _handle_process_error(self, _error: QProcess.ProcessError) -> None:
        self.server_status.setText("Server failed to start or crashed.")

    def _set_server_state(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.server_status.setText(
            f"Running at {self.server_base_url}" if running else "Server is stopped."
        )

    @staticmethod
    def _format_bytes(size_value: float | int) -> str:
        size = float(size_value)
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024
        return "0 B"

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_config()
        if self.server_process.state() != QProcess.NotRunning:
            self.server_process.kill()
            self.server_process.waitForFinished(2000)
        super().closeEvent(event)


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
