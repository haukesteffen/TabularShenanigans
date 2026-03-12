import io
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

RUNTIME_LOG_ARTIFACT_DIRNAME = "logs"
RUNTIME_LOG_FILENAME = "runtime.log"


def build_runtime_log_path(bundle_root: Path) -> Path:
    return bundle_root / RUNTIME_LOG_ARTIFACT_DIRNAME / RUNTIME_LOG_FILENAME


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append_runtime_log_message(log_path: Path, stream_name: str, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        for raw_line in message.splitlines():
            if not raw_line:
                continue
            log_file.write(f"{_timestamp_utc()} [{stream_name}] {raw_line}\n")
        log_file.flush()


class _TimestampedTeeStream(io.TextIOBase):
    def __init__(self, console_stream: TextIO, log_file: TextIO, stream_name: str) -> None:
        self._console_stream = console_stream
        self._log_file = log_file
        self._stream_name = stream_name
        self._buffer = ""

    @property
    def encoding(self) -> str:
        return getattr(self._console_stream, "encoding", "utf-8")

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._console_stream.write(text)
        for character in text:
            if character in {"\n", "\r"}:
                self._flush_buffered_line()
                continue
            self._buffer += character
        return len(text)

    def flush(self) -> None:
        self._console_stream.flush()
        if self._log_file.closed:
            self._buffer = ""
            return
        self._flush_buffered_line()
        self._log_file.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._console_stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def fileno(self) -> int:
        return self._console_stream.fileno()

    def writable(self) -> bool:
        return True

    def _flush_buffered_line(self) -> None:
        if self._log_file.closed:
            self._buffer = ""
            return
        if not self._buffer:
            return
        self._log_file.write(f"{_timestamp_utc()} [{self._stream_name}] {self._buffer}\n")
        self._log_file.flush()
        self._buffer = ""


@contextmanager
def capture_runtime_log(log_path: Path) -> Iterator[Path]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = _TimestampedTeeStream(sys.stdout, log_file, "stdout")
        stderr_tee = _TimestampedTeeStream(sys.stderr, log_file, "stderr")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_showwarning = warnings.showwarning

        def _showwarning(
            message,
            category,
            filename,
            lineno,
            file=None,
            line=None,
        ) -> None:
            formatted_warning = warnings.formatwarning(message, category, filename, lineno, line)
            if file is not None:
                file.write(formatted_warning)
                file.flush()
                return
            stderr_tee.write(formatted_warning)
            stderr_tee.flush()

        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        warnings.showwarning = _showwarning
        try:
            yield log_path
        finally:
            stdout_tee.flush()
            stderr_tee.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            warnings.showwarning = original_showwarning


def emit_runtime_log_header(
    stage_name: str,
    competition_slug: str,
    candidate_id: str,
    mlflow_run_id: str,
) -> None:
    print(
        "Runtime log started: "
        f"stage={stage_name}, competition_slug={competition_slug}, "
        f"candidate_id={candidate_id}, mlflow_run_id={mlflow_run_id}"
    )
    print(
        "Runtime environment: "
        f"pid={os.getpid()}, cwd={Path.cwd()}, python={sys.executable}, argv={sys.argv}"
    )
