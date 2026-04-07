from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
from typing import IO, cast

import webview

from mtf_calc._roi_tools import roi_from_payload, roi_to_payload


@dataclass
class _HostRequest:
    request_id: int
    command: str
    payload: dict[str, object]


class _ResponseWriter:
    def __init__(self, stream: IO[str]) -> None:
        self._stream: IO[str] = stream
        self._lock: threading.Lock = threading.Lock()

    def send(self, payload: dict[str, object]) -> None:
        message = json.dumps(payload)
        with self._lock:
            _ = self._stream.write(message)
            _ = self._stream.write("\n")
            self._stream.flush()


class _VizBridge:
    def __init__(self, response_writer: _ResponseWriter) -> None:
        self._response_writer: _ResponseWriter = response_writer
        self._window: webview.Window | None = None
        self._active_request: _HostRequest | None = None
        self._request_complete: threading.Event = threading.Event()
        self._request_lock: threading.Lock = threading.Lock()
        self._ui_path: Path = Path(__file__).with_name("ui") / "roi_picker.html"

    def attach_window(self, window: webview.Window) -> None:
        self._window = window

    def on_runtime_started(self) -> None:
        if self._window is None:
            raise RuntimeError("Host window is not attached")
        self._window.hide()
        self._response_writer.send({"type": "ready"})

    def present(self, request: _HostRequest) -> None:
        if self._window is None:
            raise RuntimeError("Host window is not attached")

        with self._request_lock:
            self._active_request = request
            self._request_complete.clear()

        self._window.set_title(_window_title_for(request.command))
        self._window.load_url(str(self._ui_path.resolve()))
        self._window.show()

    def wait_for_completion(self) -> None:
        _ = self._request_complete.wait()

    def load_config(self) -> dict[str, object]:
        active_request = self._active_request
        if active_request is None:
            raise RuntimeError("No active visualization request")
        return active_request.payload

    def submit_selection(self, payload: dict[str, object]) -> None:
        roi = roi_from_payload(payload)
        self._finish_active_request(ok=True, result=cast(dict[str, object], roi_to_payload(roi)))

    def complete_view(self) -> None:
        self._finish_active_request(ok=True)

    def cancel(self) -> None:
        active_request = self._active_request
        if active_request is not None and active_request.command == "show_anchor":
            self._finish_active_request(ok=True)
            return
        self._finish_active_request(ok=False, error="ROI selection cancelled")

    def on_window_closing(self) -> bool:
        self.cancel()
        return False

    def shutdown(self) -> None:
        if self._window is not None:
            self._window.destroy()

    def _finish_active_request(
        self,
        *,
        ok: bool,
        result: object | None = None,
        error: str | None = None,
    ) -> None:
        if self._window is None:
            raise RuntimeError("Host window is not attached")

        with self._request_lock:
            active_request = self._active_request
            if active_request is None or self._request_complete.is_set():
                self._window.hide()
                return

            response: dict[str, object] = {
                "type": "response",
                "id": active_request.request_id,
                "ok": ok,
            }
            if ok:
                response["result"] = result or {}
            else:
                response["error"] = error or "Visualization request failed"

            self._active_request = None
            self._response_writer.send(response)
            self._request_complete.set()

        self._window.hide()


def run_host(stdin: IO[str], stdout: IO[str]) -> None:
    response_writer = _ResponseWriter(stdout)
    bridge = _VizBridge(response_writer)
    request_queue: Queue[_HostRequest | None] = Queue()

    window = webview.create_window(
        title="MTF Calc",
        html="<html><body></body></html>",
        width=1440,
        height=920,
        min_size=(960, 640),
    )
    if window is None:
        raise RuntimeError("Failed to create visualization host window")

    bridge.attach_window(window)
    window.expose(
        bridge.load_config,
        bridge.submit_selection,
        bridge.complete_view,
        bridge.cancel,
    )
    window.events.closing += bridge.on_window_closing

    webview.start(
        func=_command_loop,
        args=(bridge, request_queue, stdin, response_writer),
        debug=False,
    )


def _command_loop(
    bridge: _VizBridge,
    request_queue: Queue[_HostRequest | None],
    stdin: IO[str],
    response_writer: _ResponseWriter,
) -> None:
    reader = threading.Thread(
        target=_stdin_reader,
        args=(stdin, request_queue, response_writer),
        daemon=True,
    )
    reader.start()
    bridge.on_runtime_started()

    while True:
        try:
            request = request_queue.get(timeout=0.1)
        except Empty:
            continue

        if request is None:
            bridge.shutdown()
            return

        if request.command == "shutdown":
            bridge.shutdown()
            return

        if request.command == "select_roi":
            bridge.present(request)
            bridge.wait_for_completion()
            continue

        if request.command == "show_anchor":
            bridge.present(request)
            bridge.wait_for_completion()
            continue

        response_writer.send(
            {
                "type": "response",
                "id": request.request_id,
                "ok": False,
                "error": f"Unknown visualization command: {request.command}",
            }
        )


def _stdin_reader(
    stdin: IO[str],
    request_queue: Queue[_HostRequest | None],
    response_writer: _ResponseWriter,
) -> None:
    for line in stdin:
        text = line.strip()
        if not text:
            continue

        try:
            payload = cast(dict[str, object], json.loads(text))
            request_queue.put(_coerce_request(payload))
        except Exception as exc:
            response_writer.send(
                {
                    "type": "response",
                    "id": -1,
                    "ok": False,
                    "error": str(exc),
                }
            )

    request_queue.put(None)


def _coerce_request(raw_request: dict[str, object]) -> _HostRequest:
    request_id = raw_request.get("id")
    command = raw_request.get("command")
    payload = raw_request.get("payload")

    if not isinstance(request_id, int):
        raise TypeError("Visualization request id must be an integer")
    if not isinstance(command, str):
        raise TypeError("Visualization command must be a string")
    if not isinstance(payload, dict):
        raise TypeError("Visualization payload must be a dictionary")

    return _HostRequest(
        request_id=request_id,
        command=command,
        payload=cast(dict[str, object], payload),
    )


def _window_title_for(command: str) -> str:
    if command == "select_roi":
        return "Select ROI"
    if command == "show_anchor":
        return "Anchor Preview"
    return "MTF Calc"


def main() -> None:
    run_host(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
