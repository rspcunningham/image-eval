from __future__ import annotations

import atexit
import json
from queue import Empty
import subprocess
import sys
import threading
from typing import IO, cast

import numpy as np
from numpy.typing import NDArray

from mtf_calc._roi_tools import build_select_roi_config, build_show_anchor_config, roi_from_payload
from mtf_calc.models import Anchor, MtfResult
from mtf_calc.models import Roi


class _VizHostClient:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] = subprocess.Popen(
            [sys.executable, "-m", "mtf_calc._viz_host"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Visualization host pipes are unavailable")

        self._stdin: IO[str] = self._process.stdin
        self._stdout: IO[str] = self._process.stdout
        self._request_lock: threading.Lock = threading.Lock()
        self._next_request_id: int = 1
        self._closed: bool = False
        self._await_ready()

    def select_roi(self, raw_image: NDArray[np.float32], size_ref: Roi | None = None) -> Roi:
        response = self._request(
            command="select_roi",
            payload=build_select_roi_config(raw_image, size_ref=size_ref),
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Visualization host returned an invalid ROI result")
        return roi_from_payload(cast(dict[str, object], result))

    def show_anchor(self, raw_image: NDArray[np.float32], anchor: Anchor) -> None:
        _ = self._request(
            command="show_anchor",
            payload=build_show_anchor_config(raw_image, anchor),
        )

    def close(self) -> None:
        self._shutdown(wait_timeout=5.0)

    def close_for_atexit(self) -> None:
        self._shutdown(wait_timeout=0.1)

    def _shutdown(self, *, wait_timeout: float) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self._process.poll() is None:
                self._send_message({"id": 0, "command": "shutdown", "payload": {}})
                _ = self._process.wait(timeout=wait_timeout)
        except BaseException:
            try:
                self._process.terminate()
            except Exception:
                return

    def _await_ready(self) -> None:
        while True:
            try:
                message = self._read_message()
            except Empty:
                self._raise_if_crashed("Visualization host failed to start")
                continue

            if message.get("type") == "ready":
                return

    def _request(self, *, command: str, payload: dict[str, object]) -> dict[str, object]:
        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1

            self._send_message(
                {
                    "id": request_id,
                    "command": command,
                    "payload": payload,
                }
            )

            while True:
                try:
                    message = self._read_message()
                except Empty:
                    self._raise_if_crashed(f"Visualization host died while handling {command}")
                    continue

                if message.get("type") != "response":
                    continue
                if message.get("id") != request_id:
                    continue
                if not message.get("ok"):
                    error = message.get("error")
                    if not isinstance(error, str):
                        error = f"Visualization command failed: {command}"
                    raise RuntimeError(error)
                return message

    def _send_message(self, payload: dict[str, object]) -> None:
        message = json.dumps(payload)
        _ = self._stdin.write(message)
        _ = self._stdin.write("\n")
        self._stdin.flush()

    def _read_message(self) -> dict[str, object]:
        line = self._stdout.readline()
        if line == "":
            raise Empty
        return cast(dict[str, object], json.loads(line))

    def _raise_if_crashed(self, message: str) -> None:
        if self._process.poll() is None:
            return

        exit_code = self._process.returncode
        raise RuntimeError(f"{message} (exit code {exit_code})")


_client: _VizHostClient | None = None
_client_lock = threading.Lock()


def select_roi(raw_image: NDArray[np.float32], size_ref: Roi | None = None) -> Roi:
    return _get_client().select_roi(raw_image, size_ref=size_ref)


def show_anchor(raw_image: NDArray[np.float32], anchor: Anchor) -> None:
    _get_client().show_anchor(raw_image, anchor)


def close() -> None:
    global _client

    with _client_lock:
        if _client is None:
            return
        _client.close()
        _client = None


def _close_for_atexit() -> None:
    global _client

    with _client_lock:
        if _client is None:
            return
        _client.close_for_atexit()
        _client = None


def _get_client() -> _VizHostClient:
    global _client

    with _client_lock:
        if _client is None:
            _client = _VizHostClient()
        return _client


_ = atexit.register(_close_for_atexit)


def show_mtf_graph(mtf_result: MtfResult) -> None:
    del mtf_result
    raise NotImplementedError
