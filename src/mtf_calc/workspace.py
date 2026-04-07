from __future__ import annotations

import json
import os
import shutil
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from mtf_calc.pipeline import STAGES, PipelineState, StageStatus, build_pipeline_state, initial_stage_statuses, stage_index
from mtf_calc.usaf1951 import roi_slot_metadata
from mtf_calc.workflow.stage6_fit import DEFAULT_HARMONIC_COUNT, fit_profile
from mtf_calc.workflow.stage6_profiles import build_stage6_profiles


HIST_BINS = 256


class WorkspaceError(Exception):
    """Base workspace error."""


class InvalidSourceError(WorkspaceError):
    """Raised when a source payload is invalid."""


class NoSourceLoadedError(WorkspaceError):
    """Raised when an operation needs a source."""


class InvalidStageError(WorkspaceError):
    """Raised when a stage name is unknown."""


class StageResultNotFoundError(WorkspaceError):
    """Raised when a stage result is missing."""


class StageAdvanceError(WorkspaceError):
    """Raised when a stage cannot advance yet."""


class InvalidStagePayloadError(WorkspaceError):
    """Raised when a stage-specific payload is invalid."""


class ReusableConfigNotFoundError(WorkspaceError):
    """Raised when no reusable config has been saved."""


Rect = tuple[int, int, int, int]


def default_workspace_root() -> Path:
    env_value = os.environ.get("MTF_CALC_HOME")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path.cwd() / ".mtf-calc").resolve()


class WorkspaceStore:
    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or default_workspace_root()).resolve()
        self._lock = RLock()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def current_dir(self) -> Path:
        return self._root / "current"

    @property
    def source_path(self) -> Path:
        return self.current_dir / "source.npy"

    @property
    def metadata_path(self) -> Path:
        return self.current_dir / "metadata.json"

    @property
    def state_path(self) -> Path:
        return self.current_dir / "state.json"

    @property
    def stages_dir(self) -> Path:
        return self.current_dir / "stages"

    @property
    def reusable_config_path(self) -> Path:
        return self._root / "reusable-config.json"

    def reset(self) -> None:
        with self._lock:
            if self.current_dir.exists():
                shutil.rmtree(self.current_dir)

    def has_source(self) -> bool:
        with self._lock:
            return self.source_path.exists() and self.metadata_path.exists() and self.state_path.exists()

    def load_source(self, *, payload: bytes, file_name: str) -> None:
        array, metadata = self._parse_source(payload=payload, file_name=file_name)

        with self._lock:
            if self.current_dir.exists():
                shutil.rmtree(self.current_dir)

            self.stages_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.source_path, array, allow_pickle=False)
            self._write_json(self.metadata_path, metadata)
            self._write_json(
                self.state_path,
                {
                    "current": STAGES[0].name,
                    "stages": {
                        name: status.value
                        for name, status in initial_stage_statuses(has_source=True).items()
                    },
                },
            )

    def source_summary(self) -> dict[str, Any] | None:
        with self._lock:
            if not self.metadata_path.exists():
                return None
            return self._read_json(self.metadata_path)

    def source_array(self) -> np.ndarray:
        with self._lock:
            if not self.source_path.exists():
                raise NoSourceLoadedError("No source array is loaded.")
            return np.load(self.source_path, allow_pickle=False)

    def source_buffer(self) -> bytes:
        return self.source_array().tobytes(order="C")

    def pipeline_state(self) -> PipelineState:
        with self._lock:
            if not self.state_path.exists():
                return build_pipeline_state(initial_stage_statuses(has_source=False), has_source=False)

            raw_state = self._read_json(self.state_path)
            statuses = {
                name: StageStatus(raw_state["stages"].get(name, StageStatus.PENDING.value))
                for name in [stage.name for stage in STAGES]
            }
            return build_pipeline_state(statuses, has_source=self.has_source())

    def current_stage_name(self) -> str | None:
        pipeline = self.pipeline_state()
        if pipeline.current is None:
            return None
        return pipeline.stages[pipeline.current].name

    def reusable_config_count(self) -> int:
        with self._lock:
            self._ensure_reusable_config_seeded()
            return 1 if self.reusable_config_path.exists() else 0

    def reusable_config_summary(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_reusable_config_seeded()
            if not self.reusable_config_path.exists():
                return {"count": 0}

            payload = self._read_json(self.reusable_config_path)
            source = payload.get("source", {})
            return {
                "count": 1,
                "sourceFileName": source.get("fileName"),
            }

    def stage_result_path(self, name: str) -> Path:
        self._require_stage(name)
        return self.stages_dir / f"{name}.json"

    def get_stage_result(self, name: str) -> dict[str, Any]:
        with self._lock:
            if name == "bar_rois":
                return self._load_bar_roi_payload()
            if name == "norm_rois":
                return self._load_norm_roi_payload()
            if name == "stage_6":
                return self._load_stage_6_result()
            if name == "stage_7":
                return self._load_stage_7_result()
            path = self.stage_result_path(name)
            if not path.exists():
                raise StageResultNotFoundError(f"No persisted result for stage '{name}'.")
            return self._read_json(path)

    def ensure_stage_result(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            if name == "bar_rois":
                if not self.stage_result_path(name).exists():
                    return None
                return self._load_bar_roi_payload()
            if name == "norm_rois":
                return self._load_norm_roi_payload()
            if name == "stage_6":
                if not self._can_compute_stage_6():
                    return None
                return self._load_stage_6_result()
            if name == "stage_7":
                if not self._can_compute_stage_7():
                    return None
                return self._load_stage_7_result()
            path = self.stage_result_path(name)
            if not path.exists():
                return None
            return self._read_json(path)

    def set_stage_result(self, name: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._require_source()
            self._require_stage(name)
            self.stages_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(self.stage_result_path(name), data)

    def set_scale_groups(self, groups: list[int]) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            normalized = self._validate_scale_groups(groups)
            payload = {"groups": normalized}
            self._write_json(self.stage_result_path("scale"), payload)
            self._write_json(self.stage_result_path("bar_rois"), self._build_bar_roi_sequence(normalized))
            self._write_json(self.stage_result_path("norm_rois"), self._build_norm_roi_sequence())
            self._invalidate_stage_results("stage_6", "stage_7")
            return payload

    def set_bar_roi(self, key: str, rect: Rect) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            self._validate_rect(rect)
            stage_payload = self._load_bar_roi_payload()
            slots = stage_payload["slots"]
            updated = False
            for slot in slots:
                if slot["key"] == key:
                    slot["rect"] = _rect_dict(rect)
                    updated = True
                    break
            if not updated:
                raise InvalidStagePayloadError(f"Unknown ROI slot: {key}")
            self._write_json(self.stage_result_path("bar_rois"), stage_payload)
            self._invalidate_stage_results("stage_6", "stage_7")
            return stage_payload

    def clear_bar_roi(self, key: str) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            stage_payload = self._load_bar_roi_payload()
            slots = stage_payload["slots"]
            updated = False
            for slot in slots:
                if slot["key"] == key:
                    slot["rect"] = None
                    updated = True
                    break
            if not updated:
                raise InvalidStagePayloadError(f"Unknown ROI slot: {key}")
            self._write_json(self.stage_result_path("bar_rois"), stage_payload)
            self._invalidate_stage_results("stage_6", "stage_7")
            return stage_payload

    def set_norm_roi(self, key: str, rect: Rect) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            self._validate_rect(rect)
            stage_payload = self._load_norm_roi_payload()
            slots = stage_payload["slots"]
            target = next((slot for slot in slots if slot["key"] == key), None)
            if target is None:
                raise InvalidStagePayloadError(f"Unknown normalization ROI slot: {key}")

            if target["tone"] == "white":
                black_slot = self._matching_norm_slot(slots, "black")
                if black_slot is None or black_slot["rect"] is None:
                    raise InvalidStagePayloadError("Select the black ROI before selecting the white ROI.")
                black_rect = black_slot["rect"]
                if rect[2] != black_rect["height"] or rect[3] != black_rect["width"]:
                    raise InvalidStagePayloadError("White ROI must match the black ROI dimensions.")
            else:
                white_slot = self._matching_norm_slot(slots, "white")
                if white_slot is not None and white_slot["rect"] is not None:
                    white_rect = white_slot["rect"]
                    if rect[2] != white_rect["height"] or rect[3] != white_rect["width"]:
                        white_slot["rect"] = None

            target["rect"] = _rect_dict(rect)
            self._write_json(self.stage_result_path("norm_rois"), stage_payload)
            self._invalidate_stage_results("stage_6", "stage_7")
            return stage_payload

    def clear_norm_roi(self, key: str) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            stage_payload = self._load_norm_roi_payload()
            slots = stage_payload["slots"]
            target = next((slot for slot in slots if slot["key"] == key), None)
            if target is None:
                raise InvalidStagePayloadError(f"Unknown normalization ROI slot: {key}")

            target["rect"] = None
            if target["tone"] == "black":
                white_slot = self._matching_norm_slot(slots, "white")
                if white_slot is not None:
                    white_slot["rect"] = None
            self._write_json(self.stage_result_path("norm_rois"), stage_payload)
            self._invalidate_stage_results("stage_6", "stage_7")
            return stage_payload

    def advance_stage(self) -> PipelineState:
        with self._lock:
            self._require_source()
            raw_state = self._load_state_dict()
            current_name = raw_state["current"]
            if current_name is None:
                return self.pipeline_state()

            self._assert_can_advance(current_name)
            current_idx = stage_index(current_name)
            raw_state["stages"][current_name] = StageStatus.COMPLETE.value

            next_idx = current_idx + 1
            if next_idx < len(STAGES):
                next_name = STAGES[next_idx].name
                if current_name == "norm_rois" and next_name == "stage_6":
                    self._save_reusable_config()
                raw_state["stages"][next_name] = StageStatus.ACTIVE.value
                raw_state["current"] = next_name
            else:
                raw_state["current"] = None

            self._write_json(self.state_path, raw_state)
            statuses = {name: StageStatus(value) for name, value in raw_state["stages"].items()}
            return build_pipeline_state(statuses, has_source=True)

    def retreat_stage(self) -> PipelineState:
        with self._lock:
            self._require_source()
            raw_state = self._load_state_dict()
            current_name = raw_state["current"]
            if current_name is None:
                raise StageAdvanceError("No active stage to go back from.")

            current_idx = stage_index(current_name)
            if current_idx == 0:
                raise StageAdvanceError("Already at the first stage.")

            prev_name = STAGES[current_idx - 1].name
            raw_state["stages"][current_name] = StageStatus.PENDING.value
            raw_state["stages"][prev_name] = StageStatus.ACTIVE.value
            raw_state["current"] = prev_name

            self._write_json(self.state_path, raw_state)
            statuses = {name: StageStatus(value) for name, value in raw_state["stages"].items()}
            return build_pipeline_state(statuses, has_source=True)

    def apply_reusable_config(self) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            saved = self._load_reusable_config()
            current_anchor = self.get_stage_result("anchor")

            delta_x = float(current_anchor["center_x"]) - float(saved["anchor"]["center_x"])
            delta_y = float(current_anchor["center_y"]) - float(saved["anchor"]["center_y"])

            scale_payload = saved["scale"]
            bar_rois_payload = self._translate_roi_payload(saved["bar_rois"], delta_x=delta_x, delta_y=delta_y)
            norm_rois_payload = self._translate_roi_payload(saved["norm_rois"], delta_x=delta_x, delta_y=delta_y)

            self._invalidate_stage_results("stage_6", "stage_7")
            self._write_json(self.stage_result_path("scale"), scale_payload)
            self._write_json(self.stage_result_path("bar_rois"), bar_rois_payload)
            self._write_json(self.stage_result_path("norm_rois"), norm_rois_payload)
            stage_6_payload = self._compute_stage_6_result()
            self._write_json(self.stage_result_path("stage_6"), stage_6_payload)

            raw_state = {
                "current": "stage_6",
                "stages": {
                    "view": StageStatus.COMPLETE.value,
                    "anchor": StageStatus.COMPLETE.value,
                    "scale": StageStatus.COMPLETE.value,
                    "bar_rois": StageStatus.COMPLETE.value,
                    "norm_rois": StageStatus.COMPLETE.value,
                    "stage_6": StageStatus.ACTIVE.value,
                    "stage_7": StageStatus.PENDING.value,
                },
            }
            self._write_json(self.state_path, raw_state)

            statuses = {name: StageStatus(value) for name, value in raw_state["stages"].items()}
            return {
                "pipeline": build_pipeline_state(statuses, has_source=True).model_dump(mode="json"),
                "anchor": current_anchor,
                "scale": scale_payload,
                "bar_rois": bar_rois_payload,
                "norm_rois": norm_rois_payload,
                "stage_6": stage_6_payload,
                "translation": {"x": delta_x, "y": delta_y},
            }

    def _assert_can_advance(self, name: str) -> None:
        if name == "anchor" and not self.stage_result_path("anchor").exists():
            raise StageAdvanceError("Run anchor detection before advancing.")
        if name == "scale" and not self.stage_result_path("scale").exists():
            raise StageAdvanceError("Submit the selected scale groups before advancing.")
        if name == "bar_rois":
            payload = self._load_bar_roi_payload()
            missing = [slot for slot in payload["slots"] if slot["rect"] is None]
            if missing:
                raise StageAdvanceError("Select all required ROIs before advancing.")
        if name == "norm_rois":
            payload = self._load_norm_roi_payload()
            missing = [slot for slot in payload["slots"] if slot["rect"] is None]
            if missing:
                raise StageAdvanceError("Select all normalization ROIs before advancing.")

    def _load_state_dict(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise NoSourceLoadedError("No active document.")
        return self._read_json(self.state_path)

    def _require_source(self) -> None:
        if not self.has_source():
            raise NoSourceLoadedError("No source array is loaded.")

    def _require_stage(self, name: str) -> None:
        if name not in {stage.name for stage in STAGES}:
            raise InvalidStageError(f"Unknown stage '{name}'.")

    def _parse_source(self, *, payload: bytes, file_name: str) -> tuple[np.ndarray, dict[str, Any]]:
        if not payload:
            raise InvalidSourceError("The uploaded file was empty.")

        try:
            array = np.load(BytesIO(payload), allow_pickle=False)
        except Exception as error:
            raise InvalidSourceError("Not a valid .npy file.") from error

        if not isinstance(array, np.ndarray):
            raise InvalidSourceError("Expected a NumPy array payload.")
        if array.dtype != np.float32:
            raise InvalidSourceError(f"Expected float32 data, got dtype={array.dtype}.")
        if array.ndim != 2:
            raise InvalidSourceError(f"Expected a 2D array, got shape {tuple(array.shape)}.")

        contiguous = np.ascontiguousarray(array)
        if not np.isfinite(contiguous).all():
            raise InvalidSourceError("Array contains NaN or infinite values.")

        data_min = float(contiguous.min())
        data_max = float(contiguous.max())
        if data_min < 0 or data_max > 1:
            raise InvalidSourceError(
                f"Expected values in [0, 1], got range {data_min:.4f} .. {data_max:.4f}.",
            )

        histogram, _ = np.histogram(contiguous, bins=HIST_BINS, range=(0.0, 1.0))
        metadata = {
            "fileName": file_name,
            "sourceLabel": "Local file",
            "byteLength": len(payload),
            "rows": int(contiguous.shape[0]),
            "cols": int(contiguous.shape[1]),
            "dtype": "float32",
            "dataMin": data_min,
            "dataMax": data_max,
            "histogram": histogram.astype(np.int64).tolist(),
        }
        return contiguous, metadata

    def _validate_scale_groups(self, groups: list[int]) -> list[int]:
        if not groups:
            raise InvalidStagePayloadError("Select at least one scale group.")
        normalized: list[int] = []
        seen: set[int] = set()
        for value in groups:
            if not isinstance(value, int):
                raise InvalidStagePayloadError("Scale groups must be integers.")
            if value < -2 or value > 7:
                raise InvalidStagePayloadError("Scale groups must be between -2 and 7.")
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        normalized.sort()
        return normalized

    def _validate_rect(self, rect: Rect) -> None:
        summary = self.source_summary()
        if summary is None:
            raise NoSourceLoadedError("No source loaded.")
        row, col, height, width = rect
        if height <= 0 or width <= 0:
            raise InvalidStagePayloadError("ROI must have positive dimensions.")
        if row < 0 or col < 0:
            raise InvalidStagePayloadError("ROI origin must be non-negative.")
        if row + height > summary["rows"] or col + width > summary["cols"]:
            raise InvalidStagePayloadError("ROI extends outside the source array.")

    def _save_reusable_config(self) -> None:
        payload = {
            "source": self.source_summary(),
            "anchor": self.get_stage_result("anchor"),
            "scale": self.get_stage_result("scale"),
            "bar_rois": self.get_stage_result("bar_rois"),
            "norm_rois": self.get_stage_result("norm_rois"),
        }
        self._write_json(self.reusable_config_path, payload)

    def _ensure_reusable_config_seeded(self) -> None:
        if self.reusable_config_path.exists() or not self.has_source():
            return

        raw_state = self._load_state_dict()
        required = ("anchor", "scale", "bar_rois", "norm_rois")
        if any(raw_state["stages"].get(stage_name) != StageStatus.COMPLETE.value for stage_name in required):
            return

        self._save_reusable_config()

    def _load_reusable_config(self) -> dict[str, Any]:
        if not self.reusable_config_path.exists():
            raise ReusableConfigNotFoundError("No reusable config is available.")
        return self._read_json(self.reusable_config_path)

    def _translate_roi_payload(self, payload: dict[str, Any], *, delta_x: float, delta_y: float) -> dict[str, Any]:
        translated = self._migrate_bar_roi_slots(payload)
        translated = json.loads(json.dumps(translated))
        for slot in translated.get("slots", []):
            rect = slot.get("rect")
            if rect is None:
                continue
            next_rect = (
                int(round(rect["row"] + delta_y)),
                int(round(rect["col"] + delta_x)),
                int(rect["height"]),
                int(rect["width"]),
            )
            self._validate_rect(next_rect)
            slot["rect"] = _rect_dict(next_rect)
        return translated

    def _build_bar_roi_sequence(self, groups: list[int]) -> dict[str, Any]:
        slots: list[dict[str, Any]] = []
        for group in groups:
            for roi_number in range(1, 7):
                for axis in ("X", "Y"):
                    key = f"group-{group}-roi-{roi_number}{axis}"
                    slots.append({
                        "key": key,
                        "group": group,
                        "roiNumber": roi_number,
                        "axis": axis,
                        "label": f"{group} · {roi_number}{axis}",
                        **roi_slot_metadata(group, roi_number),
                        "rect": None,
                    })
        return {"slots": slots}

    def _load_bar_roi_payload(self) -> dict[str, Any]:
        path = self.stage_result_path("bar_rois")
        if not path.exists():
            raise InvalidStagePayloadError("Select scale groups before assigning bar ROIs.")
        payload = self._read_json(path)
        migrated = self._migrate_bar_roi_slots(payload)
        if migrated != payload:
            self._write_json(path, migrated)
        return migrated

    def _migrate_bar_roi_slots(self, payload: dict[str, Any]) -> dict[str, Any]:
        migrated = json.loads(json.dumps(payload))
        changed = False
        for slot in migrated.get("slots", []):
            axis = slot.get("axis")
            if axis == "H":
                axis = "X"
                changed = True
            elif axis == "V":
                axis = "Y"
                changed = True
            if slot.get("axis") != axis:
                slot["axis"] = axis
                changed = True
            group = slot.get("group")
            roi_number = slot.get("roiNumber")
            if axis is not None and group is not None and roi_number is not None:
                expected_key = f"group-{group}-roi-{roi_number}{axis}"
                expected_label = f"{group} · {roi_number}{axis}"
                if slot.get("key") != expected_key:
                    slot["key"] = expected_key
                    changed = True
                if slot.get("label") != expected_label:
                    slot["label"] = expected_label
                    changed = True
                metadata = roi_slot_metadata(int(group), int(roi_number))
                for field, value in metadata.items():
                    if slot.get(field) != value:
                        slot[field] = value
                        changed = True
        return migrated if changed else payload

    def _build_norm_roi_sequence(self) -> dict[str, Any]:
        return {
            "slots": [
                {"key": "norm-black", "tone": "black", "label": "Black", "rect": None},
                {"key": "norm-white", "tone": "white", "label": "White", "rect": None},
            ]
        }

    def _load_stage_6_result(self) -> dict[str, Any]:
        path = self.stage_result_path("stage_6")
        if path.exists():
            payload = self._read_json(path)
            migrated = self._migrate_stage_6_result(payload)
            if migrated != payload:
                self._write_json(path, migrated)
            return migrated
        if not self._can_compute_stage_6():
            raise StageResultNotFoundError("Stage 6 requires completed bar and normalization ROIs.")
        payload = self._compute_stage_6_result()
        self._write_json(path, payload)
        return payload

    def _compute_stage_6_result(self) -> dict[str, Any]:
        source = self.source_array()
        bar_payload = self._load_bar_roi_payload()
        norm_payload = self._load_norm_roi_payload()

        missing_bar = [slot["key"] for slot in bar_payload["slots"] if slot.get("rect") is None]
        if missing_bar:
            raise InvalidStagePayloadError("Select all required bar ROIs before computing Stage 6.")

        missing_norm = [slot["key"] for slot in norm_payload["slots"] if slot.get("rect") is None]
        if missing_norm:
            raise InvalidStagePayloadError("Select all normalization ROIs before computing Stage 6.")

        return build_stage6_profiles(
            source,
            bar_slots=bar_payload["slots"],
            norm_slots=norm_payload["slots"],
        )

    def set_stage_6_crop(self, key: str, *, left: int, right: int) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            payload = self._load_stage_6_result()
            profiles = payload.get("profiles", [])
            target = next((profile for profile in profiles if profile.get("key") == key), None)
            if target is None:
                raise InvalidStagePayloadError(f"Unknown Stage 6 profile: {key}")

            profile_length = int(target.get("profileLength", 0))
            if left < 0 or right < 0:
                raise InvalidStagePayloadError("Stage 6 crop values must be non-negative.")
            if profile_length < 2:
                raise InvalidStagePayloadError("Stage 6 profile is too short to crop.")
            if left + right > profile_length - 2:
                raise InvalidStagePayloadError("Stage 6 crop leaves too few samples for fitting.")

            target["crop"] = {"left": left, "right": right}
            target["fit"] = None
            self._invalidate_stage_results("stage_7")
            self._write_json(self.stage_result_path("stage_6"), payload)
            return payload

    def run_stage_6_fit(self, key: str, *, harmonic_count: int) -> dict[str, Any]:
        with self._lock:
            self._require_source()
            if harmonic_count < 1 or harmonic_count > 15:
                raise InvalidStagePayloadError("Stage 6 harmonic count must be between 1 and 15.")

            payload = self._load_stage_6_result()
            normalization = payload.get("normalization", {})
            profiles = payload.get("profiles", [])
            target = next((profile for profile in profiles if profile.get("key") == key), None)
            if target is None:
                raise InvalidStagePayloadError(f"Unknown Stage 6 profile: {key}")

            fit = fit_profile(
                target.get("normalizedProfile"),
                raw_profile=target.get("rawProfile") or target.get("profile") or [],
                base_frequency_lp_per_mm=float(target.get("spatialFrequencyLpPerMm")),
                crop_left=int((target.get("crop") or {}).get("left", 0)),
                crop_right=int((target.get("crop") or {}).get("right", 0)),
                black_mean=normalization.get("blackMean"),
                contrast=normalization.get("contrast"),
                harmonic_count=harmonic_count,
            )
            if fit is None:
                raise InvalidStagePayloadError("Unable to fit this profile. Check normalization, crop window, and harmonic count.")

            target["fit"] = fit
            self._invalidate_stage_results("stage_7")
            self._write_json(self.stage_result_path("stage_6"), payload)
            return payload

    def _can_compute_stage_6(self) -> bool:
        bar_path = self.stage_result_path("bar_rois")
        norm_path = self.stage_result_path("norm_rois")
        return self.has_source() and bar_path.exists() and norm_path.exists()

    def _can_compute_stage_7(self) -> bool:
        return self._can_compute_stage_6()

    def _migrate_stage_6_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        migrated = json.loads(json.dumps(payload))
        changed = False
        normalization = migrated.get("normalization", {})
        black_mean = normalization.get("blackMean")
        white_mean = normalization.get("whiteMean")
        contrast = normalization.get("contrast")
        normalization_valid = (
            black_mean is not None
            and white_mean is not None
            and contrast is not None
            and abs(float(contrast)) > 1e-9
        )
        if normalization.get("normalized") != normalization_valid:
            normalization["normalized"] = normalization_valid
            changed = True
        for profile in migrated.get("profiles", []):
            profile_length = int(profile.get("profileLength", 0))
            crop = profile.get("crop")
            default_crop = {"left": 0, "right": 0}
            if (
                not isinstance(crop, dict)
                or not isinstance(crop.get("left"), int)
                or not isinstance(crop.get("right"), int)
                or crop["left"] < 0
                or crop["right"] < 0
                or (profile_length >= 2 and crop["left"] + crop["right"] > profile_length - 2)
            ):
                profile["crop"] = default_crop
                changed = True
            raw_profile = profile.get("rawProfile")
            if raw_profile is None and profile.get("profile") is not None:
                raw_profile = profile["profile"]
                profile["rawProfile"] = raw_profile
                changed = True
            if raw_profile is None:
                continue
            if profile.get("profile") != raw_profile:
                profile["profile"] = raw_profile
                changed = True
            normalized_profile = (
                [((float(value) - float(black_mean)) / float(contrast)) for value in raw_profile]
                if normalization_valid
                else None
            )
            if profile.get("normalizedProfile") != normalized_profile:
                profile["normalizedProfile"] = normalized_profile
                changed = True
            fit = profile.get("fit")
            if fit is not None:
                harmonic_count = int(fit.get("harmonicCount", DEFAULT_HARMONIC_COUNT))
                if harmonic_count < 1 or harmonic_count > 15:
                    profile["fit"] = None
                    changed = True
        return migrated if changed else payload

    def _load_stage_7_result(self) -> dict[str, Any]:
        path = self.stage_result_path("stage_7")
        if path.exists():
            return self._read_json(path)
        if not self._can_compute_stage_7():
            raise StageResultNotFoundError("Stage 7 requires completed Stage 6 profile fits.")
        payload = self._compute_stage_7_result()
        self._write_json(path, payload)
        return payload

    def _compute_stage_7_result(self) -> dict[str, Any]:
        stage_6 = self._load_stage_6_result()
        curves: dict[str, list[dict[str, Any]]] = {"X": [], "Y": []}
        total_profiles = 0

        for profile in stage_6.get("profiles", []):
            total_profiles += 1
            axis = str(profile.get("axis") or "")
            fit = profile.get("fit") or {}
            mtf = fit.get("mtf") or {}
            frequency = profile.get("spatialFrequencyLpPerMm")
            first_harmonic_mtf = mtf.get("firstHarmonicMtf")
            if axis not in curves or frequency is None or first_harmonic_mtf is None:
                continue
            curves[axis].append(
                {
                    "key": profile.get("key"),
                    "label": profile.get("label"),
                    "group": profile.get("group"),
                    "roiNumber": profile.get("roiNumber"),
                    "frequencyLpPerMm": float(frequency),
                    "mtf": float(first_harmonic_mtf),
                    "periodSamples": fit.get("periodSamples"),
                    "harmonicCount": fit.get("harmonicCount"),
                    "rmse": fit.get("rmse"),
                }
            )

        for axis in curves:
            curves[axis].sort(key=lambda point: (point["frequencyLpPerMm"], point.get("roiNumber", 0)))

        fitted_profiles = sum(len(points) for points in curves.values())
        return {
            "curves": curves,
            "summary": {
                "totalProfiles": total_profiles,
                "fittedProfiles": fitted_profiles,
                "xPoints": len(curves["X"]),
                "yPoints": len(curves["Y"]),
            },
        }

    def _invalidate_stage_results(self, *names: str) -> None:
        for name in names:
            path = self.stage_result_path(name)
            if path.exists():
                path.unlink()

    def _load_norm_roi_payload(self) -> dict[str, Any]:
        path = self.stage_result_path("norm_rois")
        default_payload = self._build_norm_roi_sequence()
        if not path.exists():
            self._write_json(path, default_payload)
            return default_payload

        payload = self._read_json(path)
        slots = payload.get("slots", [])
        tones = [slot.get("tone") for slot in slots]
        if len(slots) != 2 or tones != ["black", "white"]:
            self._write_json(path, default_payload)
            return default_payload
        return payload

    def _matching_norm_slot(self, slots: list[dict[str, Any]], tone: str) -> dict[str, Any] | None:
        return next((slot for slot in slots if slot["tone"] == tone), None)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def _rect_dict(rect: Rect | None) -> dict[str, int] | None:
    if rect is None:
        return None
    return {"row": rect[0], "col": rect[1], "height": rect[2], "width": rect[3]}
