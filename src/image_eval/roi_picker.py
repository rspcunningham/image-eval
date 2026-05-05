from dataclasses import dataclass
from typing import Literal, Sequence

import cv2
import numpy as np

from image_eval.images import ImageArray, to_display_image
from image_eval.models import Rect

DragMode = Literal["draw", "move", "resize", "pan"]
LabeledRect = tuple[str, Rect]

_PROMPT_HEIGHT = 110
_MIN_RECT_SIZE = 2.0


class PickerCancelled(RuntimeError):
    pass


@dataclass
class _Point:
    x: float
    y: float


@dataclass
class _PickerState:
    image: np.ndarray
    prompt: str
    current_label: str | None
    context_rects: tuple[LabeledRect, ...]
    view_width: int
    view_height: int
    image_width: int
    image_height: int
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    rect: Rect | None = None
    drag_mode: DragMode | None = None
    drag_handle: str | None = None
    drag_start_screen: _Point | None = None
    drag_start_image: _Point | None = None
    drag_start_rect: Rect | None = None
    needs_redraw: bool = True


def pick_rect(
    image: ImageArray,
    prompt: str,
    *,
    initial: Rect | None = None,
    current_label: str | None = None,
    context_rects: Sequence[LabeledRect] = (),
    window_name: str = "image-eval initialize",
) -> Rect:
    display_image = to_display_image(image)
    image_height, image_width = display_image.shape[:2]
    view_width = min(max(image_width, 900), 1400)
    view_height = min(max(image_height, 560), 820)

    state = _PickerState(
        image=display_image,
        prompt=prompt,
        current_label=current_label,
        context_rects=tuple(
            (label, rect.clamp(width=image_width, height=image_height))
            for label, rect in context_rects
        ),
        view_width=view_width,
        view_height=view_height,
        image_width=image_width,
        image_height=image_height,
        rect=initial.clamp(width=image_width, height=image_height) if initial is not None else None,
    )
    _fit_to_view(state)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _handle_mouse, state)

    try:
        while True:
            if state.needs_redraw:
                cv2.imshow(window_name, _render(state))
                state.needs_redraw = False

            if _window_closed(window_name):
                raise PickerCancelled()

            key = cv2.waitKeyEx(20)
            if key == -1:
                continue

            key = key & 0xFFFFFFFF
            if key in (10, 13):
                if state.rect is not None and state.rect.is_valid:
                    return state.rect.clamp(width=image_width, height=image_height)
                continue
            if key in (27, ord("q"), ord("Q")):
                raise PickerCancelled()
            if key in (ord("r"), ord("R")):
                state.rect = None
                state.needs_redraw = True
            elif key in (ord("f"), ord("F")):
                _fit_to_view(state)
            elif key in (ord("+"), ord("="), ord("i"), ord("I")):
                _zoom_at(state, factor=1.2, screen_x=view_width / 2, screen_y=view_height / 2)
            elif key in (ord("-"), ord("_"), ord("o"), ord("O")):
                _zoom_at(state, factor=1 / 1.2, screen_x=view_width / 2, screen_y=view_height / 2)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def _handle_mouse(
    event: int,
    x: int,
    y: int,
    flags: int,
    userdata: object,
) -> None:
    state = userdata
    if not isinstance(state, _PickerState):
        return

    if event == cv2.EVENT_MOUSEWHEEL:
        delta = _mouse_wheel_delta(flags)
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        _zoom_at(state, factor=factor, screen_x=x, screen_y=y - _PROMPT_HEIGHT)
        return

    screen_point = _Point(float(x), float(y - _PROMPT_HEIGHT))

    if event == cv2.EVENT_MBUTTONDOWN:
        state.drag_mode = "pan"
        state.drag_start_screen = screen_point
        state.drag_start_rect = None
        return

    if event == cv2.EVENT_MBUTTONUP:
        _stop_drag(state)
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if screen_point.y < 0:
            return

        image_point = _screen_to_image(state, screen_point)
        image_point = _clamp_point(state, image_point)

        state.drag_start_screen = screen_point
        state.drag_start_image = image_point
        state.drag_start_rect = state.rect

        hit = _hit_test(state, image_point)
        if hit == "move":
            state.drag_mode = "move"
        elif hit is not None:
            state.drag_mode = "resize"
            state.drag_handle = hit
        else:
            state.drag_mode = "draw"
            state.rect = Rect.from_bounds(
                image_point.x, image_point.y, image_point.x, image_point.y
            )

        state.needs_redraw = True
        return

    if event == cv2.EVENT_LBUTTONUP:
        _stop_drag(state)
        return

    if event != cv2.EVENT_MOUSEMOVE or state.drag_mode is None:
        return

    if state.drag_mode == "pan":
        if state.drag_start_screen is None:
            return
        state.pan_x += screen_point.x - state.drag_start_screen.x
        state.pan_y += screen_point.y - state.drag_start_screen.y
        state.drag_start_screen = screen_point
        state.needs_redraw = True
        return

    image_point = _clamp_point(state, _screen_to_image(state, screen_point))
    start_point = state.drag_start_image
    start_rect = state.drag_start_rect
    if start_point is None:
        return

    if state.drag_mode == "draw":
        state.rect = Rect.from_bounds(start_point.x, start_point.y, image_point.x, image_point.y)
    elif state.drag_mode == "move" and start_rect is not None:
        dx = image_point.x - start_point.x
        dy = image_point.y - start_point.y
        state.rect = start_rect.translated(dx, dy).clamp(
            width=state.image_width,
            height=state.image_height,
        )
    elif state.drag_mode == "resize" and start_rect is not None and state.drag_handle is not None:
        state.rect = _resize_rect(
            start_rect,
            state.drag_handle,
            image_point,
            image_width=state.image_width,
            image_height=state.image_height,
        )

    state.needs_redraw = True


def _stop_drag(state: _PickerState) -> None:
    if state.rect is not None and not state.rect.is_valid:
        state.rect = None
    state.drag_mode = None
    state.drag_handle = None
    state.drag_start_screen = None
    state.drag_start_image = None
    state.drag_start_rect = None
    state.needs_redraw = True


def _render(state: _PickerState) -> np.ndarray:
    canvas = np.full(
        (_PROMPT_HEIGHT + state.view_height, state.view_width, 3),
        (24, 24, 24),
        dtype=np.uint8,
    )
    _draw_prompt(canvas, state)

    viewport = canvas[_PROMPT_HEIGHT:, :]
    viewport[:, :] = (12, 12, 12)
    _draw_image_viewport(viewport, state)
    _draw_context_rects(canvas, state)

    if state.rect is not None:
        _draw_rect(canvas, state, state.rect)

    return canvas


def _draw_prompt(canvas: np.ndarray, state: _PickerState) -> None:
    cv2.rectangle(canvas, (0, 0), (state.view_width, _PROMPT_HEIGHT), (30, 30, 30), -1)
    cv2.rectangle(canvas, (0, 0), (7, _PROMPT_HEIGHT), (86, 225, 235), -1)

    _put_text_fit(
        canvas,
        "CURRENT ROI",
        (18, 22),
        max_width=state.view_width - 36,
        font_scale=0.45,
        min_scale=0.35,
        color=(165, 210, 215),
        thickness=1,
    )
    _put_text_fit(
        canvas,
        state.current_label or state.prompt,
        (18, 55),
        max_width=state.view_width - 36,
        font_scale=0.78,
        min_scale=0.48,
        color=(245, 245, 245),
        thickness=2,
    )

    if state.current_label is not None:
        _put_text_fit(
            canvas,
            state.prompt,
            (18, 80),
            max_width=state.view_width - 36,
            font_scale=0.43,
            min_scale=0.34,
            color=(195, 195, 195),
            thickness=1,
        )

    controls = (
        f"zoom {state.zoom:.2f}x | saved ROIs {len(state.context_rects)} | "
        "wheel/+/-/i/o zoom | middle drag pan | f fit | r reset | Enter accept | Esc cancel"
    )
    _put_text_fit(
        canvas,
        controls,
        (18, 102),
        max_width=state.view_width - 36,
        font_scale=0.39,
        min_scale=0.3,
        color=(160, 160, 160),
        thickness=1,
    )


def _draw_image_viewport(viewport: np.ndarray, state: _PickerState) -> None:
    x0 = max(0, int(np.floor((0.0 - state.pan_x) / state.zoom)))
    y0 = max(0, int(np.floor((0.0 - state.pan_y) / state.zoom)))
    x1 = min(state.image_width, int(np.ceil((state.view_width - state.pan_x) / state.zoom)))
    y1 = min(state.image_height, int(np.ceil((state.view_height - state.pan_y) / state.zoom)))
    if x1 <= x0 or y1 <= y0:
        return

    crop = state.image[y0:y1, x0:x1]
    dst_x0 = int(round((x0 * state.zoom) + state.pan_x))
    dst_y0 = int(round((y0 * state.zoom) + state.pan_y))
    dst_x1 = int(round((x1 * state.zoom) + state.pan_x))
    dst_y1 = int(round((y1 * state.zoom) + state.pan_y))
    dst_width = max(1, dst_x1 - dst_x0)
    dst_height = max(1, dst_y1 - dst_y0)
    interpolation = cv2.INTER_NEAREST if state.zoom >= 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(crop, (dst_width, dst_height), interpolation=interpolation)

    src_x0 = max(0, -dst_x0)
    src_y0 = max(0, -dst_y0)
    paste_x0 = max(0, dst_x0)
    paste_y0 = max(0, dst_y0)
    paste_x1 = min(state.view_width, dst_x0 + dst_width)
    paste_y1 = min(state.view_height, dst_y0 + dst_height)
    if paste_x1 <= paste_x0 or paste_y1 <= paste_y0:
        return

    src_x1 = src_x0 + (paste_x1 - paste_x0)
    src_y1 = src_y0 + (paste_y1 - paste_y0)
    viewport[paste_y0:paste_y1, paste_x0:paste_x1] = scaled[src_y0:src_y1, src_x0:src_x1]


def _draw_context_rects(canvas: np.ndarray, state: _PickerState) -> None:
    if not state.context_rects:
        return

    overlay = canvas.copy()
    for label, rect in state.context_rects:
        _draw_context_rect(overlay, state, label, rect)
    cv2.addWeighted(overlay, 0.52, canvas, 0.48, 0.0, dst=canvas)


def _draw_context_rect(
    canvas: np.ndarray,
    state: _PickerState,
    label: str,
    rect: Rect,
) -> None:
    left, top, right, bottom = _rect_to_canvas_bounds(state, rect)
    if right < 0 or bottom < _PROMPT_HEIGHT or left > state.view_width or top > canvas.shape[0]:
        return

    color = (150, 170, 170)
    cv2.rectangle(canvas, (left, top), (right, bottom), color, 1, cv2.LINE_AA)

    text_y = max(_PROMPT_HEIGHT + 14, top - 5)
    _put_text_fit(
        canvas,
        label,
        (max(8, left), text_y),
        max_width=max(80, state.view_width - max(8, left) - 8),
        font_scale=0.36,
        min_scale=0.28,
        color=color,
        thickness=1,
    )


def _draw_rect(canvas: np.ndarray, state: _PickerState, rect: Rect) -> None:
    left, top, right, bottom = _rect_to_canvas_bounds(state, rect)

    cv2.rectangle(canvas, (left, top), (right, bottom), (86, 225, 235), 2)
    cv2.rectangle(canvas, (left, top), (right, bottom), (86, 225, 235), 1)

    handle_radius = 4
    for handle_x, handle_y in (
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ):
        cv2.circle(canvas, (handle_x, handle_y), handle_radius, (86, 225, 235), -1)

    label = f"x {rect.left:.1f}  y {rect.top:.1f}  w {rect.width:.1f}  h {rect.height:.1f}"
    cv2.putText(
        canvas,
        label,
        (max(8, left), max(_PROMPT_HEIGHT + 18, top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (86, 225, 235),
        1,
        cv2.LINE_AA,
    )


def _rect_to_canvas_bounds(state: _PickerState, rect: Rect) -> tuple[int, int, int, int]:
    left = int(round((rect.left * state.zoom) + state.pan_x))
    top = int(round((rect.top * state.zoom) + state.pan_y)) + _PROMPT_HEIGHT
    right = int(round((rect.right * state.zoom) + state.pan_x))
    bottom = int(round((rect.bottom * state.zoom) + state.pan_y)) + _PROMPT_HEIGHT
    return left, top, right, bottom


def _put_text_fit(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    max_width: int,
    font_scale: float,
    min_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    scale = font_scale
    while scale > min_scale:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            break
        scale *= 0.92

    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        max(scale, min_scale),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _hit_test(state: _PickerState, point: _Point) -> str | None:
    rect = state.rect
    if rect is None:
        return None

    tolerance = max(5.0 / state.zoom, 3.0)
    near_left = (
        abs(point.x - rect.left) <= tolerance
        and rect.top - tolerance <= point.y <= rect.bottom + tolerance
    )
    near_right = (
        abs(point.x - rect.right) <= tolerance
        and rect.top - tolerance <= point.y <= rect.bottom + tolerance
    )
    near_top = (
        abs(point.y - rect.top) <= tolerance
        and rect.left - tolerance <= point.x <= rect.right + tolerance
    )
    near_bottom = (
        abs(point.y - rect.bottom) <= tolerance
        and rect.left - tolerance <= point.x <= rect.right + tolerance
    )

    if near_left and near_top:
        return "top_left"
    if near_right and near_top:
        return "top_right"
    if near_right and near_bottom:
        return "bottom_right"
    if near_left and near_bottom:
        return "bottom_left"
    if near_left:
        return "left"
    if near_right:
        return "right"
    if near_top:
        return "top"
    if near_bottom:
        return "bottom"
    if rect.left <= point.x <= rect.right and rect.top <= point.y <= rect.bottom:
        return "move"
    return None


def _resize_rect(
    rect: Rect,
    handle: str,
    point: _Point,
    *,
    image_width: int,
    image_height: int,
) -> Rect:
    left = rect.left
    top = rect.top
    right = rect.right
    bottom = rect.bottom

    if "left" in handle:
        left = point.x
    if "right" in handle:
        right = point.x
    if "top" in handle:
        top = point.y
    if "bottom" in handle:
        bottom = point.y

    resized = Rect.from_bounds(left, top, right, bottom)
    if resized.width < _MIN_RECT_SIZE or resized.height < _MIN_RECT_SIZE:
        return rect
    return resized.clamp(width=image_width, height=image_height)


def _fit_to_view(state: _PickerState) -> None:
    state.zoom = min(state.view_width / state.image_width, state.view_height / state.image_height)
    state.pan_x = (state.view_width - (state.image_width * state.zoom)) / 2.0
    state.pan_y = (state.view_height - (state.image_height * state.zoom)) / 2.0
    state.needs_redraw = True


def _zoom_at(state: _PickerState, *, factor: float, screen_x: float, screen_y: float) -> None:
    if screen_y < 0:
        screen_y = 0

    old_zoom = state.zoom
    new_zoom = min(max(old_zoom * factor, 0.05), 80.0)
    if new_zoom == old_zoom:
        return

    image_x = (screen_x - state.pan_x) / old_zoom
    image_y = (screen_y - state.pan_y) / old_zoom
    state.zoom = new_zoom
    state.pan_x = screen_x - (image_x * new_zoom)
    state.pan_y = screen_y - (image_y * new_zoom)
    state.needs_redraw = True


def _screen_to_image(state: _PickerState, point: _Point) -> _Point:
    return _Point(
        x=(point.x - state.pan_x) / state.zoom,
        y=(point.y - state.pan_y) / state.zoom,
    )


def _clamp_point(state: _PickerState, point: _Point) -> _Point:
    return _Point(
        x=min(max(point.x, 0.0), float(state.image_width - 1)),
        y=min(max(point.y, 0.0), float(state.image_height - 1)),
    )


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _mouse_wheel_delta(flags: int) -> int:
    high_word = (flags >> 16) & 0xFFFF
    if high_word >= 0x8000:
        high_word -= 0x10000
    return high_word
