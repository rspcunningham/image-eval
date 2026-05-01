from pathlib import Path

from image_eval.images import image_size, load_image
from image_eval.models import Rect
from image_eval.overlay import show_template_overlay
from image_eval.roi_picker import PickerCancelled, pick_rect
from image_eval.template_io import (
    get_anchor_rect,
    get_bar_rect,
    get_norm_rect,
    load_template,
    new_template,
    save_template,
    set_anchor_rect,
    set_bar_rect,
    set_norm_rect,
)

SOURCE_IMAGE_PATH = Path("samples/raw_object_intensity.npy")
TEMPLATE_PATH = Path("template.json")

GROUPS = [4, 5, 6, 7]
ELEMENTS = [1, 2, 3, 4, 5, 6]
ORIENTATIONS = ["X", "Y"]


def main() -> None:
    image = load_image(SOURCE_IMAGE_PATH)
    width, height = image_size(image)

    if TEMPLATE_PATH.exists():
        template = load_template(TEMPLATE_PATH)
    else:
        template = new_template(
            source_path=SOURCE_IMAGE_PATH,
            width=width,
            height=height,
            groups=GROUPS,
            elements=ELEMENTS,
            orientations=ORIENTATIONS,
        )
        save_template(template, TEMPLATE_PATH)

    try:
        anchor = pick_rect(
            image,
            "Select the anchor square. Enter to confirm. Esc to cancel.",
            initial=get_anchor_rect(template),
        )
        set_anchor_rect(template, anchor)
        save_template(template, TEMPLATE_PATH)

        black_norm = pick_rect(
            image,
            "Select the black normalization ROI.",
            initial=get_norm_rect(template, "black"),
        )
        set_norm_rect(template, "black", black_norm)
        save_template(template, TEMPLATE_PATH)

        white_initial = get_norm_rect(template, "white")
        if white_initial is None:
            white_initial = Rect(
                left=black_norm.right + 10,
                top=black_norm.top,
                width=black_norm.width,
                height=black_norm.height,
            ).clamp(width=width, height=height)

        white_norm = pick_rect(
            image,
            "Select the white normalization ROI.",
            initial=white_initial,
        )
        set_norm_rect(template, "white", white_norm)
        save_template(template, TEMPLATE_PATH)

        for group in GROUPS:
            for element in ELEMENTS:
                for orientation in ORIENTATIONS:
                    prompt = (
                        f"Select Group {group}, Element {element}, "
                        f"{orientation}-directed profile ROI."
                    )
                    rect = pick_rect(
                        image,
                        prompt,
                        initial=get_bar_rect(
                            template,
                            group=group,
                            element=element,
                            orientation=orientation,
                        ),
                    )
                    set_bar_rect(
                        template,
                        group=group,
                        element=element,
                        orientation=orientation,
                        rect=rect,
                    )
                    save_template(template, TEMPLATE_PATH)

    except PickerCancelled:
        save_template(template, TEMPLATE_PATH)
        print(f"Initialization cancelled. Partial template saved to {TEMPLATE_PATH}.")
        return

    save_template(template, TEMPLATE_PATH)
    print(f"Template saved to {TEMPLATE_PATH}.")
    show_template_overlay(image, template)


if __name__ == "__main__":
    main()
