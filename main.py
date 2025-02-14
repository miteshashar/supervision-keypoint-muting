"""
Script to illustrate & understand the impact of muting keypoints
below a threshold on the output pose detection models of YOLO and Mediapipe.
"""
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO


BLEED = 100
IMAGES_PATH = Path("images")
MEDIAPIPE_COLOR: sv.Color = sv.Color.RED
MEDIAPIPE_COLOR_NAME = "Red"
MEDIAPIPE_MODEL = "pose_landmarker_heavy.task"
REPLACEMENT: int | np.ndarray = 0
WRITE = True
YOLO_COLOR: sv.Color = sv.Color.YELLOW
YOLO_COLOR_NAME = "Yellow"
YOLO_MODEL = "yolo11x-pose.pt"


def show_status( # pylint: disable=too-many-arguments
    ib: tqdm, pb: tqdm, img_name: str, th: str, status: int, expand: bool = False
) -> None:
    """
    Show the status of the processing
    """

    os.system("clear")
    ib.set_description(f"Processing {img_name}")
    pb.set_description(f"Processing threshold {th}")
    print("\n" * 4)
    steps = [
        "YOLO Pose Detection",
        "Mediapipe Pose Detection",
        "Combined Annotation",
        "GIF Creation",
        "WebM Conversion",
        "GIF Optimization",
    ]
    if status <= 3 and not expand:
        steps = steps[:3]
    steps = [
        ("✅ " if status >= idx else "⌛️ ") + step
        for (idx, step) in enumerate(steps, 1)
    ]
    print(" → ".join(steps))
    print("")

def get_images() -> list[Path]:
    """
    Get all images in IMAGES_PATH
    """

    images = [
        f
        for f in IMAGES_PATH.iterdir()
        if f.is_file() and f.suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    ]
    images.sort()
    return images

def get_output_paths(img: Path) -> tuple[Path, Path, Path]:
    """
    Get output paths to place the outputs
    """

    # Define paths
    image_folder = IMAGES_PATH / img.stem
    yolo_path = image_folder / "yolo"
    mediapipe_path = image_folder / "mediapipe"
    combined_path = image_folder / "combined"

    # Create folders
    yolo_path.mkdir(exist_ok=True, parents=True)
    mediapipe_path.mkdir(exist_ok=True)
    combined_path.mkdir(exist_ok=True)
    return yolo_path, mediapipe_path, combined_path

def get_yolo_keypoints(image: cv2.typing.MatLike, confidence_threshold: float) -> sv.KeyPoints:
    """
    Get YOLO keypoints for image
    """

    model = YOLO(YOLO_MODEL)
    result = model(image)[0]
    keypoints = sv.KeyPoints.from_ultralytics(result)
    keypoints.xy[keypoints.confidence < confidence_threshold] = REPLACEMENT
    return keypoints

def get_mediapipe_keypoints(image: cv2.typing.MatLike, confidence_threshold: float) -> sv.KeyPoints:
    """
    Get Mediapipe keypoints for image
    """

    image_height, image_width, _ = image.shape
    mediapipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=3,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        pose_landmarker_result = landmarker.detect(mediapipe_image)

    keypoints = sv.KeyPoints.from_mediapipe(
        pose_landmarker_result, (image_width, image_height)
    )

    keypoints.xy[keypoints.confidence < confidence_threshold] = REPLACEMENT
    return keypoints

def annotate_image(
    image: cv2.typing.MatLike, keypoints: sv.KeyPoints, color: sv.Color
) -> cv2.typing.MatLike:
    """
    Annotate image with keypoints
    """

    edge_annotator = sv.EdgeAnnotator(color=invert_sv_color(color), thickness=6)
    vertex_annotator = sv.VertexAnnotator(color=invert_sv_color(color), radius=10)
    image = edge_annotator.annotate(scene=image.copy(), key_points=keypoints)
    image = vertex_annotator.annotate(scene=image, key_points=keypoints)
    edge_annotator.thickness = 4
    edge_annotator.color = color
    image = edge_annotator.annotate(scene=image, key_points=keypoints)
    vertex_annotator.radius = 8
    vertex_annotator.color = color
    return vertex_annotator.annotate(scene=image, key_points=keypoints)

def invert_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Invert color
    """

    return tuple(255 - c for c in color)

def invert_sv_color(color: sv.Color) -> sv.Color:
    """
    Invert color
    """

    return sv.Color.from_bgr_tuple(invert_color(color.as_bgr()))

def label_image(image: cv2.typing.MatLike, threshold: str) -> None:
    """
    Label image with threshold
    """

    labels = [
        # (label, color, font_scale, shadow_thickness, thickness, text_size)
        (f"Threshold: {threshold}", (0, 0, 0), 2, 7, 6),
        (
            f"({YOLO_MODEL})",
            YOLO_COLOR.as_bgr(),  # pylint: disable=no-member
            0.9,
            3,
            2,
        ),
        (
            f"{YOLO_COLOR_NAME} = YOLO",
            YOLO_COLOR.as_bgr(),  # pylint: disable=no-member
            1.25,
            5,
            4,
        ),
        (
            f"({MEDIAPIPE_MODEL})",
            MEDIAPIPE_COLOR.as_bgr(),  # pylint: disable=no-member
            0.9,
            3,
            2,
        ),
        (
            f"{MEDIAPIPE_COLOR_NAME} = Mediapipe",
            MEDIAPIPE_COLOR.as_bgr(),  # pylint: disable=no-member
            1.25,
            5,
            4,
        ),
    ]
    bottom_offset = BLEED
    params = {}
    params["x"] = BLEED // 2
    max_width = 0
    for idx, (label, _, font_scale, _, _) in enumerate(labels):
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 6
        )
        max_width = max(max_width, text_size[0][0])
        labels[idx] = (*labels[idx], text_size)
        bottom_offset += text_size[0][1] + text_size[1] + 20
    params["w"] = max_width + BLEED
    params["h"] = bottom_offset - BLEED // 2
    params["y"] = image.shape[0] - params["h"] - BLEED // 2
    sub_img = image[
        params["y"] : params["y"] + params["h"], params["x"] : params["x"] + params["w"]
    ]
    image[
        params["y"] : params["y"] + params["h"], params["x"] : params["x"] + params["w"]
    ] = cv2.addWeighted(
        sub_img, 0.3, np.ones(sub_img.shape, dtype=np.uint8) * 0, 0.7, 1.0
    )
    bottom_offset = BLEED
    for label, color, font_scale, shadow_thickness, thickness, text_size in labels:
        label_params = [
            image,
            label,
            (BLEED, int(image.shape[0] - bottom_offset)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
        ]
        cv2.putText(*label_params, invert_color(color), shadow_thickness, cv2.LINE_AA)
        cv2.putText(*label_params, color, thickness, cv2.LINE_AA)
        bottom_offset += text_size[0][1] + text_size[1] + 20

def process_image(img: Path, ib: tqdm) -> None:
    """
    Process an image
    """

    img: Path
    progressbar = tqdm(
        range(0, 21), total=20, desc=f"Processing {img.name}", leave=False, position=2
    )
    gif_paths = []
    for factor in progressbar:
        confidence_threshold = factor * 5 / 100
        threshold_str = str(confidence_threshold) + (
            "0" if confidence_threshold * 100 % 10 == 0 else ""
        )

        yolo_path, mediapipe_path, combined_path = get_output_paths(img)

        show_status(ib, progressbar, img.name, threshold_str, 0)

        image = cv2.imread(img.absolute())

        yolo_keypoints = get_yolo_keypoints(image, confidence_threshold)
        yolo_annotated_image = annotate_image(image, yolo_keypoints, YOLO_COLOR)
        cv2.imwrite(
            yolo_path / f"{threshold_str}.jpg",
            yolo_annotated_image,
        )

        show_status(ib, progressbar, img.name, threshold_str, 1)
        mediapipe_keypoints = get_mediapipe_keypoints(image, confidence_threshold)
        mediapipe_annotated_image = annotate_image(image, mediapipe_keypoints, MEDIAPIPE_COLOR)
        cv2.imwrite(
            mediapipe_path / f"{threshold_str}.jpg",
            mediapipe_annotated_image,
        )
        show_status(ib, progressbar, img.name, threshold_str, 2)

        combined_annotated_image = cv2.addWeighted(
            yolo_annotated_image, 0.5, mediapipe_annotated_image, 0.5, 0
        )
        label_image(combined_annotated_image, threshold_str)
        cv2.imwrite(
            combined_path / f"{threshold_str}.jpg",
            combined_annotated_image,
        )
        gif_paths.append(combined_path / f"{threshold_str}.jpg")
        show_status(ib, progressbar, img.name, threshold_str, 3)
    show_status(ib, progressbar, img.name, threshold_str, 3, True)
    print("Creating animated gif")
    gif_path = combined_path / f"{img.stem}.gif"
    cmd = (
        "magick -delay 30 "
        + " ".join([f'"{p.absolute()}"' for p in gif_paths])
        + f' -loop 0 "{gif_path.absolute()}"'
    )
    commands_path = combined_path / "commands.log"
    commands_path.write_text(cmd)
    commands = [cmd]
    os.system(cmd)
    show_status(ib, progressbar, img.name, threshold_str, 4)
    video_path = combined_path / f"{img.stem}.webm"

    cmd = [
        "ffmpeg", "-y",
        "-i", f'"{gif_path.absolute()}"',
        "-c:v", "libvpx-vp9", "-crf", "0", "-b:v", "0",
        "-lossless", "1",
        "-pix_fmt", "yuv444p",
        f'"{video_path}";',
    ]
    commands.append(" ".join(cmd))
    commands_path.write_text("\n".join(commands))
    os.system(" ".join(cmd))
    show_status(ib, progressbar, img.name, threshold_str, 5)
    optimized_gif_path = combined_path / f"{img.stem}_optimized.gif"
    cmd = [
        "ffmpeg", "-y",
        "-i", f'"{video_path.absolute()}"', "-vf", 
        '"fps=5,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"',
        f'"{optimized_gif_path.absolute()}";',
        "rm", f'"{gif_path.absolute()}";',
        "mv", f'"{optimized_gif_path.absolute()}" "{gif_path.absolute()}"',
    ]
    commands.append(" ".join(cmd))
    commands_path.write_text("\n".join(commands))
    os.system(" ".join(cmd))
    ib.update(1)
    show_status(ib, progressbar, img.name, threshold_str, 6)

def main() -> None:
    """
    Process all images
    """

    images: list[Path] = get_images()
    imagebar = tqdm(total=len(images), desc="Processing Images", leave=False, position=0)
    for img in images:
        process_image(img, imagebar)

if __name__ == "__main__":
    main()
