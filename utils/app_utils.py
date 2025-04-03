import mediapipe as mp
import numpy as np
from PIL import Image
from typing import Tuple
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def get_model_weights(repo_id="Saran30702/face_segmentation", filename="resnet34.pt") -> str:
    """
    Downloads model weights from Hugging Face Hub (cached locally).
    Returns path to downloaded weight file.
    """
    cache_dir = "./weights/"
    os.makedirs(cache_dir, exist_ok=True)

    weight_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        local_dir=cache_dir,
        local_dir_use_symlinks=False
    )

def detect_single_face(image: np.ndarray) -> bool:
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.25) as detector:
        results = detector.process(image)
        return results.detections is not None and len(results.detections) == 1
    
# Resize both images to the same width for side-by-side layout
def resize_to_fixed_size(img: np.ndarray, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Resize an image to a fixed (width, height), ignoring aspect ratio.
    """
    pil_img = Image.fromarray(img)
    return np.array(pil_img.resize(size, Image.BILINEAR))

def load_yolo_face_detector() -> YOLO:
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    return YOLO(model_path)

def count_faces(image: Image.Image, detector: YOLO) -> int:
    """
    Returns the number of faces detected in the image using YOLO.
    """
    results = detector(image)
    for result in results:
        if result.boxes is not None:
            return result.boxes.shape[0]
    return 0

def crop_face_with_padding(image: Image.Image, detector: YOLO) -> Image.Image:
    """
    Assumes exactly one face is present. Crops face with:
    - 10% padding (top/left/right)
    - 20% padding (bottom)
    """
    results = detector(image)
    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.shape[0] != 1:
            raise ValueError("crop_face_with_padding requires exactly one detected face.")

        box = boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box.astype(int)

        box_w = x2 - x1
        box_h = y2 - y1

        img_w, img_h = image.size

        # Desired padding
        desired_pad_x = int(box_w * 1.0)
        desired_pad_top = int(box_h * 1.0)
        desired_pad_bottom = int(box_h * 0.50)

        # Adjust padding to not exceed image bounds
        pad_left = min(desired_pad_x, x1)
        pad_top = min(desired_pad_top, y1)
        pad_right = min(desired_pad_x, img_w - x2)
        pad_bottom = min(desired_pad_bottom, img_h - y2)

        # Apply safe padded box
        new_x1 = x1 - pad_left
        new_y1 = y1 - pad_top
        new_x2 = x2 + pad_right
        new_y2 = y2 + pad_bottom

        return image.crop((new_x1, new_y1, new_x2, new_y2))