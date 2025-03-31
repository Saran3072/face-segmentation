import mediapipe as mp
import numpy as np
from PIL import Image
from typing import Tuple
import os
from huggingface_hub import hf_hub_download

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

