import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import mediapipe as mp
from models.bisenet import BiSeNet
from utils.common import vis_parsing_maps
from inference import prepare_image, load_model
from typing import Tuple

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

st.set_page_config(page_title="Face Segmentation", layout="wide")
st.title("Face Segmentation")

@st.cache_resource
def load_bisenet_model():
    model_name = "resnet18"
    weight_path = "./weights/resnet18.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19
    model = load_model(model_name, num_classes, weight_path, device)
    return model, device

model, device = load_bisenet_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Check if there's only one face
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # MediaPipe expects BGR
    if not detect_single_face(rgb_image):
        st.warning("Multiple or no faces detected. Please upload an image with exactly one face.")
        st.image(image_np, caption="Input Image", use_column_width=True)
    else:
        st.success("Single face detected.")

        # Run model
        input_tensor = prepare_image(image).to(device)
        with torch.no_grad():
            out = model(input_tensor)[0]
            parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)

        # Resize to original size
        parsing_anno = np.array(Image.fromarray(parsing_anno.astype(np.uint8)).resize(image.size, resample=Image.NEAREST))

        # Apply transparent mask
        masked_image = vis_parsing_maps(image, parsing_anno, save_image=False)

        # Resize both to 512x512
        target_size = (512, 512)
        resized_original = resize_to_fixed_size(image_np, size=target_size)
        resized_masked = resize_to_fixed_size(masked_image, size=target_size)

        # Show original and result side-by-side
        col1, col2 = st.columns(2)
        col1.image(image_np, caption="Original Image", use_column_width=True)
        col2.image(masked_image, caption="Masked Output (PNG)", use_column_width=True)

        # Download button
        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(masked_image).save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="⬇️ Download Masked Image (PNG)",
            data=byte_im,
            file_name="masked_face.png",
            mime="image/png"
        )