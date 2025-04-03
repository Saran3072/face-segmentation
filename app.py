import streamlit as st
import numpy as np
from PIL import Image
import torch
from io import BytesIO

from utils.common import vis_parsing_maps
from inference import prepare_image, load_model
from utils.app_utils import (
    get_model_weights,
    resize_to_fixed_size,
    load_yolo_face_detector,
    crop_face_with_padding,
    count_faces
)

# This will download or use cached weight from HF Hub
get_model_weights()

# Streamlit config
st.set_page_config(page_title="Face Segmentation", layout="wide")
st.title("Face Segmentation")

# Load segmentation model
@st.cache_resource
def load_bisenet_model():
    model_name = "resnet34"
    weight_path = "./weights/resnet34.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19
    model = load_model(model_name, num_classes, weight_path, device)
    return model, device

# Load YOLOv8 face detector
@st.cache_resource
def load_detector():
    return load_yolo_face_detector()

model, device = load_bisenet_model()
detector = load_detector()

# Image upload UI
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    face_count = count_faces(image, detector)

    if face_count != 1:
        st.warning(f"Detected {face_count} face(s). Please upload an image with exactly one face.")
        st.image(image, caption="Original Image", use_container_width=True)
    else:
        st.success("Single face detected. Proceeding with segmentation...")

        # Crop face with padding
        cropped_face = crop_face_with_padding(image, detector)

        # Run model on cropped face
        input_tensor = prepare_image(cropped_face).to(device)
        with torch.no_grad():
            out = model(input_tensor)[0]
            parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)

        # Resize parsing mask to cropped image size
        parsing_anno = np.array(Image.fromarray(parsing_anno.astype(np.uint8)).resize(cropped_face.size, resample=Image.NEAREST))

        # Generate transparent output
        masked_image = vis_parsing_maps(cropped_face, parsing_anno, save_image=False)

        # Resize both images for display
        target_size = (1024, 1024)
        resized_original = resize_to_fixed_size(np.array(cropped_face), size=target_size)
        resized_masked = resize_to_fixed_size(masked_image, size=target_size)

        # Show side-by-side
        col1, col2 = st.columns(2)
        col1.image(resized_original, caption="Original Image", use_container_width=True)
        col2.image(resized_masked, caption="Segmented Image", use_container_width=True)

        # PNG Download button
        buf = BytesIO()
        Image.fromarray(masked_image).save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Segmented PNG",
            data=byte_im,
            file_name="masked_face.png",
            mime="image/png"
        )
