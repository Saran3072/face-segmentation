import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

from utils.common import vis_parsing_maps
from inference import prepare_image, load_model
from utils.app_utils import get_model_weights, resize_to_fixed_size, detect_single_face

# This will download or use cached weight from HF Hub
get_model_weights()

st.set_page_config(page_title="Face Segmentation", layout="wide")
st.title("Face Segmentation")

@st.cache_resource
def load_bisenet_model():
    model_name = "resnet34"
    weight_path = "./weights/resnet34.pt"
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
        st.image(image_np, caption="Input Image", use_container_width=True)
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
        col1.image(image_np, caption="Original Image", use_container_width=True)
        col2.image(masked_image, caption="Masked Output (PNG)", use_container_width=True)

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