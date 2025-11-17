import streamlit as st
from PIL import Image, ImageEnhance
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.enhancer_torch import SRCNN
import torchvision.transforms as T
import numpy as np
import io

# Load model
@st.cache_resource
def load_model():
    model = SRCNN()
    model.load_state_dict(torch.load("srcnn.pt", map_location="cpu"))
    model.eval()
    return model

# Enhance image
def enhance_image(image, model):
    input_tensor = T.ToTensor()(image).unsqueeze(0)  # shape: [1, 3, H, W]
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).clamp(0.0, 1.0)
    return T.ToPILImage()(output)

# Difference heatmap
def compute_difference(original, enhanced):
    orig_np = np.array(original).astype(np.float32)
    enh_np = np.array(enhanced).astype(np.float32)
    diff = np.abs(orig_np - enh_np).mean(axis=2)  # grayscale diff
    diff = (diff / diff.max() * 255).astype(np.uint8)
    return Image.fromarray(diff).convert("L")

# UI
st.set_page_config(page_title="Image Quality Enhancer", layout="wide")
st.title("🖼️ Image Quality Enhancer")
st.markdown("Upload a low-resolution image and see it enhanced using a deep learning model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    model = load_model()

    # Downscale to simulate low-res input
    #low_res = image.resize((image.width // 2, image.height // 2), Image.BICUBIC)

    # Upscale back to original size (baseline)
    #upscaled = low_res.resize(image.size, Image.BICUBIC)

    # Enhance with SRCNN
    enhanced = enhance_image(image, model)

    # Optional: sharpen for visual pop
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)

    # Compute difference map
    diff_map = compute_difference(image, enhanced)

    # Layout
    col1, col2, col3, col4 = st.columns(4)
    col1.image(image, caption="Original", use_container_width=True)
    #col2.image(low_res, caption="Downscaled", use_container_width=True)
    col3.image(enhanced, caption="Enhanced", use_container_width=True)
    col4.image(diff_map, caption="Difference Map", use_container_width=True)

    # Download button
    buf = io.BytesIO()
    enhanced.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download Enhanced Image", buf, file_name="enhanced.png", mime="image/png")