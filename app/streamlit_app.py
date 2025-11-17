import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from PIL import Image, ImageEnhance
import torch
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

# Enhance image using Y channel
def enhance_image(image, model):
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    y_tensor = T.ToTensor()(y).unsqueeze(0)
    with torch.no_grad():
        out_y = model(y_tensor).squeeze(0).clamp(0.0, 1.0)

    out_y_img = T.ToPILImage()(out_y).resize(image.size, Image.BICUBIC)
    final_img = Image.merge("YCbCr", (out_y_img, cb, cr)).convert("RGB")
    return final_img

# Difference map
def compute_difference(original, enhanced):
    orig_np = np.array(original).astype(np.float32)
    enh_np = np.array(enhanced).astype(np.float32)
    diff = np.abs(orig_np - enh_np).mean(axis=2)
    diff = (diff / diff.max() * 255).astype(np.uint8)
    return Image.fromarray(diff).convert("L")

# Page config
st.set_page_config(
    page_title="Image Quality Enhancer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🧠 Image Enhancer")
st.sidebar.markdown("""
This app uses a deep learning model (SRCNN) trained on the Y channel of DIV2K images to upscale and enhance image quality.

**Tech Stack**: PyTorch, Streamlit, PIL  
**Model**: SRCNN (3-layer CNN)  
**Training**: L1 loss on Y channel  
**Dataset**: DIV2K HR + LR (bicubic X2)

Upload a low-res image and see the enhanced result side-by-side.
""")

# Main UI
st.title("📸 Super-Resolution Image Enhancer")
st.markdown("Upload a low-resolution image and enhance it using a deep learning model trained on luminance.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    model = load_model()

    # Downscale to simulate low-res input
    low_res = image.resize((image.width // 2, image.height // 2), Image.BICUBIC)
    upscaled = low_res.resize(image.size, Image.BICUBIC)

    # Enhance
    enhanced = enhance_image(upscaled, model)

    # Optional sharpening
    if st.sidebar.checkbox("🔧 Apply sharpening", value=True):
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)

    # Show images
    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="Original", use_column_width=True)
    col2.image(upscaled, caption="Upscaled (Bicubic)", use_column_width=True)
    col3.image(enhanced, caption="Enhanced (SRCNN)", use_column_width=True)

    # Difference map
    if st.sidebar.checkbox("🧪 Show difference map"):
        diff_map = compute_difference(upscaled, enhanced)
        st.image(diff_map, caption="Difference Map", use_column_width=True)

    # Download button
    buf = io.BytesIO()
    enhanced.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("📥 Download Enhanced Image", buf, file_name="enhanced.png", mime="image/png")