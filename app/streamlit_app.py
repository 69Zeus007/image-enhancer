import streamlit as st
from PIL import Image
import torch
from model.enhancer_torch import SRCNN
import torchvision.transforms as T

# Load model
@st.cache_resource
def load_model():
    model = SRCNN()
    model.load_state_dict(torch.load("srcnn.pt", map_location="cpu"))
    model.eval()
    return model

# Enhance image
def enhance_image(image, model):
    transform = T.ToTensor()
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor).squeeze(0)
    return T.ToPILImage()(output)

# UI
st.set_page_config(page_title="Image Quality Enhancer", layout="centered")
st.title("🖼️ Image Quality Enhancer")
st.markdown("Upload a low-resolution image and see it enhanced using a deep learning model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    model = load_model()
    enhanced = enhance_image(image, model)
    st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    st.download_button("Download Enhanced Image", enhanced.fp, file_name="enhanced.png")