"""
Streamlit demo for solar panel segmentation.
Sends uploaded image to the FastAPI backend and displays results.
"""

import base64
import io

import numpy as np
import requests
import streamlit as st
from PIL import Image

API_URL = "https://solar-panel-segmentation-655838531680.europe-west1.run.app/predict"

st.set_page_config(
    page_title="Solar Panel Detector",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Panel Segmentation")
st.markdown(
    "Upload an aerial image to detect and segment solar panels using a "
    "U-Net model trained on the BDAPPV dataset (IoU: **88.25%**)."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload an aerial image (PNG or JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:

    # show original image
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running inference on Cloud Run..."):
        # send to API
        uploaded_file.seek(0)
        response = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, uploaded_file, "image/png")},
            timeout=120,
        )

    if response.status_code != 200:
        st.error(f"API error: {response.status_code} — {response.text}")
    else:
        data = response.json()

        # decode mask
        mask_bytes = base64.b64decode(data["mask_base64"])
        mask = Image.open(io.BytesIO(mask_bytes))
        mask_arr = np.array(mask)
        img_arr = np.array(image.resize((400, 400)))

        # overlay
        overlay = img_arr.copy()
        red_channel = mask_arr > 127
        overlay[red_channel] = [255, 50, 50]
        overlay = (img_arr * 0.6 + overlay * 0.4).astype(np.uint8)

        # display
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_arr, caption="Input Image", use_container_width=True)

        with col2:
            st.image(mask_arr, caption="Predicted Mask", use_container_width=True)

        with col3:
            st.image(overlay, caption=f"Overlay", use_container_width=True)

        st.divider()

        # metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Panel Coverage", f"{data['panel_coverage_percent']}%")
        m2.metric("Inference Time", f"{data['inference_time_ms']:.0f}ms")
        m3.metric("Image Size", f"{data['image_size'][0]}×{data['image_size'][1]}px")

        st.caption(
            "Model: U-Net (ResNet34 encoder, ImageNet pretrained) | "
            "Loss: Dice | Metric: IoU | "
            "Dataset: BDAPPV (Kasmi et al., 2023)"
        )

st.divider()
st.markdown(
    "**GitHub**: [m-umar-raza/solar-panel-segmentation]"
    "(https://github.com/m-umar-raza/solar-panel-segmentation) | "
    "**API Docs**: [Cloud Run](/docs)"
)