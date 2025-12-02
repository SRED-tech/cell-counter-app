import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.title("Fluorescent Cell Counter")

st.write(
    "Upload a fluorescence image where cells are green on a dark background. "
    "The app will:\n"
    "1. Turn green cells → white and background → black\n"
    "2. Count each white cell (connected component)\n"
    "3. Mark each counted cell with a red dot\n"
    "4. Show the total cell count and overlay image"
)

uploaded_file = st.file_uploader(
    "Upload image (JPG, PNG, TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# Controls to tune detection
st.sidebar.header("Detection settings")
threshold = st.sidebar.slider(
    "Green intensity threshold", min_value=0, max_value=255, value=40, step=1
)
min_size = st.sidebar.slider(
    "Minimum cell area (pixels)", min_value=1, max_value=1000, value=20, step=1
)
max_size = st.sidebar.slider(
    "Maximum cell area (pixels)", min_value=100, max_value=20000, value=5000, step=10
)
use_clahe = st.sidebar.checkbox(
    "Enhance faint cells (CLAHE on green channel)", value=True
)

if uploaded_file is not None:
    # Load image as RGB
    pil_img = Image.open(uploaded_file).convert("RGB")
    img = np.array(pil_img)

    st.subheader("Original image")
    st.image(img, use_column_width=True)

    # --- Step 1: extract green channel ---
    green = img[:, :, 1].astype(np.uint8)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        green_proc = clahe.apply(green)
    else:
        green_proc = green

    # --- Step 2: threshold: green -> white, background -> black ---
    _, bw = cv2.threshold(green_proc, threshold, 255, cv2.THRESH_BINARY)

    st.subheader("Binary image (green → white, background → black)")
    st.image(bw, clamp=True, use_column_width=True)

    # --- Step 3: connected components = cells ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)

    # Filter by size & build a mask for valid cells
    mask_filtered = np.zeros_like(bw)
    filtered_centroids = []

    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size <= area <= max_size:
            mask_filtered[labels == i] = 255
            filtered_centroids.append(centroids[i])

    cell_count = len(filtered_centroids)

    # --- Step 4: overlay red dots on original image ---
    overlay = img.copy()
    for c in filtered_centroids:
        x, y = int(c[0]), int(c[1])
        cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)  # red dot

    st.subheader(f"Detected cells with red markers (count = {cell_count})")
    st.image(overlay, use_column_width=True)

    # --- Step 5: download button for overlay ---
    result_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download overlay image (PNG)",
        data=byte_im,
        file_name="cell_overlay.png",
        mime="image/png",
    )

    # Optional: show filtered mask
    with st.expander("Show filtered cell mask (after size filtering)"):
        st.image(mask_filtered, clamp=True, use_column_width=True)
