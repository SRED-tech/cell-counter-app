import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import zipfile
import pandas as pd


# ------------------ Core processing function ------------------ #

def process_image_array(
    img_rgb: np.ndarray,
    threshold: int,
    min_size: int,
    max_size: int,
    use_clahe: bool,
    blur_ksize: int,
    morph_kernel: int,
    split_touching: bool,
    split_strength: float,
):
    """
    Process a single RGB image and return:
      - initial_bw: initial mask after thresholding + morph
      - final_bw: mask after optional splitting
      - overlay_rgb: final_bw with red dots on each counted cell
      - count: number of counted cells
    """
    # extract green channel
    green = img_rgb[:, :, 1].astype(np.uint8)

    # CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        proc = clahe.apply(green)
    else:
        proc = green

    # Gaussian blur
    if blur_ksize > 0:
        k = blur_ksize
        if k % 2 == 0:
            k += 1
        proc = cv2.GaussianBlur(proc, (k, k), 0)

    # Threshold -> initial mask
    _, bw = cv2.threshold(proc, threshold, 255, cv2.THRESH_BINARY)

    # Morph cleanup
    if morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    initial_bw = bw.copy()

    # Optional splitting
    if split_touching:
        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
        if dist.max() > 0:
            thresh_val = (1.0 - split_strength) * dist.max()
        else:
            thresh_val = 0

        _, sure_fg = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[bw == 0] = 0

        bw_color = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(bw_color, markers)

        split_mask = np.zeros_like(bw, dtype=np.uint8)
        split_mask[markers > 1] = 255
        final_bw = split_mask
    else:
        final_bw = bw

    # Connected components on final mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_bw)

    mask_filtered = np.zeros_like(final_bw)
    filtered_centroids = []

    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size <= area <= max_size:
            mask_filtered[labels == i] = 255
            filtered_centroids.append(centroids[i])

    cell_count = len(filtered_centroids)

    # Create overlay (B/W + red dots)
    mask_bgr = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)
    for c in filtered_centroids:
        x, y = int(c[0]), int(c[1])
        cv2.circle(mask_bgr, (x, y), 5, (0, 0, 255), -1)  # red in BGR

    overlay_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    return initial_bw, final_bw, overlay_rgb, cell_count


# ------------------ ROI helper ------------------ #

def get_roi_and_scale(img_rgb: np.ndarray, use_roi: bool, roi_choice: str):
    """
    Returns:
      roi_img (RGB),
      (x0, y0, x1, y1) coordinates in original image,
      scale_factor = (full_area / roi_area)
    If use_roi is False, returns the full image and scale_factor = 1.
    """
    h, w, _ = img_rgb.shape
    if not use_roi:
        return img_rgb, (0, 0, w, h), 1.0

    # 25% area square => half width & half height (quadrants / centre)
    w_half, h_half = w // 2, h // 2

    if roi_choice == "Top-left":
        x0, y0, x1, y1 = 0, 0, w_half, h_half
    elif roi_choice == "Top-right":
        x0, y0, x1, y1 = w - w_half, 0, w, h_half
    elif roi_choice == "Bottom-left":
        x0, y0, x1, y1 = 0, h - h_half, w_half, h
    elif roi_choice == "Bottom-right":
        x0, y0, x1, y1 = w - w_half, h - h_half, w, h
    else:  # Center
        x0 = w // 4
        x1 = x0 + w_half
        y0 = h // 4
        y1 = y0 + h_half

    roi = img_rgb[y0:y1, x0:x1].copy()
    area_full = float(w * h)
    area_roi = float((x1 - x0) * (y1 - y0))
    scale_factor = area_full / area_roi if area_roi > 0 else 1.0
    return roi, (x0, y0, x1, y1), scale_factor


# ------------------ Streamlit UI ------------------ #

st.title("Fluorescent Cell Counter (B/W + Red Dots + Batch + ROI)")

st.write(
    "Workflow:\n"
    "1. Upload **one image** below and tune the detection parameters.\n"
    "2. Optionally enable **25% ROI** if parts of the image are blurry; the app will "
    "count only the selected square and scale up the count.\n"
    "3. Once happy, upload **multiple images** in the batch section.\n"
    "4. The same settings (and ROI) will be applied to all images."
)

# ---- Sidebar controls ----
st.sidebar.header("Detection settings")

threshold = st.sidebar.slider(
    "Intensity threshold (on processed green channel)",
    min_value=0,
    max_value=255,
    value=60,
    step=1,
    help="Lower = more sensitive (includes fainter cells). "
         "Higher = stricter (only brighter cells)."
)

min_size = st.sidebar.slider(
    "Minimum cell area (pixels)",
    min_value=1,
    max_value=1000,
    value=20,
    step=1,
    help="Objects smaller than this are ignored as noise."
)

max_size = st.sidebar.slider(
    "Maximum cell area (pixels)",
    min_value=100,
    max_value=20000,
    value=5000,
    step=10,
    help="Objects larger than this are ignored (aggregates, artefacts)."
)

use_clahe = st.sidebar.checkbox(
    "Enhance faint cells (CLAHE)",
    value=True,
    help="Improves local contrast before thresholding to pick up faint cells."
)

blur_ksize = st.sidebar.slider(
    "Gaussian blur (kernel size)",
    min_value=0,
    max_value=11,
    value=3,
    step=2,
    help="0 = no blur. Larger odd values smooth noise and can help connect faint cells."
)

morph_kernel = st.sidebar.slider(
    "Morphological cleanup kernel size",
    min_value=1,
    max_value=7,
    value=3,
    step=2,
    help="Used for opening (erode+dilate) to remove tiny specks. "
         "1 = no cleanup; larger = stronger."
)

st.sidebar.header("Touching cell splitting")

split_touching = st.sidebar.checkbox(
    "Split touching cells (watershed)",
    value=False,
    help="Try to separate neighbouring cells that touch each other."
)

split_strength = st.sidebar.slider(
    "Split strength",
    min_value=0.7,
    max_value=1.0,
    value=0.95,
    step=0.01,
    help="Controls how strongly touching cells are separated. "
         "Higher = more splitting. Start around 0.90–0.97 and fine-tune."
)

st.sidebar.header("Region of interest (ROI)")

use_roi = st.sidebar.checkbox(
    "Use 25% ROI and scale up",
    value=False,
    help="If parts of the image are out of focus, count only a 25% square and "
         "estimate total cells by scaling."
)

roi_choice = st.sidebar.selectbox(
    "ROI position",
    ["Center", "Top-left", "Top-right", "Bottom-left", "Bottom-right"],
    index=0,
    help="Which 25% square of the image to use for counting."
)

# ------------ Single-image tuning section ------------ #

st.header("1️⃣ Tune parameters on a single image")

single_file = st.file_uploader(
    "Upload ONE image to tune settings", type=["jpg", "jpeg", "png", "tif", "tiff"], key="single"
)

if single_file is not None:
    pil_img = Image.open(single_file).convert("RGB")
    img_rgb_full = np.array(pil_img)

    # Get ROI + scale factor
    roi_img, (x0, y0, x1, y1), scale_factor = get_roi_and_scale(img_rgb_full, use_roi, roi_choice)

    # Show original with ROI rectangle (if enabled)
    orig_disp = img_rgb_full.copy()
    if use_roi:
        cv2.rectangle(orig_disp, (x0, y0), (x1, y1), (0, 255, 0), 2)  # green box
    st.subheader("Original uploaded image (green box = region counted)" if use_roi else "Original uploaded image")
    st.image(orig_disp, use_column_width=True)

    # Process ROI
    initial_bw, final_bw, overlay_rgb, roi_count = process_image_array(
        roi_img,
        threshold,
        min_size,
        max_size,
        use_clahe,
        blur_ksize,
        morph_kernel,
        split_touching,
        split_strength,
    )

    st.subheader("ROI black & white mask (white = detected signal)")
    st.image(initial_bw, clamp=True, use_column_width=True)

    if split_touching:
        st.subheader("ROI mask after splitting touching cells")
        st.image(final_bw, clamp=True, use_column_width=True)

    # Estimated total count (scaled)
    est_total = int(round(roi_count * scale_factor))

    st.subheader(
        f"Detected cells in ROI with RED markers (ROI count = {roi_count}, "
        f"scale ×{scale_factor:.2f}, estimated total ≈ {est_total})"
    )
    st.image(overlay_rgb, use_column_width=True)

    # Downloads for this image (ROI-level)
    buf_final = io.BytesIO()
    Image.fromarray(final_bw).save(buf_final, format="PNG")
    buf_final.seek(0)
    st.download_button(
        "Download ROI final mask (PNG)",
        data=buf_final,
        file_name="single_image_roi_final_mask.png",
        mime="image/png",
        key="dl_single_mask",
    )

    buf_overlay = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf_overlay, format="PNG")
    buf_overlay.seek(0)
    st.download_button(
        "Download ROI mask + RED dots (PNG)",
        data=buf_overlay,
        file_name="single_image_roi_overlay_red_dots.png",
        mime="image/png",
        key="dl_single_overlay",
    )

else:
    st.info("Upload one image above to fine-tune your settings (and ROI) before batch counting.")

# ------------ Batch processing section ------------ #

st.header("2️⃣ Batch count multiple images with current settings + ROI")

batch_files = st.file_uploader(
    "Upload multiple images for batch counting",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
    key="batch",
)

if batch_files:
    st.write(
        f"Processing **{len(batch_files)}** images with the current settings "
        "(threshold, area filters, CLAHE, blur, splitting, ROI, etc.)."
    )

    results = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for f in batch_files:
            pil_img = Image.open(f).convert("RGB")
            img_rgb_full = np.array(pil_img)

            # ROI for this image
            roi_img, (_, _, _, _), scale_factor = get_roi_and_scale(
                img_rgb_full, use_roi, roi_choice
            )

            # Process ROI
            _, final_bw, overlay_rgb, roi_count = process_image_array(
                roi_img,
                threshold,
                min_size,
                max_size,
                use_clahe,
                blur_ksize,
                morph_kernel,
                split_touching,
                split_strength,
            )

            est_total = int(round(roi_count * scale_factor))
            filename = f.name
            results.append(
                {
                    "filename": filename,
                    "roi_count": roi_count,
                    "scale_factor": scale_factor,
                    "estimated_total": est_total,
                }
            )

            # Save ROI overlay into the zip
            overlay_bytes = io.BytesIO()
            Image.fromarray(overlay_rgb).save(overlay_bytes, format="PNG")
            overlay_bytes.seek(0)
            zip_out.writestr(f"{filename}_ROI_overlay.png", overlay_bytes.read())

    # Show results table
    df = pd.DataFrame(results)
    st.subheader("Batch results (ROI counts and estimated totals)")
    st.dataframe(df)

    # Download CSV of counts
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download counts as CSV",
        data=csv_buf.getvalue(),
        file_name="batch_cell_counts_with_roi.csv",
        mime="text/csv",
        key="dl_csv",
    )

    # Download ZIP of overlays
    zip_buffer.seek(0)
    st.download_button(
        "Download all ROI overlay images (ZIP)",
        data=zip_buffer,
        file_name="batch_roi_overlays_red_dots.zip",
        mime="application/zip",
        key="dl_zip",
    )
