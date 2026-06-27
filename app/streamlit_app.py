"""Aircraft Classifier — Streamlit app (single page, horizontal tabs).

Run locally:   streamlit run app/streamlit_app.py
"""
import io
import sys
import base64
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_paste_button import paste_image_button
from torchvision import transforms

# Make the project's src/ package importable when run from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict
from src.explain import grad_cam
from src.data import IMAGENET_MEAN, IMAGENET_STD

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_aircraft5.pt"
SAMPLES_DIR = PROJECT_ROOT / "app" / "samples"
ASSETS = PROJECT_ROOT / "app" / "assets"
HEADER_IMG = ASSETS / "terbe_rezso-aircraft-9927406.jpg"

# App preprocessing: squash the WHOLE image to 224x224 (no crop), so the model
# sees the full frame and Grad-CAM covers the entire original image.
APP_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

st.set_page_config(page_title="Aircraft Classifier", layout="wide",
                   initial_sidebar_state="collapsed")


def inject_css():
    """Baby-blue -> white gradient, thin full-bleed header, no sidebar, clean tabs."""
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #a9def7 0%, #d8f0fd 38%, #ffffff 100%);
            overflow-x: hidden;   /* stop the 100vw header causing a scrollbar */
        }
        [data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }
        /* No left sidebar — navigation is the horizontal tabs below the header. */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] { display: none; }
        .block-container { padding-top: 0.5rem; max-width: 1100px; }

        /* Full-bleed header that stays THIN on desktop. Tune the height clamp
           (min, preferred, max) and object-position (vertical framing) below. */
        .full-bleed { width: 92vw; margin-left: calc(50% - 46vw); line-height: 0; }
        .full-bleed img {
            width: 100%;
            height: clamp(180px, 24vw, 340px);
            object-fit: cover;
            object-position: center 60%;   /* higher % = show lower part of photo */
            border-radius: 14px;
            display: block;
        }

        [data-testid="stImage"] img { border-radius: 14px; }

        /* Horizontal nav tabs: centered, larger, NO box background. */
        .stTabs [data-baseweb="tab-list"] { justify-content: center; gap: 28px; }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem; font-weight: 600; padding: 8px 4px;
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def header():
    """Render a full-width, thin banner + title. CSS handles the cropping/height."""
    if HEADER_IMG.exists():
        img = Image.open(HEADER_IMG).convert("RGB")
        if img.width > 1600:                       # downscale for a lighter payload
            img = img.resize((1600, round(img.height * 1600 / img.width)))
        st.markdown(
            f'<div class="full-bleed"><img src="data:image/jpeg;base64,'
            f'{_img_to_b64(img)}"/></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        "<h1 style='text-align:center;margin:0.6rem 0 0;'>Aircraft Classifier</h1>"
        "<p style='text-align:center;color:#2b5876;margin-top:0.2rem;'>"
        "ResNet-18 transfer learning · Grad-CAM explanations</p>",
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    # CPU: Streamlit Cloud has no GPU, and CPU is plenty for single-image inference.
    return load_model(str(MODEL_PATH), device="cpu")


def list_samples():
    """Return {class_name: [image_paths]} from app/samples/<class>/, if present."""
    out = {}
    if SAMPLES_DIR.exists():
        for cls_dir in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
            imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
            if imgs:
                out[cls_dir.name] = imgs
    return out


def cam_overlay(model, pil, class_idx, size):
    """Grad-CAM heatmap for one class, resized to `size` (w, h) of the display image."""
    cam, _ = grad_cam(model, pil, class_idx=class_idx, device="cpu", tf=APP_TF)
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(size, Image.BILINEAR)
    return np.asarray(cam_img) / 255.0


def show_prediction(model, classes, image, true_name=None):
    """Run prediction + Grad-CAM on a PIL image (or path) and render it."""
    pil = image if isinstance(image, Image.Image) else Image.open(image)
    pil = pil.convert("RGB")

    ranked = predict(model, classes, pil, device="cpu", tf=APP_TF)
    pred_name, conf = ranked[0]

    base = np.asarray(pil)  # original image, original aspect ratio
    cam_pred = cam_overlay(model, pil, classes.index(pred_name), pil.size)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil, caption=(f"True: {true_name}" if true_name else "Your image"),
                 use_container_width=True)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(base)
        ax.imshow(cam_pred, cmap="jet", alpha=0.45)
        ax.axis("off")
        ax.set_title(f'Grad-CAM: why "{pred_name}"')
        st.pyplot(fig)

    if true_name is None:
        st.info(f"Prediction: **{pred_name}**  ({conf * 100:.1f}%)")
    elif pred_name == true_name:
        st.success(f"Correct — **{pred_name}**  ({conf * 100:.1f}%)")
    else:
        st.error(f"Wrong — guessed **{pred_name}** ({conf * 100:.1f}%), "
                 f"true is **{true_name}**")

    st.subheader("Confidence per class")
    for name, p in ranked:
        st.progress(min(max(p, 0.0), 1.0), text=f"{name} — {p * 100:.1f}%")

    # Grad-CAM for EVERY class — where the model looks to argue each one.
    st.subheader("Where the model looks, per class")
    gw = 300
    grid = pil.resize((gw, max(1, round(pil.height * gw / pil.width))))
    grid_np = np.asarray(grid)
    fig2, axes = plt.subplots(1, len(classes), figsize=(2.6 * len(classes), 2.9))
    for ax, cls in zip(np.atleast_1d(axes), classes):
        ax.imshow(grid_np)
        ax.imshow(cam_overlay(model, pil, classes.index(cls), grid.size),
                  cmap="jet", alpha=0.45)
        ax.axis("off")
        ax.set_title(cls, fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)


def render_predict(model, classes):
    st.write("**Recognises:** " + " · ".join(classes))
    mode = st.radio("Image source",
                    ["Upload your own", "Paste from clipboard", "Random sample"],
                    horizontal=True)

    if mode == "Upload your own":
        up = st.file_uploader("Upload an aircraft photo",
                              type=["jpg", "jpeg", "png"])
        if up is not None:
            show_prediction(model, classes, Image.open(up))

    elif mode == "Paste from clipboard":
        pasted = paste_image_button("Paste image from clipboard", errors="ignore")
        if pasted.image_data is not None:
            show_prediction(model, classes, pasted.image_data)

    else:
        samples = list_samples()
        if not samples:
            st.warning("No sample images bundled. Add some to `app/samples/<class>/` "
                       "(or run `python scripts/build_app_assets.py` locally).")
        elif st.button("Pick a random aircraft"):
            cls = random.choice(list(samples))
            path = random.choice(samples[cls])
            show_prediction(model, classes, path, true_name=cls)


def render_about():
    st.markdown(
        "Identify a commercial aircraft (Boeing / Airbus) from a photo, using deep "
        "learning. Built to do the **full ML pipeline properly** — from raw data to "
        "a deployed, explainable model."
    )
    st.subheader("The pipeline")
    st.markdown(
        """
1. **Data** — FGVC-Aircraft, filtered to 5 classes, official train/val/test splits
   (no leakage), copyright banner cropped off.
2. **Model** — ResNet-18 pretrained on ImageNet (transfer learning).
3. **Phase 1** — train a fresh 5-class head with the backbone frozen.
4. **Phase 2** — class weights (fix imbalance) + fine-tune the whole backbone.
5. **Evaluate** — balanced accuracy & macro-F1, confusion matrix.
6. **Explain** — Grad-CAM heatmaps to see *where* the model looks.
"""
    )

    col_cm, col_report = st.columns([1, 1])
    with col_cm:
        cm_path = ASSETS / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
    with col_report:
        report_path = ASSETS / "classification_report.txt"
        if report_path.exists():
            st.code(report_path.read_text(), language="text")


def main():
    inject_css()
    header()
    model, classes = get_model()

    tab_predict, tab_about = st.tabs(["Predict", "About"])
    with tab_predict:
        render_predict(model, classes)
    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
