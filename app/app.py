
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.car_damage.predict import (
    load_trained_model,
    predict_image
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "artifacts"
    / "best_model_exp2.pth"
)

CLASS_DESCRIPTIONS = {
    "F_Breakage": "Front-side structural or component breakage",
    "F_Crushed": "Crushed or compressed damage at the front",
    "F_Normal": "No visible front-side damage",
    "R_Breakage": "Rear-side structural or component breakage",
    "R_Crushed": "Crushed or compressed damage at the rear",
    "R_Normal": "No visible rear-side damage"
}

# --------------------------------------------------
# 2. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AutoInspect | Car Damage AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# 3. Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F7F9FC;
    color: #172B4D;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1400px !important;
    width: 100%;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

h1, h2, h3 {
    color: #172B4D;
    font-weight: 650;
}

.subtitle {
    color: #667085;
    font-size: 16px;
    margin-top: -10px;
}

.top-label {
    color: #2563EB;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.section-heading {
    font-size: 21px;
    font-weight: 650;
    color: #172B4D;
    margin-bottom: 12px;
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 14px;
}

.prediction-label {
    color: #667085;
    font-size: 13px;
    font-weight: 600;
}

.prediction-value {
    color: #172B4D;
    font-size: 27px;
    font-weight: 700;
    margin-top: 5px;
}

.confidence-value {
    color: #2563EB;
    font-size: 34px;
    font-weight: 750;
}

.small-muted {
    color: #667085;
    font-size: 13px;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #B8C5D6;
    border-radius: 12px;
    padding: 12px;
}

hr {
    border-color: #E5EAF2;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 4. Load trained model
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading trained model...")
def load_model():

    model, device = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH
    )

    return model, device


# --------------------------------------------------
# 5. Header
# --------------------------------------------------

st.markdown(
    '<div class="top-label">AI-POWERED VEHICLE INSPECTION</div>',
    unsafe_allow_html=True
)

st.title("Car Damage Intelligence")

st.markdown(
    '<p class="subtitle">'
    'Analyze vehicle images and identify potential damage '
    'categories using deep learning.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# 6. Image upload section
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">Vehicle Image Analysis</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your vehicle image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP."
)


MAX_FILE_SIZE_MB = 10

if uploaded_file is not None:

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(
            f"Image size exceeds {MAX_FILE_SIZE_MB} MB. "
            "Please upload a smaller image."
        )
        st.stop()

    try:
        uploaded_file.seek(0)

        image = Image.open(uploaded_file)
        image.verify()

        uploaded_file.seek(0)

        image = Image.open(uploaded_file).convert("RGB")

        # Validate image dimensions
        MIN_IMAGE_DIMENSION = 100
        MAX_IMAGE_DIMENSION = 10000

        width, height = image.size

        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            st.error(
                "Image resolution is too low. "
                "Please upload an image of at least 100 × 100 pixels."
            )
            st.stop()

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            st.error(
                "Image dimensions are too large. "
                "Please upload a smaller image."
            )
            st.stop()

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Invalid or corrupted image. Please upload a valid image.")
        st.stop()

# Reset previous prediction when a new image is uploaded
if uploaded_file is not None:

    current_file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    if st.session_state.get("current_file_id") != current_file_id:

        st.session_state["current_file_id"] = current_file_id
        st.session_state.pop("prediction_result", None)

else:
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("current_file_id", None)

if uploaded_file is not None:

    try:
        # image = Image.open(uploaded_file).convert("RGB")

        left_col, right_col = st.columns(
            [1, 1],
            gap="large"
        )

        # --------------------------------------------------
        # 7. Image preview
        # --------------------------------------------------

        with left_col:

            with st.container(border=True):

                st.subheader("Uploaded Vehicle")

                st.image(
                    image,
                    use_container_width=True
                )

                st.caption(
                    f"Image dimensions: {image.width} × {image.height}"
                )

        # --------------------------------------------------
        # 8. Prediction section
        # --------------------------------------------------

        with right_col:

            with st.container(border=True):

                st.subheader("Damage Assessment")

                st.markdown(
                    '<p class="small-muted">'
                    'Run the AI model to analyze the uploaded image.'
                    '</p>',
                    unsafe_allow_html=True
                )

                predict_button = st.button(
                    "Analyze Vehicle Image",
                    type="primary"
                )

                if predict_button:

                    try:

                        with st.spinner(
                            "Analyzing vehicle image..."
                        ):

                            model, device = load_model()

                            result = predict_image(
                                image_input=image,
                                model=model,
                                device=device
                            )

                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"]

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="prediction-label">
                                    PREDICTED DAMAGE CATEGORY
                                </div>
                                <div class="prediction-value">
                                    {predicted_class}
                                </div>
                                <p class="small-muted">
                                    {CLASS_DESCRIPTIONS[predicted_class]}
                                </p>
                                <hr>
                                <div class="prediction-label">
                                    MODEL CONFIDENCE
                                </div>
                                <div class="confidence-value">
                                    {confidence * 100:.2f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Save results in session state
                        st.session_state["prediction_result"] = result

                    except Exception as error:

                        st.error(
                            f"Prediction failed: {error}"
                        )

            

    except (UnidentifiedImageError, OSError, ValueError):

        st.error(
            "Unable to read this image. Please upload a valid image file."
        )

else:

    st.info(
        "Upload a vehicle image to begin the damage assessment."
    )

# --------------------------------------------------
# 9. Probability visualization
# --------------------------------------------------

result = st.session_state.get(
    "prediction_result"
)

if result is not None:

    st.subheader("Class Probability Distribution")

    probability_df = pd.DataFrame({
        "Damage Category": list(
            result["class_probabilities"].keys()
        ),
        "Probability (%)": [
            value * 100
            for value in result["class_probabilities"].values()
        ]
    })

    probability_df = probability_df.sort_values(
        "Probability (%)",
        ascending=False
    )

    st.bar_chart(
        probability_df.set_index("Damage Category"),
        horizontal=True
    )

    st.caption(
        "Probabilities represent the model's "
        "estimated confidence across the six categories."
    )

# --------------------------------------------------
# 10. Damage category reference
# --------------------------------------------------

st.divider()

st.subheader("Understanding Damage Categories")

category_col1, category_col2, category_col3 = st.columns(3)

with category_col1:
    st.markdown("**Front Damage**")
    st.caption("F_Breakage · F_Crushed · F_Normal")

with category_col2:
    st.markdown("**Rear Damage**")
    st.caption("R_Breakage · R_Crushed · R_Normal")

with category_col3:
    st.markdown("**Model Architecture**")
    st.caption("ResNet50 · Transfer Learning")

# --------------------------------------------------
# 11. Disclaimer and footer
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#667085; font-size:12px;">
        <b>Car Damage Intelligence</b><br>
        Powered by ResNet50 Deep Learning Model<br><br>
        AI predictions are for informational purposes only.
        They should not replace professional vehicle inspection.
    </div>
    """,
    unsafe_allow_html=True
)
