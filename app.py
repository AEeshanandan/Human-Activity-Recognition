import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv("HAR_API_URL", "").rstrip("/")

ACTIVITY_CLASSES = [
    "JumpRope",
    "WalkingWithDog",
    "BenchPress",
    "Biking",
    "HighJump",
    "PullUps",
    "PushUps",
    "RopeClimbing",
    "SkateBoarding",
    "Swing",
]

st.set_page_config(
    page_title="MotionIQ | Human Activity Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #101827 55%, #101323 100%);
        color: #edf2f7;
    }

    [data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid #263246;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid #29364c;
        border-radius: 20px;
        background: linear-gradient(120deg, #17243b, #111827);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #72e4d2;
        font-size: 0.78rem;
        letter-spacing: 2px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: 2.45rem;
        margin: 8px 0;
        color: #f8fafc;
    }

    .hero p {
        color: #b6c4d8;
        font-size: 1.03rem;
        margin-bottom: 0;
    }

    .section-label {
        color: #72e4d2;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .info-card {
        background: #121c2d;
        border: 1px solid #2a374c;
        border-radius: 15px;
        padding: 18px;
        min-height: 120px;
        margin-bottom: 12px;
    }

    .info-card h3 {
        color: #f1f5f9;
        font-size: 1.05rem;
        margin-top: 0;
    }

    .info-card p {
        color: #b5c2d5;
        font-size: 0.92rem;
        margin-bottom: 0;
    }

    .class-chip {
        display: inline-block;
        background: #172b3e;
        border: 1px solid #2a5261;
        color: #a8f0e5;
        border-radius: 20px;
        padding: 6px 11px;
        margin: 4px 4px 4px 0;
        font-size: 0.82rem;
    }

    .result-box {
        background: #102c2a;
        border: 1px solid #28766d;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0 18px 0;
    }

    div[data-testid="stMetric"] {
        background: #121c2d;
        border: 1px solid #2a374c;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #b5c2d5;
    }

    .stButton > button[kind="primary"] {
        background: #55d6c2;
        color: #07131c;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    .stButton > button[kind="primary"]:hover {
        background: #85e9d9;
        color: #07131c;
    }

    hr {
        border-color: #29364c;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR — NO API TEST BUTTON
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎯 MotionIQ")
    st.caption("Human Activity Recognition")
    st.divider()

    st.markdown("### Model configuration")
    st.markdown("**Architecture**")
    st.write("CNN + LSTM")

    st.markdown("**Input sequence**")
    st.code("20 frames × 64 × 64 × 3")

    st.markdown("**Output classes**")
    st.write("10 activities")

    st.markdown("**Input normalization**")
    st.code("Pixel values ÷ 255.0")

    st.divider()
    st.caption(
        "Inference runs through your Flask API hosted by the active "
        "Kaggle session and ngrok tunnel."
    )

# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Computer Vision · Deep Learning</div>
        <h1>Human Activity Recognition</h1>
        <p>
            Analyze a video and identify a human activity using a
            Convolutional Neural Network and Long Short-Term Memory model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

m1.metric("Model", "CNN–LSTM")
m2.metric("Supported classes", "10")
m3.metric("Frames per video", "20")
m4.metric("Frame resolution", "64 × 64")

st.markdown("")

# --------------------------------------------------
# VIDEO PREDICTION
# --------------------------------------------------

st.markdown('<div class="section-label">01 / Live inference</div>',
            unsafe_allow_html=True)
st.header("Analyze a video")
st.write(
    "Upload a clip containing one of the supported activities. "
    "The model samples video frames and predicts the most likely class."
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Use a short, clear video for the first test.",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        recognize = st.button(
            "▶  Recognize activity",
            type="primary",
            use_container_width=True,
        )
    else:
        recognize = False
        st.info("Supported formats: MP4, AVI, MOV and MKV.")

with right:
    st.markdown(
        """
        <div class="info-card">
            <h3>How it works</h3>
            <p>
                <b>1. Upload:</b> Select a video clip.<br><br>
                <b>2. Sample:</b> Extract and resize 20 frames.<br><br>
                <b>3. Normalize:</b> Scale pixel values to 0–1.<br><br>
                <b>4. Predict:</b> CNN–LSTM estimates the activity class.
            </p>
        </div>
        <div class="info-card">
            <h3>Before you begin</h3>
            <p>
                Keep the Kaggle notebook and ngrok tunnel running.
                Use videos with a visible subject and minimal obstruction.
                The model recognizes only its 10 configured classes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Persist results across Streamlit reruns.
if "har_result" not in st.session_state:
    st.session_state.har_result = None

if recognize and uploaded_video is not None:
    if not API_URL:
        st.error(
            "The API address is not configured. Set HAR_API_URL in "
            "PowerShell, then restart Streamlit."
        )
    else:
        try:
            with st.spinner("Analyzing video with the CNN–LSTM model..."):
                files = {
                    "video": (
                        uploaded_video.name,
                        uploaded_video.getvalue(),
                        uploaded_video.type or "application/octet-stream",
                    )
                }

                start_time = time.time()

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=180,
                )

                elapsed = time.time() - start_time
                response.raise_for_status()
                result = response.json()

                result["_client_elapsed_seconds"] = elapsed
                st.session_state.har_result = result

        except requests.Timeout:
            st.error(
                "The request timed out. Try a shorter video and confirm "
                "that the Kaggle session is still active."
            )
        except requests.RequestException as exc:
            st.error(f"Could not obtain a prediction: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text[:2000])
        except ValueError:
            st.error("The API returned a response that was not valid JSON.")

# --------------------------------------------------
# PREDICTION RESULTS
# --------------------------------------------------

result = st.session_state.har_result

if result:
    st.divider()
    st.markdown('<div class="section-label">02 / Prediction report</div>',
                unsafe_allow_html=True)
    st.header("Recognition results")

    predicted_activity = result.get("predicted_activity", "Unknown")

    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    # Handle confidence returned as either a fraction or percentage.
    confidence_fraction = confidence / 100 if confidence > 1 else confidence
    confidence_fraction = min(max(confidence_fraction, 0), 1)

    st.markdown(
        f"""
        <div class="result-box">
            <div class="eyebrow">Predicted activity</div>
            <h1 style="margin: 6px 0; color: #9af0dc;">
                {predicted_activity}
            </h1>
            <p style="color: #c5d8d6; margin-bottom: 0;">
                The model's highest-scoring activity class for this clip.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)

    r1.metric("Top-class confidence", f"{confidence_fraction:.1%}")

    try:
        inference_time = float(result.get("inference_time_seconds", 0))
    except (TypeError, ValueError):
        inference_time = 0.0

    r2.metric(
        "Model inference time",
        f"{inference_time:.2f} s" if inference_time else "Not provided",
    )

    client_time = result.get("_client_elapsed_seconds")
    r3.metric(
        "Total API response time",
        f"{client_time:.2f} s" if client_time is not None else "Not provided",
    )

    st.caption(
        "Confidence is the model's output score, not a guarantee that "
        "the prediction is correct."
    )

    # Normalize the API's top-predictions response to activity/confidence.
    raw_predictions = result.get("predictions", [])
    chart_rows = []

    if isinstance(raw_predictions, list):
        for item in raw_predictions:
            if not isinstance(item, dict):
                continue

            label = (
                item.get("activity")
                or item.get("class")
                or item.get("predicted_activity")
                or item.get("label")
                or item.get("name")
            )

            score = (
                item.get("confidence")
                if item.get("confidence") is not None
                else item.get("probability")
            )

            if label is None or score is None:
                continue

            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            if score > 1:
                score /= 100

            chart_rows.append({
                "Activity": str(label),
                "Confidence": max(0, min(score, 1)),
            })

    # If the API only returns a single prediction, show that result.
    if not chart_rows:
        chart_rows = [{
            "Activity": str(predicted_activity),
            "Confidence": confidence_fraction,
        }]

    chart_df = pd.DataFrame(chart_rows)
    chart_df = chart_df.sort_values("Confidence", ascending=True)

    st.subheader("Prediction confidence by activity")

    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(chart_df))))
    fig.patch.set_facecolor("#121c2d")
    ax.set_facecolor("#121c2d")

    bars = ax.barh(
        chart_df["Activity"],
        chart_df["Confidence"] * 100,
    )

    ax.set_xlabel("Model score (%)", color="#dbe7f3")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", colors="#c0ccdc")
    ax.tick_params(axis="y", colors="#e7edf6", labelsize=9)
    ax.spines["bottom"].set_color("#40506a")
    ax.spines["left"].set_color("#40506a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, chart_df["Confidence"] * 100):
        ax.text(
            min(value + 1.2, 91),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            color="#f1f5f9",
            fontsize=9,
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("View raw API response"):
        st.json({
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        })

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()
st.markdown('<div class="section-label">03 / Under the hood</div>',
            unsafe_allow_html=True)
st.header("Model architecture")

st.write(
    "This project uses a CNN–LSTM architecture. A CNN extracts spatial "
    "features from individual video frames, while an LSTM models temporal "
    "patterns across the frame sequence."
)

# Architecture illustration. This is a conceptual diagram, not a
# layer-by-layer export of the trained Keras model.
fig, ax = plt.subplots(figsize=(12, 3.2))
fig.patch.set_facecolor("#121c2d")
ax.set_facecolor("#121c2d")
ax.axis("off")

blocks = [
    (0.02, "Input video", "20 frames"),
    (0.22, "Preprocessing", "64 × 64 · RGB"),
    (0.42, "CNN", "Spatial features"),
    (0.62, "LSTM", "Temporal patterns"),
    (0.82, "Softmax", "10 classes"),
]

for x, title, subtitle in blocks:
    ax.text(
        x + 0.075,
        0.58,
        f"{title}\n\n{subtitle}",
        ha="center",
        va="center",
        color="#f1f5f9",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#1b3044",
            "edgecolor": "#55d6c2",
            "linewidth": 1.3,
        },
        transform=ax.transAxes,
    )

for i in range(len(blocks) - 1):
    start_x = blocks[i][0] + 0.14
    end_x = blocks[i + 1][0] - 0.01

    ax.annotate(
        "",
        xy=(end_x, 0.58),
        xytext=(start_x, 0.58),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "color": "#8ca4bd", "lw": 1.5},
    )

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.caption(
    "Conceptual architecture diagram. It summarizes the processing "
    "pipeline and is not an exact visualization of every layer in the model."
)

a, b = st.columns(2)

with a:
    st.markdown(
        """
        <div class="info-card">
            <h3>Convolutional Neural Network (CNN)</h3>
            <p>
                Learns visual patterns such as edges, shapes, body parts,
                and other spatial features from individual frames.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        """
        <div class="info-card">
            <h3>Long Short-Term Memory (LSTM)</h3>
            <p>
                Processes a sequence of extracted features to learn how
                movement changes over time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Input processing")

p1, p2, p3 = st.columns(3)

p1.markdown(
    """
    <div class="info-card">
        <h3>Frame sampling</h3>
        <p>Samples up to 20 frames from the input video and pads short sequences
        by repeating the last readable frame.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

p2.markdown(
    """
    <div class="info-card">
        <h3>Image resizing</h3>
        <p>Each selected frame is resized to 64 × 64 pixels with three
        color channels.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

p3.markdown(
    """
    <div class="info-card">
        <h3>Normalization</h3>
        <p>Pixel intensities are divided by 255 so the model receives
        values in the 0–1 range.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# DATASET AND SUPPORTED CLASSES
# --------------------------------------------------

st.divider()
st.markdown('<div class="section-label">04 / Data</div>',
            unsafe_allow_html=True)
st.header("Dataset and activity classes")

st.markdown(
    """
    This implementation uses a UCF50-based dataset and a selected subset
    of ten activity labels. The model's final classification layer has
    ten outputs, so it is not configured to recognize every possible
    human activity.
    """
)

st.markdown("#### Supported activities")

chips_html = "".join(
    f'<span class="class-chip">{name}</span>'
    for name in ACTIVITY_CLASSES
)

st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("")
d1, d2 = st.columns(2)

with d1:
    st.markdown(
        """
        <div class="info-card">
            <h3>Dataset: UCF50</h3>
            <p>
                UCF50 is a human action-recognition video dataset.
                This project uses selected activity classes from the
                dataset rather than the full activity vocabulary.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with d2:
    st.markdown(
        """
        <div class="info-card">
            <h3>Class selection</h3>
            <p>
                The configured model predicts one of ten labels.
                Performance depends on the training examples, class balance,
                data splits, and similarity between training and test videos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info(
    "The charts on this page do not claim a training-set class distribution. "
    "To show actual video counts per class, load the dataset and calculate "
    "counts from your notebook's training and validation splits."
)

# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------

st.divider()
st.markdown('<div class="section-label">05 / Responsible interpretation</div>',
            unsafe_allow_html=True)
st.header("Limitations")

limitations = [
    (
        "Limited activity vocabulary",
        "The model only predicts its ten configured classes. An unseen "
        "activity may still be assigned one of these labels."
    ),
    (
        "Dependence on video quality",
        "Poor lighting, occlusion, motion blur, unusual camera angles, "
        "or a small subject can reduce prediction reliability."
    ),
    (
        "Temporal information is sampled",
        "Using 20 frames may miss brief movements or important events "
        "between sampled frames."
    ),
    (
        "Generalization",
        "A model trained on a particular dataset may perform differently "
        "on real-world videos, different backgrounds, or different people."
    ),
    (
        "Confidence is not accuracy",
        "A high output score does not prove that the prediction is correct. "
        "Accuracy must be measured on labeled, held-out test data."
    ),
    (
        "Evaluation metrics not yet displayed",
        "Training accuracy, validation accuracy, confusion matrix, and loss "
        "curves should be added only after retrieving the actual values "
        "from the model training history and test evaluation."
    ),
]

for title, description in limitations:
    with st.expander(title):
        st.write(description)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8797ad; padding:10px;">
        <b>MotionIQ</b> · Human Activity Recognition<br>
        <span style="font-size:0.85rem;">
            CNN–LSTM · Computer Vision · Video Classification
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)