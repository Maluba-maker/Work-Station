import streamlit as st
import hashlib
import cv2
import numpy as np
from PIL import Image
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Maluz Signal Engine V2",
    layout="wide"
)

st.title("🔹 Maluz Signal Engine V2")
st.caption("Vision Diagnostic • Candle Detection • NO TRADING SIGNALS YET")


# ============================================================
# PASSWORD
# ============================================================

PASSWORD = "maluz123"
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()


def check_password():

    def entered():

        entered_password = st.session_state["pw"]

        if hashlib.sha256(
            entered_password.encode()
        ).hexdigest() == PASSWORD_HASH:

            st.session_state.auth = True

        else:

            st.session_state.auth = False

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:

        st.text_input(
            "🔐 Password",
            type="password",
            key="pw",
            on_change=entered
        )

        if "auth" in st.session_state and st.session_state.auth:
            pass

        elif st.session_state.get("pw"):
            st.error("Incorrect password")

        st.stop()


check_password()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def crop_chart(image, x1, y1, x2, y2):

    h, w = image.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return image[y1:y2, x1:x2]


def create_color_masks(image):

    """
    Detect strongly colored pixels.

    Green:
        Used for bullish candles.

    Red:
        Used for bearish candles.

    We deliberately use HSV rather than RGB because
    HSV makes color segmentation more stable.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    H, S, V = cv2.split(hsv)

    # Green
    green_mask = (
        (H >= 35) &
        (H <= 90) &
        (S >= 100) &
        (V >= 70)
    ).astype(np.uint8) * 255

    # Red wraps around the HSV hue scale
    red_mask = (
        ((H <= 10) | (H >= 170)) &
        (S >= 100) &
        (V >= 70)
    ).astype(np.uint8) * 255

    return green_mask, red_mask


def find_candidates(mask, color_name):

    """
    Find narrow vertical colored objects.

    Candle bodies/wicks tend to be:
        - narrow
        - vertical
        - relatively tall

    Moving averages tend to be:
        - long
        - thin
        - horizontally connected

    Therefore we reject very wide objects.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)

        # Candle candidate filters
        if w < 3:
            continue

        if w > 15:
            continue

        if h < 15:
            continue

        if h > 125:
            continue

        if area < 15:
            continue

        if area > 1000:
            continue

        candidates.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": float(area),
            "color": color_name
        })

    return candidates


def merge_candle_fragments(candidates):

    """
    Sometimes a candle is split into several colored pieces.

    For example:

          wick
           |
        ┌──┐
        │  │
        └──┘
           |
          wick

    Other chart elements can interrupt the colored pixels.

    We therefore group candidates that occupy almost
    the same X position.
    """

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c["x"] + c["w"] / 2
    )

    groups = []

    for candidate in candidates:

        center_x = candidate["x"] + candidate["w"] / 2

        if not groups:

            groups.append([candidate])

            continue

        previous_group = groups[-1]

        previous_centers = [
            c["x"] + c["w"] / 2
            for c in previous_group
        ]

        previous_center = np.mean(previous_centers)

        # Same candle if centers are close
        if abs(center_x - previous_center) <= 4:

            previous_group.append(candidate)

        else:

            groups.append([candidate])

    merged = []

    for group in groups:

        # Keep dominant color
        colors = [c["color"] for c in group]

        if colors.count("GREEN") >= colors.count("RED"):
            color = "GREEN"
        else:
            color = "RED"

        x1 = min(c["x"] for c in group)
        y1 = min(c["y"] for c in group)

        x2 = max(
            c["x"] + c["w"]
            for c in group
        )

        y2 = max(
            c["y"] + c["h"]
            for c in group
        )

        area = sum(
            c["area"]
            for c in group
        )

        merged.append({
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "area": area,
            "color": color
        })

    return merged


def filter_spacing(candles):

    """
    Remove extremely suspicious detections.

    Real candles should normally appear at roughly
    regular horizontal intervals.
    """

    if len(candles) < 5:
        return candles

    candles = sorted(
        candles,
        key=lambda c: c["x"]
    )

    centers = np.array([
        c["x"] + c["w"] / 2
        for c in candles
    ])

    gaps = np.diff(centers)

    positive_gaps = gaps[gaps > 0]

    if len(positive_gaps) == 0:
        return candles

    median_gap = np.median(positive_gaps)

    # Very wide gaps may indicate missing candles,
    # but we don't delete them. We only reject objects
    # that are extremely close together.
    filtered = []

    for i, candle in enumerate(candles):

        if i == 0:

            filtered.append(candle)
            continue

        gap = centers[i] - centers[i - 1]

        # If objects overlap heavily, likely duplicate detection
        if gap < max(4, median_gap * 0.35):

            previous = filtered[-1]

            if candle["h"] > previous["h"]:

                filtered[-1] = candle

        else:

            filtered.append(candle)

    return filtered


def calculate_detection_quality(candles):

    """
    This is NOT a probability.

    It is a diagnostic score based on:
        - number of detections
        - candle width consistency
        - spacing consistency
        - reasonable candle geometry
    """

    if len(candles) < 5:
        return 0

    widths = np.array([
        c["w"]
        for c in candles
    ])

    centers = np.array([
        c["x"] + c["w"] / 2
        for c in candles
    ])

    gaps = np.diff(centers)

    # Width consistency
    width_median = np.median(widths)

    if width_median == 0:
        width_score = 0
    else:

        width_deviation = np.mean(
            np.abs(widths - width_median)
            / width_median
        )

        width_score = max(
            0,
            1 - width_deviation
        )

    # Spacing consistency
    if len(gaps) > 1:

        gap_median = np.median(gaps)

        if gap_median > 0:

            gap_deviation = np.mean(
                np.abs(gaps - gap_median)
                / gap_median
            )

            spacing_score = max(
                0,
                1 - gap_deviation
            )

        else:

            spacing_score = 0

    else:

        spacing_score = 0

    # Count score
    count_score = min(
        len(candles) / 60,
        1
    )

    score = (
        width_score * 0.25 +
        spacing_score * 0.35 +
        count_score * 0.40
    )

    return round(
        score * 100,
        1
    )


def annotate_candles(image, candles):

    """
    Draw detection results on the chart.
    """

    output = image.copy()

    for index, candle in enumerate(candles):

        x = candle["x"]
        y = candle["y"]
        w = candle["w"]
        h = candle["h"]

        if candle["color"] == "GREEN":

            box_color = (0, 255, 0)

        else:

            box_color = (255, 60, 60)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )

        cv2.putText(
            output,
            str(index + 1),
            (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            box_color,
            1,
            cv2.LINE_AA
        )

    return output


# ============================================================
# INPUT
# ============================================================

st.subheader("1️⃣ Upload Chart")

mode = st.radio(
    "Input Mode",
    ["Upload Screenshot", "Camera"],
    horizontal=True
)

image = None

if mode == "Upload Screenshot":

    uploaded = st.file_uploader(
        "Upload your Pocket Option chart",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:

        image = np.array(
            Image.open(uploaded).convert("RGB")
        )


else:

    camera = st.camera_input(
        "Capture chart"
    )

    if camera:

        image = np.array(
            Image.open(camera).convert("RGB")
        )


if image is None:

    st.info(
        "Upload a chart screenshot to begin."
    )

    st.stop()


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

height, width = image.shape[:2]

st.write(
    f"Image size: **{width} × {height} px**"
)


# ============================================================
# CROP CONTROLS
# ============================================================

st.subheader("2️⃣ Chart Region")

st.caption(
    "For this version, we deliberately crop the chart manually. "
    "This removes the trading controls and stochastic panel."
)

# Defaults based on the screenshot you provided
default_x1 = int(width * 0.01)
default_x2 = int(width * 0.76)

default_y1 = int(height * 0.105)
default_y2 = int(height * 0.80)


with st.expander(
    "⚙️ Adjust chart crop",
    expanded=False
):

    c1, c2 = st.columns(2)

    with c1:

        x1 = st.slider(
            "Left",
            0,
            width - 2,
            default_x1
        )

        y1 = st.slider(
            "Top",
            0,
            height - 2,
            default_y1
        )

    with c2:

        x2 = st.slider(
            "Right",
            x1 + 1,
            width,
            default_x2
        )

        y2 = st.slider(
            "Bottom",
            y1 + 1,
            height,
            default_y2
        )


crop = crop_chart(
    image,
    x1,
    y1,
    x2,
    y2
)


st.image(
    crop,
    caption="Chart region used by the vision engine",
    width="stretch"
)


# ============================================================
# ANALYSIS
# ============================================================

st.subheader("3️⃣ Vision Diagnostic")

analyse = st.button(
    "👁️ Detect Candles",
    type="primary"
)


if analyse:

    with st.spinner(
        "Analysing chart geometry..."
    ):

        # ----------------------------------------------------
        # COLOR MASKS
        # ----------------------------------------------------

        green_mask, red_mask = create_color_masks(
            crop
        )

        # ----------------------------------------------------
        # FIND CANDIDATES
        # ----------------------------------------------------

        green_candidates = find_candidates(
            green_mask,
            "GREEN"
        )

        red_candidates = find_candidates(
            red_mask,
            "RED"
        )

        all_candidates = (
            green_candidates +
            red_candidates
        )

        # ----------------------------------------------------
        # MERGE FRAGMENTS
        # ----------------------------------------------------

        candles = merge_candle_fragments(
            all_candidates
        )

        # ----------------------------------------------------
        # FILTER SPACING
        # ----------------------------------------------------

        candles = filter_spacing(
            candles
        )

        candles = sorted(
            candles,
            key=lambda c: c["x"]
        )

        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        quality = calculate_detection_quality(
            candles
        )

        # ----------------------------------------------------
        # ANNOTATION
        # ----------------------------------------------------

        annotated = annotate_candles(
            crop,
            candles
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("4️⃣ Detection Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Candles detected",
            len(candles)
        )

    with col2:

        st.metric(
            "Green candidates",
            len(green_candidates)
        )

    with col3:

        st.metric(
            "Red candidates",
            len(red_candidates)
        )

    with col4:

        st.metric(
            "Diagnostic score",
            f"{quality}%"
        )


    # ========================================================
    # WARNING
    # ========================================================

    if len(candles) < 20:

        st.warning(
            "⚠️ Too few candle candidates detected. "
            "Do NOT use this output for trading."
        )

    elif quality < 60:

        st.warning(
            "⚠️ Detection quality is currently low. "
            "The engine needs calibration."
        )

    else:

        st.success(
            "Candle candidates detected. "
            "Now inspect the annotated image carefully."
        )


    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    st.subheader(
        "5️⃣ What the Computer Thinks Are Candles"
    )

    st.image(
        annotated,
        caption=(
            "Green boxes = detected bullish candle candidates | "
            "Red boxes = detected bearish candle candidates"
        ),
        width="stretch"
    )


    # ========================================================
    # MASKS
    # ========================================================

    st.subheader(
        "6️⃣ Raw Color Detection"
    )

    mask_col1, mask_col2 = st.columns(2)

    with mask_col1:

        st.image(
            green_mask,
            caption="Green pixel mask",
            width="stretch"
        )

    with mask_col2:

        st.image(
            red_mask,
            caption="Red pixel mask",
            width="stretch"
        )


    # ========================================================
    # CANDLE DATA
    # ========================================================

    if candles:

        st.subheader(
            "7️⃣ Detected Candle Data"
        )

        table_data = []

        for i, candle in enumerate(candles):

            table_data.append({
                "Candle": i + 1,
                "X": candle["x"],
                "Y": candle["y"],
                "Width": candle["w"],
                "Height": candle["h"],
                "Area": round(
                    candle["area"],
                    1
                ),
                "Color": candle["color"]
            })

        df = pd.DataFrame(
            table_data
        )

        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )


    # ========================================================
    # IMPORTANT STATUS
    # ========================================================

    st.divider()

    st.warning(
        """
        IMPORTANT:

        This version does NOT generate BUY or SELL signals.

        The purpose of V2.0 is to verify that the computer
        correctly identifies candles from the screenshot.

        Do not trade based on the diagnostic score.
        """
    )
