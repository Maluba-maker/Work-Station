import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Maluz Signal Engine V2.1",
    layout="wide"
)

st.title("🔹 Maluz Signal Engine V2.1")
st.caption(
    "Vision Diagnostic • Candle Geometry • OHLC Reconstruction • NO TRADING SIGNALS"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def crop_chart(image, left, right, top, bottom):
    h, w = image.shape[:2]

    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))
    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))

    return image[top:bottom, left:right]


# ============================================================
# COLOR MASKS
# ============================================================

def create_color_masks(image):
    """
    Pocket Option style:
    Green candles = strong green
    Red candles   = strong red

    HSV is used because it separates color from brightness
    better than raw RGB.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # GREEN
    green_lower = np.array([35, 80, 50])
    green_upper = np.array([95, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # RED
    red_lower1 = np.array([0, 80, 50])
    red_upper1 = np.array([12, 255, 255])

    red_lower2 = np.array([165, 80, 50])
    red_upper2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(
        hsv,
        red_lower1,
        red_upper1
    )

    red_mask2 = cv2.inRange(
        hsv,
        red_lower2,
        red_upper2
    )

    red_mask = cv2.bitwise_or(
        red_mask1,
        red_mask2
    )

    # Remove tiny isolated noise
    kernel = np.ones((2, 2), np.uint8)

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return green_mask, red_mask


# ============================================================
# COMPONENT DETECTION
# ============================================================

def find_components(mask, color_name):

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    components = []

    for i in range(1, num_labels):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Basic noise filtering
        if area < 15:
            continue

        if w < 2:
            continue

        if h < 5:
            continue

        components.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": area,
            "color": color_name
        })

    return components


# ============================================================
# COMPONENT GROUPING
# ============================================================

def group_components(components, x_tolerance=5):

    """
    A candle may consist of several connected components:

        wick
         |
       ███
       ███
         |

    Therefore we group objects that occupy approximately
    the same X position.
    """

    if not components:
        return []

    components = sorted(
        components,
        key=lambda c: c["x"]
    )

    groups = []

    for component in components:

        center_x = component["x"] + component["width"] / 2

        placed = False

        for group in groups:

            group_center = np.mean([
                c["x"] + c["width"] / 2
                for c in group
            ])

            if abs(center_x - group_center) <= x_tolerance:

                group.append(component)
                placed = True
                break

        if not placed:
            groups.append([component])

    return groups


# ============================================================
# CANDLE RECONSTRUCTION
# ============================================================

def reconstruct_candle(group):

    if not group:
        return None

    # Determine overall geometry
    left = min(c["x"] for c in group)
    right = max(
        c["x"] + c["width"]
        for c in group
    )

    top = min(c["y"] for c in group)
    bottom = max(
        c["y"] + c["height"]
        for c in group
    )

    width = right - left
    height = bottom - top

    # Determine dominant color
    green_area = sum(
        c["area"]
        for c in group
        if c["color"] == "GREEN"
    )

    red_area = sum(
        c["area"]
        for c in group
        if c["color"] == "RED"
    )

    if green_area >= red_area:
        color = "GREEN"
    else:
        color = "RED"

    # --------------------------------------------------------
    # Find likely BODY
    # --------------------------------------------------------

    body_candidates = sorted(
        group,
        key=lambda c: c["area"],
        reverse=True
    )

    body = body_candidates[0]

    body_top = body["y"]
    body_bottom = body["y"] + body["height"]

    body_height = max(
        1,
        body_bottom - body_top
    )

    # --------------------------------------------------------
    # Wick estimation
    # --------------------------------------------------------

    upper_wick = max(
        0,
        body_top - top
    )

    lower_wick = max(
        0,
        bottom - body_bottom
    )

    # --------------------------------------------------------
    # OHLC in PIXEL coordinates
    #
    # IMPORTANT:
    # This is NOT real price yet.
    # --------------------------------------------------------

    high = top
    low = bottom

    if color == "GREEN":

        open_price = body_bottom
        close_price = body_top

    else:

        open_price = body_top
        close_price = body_bottom

    # --------------------------------------------------------
    # Geometry ratios
    # --------------------------------------------------------

    body_ratio = body_height / max(height, 1)

    aspect_ratio = height / max(width, 1)

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = 100

    # Extremely thin object
    if width < 3:
        confidence -= 25

    # Extremely short object
    if height < 10:
        confidence -= 20

    # Suspiciously wide object
    if width > 20:
        confidence -= 20

    # Moving-average-like shape
    if aspect_ratio < 1.2:
        confidence -= 20

    # Very tiny body
    if body_height <= 2:
        confidence -= 15

    confidence = max(
        0,
        min(100, confidence)
    )

    return {
        "x": left,
        "y": top,
        "width": width,
        "height": height,
        "color": color,

        "open": open_price,
        "high": high,
        "low": low,
        "close": close_price,

        "body_height": body_height,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,

        "body_ratio": round(body_ratio, 3),
        "aspect_ratio": round(aspect_ratio, 3),

        "confidence": confidence
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_candle(candle):

    if candle is None:
        return False

    if candle["width"] < 3:
        return False

    if candle["height"] < 10:
        return False

    if candle["width"] > 20:
        return False

    if candle["confidence"] < 45:
        return False

    return True


# ============================================================
# DETECT CANDLES
# ============================================================

def detect_candles(image):

    green_mask, red_mask = create_color_masks(image)

    green_components = find_components(
        green_mask,
        "GREEN"
    )

    red_components = find_components(
        red_mask,
        "RED"
    )

    all_components = (
        green_components +
        red_components
    )

    groups = group_components(
        all_components,
        x_tolerance=5
    )

    candles = []

    for group in groups:

        candle = reconstruct_candle(group)

        if validate_candle(candle):

            candles.append(candle)

    candles = sorted(
        candles,
        key=lambda c: c["x"]
    )

    return (
        candles,
        green_mask,
        red_mask,
        all_components
    )


# ============================================================
# ANNOTATION
# ============================================================

def annotate_candles(image, candles):

    annotated = image.copy()

    for index, candle in enumerate(candles, start=1):

        x = candle["x"]
        y = candle["y"]
        w = candle["width"]
        h = candle["height"]

        # Green / red annotation
        if candle["color"] == "GREEN":
            color = (0, 255, 0)
        else:
            color = (255, 60, 60)

        # Bounding box
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Number
        cv2.putText(
            annotated,
            str(index),
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    return annotated


# ============================================================
# MAIN UI
# ============================================================

st.header("1️⃣ Upload Chart")

uploaded = st.file_uploader(
    "Upload your Pocket Option chart",
    type=["png", "jpg", "jpeg"]
)

if uploaded is None:

    st.info("Upload a screenshot to begin.")

    st.stop()

image = load_image(uploaded)

h, w = image.shape[:2]

st.write(
    f"**Image size:** {w} × {h} px"
)

# ============================================================
# CROP
# ============================================================

st.header("2️⃣ Chart Region")

st.caption(
    "Manually crop out trading controls and indicators "
    "that are not part of the candle chart."
)

with st.expander("⚙️ Adjust chart crop", expanded=False):

    left = st.slider(
        "Left",
        0,
        w - 1,
        min(34, w - 1)
    )

    right = st.slider(
        "Right",
        1,
        w,
        min(1164, w)
    )

    top = st.slider(
        "Top",
        0,
        h - 1,
        min(122, h - 1)
    )

    bottom = st.slider(
        "Bottom",
        1,
        h,
        min(734, h)
    )

chart = crop_chart(
    image,
    left,
    right,
    top,
    bottom
)

st.image(
    chart,
    caption="Chart region used by detector",
    use_container_width=True
)

# ============================================================
# DETECTION
# ============================================================

st.header("3️⃣ Vision Diagnostic")

if st.button(
    "👁️ Detect & Reconstruct Candles",
    type="primary"
):

    candles, green_mask, red_mask, components = detect_candles(
        chart
    )

    # Store results
    st.session_state["candles"] = candles
    st.session_state["green_mask"] = green_mask
    st.session_state["red_mask"] = red_mask
    st.session_state["components"] = components


# ============================================================
# RESULTS
# ============================================================

if "candles" in st.session_state:

    candles = st.session_state["candles"]
    green_mask = st.session_state["green_mask"]
    red_mask = st.session_state["red_mask"]
    components = st.session_state["components"]

    st.header("4️⃣ Detection Result")

    green_count = sum(
        c["color"] == "GREEN"
        for c in candles
    )

    red_count = sum(
        c["color"] == "RED"
        for c in candles
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Confirmed Candles",
        len(candles)
    )

    col2.metric(
        "Green",
        green_count
    )

    col3.metric(
        "Red",
        red_count
    )

    col4.metric(
        "Rejected Candidates",
        max(
            0,
            len(components) - len(candles)
        )
    )

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    st.header("5️⃣ What the Computer Thinks Are Candles")

    annotated = annotate_candles(
        chart,
        candles
    )

    st.image(
        annotated,
        use_container_width=True
    )

    st.caption(
        "Each numbered box represents a candle candidate "
        "that survived the geometry filters."
    )

    # ========================================================
    # MASKS
    # ========================================================

    st.header("6️⃣ Color Segmentation")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            green_mask,
            caption="Green mask",
            use_container_width=True
        )

    with col2:

        st.image(
            red_mask,
            caption="Red mask",
            use_container_width=True
        )

    # ========================================================
    # CANDLE DATA
    # ========================================================

    st.header("7️⃣ Reconstructed Candle Data")

    if candles:

        df = pd.DataFrame(candles)

        df.insert(
            0,
            "Candle",
            range(1, len(df) + 1)
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "OHLC values are currently PIXEL coordinates, "
            "not actual market prices."
        )

    else:

        st.warning(
            "No candles survived the geometry filters."
        )

    # ========================================================
    # SPACING ANALYSIS
    # ========================================================

    st.header("8️⃣ Candle Spacing")

    if len(candles) >= 3:

        x_positions = np.array([
            c["x"] + c["width"] / 2
            for c in candles
        ])

        spacing = np.diff(x_positions)

        st.write(
            f"Median candle spacing: "
            f"**{np.median(spacing):.2f} px**"
        )

        st.write(
            f"Minimum spacing: "
            f"**{np.min(spacing):.2f} px**"
        )

        st.write(
            f"Maximum spacing: "
            f"**{np.max(spacing):.2f} px**"
        )

        spacing_df = pd.DataFrame({
            "From Candle": range(1, len(spacing) + 1),
            "To Candle": range(2, len(spacing) + 2),
            "Spacing (px)": np.round(spacing, 2)
        })

        st.dataframe(
            spacing_df,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # QUALITY SUMMARY
    # ========================================================

    st.header("9️⃣ Detection Quality")

    if candles:

        confidence = np.array([
            c["confidence"]
            for c in candles
        ])

        average_confidence = np.mean(
            confidence
        )

        high_confidence = np.sum(
            confidence >= 80
        )

        low_confidence = np.sum(
            confidence < 60
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Confidence",
            f"{average_confidence:.1f}%"
        )

        col2.metric(
            "High Confidence",
            int(high_confidence)
        )

        col3.metric(
            "Low Confidence",
            int(low_confidence)
        )

        if average_confidence >= 80:

            st.success(
                "The geometry detector is producing "
                "promising candidates."
            )

        elif average_confidence >= 65:

            st.warning(
                "Detection is usable for debugging, "
                "but not yet reliable."
            )

        else:

            st.error(
                "Detection quality is poor. "
                "Do NOT proceed to signal generation."
            )

    # ========================================================
    # IMPORTANT
    # ========================================================

    st.warning(
        """
        IMPORTANT:

        This version does NOT generate BUY or SELL signals.

        The purpose of V2.1 is to determine whether the
        screenshot can be converted into a reliable candle
        sequence.

        We will not build a trading signal layer until this
        extraction layer has been validated.
        """
    )
