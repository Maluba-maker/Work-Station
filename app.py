import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Maluz Signal Engine V2.2",
    layout="wide"
)

st.title("🔹 Maluz Signal Engine V2.2")
st.caption(
    "Vision Diagnostic • Candle Reconstruction • Sequence Validation • NO TRADING SIGNALS"
)

# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def crop_chart(image, left, right, top, bottom):
    h, w = image.shape[:2]

    left = max(0, min(int(left), w - 1))
    right = max(left + 1, min(int(right), w))
    top = max(0, min(int(top), h - 1))
    bottom = max(top + 1, min(int(bottom), h))

    return image[top:bottom, left:right]


# ============================================================
# COLOR SEGMENTATION
# ============================================================

def create_color_masks(image):
    """
    Detect red and green chart objects.

    IMPORTANT:
    These masks identify COLOUR candidates.
    They do not automatically mean that every candidate
    is a candle.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # -----------------------------
    # GREEN
    # -----------------------------

    green_lower = np.array([35, 70, 40])
    green_upper = np.array([95, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # -----------------------------
    # RED
    # -----------------------------

    red_lower1 = np.array([0, 70, 40])
    red_upper1 = np.array([15, 255, 255])

    red_lower2 = np.array([165, 70, 40])
    red_upper2 = np.array([180, 255, 255])

    red_mask_1 = cv2.inRange(
        hsv,
        red_lower1,
        red_upper1
    )

    red_mask_2 = cv2.inRange(
        hsv,
        red_lower2,
        red_upper2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    # -----------------------------
    # Remove tiny noise
    # -----------------------------

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

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    for i in range(1, num_labels):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < 10:
            continue

        if w < 2:
            continue

        if h < 4:
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
# INITIAL X GROUPING
# ============================================================

def group_components(components, x_tolerance=5):

    if not components:
        return []

    components = sorted(
        components,
        key=lambda c: c["x"]
    )

    groups = []

    for component in components:

        center_x = (
            component["x"] +
            component["width"] / 2
        )

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

            groups.append([
                component
            ])

    return groups


# ============================================================
# CANDIDATE GEOMETRY
# ============================================================

def calculate_group_geometry(group):

    if not group:
        return None

    left = min(
        c["x"]
        for c in group
    )

    right = max(
        c["x"] + c["width"]
        for c in group
    )

    top = min(
        c["y"]
        for c in group
    )

    bottom = max(
        c["y"] + c["height"]
        for c in group
    )

    width = right - left
    height = bottom - top

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

    total_area = green_area + red_area

    color_confidence = (
        max(green_area, red_area) /
        max(total_area, 1)
    )

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "width": width,
        "height": height,
        "color": color,
        "green_area": green_area,
        "red_area": red_area,
        "color_confidence": color_confidence
    }


# ============================================================
# BODY DETECTION
# ============================================================

def detect_body_from_mask(
    mask,
    x1,
    x2,
    y1,
    y2
):

    h, w = mask.shape

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    roi = mask[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Count coloured pixels on every row.
    row_counts = np.sum(
        roi > 0,
        axis=1
    )

    if len(row_counts) == 0:
        return None

    max_count = np.max(row_counts)

    if max_count <= 0:
        return None

    # A candle body generally occupies a wider
    # horizontal section than its wick.
    threshold = max(
        2,
        max_count * 0.35
    )

    body_rows = np.where(
        row_counts >= threshold
    )[0]

    if len(body_rows) == 0:
        return None

    body_top = int(body_rows[0] + y1)
    body_bottom = int(body_rows[-1] + y1)

    body_height = (
        body_bottom -
        body_top +
        1
    )

    # Estimate body width from the relevant rows.
    relevant = roi[
        body_rows[0]:
        body_rows[-1] + 1
    ]

    ys, xs = np.where(
        relevant > 0
    )

    if len(xs) == 0:
        body_width = 0
    else:
        body_width = (
            int(xs.max()) -
            int(xs.min()) +
            1
        )

    return {
        "top": body_top,
        "bottom": body_bottom,
        "height": body_height,
        "width": body_width
    }


# ============================================================
# CANDLE RECONSTRUCTION
# ============================================================

def reconstruct_candle(
    group,
    green_mask,
    red_mask,
    expected_spacing=None
):

    geometry = calculate_group_geometry(
        group
    )

    if geometry is None:
        return None

    left = geometry["left"]
    right = geometry["right"]
    top = geometry["top"]
    bottom = geometry["bottom"]

    width = geometry["width"]
    height = geometry["height"]

    if width <= 0 or height <= 0:
        return None

    # Select the dominant colour mask.
    if geometry["color"] == "GREEN":
        mask = green_mask
    else:
        mask = red_mask

    # Expand slightly around candidate.
    x_pad = 2
    y_pad = 1

    x1 = max(0, left - x_pad)
    x2 = min(
        mask.shape[1],
        right + x_pad
    )

    y1 = max(0, top - y_pad)
    y2 = min(
        mask.shape[0],
        bottom + y_pad
    )

    body = detect_body_from_mask(
        mask,
        x1,
        x2,
        y1,
        y2
    )

    # --------------------------------------------------------
    # If body cannot be identified
    # --------------------------------------------------------

    if body is None:

        return {
            "x": left,
            "y": top,
            "width": width,
            "height": height,
            "color": geometry["color"],
            "open": np.nan,
            "high": top,
            "low": bottom,
            "close": np.nan,
            "body_top": np.nan,
            "body_bottom": np.nan,
            "body_height": 0,
            "upper_wick": 0,
            "lower_wick": 0,
            "body_ratio": 0,
            "wick_ratio": 0,
            "color_confidence": geometry[
                "color_confidence"
            ],
            "spacing_score": 0,
            "geometry_score": 0,
            "confidence": 0
        }

    body_top = body["top"]
    body_bottom = body["bottom"]
    body_height = body["height"]

    upper_wick = max(
        0,
        body_top - top
    )

    lower_wick = max(
        0,
        bottom - body_bottom
    )

    # --------------------------------------------------------
    # Pixel-space OHLC
    #
    # Y increases downward.
    # --------------------------------------------------------

    if geometry["color"] == "GREEN":

        open_pixel = body_bottom
        close_pixel = body_top

    else:

        open_pixel = body_top
        close_pixel = body_bottom

    high_pixel = top
    low_pixel = bottom

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    body_ratio = (
        body_height /
        max(height, 1)
    )

    wick_ratio = (
        (upper_wick + lower_wick) /
        max(height, 1)
    )

    aspect_ratio = (
        height /
        max(width, 1)
    )

    # --------------------------------------------------------
    # Geometry score
    # --------------------------------------------------------

    geometry_score = 100.0

    if width < 3:
        geometry_score -= 30

    if width > 25:
        geometry_score -= 30

    if height < 8:
        geometry_score -= 20

    if body_height < 2:
        geometry_score -= 20

    if body_height > height:
        geometry_score -= 20

    if aspect_ratio < 1.0:
        geometry_score -= 20

    geometry_score = max(
        0,
        min(100, geometry_score)
    )

    # --------------------------------------------------------
    # Spacing score
    # --------------------------------------------------------

    spacing_score = 100.0

    if expected_spacing is not None:

        center_x = (
            left +
            width / 2
        )

        # expected_spacing is only used as
        # contextual information here.
        #
        # Individual spacing is evaluated
        # later in sequence validation.

        if expected_spacing <= 0:
            spacing_score = 50

    else:

        spacing_score = 70

    # --------------------------------------------------------
    # Overall confidence
    # --------------------------------------------------------

    confidence = (
        geometry["color_confidence"] * 25
        +
        geometry_score / 100 * 35
        +
        min(body_ratio * 2, 1) * 20
        +
        min(wick_ratio * 2, 1) * 10
        +
        spacing_score / 100 * 10
    )

    confidence = max(
        0,
        min(100, confidence)
    )

    return {
        "x": left,
        "y": top,
        "width": width,
        "height": height,

        "color": geometry["color"],

        "open": open_pixel,
        "high": high_pixel,
        "low": low_pixel,
        "close": close_pixel,

        "body_top": body_top,
        "body_bottom": body_bottom,

        "body_height": body_height,

        "upper_wick": upper_wick,
        "lower_wick": lower_wick,

        "body_ratio": round(
            body_ratio,
            3
        ),

        "wick_ratio": round(
            wick_ratio,
            3
        ),

        "color_confidence": round(
            geometry["color_confidence"] * 100,
            1
        ),

        "spacing_score": round(
            spacing_score,
            1
        ),

        "geometry_score": round(
            geometry_score,
            1
        ),

        "confidence": round(
            confidence,
            1
        )
    }


# ============================================================
# ADAPTIVE SPACING
# ============================================================

def estimate_spacing(candles):

    if len(candles) < 3:
        return None

    centers = np.array([
        c["x"] + c["width"] / 2
        for c in candles
    ])

    spacing = np.diff(
        centers
    )

    spacing = spacing[
        spacing > 0
    ]

    if len(spacing) == 0:
        return None

    # Median is more resistant to
    # missing candles and outliers.
    return float(
        np.median(spacing)
    )


# ============================================================
# SEQUENCE VALIDATION
# ============================================================

def validate_sequence(candles):

    if len(candles) < 3:

        return candles, {
            "median_spacing": None,
            "missing_gaps": [],
            "duplicate_gaps": [],
            "spacing_consistency": 0
        }

    candles = sorted(
        candles,
        key=lambda c: c["x"]
    )

    centers = np.array([
        c["x"] + c["width"] / 2
        for c in candles
    ])

    spacing = np.diff(
        centers
    )

    valid_spacing = spacing[
        spacing > 0
    ]

    if len(valid_spacing) == 0:

        return candles, {
            "median_spacing": None,
            "missing_gaps": [],
            "duplicate_gaps": [],
            "spacing_consistency": 0
        }

    median_spacing = float(
        np.median(valid_spacing)
    )

    missing_gaps = []
    duplicate_gaps = []

    tolerance = median_spacing * 0.45

    for i, gap in enumerate(
        valid_spacing
    ):

        ratio = (
            gap /
            max(median_spacing, 0.001)
        )

        # Approximately 2x, 3x etc.
        # may indicate missing candles.
        nearest_multiple = round(
            ratio
        )

        if (
            nearest_multiple >= 2
            and
            abs(
                ratio -
                nearest_multiple
            ) < 0.25
        ):

            missing_gaps.append({
                "after_candle": i + 1,
                "gap": round(
                    float(gap),
                    2
                ),
                "ratio": round(
                    float(ratio),
                    2
                ),
                "possible_missing": (
                    nearest_multiple - 1
                )
            })

        elif gap < (
            median_spacing -
            tolerance
        ):

            duplicate_gaps.append({
                "after_candle": i + 1,
                "gap": round(
                    float(gap),
                    2
                )
            })

    # --------------------------------------------------------
    # Spacing consistency
    # --------------------------------------------------------

    deviation = np.abs(
        valid_spacing -
        median_spacing
    )

    consistency = 100 * (
        1 -
        np.mean(
            deviation /
            max(median_spacing, 0.001)
        )
    )

    consistency = max(
        0,
        min(100, consistency)
    )

    # --------------------------------------------------------
    # Apply spacing score to candles
    # --------------------------------------------------------

    for i, candle in enumerate(
        candles
    ):

        if i == 0:

            candle["spacing_score"] = round(
                consistency,
                1
            )

        else:

            gap = (
                centers[i] -
                centers[i - 1]
            )

            ratio = (
                gap /
                max(median_spacing, 0.001)
            )

            if (
                abs(ratio - 1)
                <= 0.25
            ):

                score = 100

            elif (
                abs(ratio - 1)
                <= 0.5
            ):

                score = 75

            elif ratio >= 1.75:

                score = 40

            else:

                score = 55

            candle["spacing_score"] = score

    return candles, {
        "median_spacing": median_spacing,
        "missing_gaps": missing_gaps,
        "duplicate_gaps": duplicate_gaps,
        "spacing_consistency": round(
            consistency,
            1
        )
    }


# ============================================================
# CANDLE VALIDATION
# ============================================================

def validate_candle(candle):

    if candle is None:
        return False, "No candle data"

    if candle["width"] < 3:
        return False, "Too narrow"

    if candle["width"] > 25:
        return False, "Too wide"

    if candle["height"] < 8:
        return False, "Too short"

    if candle["body_height"] < 2:
        return False, "No clear body"

    if candle["confidence"] < 50:
        return False, "Low confidence"

    if (
        candle["upper_wick"] < 0
        or
        candle["lower_wick"] < 0
    ):
        return False, "Invalid wick geometry"

    return True, "Accepted"


# ============================================================
# COMPLETE CANDLE DETECTION
# ============================================================

def detect_candles(image):

    green_mask, red_mask = (
        create_color_masks(image)
    )

    green_components = (
        find_components(
            green_mask,
            "GREEN"
        )
    )

    red_components = (
        find_components(
            red_mask,
            "RED"
        )
    )

    all_components = (
        green_components +
        red_components
    )

    # --------------------------------------------------------
    # First candidate grouping
    # --------------------------------------------------------

    groups = group_components(
        all_components,
        x_tolerance=5
    )

    raw_candidates = []

    for group in groups:

        candle = reconstruct_candle(
            group,
            green_mask,
            red_mask
        )

        if candle is not None:

            raw_candidates.append(
                candle
            )

    # --------------------------------------------------------
    # Estimate spacing
    # --------------------------------------------------------

    estimated_spacing = (
        estimate_spacing(
            raw_candidates
        )
    )

    # --------------------------------------------------------
    # Reconstruct again with spacing context
    # --------------------------------------------------------

    candles = []

    for group in groups:

        candle = reconstruct_candle(
            group,
            green_mask,
            red_mask,
            estimated_spacing
        )

        if candle is None:
            continue

        valid, reason = (
            validate_candle(
                candle
            )
        )

        candle["validation"] = reason

        if valid:

            candles.append(
                candle
            )

    # --------------------------------------------------------
    # Sequence validation
    # --------------------------------------------------------

    candles, sequence = (
        validate_sequence(
            candles
        )
    )

    return (
        candles,
        green_mask,
        red_mask,
        all_components,
        raw_candidates,
        sequence
    )


# ============================================================
# ANNOTATION
# ============================================================

def annotate_candles(
    image,
    candles
):

    annotated = image.copy()

    for index, candle in enumerate(
        candles,
        start=1
    ):

        x = int(candle["x"])
        y = int(candle["y"])
        w = int(candle["width"])
        h = int(candle["height"])

        # ----------------------------------------------------
        # Candle colour
        # ----------------------------------------------------

        if candle["color"] == "GREEN":

            colour = (
                0,
                255,
                0
            )

        else:

            colour = (
                255,
                70,
                70
            )

        # ----------------------------------------------------
        # Body
        # ----------------------------------------------------

        body_top = int(
            candle["body_top"]
        )

        body_bottom = int(
            candle["body_bottom"]
        )

        body_left = x
        body_right = x + w

        cv2.rectangle(
            annotated,
            (
                body_left,
                body_top
            ),
            (
                body_right,
                body_bottom
            ),
            colour,
            2
        )

        # ----------------------------------------------------
        # Upper wick
        # ----------------------------------------------------

        center_x = int(
            x + w / 2
        )

        cv2.line(
            annotated,
            (
                center_x,
                y
            ),
            (
                center_x,
                body_top
            ),
            colour,
            2
        )

        # ----------------------------------------------------
        # Lower wick
        # ----------------------------------------------------

        cv2.line(
            annotated,
            (
                center_x,
                body_bottom
            ),
            (
                center_x,
                y + h
            ),
            colour,
            2
        )

        # ----------------------------------------------------
        # Candle number
        # ----------------------------------------------------

        label = (
            f"{index}"
        )

        cv2.putText(
            annotated,
            label,
            (
                x,
                max(
                    15,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence_text = (
            f"{candle['confidence']:.0f}%"
        )

        cv2.putText(
            annotated,
            confidence_text,
            (
                x,
                y + h + 14
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colour,
            1,
            cv2.LINE_AA
        )

    return annotated


# ============================================================
# GAP ANNOTATION
# ============================================================

def annotate_missing_gaps(
    image,
    candles,
    sequence
):

    annotated = image.copy()

    missing = sequence[
        "missing_gaps"
    ]

    if not missing:
        return annotated

    centers = np.array([
        c["x"] + c["width"] / 2
        for c in candles
    ])

    for item in missing:

        i = item[
            "after_candle"
        ]

        if i >= len(centers):
            continue

        x1 = int(
            centers[i - 1]
        )

        x2 = int(
            centers[i]
        )

        mid_x = int(
            (x1 + x2) / 2
        )

        cv2.line(
            annotated,
            (
                mid_x,
                0
            ),
            (
                mid_x,
                annotated.shape[0]
            ),
            (255, 255, 0),
            1
        )

        cv2.putText(
            annotated,
            "POSSIBLE MISSING",
            (
                max(0, mid_x - 45),
                20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    return annotated


# ============================================================
# DATAFRAME
# ============================================================

def candles_to_dataframe(
    candles
):

    if not candles:

        return pd.DataFrame()

    df = pd.DataFrame(
        candles
    )

    df.insert(
        0,
        "Candle",
        range(
            1,
            len(df) + 1
        )
    )

    return df


# ============================================================
# APPLICATION
# ============================================================

st.header("1️⃣ Upload Chart")

uploaded = st.file_uploader(
    "Upload your Pocket Option chart",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)

if uploaded is None:

    st.info(
        "Upload a screenshot to begin."
    )

    st.stop()

image = load_image(
    uploaded
)

h, w = image.shape[:2]

st.write(
    f"**Image size:** {w} × {h} px"
)

# ============================================================
# CHART CROP
# ============================================================

st.header(
    "2️⃣ Chart Region"
)

st.caption(
    "Crop out controls, menus and unrelated interface elements."
)

with st.expander(
    "⚙️ Adjust chart crop",
    expanded=False
):

    left = st.slider(
        "Left",
        0,
        max(1, w - 1),
        min(34, max(0, w - 1))
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
        max(1, h - 1),
        min(122, max(0, h - 1))
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
    width="stretch"
)

# ============================================================
# DETECTION
# ============================================================

st.header(
    "3️⃣ Vision Diagnostic"
)

if st.button(
    "👁️ Detect & Reconstruct Candles",
    type="primary"
):

    (
        candles,
        green_mask,
        red_mask,
        components,
        raw_candidates,
        sequence
    ) = detect_candles(
        chart
    )

    st.session_state[
        "candles"
    ] = candles

    st.session_state[
        "green_mask"
    ] = green_mask

    st.session_state[
        "red_mask"
    ] = red_mask

    st.session_state[
        "components"
    ] = components

    st.session_state[
        "raw_candidates"
    ] = raw_candidates

    st.session_state[
        "sequence"
    ] = sequence


# ============================================================
# RESULTS
# ============================================================

if "candles" in st.session_state:

    candles = st.session_state[
        "candles"
    ]

    green_mask = st.session_state[
        "green_mask"
    ]

    red_mask = st.session_state[
        "red_mask"
    ]

    components = st.session_state[
        "components"
    ]

    raw_candidates = st.session_state[
        "raw_candidates"
    ]

    sequence = st.session_state[
        "sequence"
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    st.header(
        "4️⃣ Detection Summary"
    )

    green_count = sum(
        c["color"] == "GREEN"
        for c in candles
    )

    red_count = sum(
        c["color"] == "RED"
        for c in candles
    )

    rejected_count = max(
        0,
        len(raw_candidates) -
        len(candles)
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

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
        "Rejected",
        rejected_count
    )

    # ========================================================
    # SPACING
    # ========================================================

    st.header(
        "5️⃣ Candle Sequence Integrity"
    )

    median_spacing = sequence[
        "median_spacing"
    ]

    consistency = sequence[
        "spacing_consistency"
    ]

    if median_spacing is not None:

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Expected Spacing",
            f"{median_spacing:.2f} px"
        )

        col2.metric(
            "Spacing Consistency",
            f"{consistency:.1f}%"
        )

        col3.metric(
            "Possible Missing Gaps",
            len(
                sequence[
                    "missing_gaps"
                ]
            )
        )

        if sequence[
            "missing_gaps"
        ]:

            st.warning(
                "Potential missing candles were detected."
            )

            st.dataframe(
                pd.DataFrame(
                    sequence[
                        "missing_gaps"
                    ]
                ),
                width="stretch",
                hide_index=True
            )

        else:

            st.success(
                "No obvious multi-candle gaps detected."
            )

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    st.header(
        "6️⃣ Candle Geometry"
    )

    annotated = annotate_candles(
        chart,
        candles
    )

    annotated = annotate_missing_gaps(
        annotated,
        candles,
        sequence
    )

    st.image(
        annotated,
        caption=(
            "Green/red boxes = candle bodies. "
            "Lines = estimated wicks. "
            "Yellow = possible missing candle."
        ),
        width="stretch"
    )

    # ========================================================
    # COLOR MASKS
    # ========================================================

    st.header(
        "7️⃣ Colour Segmentation"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.image(
            green_mask,
            caption="Green mask",
            width="stretch"
        )

    with col2:

        st.image(
            red_mask,
            caption="Red mask",
            width="stretch"
        )

    # ========================================================
    # RECONSTRUCTED DATA
    # ========================================================

    st.header(
        "8️⃣ Reconstructed Candle Data"
    )

    if candles:

        df = candles_to_dataframe(
            candles
        )

        display_columns = [
            "Candle",
            "color",
            "open",
            "high",
            "low",
            "close",
            "body_height",
            "upper_wick",
            "lower_wick",
            "body_ratio",
            "wick_ratio",
            "color_confidence",
            "geometry_score",
            "spacing_score",
            "confidence",
            "validation"
        ]

        available_columns = [
            col
            for col in display_columns
            if col in df.columns
        ]

        display_df = df[
            available_columns
        ].copy()

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "OHLC values are still PIXEL coordinates. "
            "They are not real market prices yet."
        )

    else:

        st.error(
            "No valid candles survived the validation layer."
        )

    # ========================================================
    # QUALITY
    # ========================================================

    st.header(
        "9️⃣ Detection Quality"
    )

    if candles:

        confidence = np.array([
            c["confidence"]
            for c in candles
        ])

        geometry_scores = np.array([
            c["geometry_score"]
            for c in candles
        ])

        color_scores = np.array([
            c["color_confidence"]
            for c in candles
        ])

        avg_confidence = float(
            np.mean(confidence)
        )

        avg_geometry = float(
            np.mean(
                geometry_scores
            )
        )

        avg_color = float(
            np.mean(
                color_scores
            )
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Average Confidence",
            f"{avg_confidence:.1f}%"
        )

        col2.metric(
            "Geometry Quality",
            f"{avg_geometry:.1f}%"
        )

        col3.metric(
            "Colour Confidence",
            f"{avg_color:.1f}%"
        )

        if avg_confidence >= 85:

            st.success(
                "Detection quality looks promising."
            )

        elif avg_confidence >= 70:

            st.warning(
                "Detection is usable for development, "
                "but still needs validation."
            )

        else:

            st.error(
                "Detection quality is poor. "
                "Do not build signals on this result."
            )

    # ========================================================
    # CANDLE GEOMETRY STATISTICS
    # ========================================================

    st.header(
        "🔟 Candle Geometry Statistics"
    )

    if candles:

        body_sizes = np.array([
            c["body_height"]
            for c in candles
        ])

        upper_wicks = np.array([
            c["upper_wick"]
            for c in candles
        ])

        lower_wicks = np.array([
            c["lower_wick"]
            for c in candles
        ])

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Median Body",
            f"{np.median(body_sizes):.1f} px"
        )

        col2.metric(
            "Median Upper Wick",
            f"{np.median(upper_wicks):.1f} px"
        )

        col3.metric(
            "Median Lower Wick",
            f"{np.median(lower_wicks):.1f} px"
        )

    # ========================================================
    # REJECTED CANDIDATE DIAGNOSTICS
    # ========================================================

    st.header(
        "1️⃣1️⃣ Candidate Diagnostics"
    )

    if raw_candidates:

        rejected = []

        for candidate in raw_candidates:

            valid, reason = (
                validate_candle(
                    candidate
                )
            )

            if not valid:

                rejected.append({
                    "x": candidate["x"],
                    "width": candidate["width"],
                    "height": candidate["height"],
                    "color": candidate["color"],
                    "confidence": candidate[
                        "confidence"
                    ],
                    "reason": reason
                })

        if rejected:

            st.dataframe(
                pd.DataFrame(
                    rejected
                ),
                width="stretch",
                hide_index=True
            )

        else:

            st.success(
                "No candidates were rejected by the current validation rules."
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    st.header(
        "1️⃣2️⃣ Engine Status"
    )

    st.warning(
        """
        V2.2 is intentionally NOT generating BUY or SELL signals.

        Current objective:

        1. Detect candles.
        2. Reconstruct body and wick geometry.
        3. Validate candle spacing.
        4. Identify possible missing candles.
        5. Produce a trustworthy pixel-space OHLC series.

        Real price calibration and technical indicators come later.
        """
    )
