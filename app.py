import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import hashlib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Maluz Signal Engine V2.5",
    layout="wide"
)


# ============================================================
# PASSWORD PROTECTION
# ============================================================

# Password used for the current protected version.
# Change this value before deploying if you want a different password.
PASSWORD_HASH = "73e5f1c0605c49650e419c6486a26c31721235175b285331b8ea32bf12cd6677"

def check_password(password):
    """Check the entered password against the stored SHA-256 hash."""
    entered_hash = hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()

    return entered_hash == PASSWORD_HASH

def require_password():

    # Already authenticated during this browser session
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔒 Maluz Signal Engine")
    st.caption(
        "This application is protected. "
        "Enter the password to continue."
    )

    password = st.text_input(
        "Password",
        type="password",
        key="login_password"
    )

    if st.button(
        "🔓 Unlock",
        type="primary"
    ):

        if check_password(password):

            st.session_state["authenticated"] = True
            st.rerun()

        else:

            st.error(
                "Incorrect password."
            )

    return False


if not require_password():
    st.stop()


# ------------------------------------------------------------
# LOGOUT
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔐 Access")
    if st.button("Log out"):
        st.session_state["authenticated"] = False
        st.session_state.pop("login_password", None)
        st.rerun()

st.title("🔹 Maluz Signal Engine V2.5")
st.caption(
    "Vision Diagnostic • Candle Geometry • OHLC Reconstruction "
    "• Structural Validation • NO TRADING SIGNALS"
)

# ============================================================
# CONFIGURATION
# ============================================================

MIN_COMPONENT_AREA = 8
MIN_COMPONENT_WIDTH = 1
MIN_COMPONENT_HEIGHT = 4

MIN_CANDLE_WIDTH = 2
MAX_CANDLE_WIDTH = 35
MIN_CANDLE_HEIGHT = 7

# Detection stage should be permissive.
MIN_CONFIDENCE_ACCEPT = 35

# ============================================================
# MISSING-CANDLE DETECTION
# ============================================================

# Gaps below this are considered normal.
SUSPICIOUS_GAP_RATIO = 1.35

# A candidate must be reasonably close to TWO normal
# candle intervals to represent one missing candle.
MISSING_TARGET_RATIO = 2.00

# Allowed deviation around the 2x target.
MISSING_RATIO_TOLERANCE = 0.22

# Maximum ratio we will still consider for ONE missing candle.
MAX_MISSING_RATIO = 2.35

# Neighboring spacing must remain close to the local baseline.
LOCAL_SPACING_TOLERANCE = 0.20

# Candle widths on both sides of a suspected gap should
# be reasonably similar.
MISSING_WIDTH_TOLERANCE = 0.35

# Minimum final confidence required before drawing
# a yellow missing-candle marker.
MIN_MISSING_CONFIDENCE = 78.0

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


def clamp_score(value):
    return max(0.0, min(100.0, float(value)))


def score_from_distance(value, target, tolerance):

    if target <= 0:
        return 100.0

    deviation = abs(value - target) / target

    score = 100.0 - (
        deviation / max(tolerance, 0.01) * 100.0
    )

    return clamp_score(score)


# ============================================================
# COLOR MASKS
# ============================================================

def create_color_masks(image):

    """
    Detect strong Pocket Option style candle colours.

    HSV is used because it separates hue from brightness
    better than raw RGB.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    # --------------------------------------------------------
    # GREEN
    # --------------------------------------------------------

    green_lower = np.array(
        [35, 80, 50]
    )

    green_upper = np.array(
        [95, 255, 255]
    )

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    # --------------------------------------------------------
    # RED
    # --------------------------------------------------------

    red_lower1 = np.array(
        [0, 80, 50]
    )

    red_upper1 = np.array(
        [12, 255, 255]
    )

    red_lower2 = np.array(
        [165, 80, 50]
    )

    red_upper2 = np.array(
        [180, 255, 255]
    )

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

    # ------------------------------------------------------------
    # CLEAN / RECONNECT CANDLE COLOUR STRUCTURES
    # ------------------------------------------------------------
    
    kernel = np.ones((2, 2), np.uint8)
    
    # IMPORTANT:
    # Do NOT use MORPH_OPEN here.
    # Opening can destroy thin candle wicks and narrow bodies.
    
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel
    )
    
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )
    
    # Small dilation helps reconnect anti-aliased candle edges
    # without aggressively expanding the candles.
    small_kernel = np.ones((2, 1), np.uint8)
    
    green_mask = cv2.dilate(
        green_mask,
        small_kernel,
        iterations=1
    )
    
    red_mask = cv2.dilate(
        red_mask,
        small_kernel,
        iterations=1
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

        x = int(
            stats[i, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[i, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[i, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[i, cv2.CC_STAT_HEIGHT]
        )

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        if w < MIN_COMPONENT_WIDTH:
            continue

        if h < MIN_COMPONENT_HEIGHT:
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

def group_components(components, x_tolerance=7):
    """
    Group colour components that belong to the same candle.

    Components are grouped primarily by X-centre.
    The grouping is deliberately permissive because a candle
    may have separate body and wick components.
    """

    if not components:
        return []

    components = sorted(
        components,
        key=lambda c: c["x"] + c["width"] / 2
    )

    groups = []

    for component in components:

        center_x = (
            component["x"]
            + component["width"] / 2
        )

        best_group = None
        best_distance = float("inf")

        for group in groups:

            group_center = np.mean([
                c["x"] + c["width"] / 2
                for c in group
            ])

            distance = abs(
                center_x - group_center
            )

            if distance <= x_tolerance and distance < best_distance:
                best_group = group
                best_distance = distance

        if best_group is not None:

            best_group.append(component)

        else:

            groups.append([component])

    return groups

# ============================================================
# BASIC CANDLE RECONSTRUCTION
# ============================================================

def reconstruct_candle(group):

    if not group:
        return None

    # --------------------------------------------------------
    # Overall bounding geometry
    # --------------------------------------------------------

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

    if width <= 0 or height <= 0:
        return None

    # --------------------------------------------------------
    # Dominant colour
    # --------------------------------------------------------

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

    total_color_area = (
        green_area + red_area
    )

    if green_area >= red_area:
        color = "GREEN"
        dominant_area = green_area
    else:
        color = "RED"
        dominant_area = red_area

    # --------------------------------------------------------
    # Colour purity
    # --------------------------------------------------------

    if total_color_area > 0:

        color_purity = (
            dominant_area
            / total_color_area
        )

    else:

        color_purity = 0.0

    color_score = clamp_score(
        color_purity * 100
    )

    # --------------------------------------------------------
    # Likely candle body
    # --------------------------------------------------------

    body_candidates = sorted(
        group,
        key=lambda c: c["area"],
        reverse=True
    )

    body = body_candidates[0]

    body_top = body["y"]

    body_bottom = (
        body["y"]
        + body["height"]
    )

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
    # These are NOT actual market prices.
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

    body_ratio = (
        body_height
        / max(height, 1)
    )

    aspect_ratio = (
        height
        / max(width, 1)
    )

    wick_ratio = (
        (upper_wick + lower_wick)
        / max(height, 1)
    )

    # --------------------------------------------------------
    # Initial geometry score
    # --------------------------------------------------------

    geometry_score = 100.0

    # Width
    if width < 3:

        geometry_score -= 35

    elif width > 20:

        geometry_score -= min(
            35,
            (width - 20) * 3
        )

    # Height
    if height < 10:

        geometry_score -= 35

    elif height < 14:

        geometry_score -= 10

    # Aspect ratio
    if aspect_ratio < 1.2:

        geometry_score -= 30

    elif aspect_ratio < 1.8:

        geometry_score -= 10

    # Body ratio
    if body_ratio < 0.03:

        geometry_score -= 25

    elif body_ratio < 0.06:

        geometry_score -= 10

    # Extremely dominant wick structure
    if wick_ratio > 0.90:

        geometry_score -= 15

    geometry_score = clamp_score(
        geometry_score
    )

    # --------------------------------------------------------
    # Group support score
    # --------------------------------------------------------

    component_count = len(group)

    if component_count >= 3:

        detection_score = 100.0

    elif component_count == 2:

        detection_score = 92.0

    else:

        detection_score = 82.0

    # --------------------------------------------------------
    # Return reconstructed candle
    # --------------------------------------------------------

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

        "body_ratio": round(
            body_ratio,
            3
        ),

        "wick_ratio": round(
            wick_ratio,
            3
        ),

        "aspect_ratio": round(
            aspect_ratio,
            3
        ),

        "color_confidence": round(
            color_score,
            1
        ),

        "geometry_score": round(
            geometry_score,
            1
        ),

        "detection_score": round(
            detection_score,
            1
        ),

        "structure_score": 0.0,

        "confidence": 0.0,

        "validation": "Pending"
    }


# ============================================================
# STRUCTURAL SCORING
# ============================================================

def apply_structural_scores(candles):

    """
    Compare each candle against its neighbours.

    This is deliberately relative rather than using a single
    fixed candle size, because screenshots can be scaled.
    """

    if not candles:
        return candles

    widths = np.array([
        c["width"]
        for c in candles
    ], dtype=float)

    heights = np.array([
        c["height"]
        for c in candles
    ], dtype=float)

    # Robust medians
    median_width = np.median(
        widths
    )

    median_height = np.median(
        heights
    )

    for i, candle in enumerate(
        candles
    ):

        width_score = score_from_distance(
            candle["width"],
            median_width,
            0.75
        )

        height_score = score_from_distance(
            candle["height"],
            median_height,
            1.00
        )

        # ----------------------------------------------------
        # Neighbour spacing score
        # ----------------------------------------------------

        spacing_values = []

        if i > 0:

            previous_center = (
                candles[i - 1]["x"]
                + candles[i - 1]["width"] / 2
            )

            current_center = (
                candle["x"]
                + candle["width"] / 2
            )

            spacing_values.append(
                current_center
                - previous_center
            )

        if i < len(candles) - 1:

            current_center = (
                candle["x"]
                + candle["width"] / 2
            )

            next_center = (
                candles[i + 1]["x"]
                + candles[i + 1]["width"] / 2
            )

            spacing_values.append(
                next_center
                - current_center
            )

        if spacing_values:

            local_spacing = np.median(
                spacing_values
            )

            all_centers = np.array([
                c["x"] + c["width"] / 2
                for c in candles
            ])

            all_spacing = np.diff(
                all_centers
            )

            global_spacing = np.median(
                all_spacing
            )

            spacing_score = score_from_distance(
                local_spacing,
                global_spacing,
                0.75
            )

        else:

            spacing_score = 80.0

        # ----------------------------------------------------
        # Combine structure
        # ----------------------------------------------------

        structure_score = (
            width_score * 0.35
            + height_score * 0.35
            + spacing_score * 0.30
        )

        candle["structure_score"] = round(
            clamp_score(structure_score),
            1
        )

        # ----------------------------------------------------
        # Final weighted confidence
        #
        # Geometry  = 40%
        # Colour    = 25%
        # Structure = 25%
        # Detection = 10%
        # ----------------------------------------------------

        final_confidence = (
            candle["geometry_score"] * 0.40
            + candle["color_confidence"] * 0.25
            + candle["structure_score"] * 0.25
            + candle["detection_score"] * 0.10
        )

        candle["confidence"] = round(
            clamp_score(final_confidence),
            1
        )

    return candles


# ============================================================
# VALIDATION
# ============================================================

def validate_candle(candle):

    if candle is None:
        return False

    if candle["width"] < MIN_CANDLE_WIDTH:
        return False

    if candle["height"] < MIN_CANDLE_HEIGHT:
        return False

    if candle["width"] > MAX_CANDLE_WIDTH:
        return False

    if candle["confidence"] < MIN_CONFIDENCE_ACCEPT:
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
        x_tolerance=7
    )

    candidates = []

    for group in groups:

        candle = reconstruct_candle(group)

        if candle is None:
            continue

        candidates.append(candle)

    # --------------------------------------------------------
    # SORT BY X POSITION
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda c: c["x"]
    )

    # --------------------------------------------------------
    # STRUCTURAL SCORING
    # --------------------------------------------------------

    candidates = apply_structural_scores(
        candidates
    )

    # --------------------------------------------------------
    # DO NOT DELETE WEAK CANDIDATES
    # --------------------------------------------------------

    accepted = []

    for candle in candidates:

        if validate_candle(candle):

            candle["validation"] = "Accepted"

        else:

            candle["validation"] = "Review"

        accepted.append(candle)

    return (
        accepted,
        green_mask,
        red_mask,
        all_components,
        candidates
    )

# ============================================================
# ROBUST MISSING-CANDLE ANALYSIS
# ============================================================

def analyze_spacing(candles):

    if len(candles) < 3:

        return {
            "median": None,
            "minimum": None,
            "maximum": None,
            "rows": [],
            "possible_missing": []
        }

    # --------------------------------------------------------
    # Candle centres
    # --------------------------------------------------------

    centers = np.array([
        c["x"] + c["width"] / 2
        for c in candles
    ], dtype=float)

    spacing = np.diff(centers)

    # --------------------------------------------------------
    # Robust global baseline
    #
    # Median is resistant to a few abnormal gaps.
    # --------------------------------------------------------

    baseline = float(np.median(spacing))

    minimum = float(np.min(spacing))
    maximum = float(np.max(spacing))

    rows = []
    possible_missing = []

    # --------------------------------------------------------
    # Analyse every gap
    # --------------------------------------------------------

    for i, gap in enumerate(spacing):

        if baseline <= 0:
            continue

        ratio = float(gap / baseline)

        status = "Normal"
        status_score = 100.0

        # ====================================================
        # NORMAL GAP
        # ====================================================

        if ratio < SUSPICIOUS_GAP_RATIO:

            status = "Normal"
            status_score = 100.0

        # ====================================================
        # SUSPICIOUS BUT NOT MISSING
        # ====================================================

        elif ratio < MISSING_TARGET_RATIO - MISSING_RATIO_TOLERANCE:

            status = "Suspicious Gap"
            status_score = 55.0

        # ====================================================
        # POSSIBLE MISSING CANDLE
        # ====================================================

        else:

            # ------------------------------------------------
            # We need a candle on both sides.
            #
            # This prevents the first/last candle from
            # generating unreliable missing-candle claims.
            # ------------------------------------------------

            has_left = i > 0
            has_right = i < len(spacing) - 1

            if not (has_left and has_right):

                status = "Edge Gap"
                status_score = 40.0

            else:

                previous_gap = float(spacing[i - 1])
                next_gap = float(spacing[i + 1])

                # ------------------------------------------------
                # Local baseline
                #
                # Rather than trusting the entire chart,
                # estimate normal spacing from the immediate
                # neighbors.
                # ------------------------------------------------

                local_baseline = float(
                    np.median([
                        previous_gap,
                        next_gap
                    ])
                )

                if local_baseline <= 0:

                    status = "Unreliable Gap"
                    status_score = 40.0

                else:

                    previous_ratio = (
                        previous_gap / local_baseline
                    )

                    next_ratio = (
                        next_gap / local_baseline
                    )

                    # ------------------------------------------------
                    # Neighbor spacing scores
                    # ------------------------------------------------

                    left_spacing_score = clamp_score(
                        100.0
                        - (
                            abs(previous_ratio - 1.0)
                            / LOCAL_SPACING_TOLERANCE
                            * 100.0
                        )
                    )

                    right_spacing_score = clamp_score(
                        100.0
                        - (
                            abs(next_ratio - 1.0)
                            / LOCAL_SPACING_TOLERANCE
                            * 100.0
                        )
                    )

                    neighbor_score = (
                        left_spacing_score
                        + right_spacing_score
                    ) / 2.0

                    # ------------------------------------------------
                    # Gap-to-2x score
                    #
                    # Perfect missing candle:
                    #
                    # normal gap
                    #      +
                    # missing candle interval
                    #
                    # therefore approximately 2x.
                    # ------------------------------------------------

                    gap_deviation = abs(
                        ratio - MISSING_TARGET_RATIO
                    )

                    gap_score = clamp_score(
                        100.0
                        - (
                            gap_deviation
                            / MISSING_RATIO_TOLERANCE
                            * 100.0
                        )
                    )

                    # ------------------------------------------------
                    # Width consistency
                    # ------------------------------------------------

                    left_candle = candles[i]
                    right_candle = candles[i + 1]

                    left_width = float(
                        left_candle["width"]
                    )

                    right_width = float(
                        right_candle["width"]
                    )

                    average_width = (
                        left_width + right_width
                    ) / 2.0

                    if average_width > 0:

                        width_difference = (
                            abs(
                                left_width
                                - right_width
                            )
                            / average_width
                        )

                        width_score = clamp_score(
                            100.0
                            - (
                                width_difference
                                / MISSING_WIDTH_TOLERANCE
                                * 100.0
                            )
                        )

                    else:

                        width_score = 0.0

                    # ------------------------------------------------
                    # Final missing-candle confidence
                    #
                    # Gap alignment       = 45%
                    # Neighbor spacing    = 35%
                    # Width consistency   = 20%
                    # ------------------------------------------------

                    missing_confidence = (
                        gap_score * 0.45
                        + neighbor_score * 0.35
                        + width_score * 0.20
                    )

                    missing_confidence = round(
                        clamp_score(
                            missing_confidence
                        ),
                        1
                    )

                    # ------------------------------------------------
                    # Decision
                    # ------------------------------------------------

                    if (
                        MISSING_TARGET_RATIO
                        - MISSING_RATIO_TOLERANCE
                        <= ratio
                        <=
                        MISSING_TARGET_RATIO
                        + MISSING_RATIO_TOLERANCE
                        and
                        missing_confidence
                        >= MIN_MISSING_CONFIDENCE
                    ):

                        status = (
                            "Possible Missing Candle"
                        )

                        status_score = (
                            missing_confidence
                        )

                        # --------------------------------------------
                        # Estimated position of missing candle
                        # --------------------------------------------

                        estimated_x = (
                            centers[i]
                            + local_baseline
                        )

                        possible_missing.append({

                            "from": i + 1,

                            "to": i + 2,

                            "spacing": round(
                                float(gap),
                                2
                            ),

                            "baseline": round(
                                float(local_baseline),
                                2
                            ),

                            "ratio": round(
                                float(ratio),
                                2
                            ),

                            "gap_score": round(
                                float(gap_score),
                                1
                            ),

                            "neighbor_score": round(
                                float(neighbor_score),
                                1
                            ),

                            "width_score": round(
                                float(width_score),
                                1
                            ),

                            "confidence": round(
                                float(missing_confidence),
                                1
                            ),

                            "estimated_x": round(
                                float(estimated_x),
                                1
                            )
                        })

                    else:

                        # --------------------------------------------
                        # It may be a large gap, but not enough
                        # evidence to call it missing.
                        # --------------------------------------------

                        if ratio <= MAX_MISSING_RATIO:

                            status = (
                                "Suspicious Gap"
                            )

                        else:

                            status = (
                                "Large / Unreliable Gap"
                            )

                        status_score = round(
                            missing_confidence,
                            1
                        )

        # ----------------------------------------------------
        # Store every gap in the diagnostic table
        # ----------------------------------------------------

        rows.append({

            "From Candle": i + 1,

            "To Candle": i + 2,

            "Spacing (px)": round(
                float(gap),
                2
            ),

            "Baseline (px)": round(
                float(baseline),
                2
            ),

            "Ratio": round(
                float(ratio),
                2
            ),

            "Status": status,

            "Gap Score": round(
                float(status_score),
                1
            )
        })

    return {

        "median": baseline,

        "minimum": minimum,

        "maximum": maximum,

        "rows": rows,

        "possible_missing": possible_missing
    }

# ============================================================
# SEQUENCE VALIDATION / MARKET STRUCTURE
# ============================================================

def analyze_candle_sequence(candles):

    # --------------------------------------------------------
    # EMPTY DATA
    # --------------------------------------------------------

    if not candles:
        return {
            "count": 0,
            "sequence_integrity": 0.0,
            "ohlc_validity": 0.0,
            "duplicate_centers": 0,
            "spacing_consistency": 0.0,

            "higher_highs": 0,
            "higher_lows": 0,
            "lower_highs": 0,
            "lower_lows": 0,

            "trend": "UNKNOWN",
            "current_structure": "INSUFFICIENT DATA",

            "swing_highs": [],
            "swing_lows": [],

            "current_direction": "UNKNOWN",
            "body_percentage": 0.0,
            "upper_wick_percentage": 0.0,
            "lower_wick_percentage": 0.0,
            "current_confidence": 0.0
        }

    # --------------------------------------------------------
    # BASIC COUNT
    # --------------------------------------------------------

    count = len(candles)

    # ========================================================
    # SWING STRUCTURE
    # ========================================================
    
    swing_analysis = detect_swings(
        candles,
        lookback=2
    )
    
    swing_highs = swing_analysis["swing_highs"]
    swing_lows = swing_analysis["swing_lows"]
    
    swing_high_count = len(swing_highs)
    swing_low_count = len(swing_lows)
    
    
    # ========================================================
    # CLASSIFY SWING STRUCTURE
    # ========================================================
    
    swing_structure = classify_swing_structure(
        swing_highs,
        swing_lows
    )
    
    
    # ========================================================
    # GET CLASSIFIED SWING DATA
    # ========================================================
    
    swing_highs = swing_structure["swing_highs"]
    swing_lows = swing_structure["swing_lows"]
    
    swing_high_count = len(swing_highs)
    swing_low_count = len(swing_lows)
    
    
    # ========================================================
    # BOS / CHoCH DETECTION
    # ========================================================
    
    bos_choch_analysis = detect_bos_choch(
        candles,
        swing_highs,
        swing_lows,
        lookback=2
    )
    
    bos_choch_events = (
        bos_choch_analysis["events"]
    )
    
    bos_choch_bias = (
        bos_choch_analysis["current_bias"]
    )
    
    last_bos_choch = (
        bos_choch_analysis["last_event"]
    )

    # --------------------------------------------------------
    # GET STRUCTURE COUNTS
    # --------------------------------------------------------

    higher_highs = (
        swing_structure["higher_highs"]
    )

    higher_lows = (
        swing_structure["higher_lows"]
    )

    lower_highs = (
        swing_structure["lower_highs"]
    )

    lower_lows = (
        swing_structure["lower_lows"]
    )

    # --------------------------------------------------------
    # GET TREND / STRUCTURE
    # --------------------------------------------------------

    swing_trend = (
        swing_structure["trend"]
    )

    swing_current_structure = (
        swing_structure["current_structure"]
    )

    # --------------------------------------------------------
    # STRUCTURE DEBUG
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SWING STRUCTURE")
    print("=" * 70)

    print(
        "Swing highs:",
        swing_high_count
    )

    print(
        "Swing lows:",
        swing_low_count
    )

    print(
        "Higher highs:",
        higher_highs
    )

    print(
        "Higher lows:",
        higher_lows
    )

    print(
        "Lower highs:",
        lower_highs
    )

    print(
        "Lower lows:",
        lower_lows
    )

    print(
        "Trend:",
        swing_trend
    )

    print(
        "Current structure:",
        swing_current_structure
    )

    print("=" * 70)
    print("\n")
    
    # --------------------------------------------------------
    # TERMINAL DEBUG
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SWING ANALYSIS")
    print("=" * 70)

    print("Total candles:", count)
    print("Swing highs:", swing_high_count)
    print("Swing lows:", swing_low_count)

    print("-" * 70)
    print("SWING HIGH DATA")

    for swing in swing_highs:
        print(swing)

    print("-" * 70)
    print("SWING LOW DATA")

    for swing in swing_lows:
        print(swing)

    print("=" * 70)
    print("\n")

    # ========================================================
    # CENTRES
    # ========================================================

    centers = np.array(
        [
            c["x"] + c["width"] / 2
            for c in candles
        ],
        dtype=float
    )

    # ========================================================
    # DUPLICATE CENTRES
    # ========================================================

    duplicate_centers = 0

    if len(centers) > 1:

        for i in range(
            1,
            len(centers)
        ):

            if abs(
                centers[i] -
                centers[i - 1]
            ) < 1.0:

                duplicate_centers += 1

    # ========================================================
    # OHLC VALIDATION
    #
    # IMPORTANT:
    # These are PIXEL coordinates.
    #
    # Smaller Y = higher price
    # Larger Y = lower price
    # ========================================================

    valid_ohlc = 0

    for candle in candles:

        high = candle["high"]
        low = candle["low"]

        open_price = candle["open"]
        close_price = candle["close"]

        valid = (
            high <= open_price
            and
            high <= close_price
            and
            low >= open_price
            and
            low >= close_price
            and
            high <= low
        )

        if valid:
            valid_ohlc += 1

    ohlc_validity = (
        valid_ohlc / count * 100
        if count > 0
        else 0
    )

    # ========================================================
    # SPACING CONSISTENCY
    # ========================================================

    spacing_consistency = 0.0

    if len(centers) >= 3:

        spacing = np.diff(centers)

        median_spacing = np.median(
            spacing
        )

        if median_spacing > 0:

            deviations = (
                np.abs(
                    spacing -
                    median_spacing
                )
                /
                median_spacing
            )

            average_deviation = np.mean(
                deviations
            )

            spacing_consistency = clamp_score(
                100 -
                (
                    average_deviation *
                    100
                )
            )

    elif len(centers) == 2:

        spacing_consistency = 100.0

    # ========================================================
    # MARKET STRUCTURE
    #
    # Use the detected candle sequence to determine
    # higher highs / higher lows / lower highs / lower lows.
    # ========================================================

    higher_highs = 0
    higher_lows = 0
    lower_highs = 0
    lower_lows = 0

    # --------------------------------------------------------
    # Compare consecutive candle highs/lows
    # --------------------------------------------------------

    for i in range(1, count):

        previous = candles[i - 1]
        current = candles[i]

        # Smaller Y = higher price
        if current["high"] < previous["high"]:
            higher_highs += 1

        elif current["high"] > previous["high"]:
            lower_highs += 1

        # Larger Y = lower price
        if current["low"] < previous["low"]:
            higher_lows += 1

        elif current["low"] > previous["low"]:
            lower_lows += 1

    # ========================================================
    # RECENT STRUCTURE
    # ========================================================

    recent_window = min(
        10,
        count
    )

    recent_candles = candles[
        -recent_window:
    ]

    recent_hh = 0
    recent_hl = 0
    recent_lh = 0
    recent_ll = 0

    for i in range(
        1,
        len(recent_candles)
    ):

        previous = recent_candles[
            i - 1
        ]

        current = recent_candles[i]

        # Higher high
        if current["high"] < previous["high"]:
            recent_hh += 1

        # Lower high
        elif current["high"] > previous["high"]:
            recent_lh += 1

        # Higher low
        if current["low"] < previous["low"]:
            recent_hl += 1

        # Lower low
        elif current["low"] > previous["low"]:
            recent_ll += 1

    # ========================================================
    # TREND
    # ========================================================

    bullish_score = (
        recent_hh +
        recent_hl
    )

    bearish_score = (
        recent_lh +
        recent_ll
    )

    if bullish_score >= bearish_score + 2:

        trend = "BULLISH"

    elif bearish_score >= bullish_score + 2:

        trend = "BEARISH"

    else:

        trend = "SIDEWAYS / MIXED"

    # ========================================================
    # CURRENT CANDLE
    # ========================================================

    current = candles[-1]

    if current["color"] == "GREEN":

        direction = "GREEN"

    else:

        direction = "RED"

    # ========================================================
    # BODY / WICK ANALYSIS
    # ========================================================

    body_height = float(
        current["body_height"]
    )

    total_height = max(
        float(current["height"]),
        1.0
    )

    body_percentage = (
        body_height /
        total_height *
        100
    )

    upper_wick_percentage = (
        current["upper_wick"] /
        total_height *
        100
    )

    lower_wick_percentage = (
        current["lower_wick"] /
        total_height *
        100
    )

    # ========================================================
    # CURRENT STRUCTURE DESCRIPTION
    # ========================================================

    if (
        recent_hh >= 2
        and
        recent_hl >= 2
    ):

        current_structure = (
            "HIGHER HIGH + HIGHER LOW"
        )

    elif (
        recent_lh >= 2
        and
        recent_ll >= 2
    ):

        current_structure = (
            "LOWER HIGH + LOWER LOW"
        )

    elif (
        recent_hh > recent_lh
        and
        recent_hl > recent_ll
    ):

        current_structure = (
            "BULLISH STRUCTURE DEVELOPING"
        )

    elif (
        recent_lh > recent_hh
        and
        recent_ll > recent_hl
    ):

        current_structure = (
            "BEARISH STRUCTURE DEVELOPING"
        )

    else:

        current_structure = (
            "CONSOLIDATION / MIXED"
        )

    # ========================================================
    # SEQUENCE INTEGRITY
    # ========================================================

    integrity_components = []

    duplicate_score = (
        100.0
        if duplicate_centers == 0
        else clamp_score(
            100 -
            duplicate_centers /
            count *
            100
        )
    )

    integrity_components.append(
        duplicate_score
    )

    integrity_components.append(
        ohlc_validity
    )

    integrity_components.append(
        spacing_consistency
    )

    sequence_integrity = round(
        np.mean(
            integrity_components
        ),
        1
    )

    # ========================================================
    # FINAL RETURN
    # ========================================================

    return {

        "count": count,

        "sequence_integrity":
            sequence_integrity,

        "ohlc_validity":
            round(
                ohlc_validity,
                1
            ),

        "duplicate_centers":
            duplicate_centers,

        "spacing_consistency":
            round(
                spacing_consistency,
                1
            ),

        "higher_highs":
            higher_highs,

        "higher_lows":
            higher_lows,

        "lower_highs":
            lower_highs,

        "lower_lows":
            lower_lows,

        "trend":
            trend,

        "current_structure":
            current_structure,

        "swing_highs":
            swing_highs,

        "swing_lows":
            swing_lows,

        # ====================================================
        # BOS / CHoCH
        # ====================================================

        "bos_choch_events":
            bos_choch_events,

        "bos_choch_bias":
            bos_choch_bias,

        "last_bos_choch":
            last_bos_choch,

        # ====================================================
        # CURRENT CANDLE
        # ====================================================

        "current_direction":
            direction,

        "body_percentage":
            round(
                body_percentage,
                1
            ),

        "upper_wick_percentage":
            round(
                upper_wick_percentage,
                1
            ),

        "lower_wick_percentage":
            round(
                lower_wick_percentage,
                1
            ),

        "current_confidence":
            round(
                current["confidence"],
                1
            )
    }

# ============================================================
# SWING STRUCTURE DETECTION
# ============================================================

def detect_swings(candles, lookback=2):

    """
    Detect confirmed swing highs and swing lows.

    Candle coordinates are PIXEL coordinates.

    Smaller Y = higher market price.
    Larger Y = lower market price.

    Swing High:
        Current HIGH is at or above the surrounding highs.

    Swing Low:
        Current LOW is at or below the surrounding lows.

    Equal pixel values are allowed because screenshot
    reconstruction can produce flat coordinates.
    """

    # --------------------------------------------------------
    # NOT ENOUGH DATA
    # --------------------------------------------------------

    minimum_candles = (
        lookback * 2 + 1
    )

    if len(candles) < minimum_candles:

        return {
            "swing_highs": [],
            "swing_lows": []
        }

    swing_highs = []
    swing_lows = []

    # ========================================================
    # TEST EACH POSSIBLE SWING
    # ========================================================

    for i in range(
        lookback,
        len(candles) - lookback
    ):

        current = candles[i]

        # ====================================================
        # SURROUNDING HIGHS
        # ====================================================

        current_high = float(
            current["high"]
        )

        left_highs = [
            float(
                candles[j]["high"]
            )
            for j in range(
                i - lookback,
                i
            )
        ]

        right_highs = [
            float(
                candles[j]["high"]
            )
            for j in range(
                i + 1,
                i + lookback + 1
            )
        ]

        # ====================================================
        # SWING HIGH
        #
        # Smaller Y = higher price.
        #
        # Therefore current high must be <= the surrounding
        # high coordinates.
        #
        # We also require that at least one neighbouring
        # value is strictly lower in market height
        # (larger Y), otherwise a completely flat area
        # would generate multiple swing highs.
        # ====================================================

        is_swing_high = (

            current_high <=
            min(left_highs)

            and

            current_high <=
            min(right_highs)

            and

            (
                current_high <
                max(left_highs)

                or

                current_high <
                max(right_highs)
            )
        )

        if is_swing_high:

            swing_highs.append({

                "index":
                    i,

                "price":
                    current_high,

                "x":
                    (
                        current["x"]
                        +
                        current["width"] / 2
                    ),

                "type":
                    "SWING HIGH"
            })

        # ====================================================
        # SURROUNDING LOWS
        # ====================================================

        current_low = float(
            current["low"]
        )

        left_lows = [
            float(
                candles[j]["low"]
            )
            for j in range(
                i - lookback,
                i
            )
        ]

        right_lows = [
            float(
                candles[j]["low"]
            )
            for j in range(
                i + 1,
                i + lookback + 1
            )
        ]

        # ====================================================
        # SWING LOW
        #
        # Larger Y = lower price.
        #
        # Therefore current low must be >= the surrounding
        # low coordinates.
        # ====================================================

        is_swing_low = (

            current_low >=
            max(left_lows)

            and

            current_low >=
            max(right_lows)

            and

            (
                current_low >
                min(left_lows)

                or

                current_low >
                min(right_lows)
            )
        )

        if is_swing_low:

            swing_lows.append({

                "index":
                    i,

                "price":
                    current_low,

                "x":
                    (
                        current["x"]
                        +
                        current["width"] / 2
                    ),

                "type":
                    "SWING LOW"
            })

    # ========================================================
    # DEBUG
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SWING DETECTION RESULT")
    print("=" * 70)

    print(
        "Total candles:",
        len(candles)
    )

    print(
        "Swing highs found:",
        len(swing_highs)
    )

    print(
        "Swing lows found:",
        len(swing_lows)
    )

    print("-" * 70)

    print("SWING HIGH DATA")

    for swing in swing_highs:
        print(swing)

    print("-" * 70)

    print("SWING LOW DATA")

    for swing in swing_lows:
        print(swing)

    print("=" * 70)
    print("\n")

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "swing_highs":
            swing_highs,

        "swing_lows":
            swing_lows
        
    }

# ============================================================
# SWING STRUCTURE CLASSIFICATION
# ============================================================

def classify_swing_structure(swing_highs, swing_lows):

    """
    Classify detected swing points as:

        HH = Higher High
        LH = Lower High
        HL = Higher Low
        LL = Lower Low

    Pixel coordinates:

        Smaller Y = higher market price
        Larger Y = lower market price
    """

    # --------------------------------------------------------
    # COPY THE ORIGINAL SWING DATA
    # --------------------------------------------------------

    classified_highs = []

    classified_lows = []

    # ========================================================
    # CLASSIFY SWING HIGHS
    # ========================================================

    for i, swing in enumerate(swing_highs):

        item = swing.copy()

        if i == 0:

            item["structure"] = "INITIAL HIGH"

        else:

            previous = swing_highs[i - 1]

            # Smaller Y = higher price

            if swing["price"] < previous["price"]:

                item["structure"] = "HH"

            elif swing["price"] > previous["price"]:

                item["structure"] = "LH"

            else:

                item["structure"] = "EQUAL HIGH"

        classified_highs.append(item)

    # ========================================================
    # CLASSIFY SWING LOWS
    # ========================================================

    for i, swing in enumerate(swing_lows):

        item = swing.copy()

        if i == 0:

            item["structure"] = "INITIAL LOW"

        else:

            previous = swing_lows[i - 1]

            # Larger Y = lower price

            if swing["price"] < previous["price"]:

                item["structure"] = "HL"

            elif swing["price"] > previous["price"]:

                item["structure"] = "LL"

            else:

                item["structure"] = "EQUAL LOW"

        classified_lows.append(item)

    # ========================================================
    # COUNT STRUCTURE
    # ========================================================

    higher_highs = sum(
        1
        for swing in classified_highs
        if swing["structure"] == "HH"
    )

    lower_highs = sum(
        1
        for swing in classified_highs
        if swing["structure"] == "LH"
    )

    higher_lows = sum(
        1
        for swing in classified_lows
        if swing["structure"] == "HL"
    )

    lower_lows = sum(
        1
        for swing in classified_lows
        if swing["structure"] == "LL"
    )

    # ========================================================
    # DETERMINE STRUCTURAL BIAS
    # ========================================================

    bullish_score = (
        higher_highs +
        higher_lows
    )

    bearish_score = (
        lower_highs +
        lower_lows
    )

    if bullish_score > bearish_score:

        trend = "BULLISH"

    elif bearish_score > bullish_score:

        trend = "BEARISH"

    else:

        trend = "NEUTRAL"

    # ========================================================
    # CURRENT STRUCTURE
    # ========================================================

    if (
        higher_highs > 0
        and
        higher_lows > 0
        and
        higher_highs >= lower_highs
        and
        higher_lows >= lower_lows
    ):

        current_structure = (
            "HIGHER HIGH + HIGHER LOW"
        )

    elif (
        lower_highs > 0
        and
        lower_lows > 0
        and
        lower_highs >= higher_highs
        and
        lower_lows >= higher_lows
    ):

        current_structure = (
            "LOWER HIGH + LOWER LOW"
        )

    elif bullish_score > bearish_score:

        current_structure = (
            "BULLISH STRUCTURE"
        )

    elif bearish_score > bullish_score:

        current_structure = (
            "BEARISH STRUCTURE"
        )

    else:

        current_structure = (
            "MIXED STRUCTURE"
        )

    # ========================================================
    # MOST RECENT SWINGS
    # ========================================================

    if len(classified_highs) >= 2:

        previous_high = classified_highs[-2]
        latest_high = classified_highs[-1]

    else:

        previous_high = None
        latest_high = None

    if len(classified_lows) >= 2:

        previous_low = classified_lows[-2]
        latest_low = classified_lows[-1]

    else:

        previous_low = None
        latest_low = None

    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return {

        # Actual classified swing data
        "swing_highs": classified_highs,

        "swing_lows": classified_lows,

        # Counts
        "higher_highs": higher_highs,

        "higher_lows": higher_lows,

        "lower_highs": lower_highs,

        "lower_lows": lower_lows,

        # Overall structure
        "trend": trend,

        "current_structure": current_structure,

        # Recent swing information
        "previous_high": previous_high,

        "latest_high": latest_high,

        "previous_low": previous_low,

        "latest_low": latest_low,

        # Detailed latest structure
        "high_structure": (
            latest_high["structure"]
            if latest_high is not None
            else "INSUFFICIENT DATA"
        ),

        "low_structure": (
            latest_low["structure"]
            if latest_low is not None
            else "INSUFFICIENT DATA"
        )
    }

# ============================================================
# BOS / CHoCH DETECTION
# ============================================================

def detect_bos_choch(
    candles,
    swing_highs,
    swing_lows,
    lookback=2
):
    """
    Detect structural BOS and CHoCH using actual price
    movement through confirmed swing levels.

    IMPORTANT:
    Candle coordinates are PIXELS.

        Smaller Y = higher market price
        Larger Y = lower market price

    Logic:

    BULLISH STRUCTURE
        - Close above a protected swing high
          = BULLISH BOS
        - Close below a protected swing low
          = BEARISH CHoCH

    BEARISH STRUCTURE
        - Close below a protected swing low
          = BEARISH BOS
        - Close above a protected swing high
          = BULLISH CHoCH

    This is a diagnostic structural model.
    It is NOT a trading signal generator.
    """

    events = []

    # ========================================================
    # COMBINE SWING DATA
    # ========================================================

    all_swings = []

    for swing in swing_highs:

        all_swings.append({
            "index": int(swing["index"]),
            "price": float(swing["price"]),
            "x": swing.get("x"),
            "type": "HIGH",
            "structure": swing.get(
                "structure",
                "UNKNOWN"
            )
        })

    for swing in swing_lows:

        all_swings.append({
            "index": int(swing["index"]),
            "price": float(swing["price"]),
            "x": swing.get("x"),
            "type": "LOW",
            "structure": swing.get(
                "structure",
                "UNKNOWN"
            )
        })

    # ========================================================
    # CHRONOLOGICAL ORDER
    # ========================================================

    all_swings.sort(
        key=lambda s: s["index"]
    )

    # ========================================================
    # NOT ENOUGH DATA
    # ========================================================

    if len(all_swings) < 2 or len(candles) < 3:

        return {
            "events": [],
            "current_bias": "UNKNOWN",
            "last_event": None
        }

    # ========================================================
    # DETERMINE INITIAL BIAS
    #
    # We do NOT automatically call every HH bullish BOS
    # or every LL bearish BOS.
    #
    # We first establish the prevailing structural direction.
    # ========================================================

    bias = "UNKNOWN"

    for swing in all_swings:

        structure = swing["structure"]

        if structure == "HH":

            bias = "BULLISH"
            break

        elif structure == "LL":

            bias = "BEARISH"
            break

    # ========================================================
    # IF NO HH/LL HAS ESTABLISHED A DIRECTION
    # USE THE CLASSIFIED SWING COUNTS
    # ========================================================

    if bias == "UNKNOWN":

        hh_count = sum(
            1
            for swing in swing_highs
            if swing.get("structure") == "HH"
        )

        hl_count = sum(
            1
            for swing in swing_lows
            if swing.get("structure") == "HL"
        )

        lh_count = sum(
            1
            for swing in swing_highs
            if swing.get("structure") == "LH"
        )

        ll_count = sum(
            1
            for swing in swing_lows
            if swing.get("structure") == "LL"
        )
# ============================================================
# BOS / CHoCH DETECTION
# ============================================================

def detect_bos_choch(
    candles,
    swing_highs,
    swing_lows,
    lookback=2
):
    """
    BOS / CHoCH STRUCTURAL ENGINE - VERSION 3

    Pixel coordinates:
        Smaller Y = higher market price
        Larger Y = lower market price

    STRUCTURAL RULES
    ----------------

    BULLISH STRUCTURE:

        HH
          \
           HL
            \
             price breaks HH
                    ↓
               BULLISH BOS

        If price breaks the protected HL:
                    ↓
               BEARISH CHoCH


    BEARISH STRUCTURE:

        LH
          \
           LL
            \
             price breaks LL
                    ↓
               BEARISH BOS

        If price breaks the protected LH:
                    ↓
               BULLISH CHoCH


    IMPORTANT:

    1. HH or LL by itself is NOT a BOS.
    2. Price must actually close through the structural level.
    3. A continuation BOS requires a completed retracement.
    4. CHoCH breaks the protected opposite-side structure.
    5. A structural level can only trigger one event.
    """

    # ========================================================
    # INITIAL RESULT
    # ========================================================

    events = []

    # ========================================================
    # BUILD COMBINED SWING LIST
    # ========================================================

    all_swings = []

    for swing in swing_highs:

        all_swings.append({
            "index": int(
                swing["index"]
            ),

            "price": float(
                swing["price"]
            ),

            "x": swing.get("x"),

            "type": "HIGH",

            "structure": swing.get(
                "structure",
                "UNKNOWN"
            )
        })

    for swing in swing_lows:

        all_swings.append({
            "index": int(
                swing["index"]
            ),

            "price": float(
                swing["price"]
            ),

            "x": swing.get("x"),

            "type": "LOW",

            "structure": swing.get(
                "structure",
                "UNKNOWN"
            )
        })

    # ========================================================
    # CHRONOLOGICAL ORDER
    # ========================================================

    all_swings.sort(
        key=lambda swing:
        swing["index"]
    )

    # ========================================================
    # NOT ENOUGH DATA
    # ========================================================

    if (
        len(candles) < 3
        or
        len(all_swings) < 3
    ):

        return {
            "events": [],
            "current_bias": "UNKNOWN",
            "last_event": None
        }

    # ========================================================
    # INITIAL STRUCTURAL BIAS
    #
    # Determine the earliest meaningful directional sequence.
    # ========================================================

    bias = "UNKNOWN"

    recent_high = None
    recent_low = None

    for swing in all_swings:

        if swing["type"] == "HIGH":

            recent_high = swing

        elif swing["type"] == "LOW":

            recent_low = swing

        # ----------------------------------------------------
        # Bullish structure
        # ----------------------------------------------------

        if (
            recent_high is not None
            and
            recent_low is not None
        ):

            if (
                recent_high["structure"]
                == "HH"
                and
                recent_low["structure"]
                == "HL"
            ):

                bias = "BULLISH"
                break

            # ------------------------------------------------
            # Bearish structure
            # ------------------------------------------------

            if (
                recent_high["structure"]
                == "LH"
                and
                recent_low["structure"]
                == "LL"
            ):

                bias = "BEARISH"
                break

    # ========================================================
    # FALLBACK BIAS
    # ========================================================

    if bias == "UNKNOWN":

        hh_count = sum(
            1
            for swing in swing_highs
            if swing.get("structure") == "HH"
        )

        hl_count = sum(
            1
            for swing in swing_lows
            if swing.get("structure") == "HL"
        )

        lh_count = sum(
            1
            for swing in swing_highs
            if swing.get("structure") == "LH"
        )

        ll_count = sum(
            1
            for swing in swing_lows
            if swing.get("structure") == "LL"
        )

        bullish_score = (
            hh_count + hl_count
        )

        bearish_score = (
            lh_count + ll_count
        )

        if bullish_score > bearish_score:

            bias = "BULLISH"

        elif bearish_score > bullish_score:

            bias = "BEARISH"

    # ========================================================
    # STATE VARIABLES
    # ========================================================

    latest_high = None
    latest_low = None

    # --------------------------------------------------------
    # Bullish continuation needs:
    #
    # 1. A reference HH
    # 2. Then a retracement HL
    # 3. Then price must break that HH
    # --------------------------------------------------------

    bullish_reference_high = None
    bullish_retracement_low = None

    # --------------------------------------------------------
    # Bearish continuation needs:
    #
    # 1. A reference LL
    # 2. Then a retracement LH
    # 3. Then price must break that LL
    # --------------------------------------------------------

    bearish_reference_low = None
    bearish_retracement_high = None

    # --------------------------------------------------------
    # Protected levels for CHoCH
    # --------------------------------------------------------

    protected_high = None
    protected_low = None

    # ========================================================
    # PREVENT REPEATED BREAKS
    # ========================================================

    broken_highs = set()
    broken_lows = set()

    # ========================================================
    # HELPER TO ADD EVENTS
    # ========================================================

    def add_event(
        candle_index,
        price,
        event,
        direction,
        swing_type,
        level_type,
        level_index
    ):

        events.append({

            "candle_index":
                int(candle_index),

            "price":
                float(price),

            "event":
                event,

            "direction":
                direction,

            "swing_type":
                swing_type,

            "level_type":
                level_type,

            "level_index":
                int(level_index)
        })

    # ========================================================
    # PROCESS MARKET FROM LEFT TO RIGHT
    # ========================================================

    swing_pointer = 0

    for candle_index in range(
        len(candles)
    ):

        candle = candles[
            candle_index
        ]

        close_price = float(
            candle["close"]
        )

        # ====================================================
        # REGISTER SWINGS THAT HAVE ALREADY FORMED
        # ====================================================

        while (
            swing_pointer
            < len(all_swings)
            and
            all_swings[
                swing_pointer
            ]["index"]
            < candle_index
        ):

            swing = all_swings[
                swing_pointer
            ]

            # =================================================
            # HIGH SWING
            # =================================================

            if swing["type"] == "HIGH":

                latest_high = swing

                structure = (
                    swing["structure"]
                )

                # ---------------------------------------------
                # BEARISH RETRACEMENT
                #
                # LH after a LL gives us a complete bearish leg.
                # ---------------------------------------------

                if structure == "LH":

                    bearish_retracement_high = swing

                    # During bearish structure, this is the
                    # protected high.
                    if bias == "BEARISH":

                        protected_high = swing

                # ---------------------------------------------
                # BULLISH REFERENCE HIGH
                #
                # HH becomes a level that can later be broken
                # for bullish BOS.
                # ---------------------------------------------

                elif structure == "HH":

                    bullish_reference_high = swing

            # =================================================
            # LOW SWING
            # =================================================

            elif swing["type"] == "LOW":

                latest_low = swing

                structure = (
                    swing["structure"]
                )

                # ---------------------------------------------
                # BULLISH RETRACEMENT
                #
                # HL after an HH gives us a complete bullish leg.
                # ---------------------------------------------

                if structure == "HL":

                    bullish_retracement_low = swing

                    # During bullish structure, this is the
                    # protected low.
                    if bias == "BULLISH":

                        protected_low = swing

                # ---------------------------------------------
                # BEARISH REFERENCE LOW
                #
                # LL becomes a level that can later be broken
                # for bearish BOS.
                # ---------------------------------------------

                elif structure == "LL":

                    bearish_reference_low = swing

            swing_pointer += 1

        # ====================================================
        # BULLISH STRUCTURE
        # ====================================================

        if bias == "BULLISH":

            # =================================================
            # BEARISH CHoCH
            #
            # Close below protected Higher Low.
            #
            # Pixel rule:
            # larger Y = lower price
            # =================================================

            if protected_low is not None:

                level_index = (
                    protected_low["index"]
                )

                level_price = (
                    protected_low["price"]
                )

                broke_protected_low = (
                    close_price > level_price
                )

                if (
                    broke_protected_low
                    and
                    level_index
                    not in broken_lows
                ):

                    add_event(

                        candle_index,

                        close_price,

                        "BEARISH CHoCH",

                        "BEARISH",

                        "HL",

                        "PROTECTED LOW",

                        level_index
                    )

                    broken_lows.add(
                        level_index
                    )

                    # -----------------------------------------
                    # STRUCTURE HAS CHANGED.
                    # -----------------------------------------

                    bias = "BEARISH"

                    protected_low = None

                    # Reset bullish continuation state.
                    bullish_reference_high = None
                    bullish_retracement_low = None

                    continue

            # =================================================
            # BULLISH BOS
            #
            # REQUIRE BOTH:
            #
            # 1. A bullish HH reference
            # 2. A later HL retracement
            #
            # Only then can price breaking the HH become BOS.
            # =================================================

            if (
                bullish_reference_high
                is not None
                and
                bullish_retracement_low
                is not None
                and
                bullish_retracement_low[
                    "index"
                ]
                >
                bullish_reference_high[
                    "index"
                ]
            ):

                level_index = (
                    bullish_reference_high[
                        "index"
                    ]
                )

                level_price = (
                    bullish_reference_high[
                        "price"
                    ]
                )

                broke_reference_high = (
                    close_price < level_price
                )

                if (
                    broke_reference_high
                    and
                    level_index
                    not in broken_highs
                ):

                    add_event(

                        candle_index,

                        close_price,

                        "BULLISH BOS",

                        "BULLISH",

                        "HH",

                        "SWING HIGH",

                        level_index
                    )

                    broken_highs.add(
                        level_index
                    )

                    # -----------------------------------------
                    # This HH has now been broken.
                    #
                    # The next HH can become the new reference.
                    # -----------------------------------------

                    bullish_reference_high = None

                    continue

        # ====================================================
        # BEARISH STRUCTURE
        # ====================================================

        elif bias == "BEARISH":

            # =================================================
            # BULLISH CHoCH
            #
            # Close above protected Lower High.
            #
            # Pixel rule:
            # smaller Y = higher price
            # =================================================

            if protected_high is not None:

                level_index = (
                    protected_high["index"]
                )

                level_price = (
                    protected_high["price"]
                )

                broke_protected_high = (
                    close_price < level_price
                )

                if (
                    broke_protected_high
                    and
                    level_index
                    not in broken_highs
                ):

                    add_event(

                        candle_index,

                        close_price,

                        "BULLISH CHoCH",

                        "BULLISH",

                        "LH",

                        "PROTECTED HIGH",

                        level_index
                    )

                    broken_highs.add(
                        level_index
                    )

                    # -----------------------------------------
                    # STRUCTURE HAS CHANGED.
                    # -----------------------------------------

                    bias = "BULLISH"

                    protected_high = None

                    # Reset bearish continuation state.
                    bearish_reference_low = None
                    bearish_retracement_high = None

                    continue

            # =================================================
            # BEARISH BOS
            #
            # REQUIRE BOTH:
            #
            # 1. A bearish LL reference
            # 2. A later LH retracement
            #
            # Only then can price breaking the LL become BOS.
            # =================================================

            if (
                bearish_reference_low
                is not None
                and
                bearish_retracement_high
                is not None
                and
                bearish_retracement_high[
                    "index"
                ]
                >
                bearish_reference_low[
                    "index"
                ]
            ):

                level_index = (
                    bearish_reference_low[
                        "index"
                    ]
                )

                level_price = (
                    bearish_reference_low[
                        "price"
                    ]
                )

                broke_reference_low = (
                    close_price > level_price
                )

                if (
                    broke_reference_low
                    and
                    level_index
                    not in broken_lows
                ):

                    add_event(

                        candle_index,

                        close_price,

                        "BEARISH BOS",

                        "BEARISH",

                        "LL",

                        "SWING LOW",

                        level_index
                    )

                    broken_lows.add(
                        level_index
                    )

                    # -----------------------------------------
                    # This LL has now been broken.
                    #
                    # Wait for the next bearish structural leg.
                    # -----------------------------------------

                    bearish_reference_low = None

                    continue

    # ========================================================
    # SORT EVENTS
    # ========================================================

    events.sort(
        key=lambda event:
        event["candle_index"]
    )

    # ========================================================
    # DETERMINE LAST EVENT
    # ========================================================

    last_event = (
        events[-1]
        if events
        else None
    )

    # ========================================================
    # FINAL BIAS CONSISTENCY
    #
    # If the last structural event was a CHoCH or BOS,
    # its direction becomes the final structural bias.
    # ========================================================

    if last_event is not None:

        event_direction = (
            last_event["direction"]
        )

        if event_direction in (
            "BULLISH",
            "BEARISH"
        ):

            bias = event_direction

    # ========================================================
    # DEBUG
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BOS / CHoCH STRUCTURAL ANALYSIS V3")
    print("=" * 70)

    print(
        "Final structural bias:",
        bias
    )

    print(
        "Total BOS / CHoCH events:",
        len(events)
    )

    print("-" * 70)

    for event in events:

        print(
            f"Candle {event['candle_index']} | "
            f"{event['event']} | "
            f"Level {event['level_index']} | "
            f"Price {event['price']}"
        )

    print("=" * 70)
    print("\n")

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "events":
            events,

        "current_bias":
            bias,

        "last_event":
            last_event
    }
# ============================================================
# ANNOTATION
# ============================================================

def annotate_candles(
    image,
    candles,
    spacing_analysis=None
):

    annotated = image.copy()

    # --------------------------------------------------------
    # Candle annotations
    # --------------------------------------------------------

    for index, candle in enumerate(
        candles,
        start=1
    ):

        x = candle["x"]
        y = candle["y"]
        w = candle["width"]
        h = candle["height"]

        if candle["color"] == "GREEN":

            color = (
                0,
                255,
                0
            )

        else:

            color = (
                255,
                60,
                60
            )

        # Bounding box
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Candle number
        cv2.putText(
            annotated,
            str(index),
            (
                x,
                max(
                    15,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

        # Confidence
        confidence_text = (
            f"{candle['confidence']:.0f}%"
        )

        cv2.putText(
            annotated,
            confidence_text,
            (
                x,
                min(
                    image.shape[0] - 5,
                    y + h + 14
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

    # ========================================================
    # MISSING-CANDLE MARKERS
    #
    # Draw ONLY after all candle boxes are complete.
    # ========================================================

    if spacing_analysis:

        for candidate in spacing_analysis.get(
            "possible_missing",
            []
        ):

            estimated_x = int(
                candidate["estimated_x"]
            )

            confidence = candidate[
                "confidence"
            ]

            # Yellow
            marker_color = (
                255,
                255,
                0
            )

            # Vertical marker
            cv2.line(
                annotated,
                (
                    estimated_x,
                    0
                ),
                (
                    estimated_x,
                    image.shape[0]
                ),
                marker_color,
                1
            )

            # Label
            label = (
                "POSSIBLE MISSING "
                f"{confidence:.0f}%"
            )

            cv2.putText(
                annotated,
                label,
                (
                    max(
                        5,
                        estimated_x - 80
                    ),
                    18
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                marker_color,
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
# CROP
# ============================================================

st.header("2️⃣ Chart Region")

st.caption(
    "Manually crop out trading controls and indicators "
    "that are not part of the candle chart."
)

with st.expander(
    "⚙️ Adjust chart crop",
    expanded=False
):

    left = st.slider(
        "Left",
        0,
        w - 1,
        min(
            34,
            w - 1
        )
    )

    right = st.slider(
        "Right",
        1,
        w,
        min(
            1164,
            w
        )
    )

    top = st.slider(
        "Top",
        0,
        h - 1,
        min(
            122,
            h - 1
        )
    )

    bottom = st.slider(
        "Bottom",
        1,
        h,
        min(
            734,
            h
        )
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
        all_candles
    ) = detect_candles(
        chart
    )

    spacing_analysis = (
        analyze_spacing(candles)
    )

    # ========================================================
    # V2.4 SEQUENCE VALIDATION
    # ========================================================

    sequence_analysis = (
        analyze_candle_sequence(candles)
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    st.session_state["candles"] = candles

    st.session_state["all_candles"] = all_candles

    st.session_state["green_mask"] = green_mask

    st.session_state["red_mask"] = red_mask

    st.session_state["components"] = components

    st.session_state["spacing_analysis"] = (
        spacing_analysis
    )

    st.session_state["sequence_analysis"] = (
        sequence_analysis
    )

# ============================================================
# RESULTS
# ============================================================

if "candles" in st.session_state:

    candles = st.session_state[
        "candles"
    ]

    all_candles = st.session_state[
        "all_candles"
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

    spacing_analysis = st.session_state[
        "spacing_analysis"
    ]


    sequence_analysis = st.session_state[
        "sequence_analysis"
    ]
    
    # ========================================================
    # DETECTION RESULT
    # ========================================================

    st.header(
        "4️⃣ Detection Result"
    )

    green_count = sum(
        c["color"] == "GREEN"
        for c in candles
    )

    red_count = sum(
        c["color"] == "RED"
        for c in candles
    )

    rejected_count = sum(
        c["validation"] == "Rejected"
        for c in all_candles
    )

    possible_missing_count = len(
        spacing_analysis[
            "possible_missing"
        ]
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
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

    col5.metric(
        "Possible Missing",
        possible_missing_count
    )

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    st.header(
        "5️⃣ What the Computer Thinks Are Candles"
    )

    annotated = annotate_candles(
        chart,
        candles,
        spacing_analysis
    )

    st.image(
        annotated,
        use_container_width=True
    )

    st.caption(
        "Green/red boxes are detected candles. "
        "Confidence is shown below each candle. "
        "Yellow lines indicate high-confidence possible "
        "missing candles based on gap periodicity, "
        "neighbor spacing, and candle-width consistency."
    )

    # ========================================================
    # MASKS
    # ========================================================

    st.header(
        "6️⃣ Color Segmentation"
    )

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

    st.header(
        "7️⃣ Reconstructed Candle Data"
    )

    if candles:

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

        # Round numerical values
        for column in [
            "body_ratio",
            "wick_ratio",
            "aspect_ratio",
            "color_confidence",
            "geometry_score",
            "detection_score",
            "structure_score",
            "confidence"
        ]:

            if column in df.columns:

                df[column] = df[
                    column
                ].round(2)

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
            "No candles survived the validation filters."
        )

    # ========================================================
    # SPACING ANALYSIS
    # ========================================================

    st.header(
        "8️⃣ Candle Spacing"
    )

    if spacing_analysis[
        "median"
    ] is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Median Spacing",
            f"{spacing_analysis['median']:.2f} px"
        )

        col2.metric(
            "Minimum Spacing",
            f"{spacing_analysis['minimum']:.2f} px"
        )

        col3.metric(
            "Maximum Spacing",
            f"{spacing_analysis['maximum']:.2f} px"
        )

        spacing_df = pd.DataFrame(
            spacing_analysis[
                "rows"
            ]
        )

        st.dataframe(
            spacing_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Missing candidates
        # ----------------------------------------------------

        if spacing_analysis[
            "possible_missing"
        ]:

            st.warning(
                "Strong spacing anomalies detected. "
                "These are possible missing candles, "
                "not confirmed missing candles."
            )

            missing_df = pd.DataFrame(
                spacing_analysis[
                    "possible_missing"
                ]
            )

            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No strong evidence of missing candles "
                "was found in the detected sequence."
            )

    else:

        st.info(
            "At least three candles are required "
            "for spacing analysis."
        )


    # ========================================================
    # QUALITY SUMMARY
    # ========================================================

    st.header(
        "9️⃣ Detection Quality"
    )

    if candles:

        confidence = np.array([
            c["confidence"]
            for c in candles
        ])

        geometry = np.array([
            c["geometry_score"]
            for c in candles
        ])

        color_scores = np.array([
            c["color_confidence"]
            for c in candles
        ])

        structure = np.array([
            c["structure_score"]
            for c in candles
        ])

        average_confidence = np.mean(
            confidence
        )

        average_geometry = np.mean(
            geometry
        )

        average_color = np.mean(
            color_scores
        )

        average_structure = np.mean(
            structure
        )

        high_confidence = np.sum(
            confidence >= 80
        )

        review_count = np.sum(
            (confidence >= 60)
            & (confidence < 80)
        )

        low_confidence = np.sum(
            confidence < 60
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Average Confidence",
            f"{average_confidence:.1f}%"
        )

        col2.metric(
            "Average Geometry",
            f"{average_geometry:.1f}%"
        )

        col3.metric(
            "Average Structure",
            f"{average_structure:.1f}%"
        )

        col4.metric(
            "High Confidence",
            int(high_confidence)
        )

        st.write(
            f"**Average colour confidence:** "
            f"{average_color:.1f}%"
        )

        st.write(
            f"**Review candidates:** "
            f"{int(review_count)}"
        )

        st.write(
            f"**Low-confidence candidates:** "
            f"{int(low_confidence)}"
        )

        if average_confidence >= 85:

            st.success(
                "Detection quality is strong. "
                "Continue validating against additional screenshots."
            )

        elif average_confidence >= 70:

            st.warning(
                "Detection is usable for diagnostics, "
                "but still requires validation."
            )

        else:

            st.error(
                "Detection quality is weak. "
                "Do NOT proceed to signal generation."
            )

    else:

        st.error(
            "No accepted candles are available "
            "for quality analysis."
        )

    # ============================================================
    # SEQUENCE VALIDATION
    # ============================================================
    
    st.header(
        "1️⃣1️⃣ Candle Sequence Validation"
    )
    
    if "sequence_analysis" in st.session_state:
    
        sequence = st.session_state[
            "sequence_analysis"
        ]
    
        # --------------------------------------------------------
        # TOP METRICS
        # --------------------------------------------------------
    
        col1, col2, col3, col4 = st.columns(4)
    
        col1.metric(
            "Sequence Integrity",
            f"{sequence['sequence_integrity']:.1f}%"
        )
    
        col2.metric(
            "OHLC Validity",
            f"{sequence['ohlc_validity']:.1f}%"
        )
    
        col3.metric(
            "Spacing Consistency",
            f"{sequence['spacing_consistency']:.1f}%"
        )
    
        col4.metric(
            "Duplicate Centres",
            sequence["duplicate_centers"]
        )
    
        st.divider()
    
        # --------------------------------------------------------
        # STRUCTURE
        # --------------------------------------------------------
    
        st.subheader(
            "Market Structure Diagnostic"
        )
    
        col1, col2, col3, col4 = st.columns(4)
    
        col1.metric(
            "Higher Highs",
            sequence["higher_highs"]
        )
    
        col2.metric(
            "Higher Lows",
            sequence["higher_lows"]
        )
    
        col3.metric(
            "Lower Highs",
            sequence["lower_highs"]
        )
    
        col4.metric(
            "Lower Lows",
            sequence["lower_lows"]
        )
    
        st.write(
            f"**Current Trend:** "
            f"`{sequence['trend']}`"
        )
    
        st.write(
            f"**Current Structure:** "
            f"`{sequence['current_structure']}`"
        )

        st.subheader("Swing Structure Diagnostic")

        swing_highs = sequence_analysis.get(
            "swing_highs",
            []
        )
        
        swing_lows = sequence_analysis.get(
            "swing_lows",
            []
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Swing Highs",
                len(swing_highs)
            )
        
        with col2:
            st.metric(
                "Swing Lows",
                len(swing_lows)
            )
        
        if swing_highs:
            swing_high_df = pd.DataFrame(
                swing_highs
            )
        
            st.write("Detected Swing Highs")
            st.dataframe(
                swing_high_df,
                use_container_width=True,
                hide_index=True
            )
        
        if swing_lows:
            swing_low_df = pd.DataFrame(
                swing_lows
            )
        
            st.write("Detected Swing Lows")
            st.dataframe(
                swing_low_df,
                use_container_width=True,
                hide_index=True
            )
        
                # ========================================================
        # BOS / CHoCH DIAGNOSTIC
        # ========================================================

        st.subheader(
            "BOS / CHoCH Diagnostic"
        )

        bos_events = sequence.get(
            "bos_choch_events",
            []
        )

        bos_bias = sequence.get(
            "bos_choch_bias",
            "UNKNOWN"
        )

        last_event = sequence.get(
            "last_bos_choch",
            None
        )

        # --------------------------------------------------------
        # CURRENT STRUCTURAL BIAS
        # --------------------------------------------------------

        st.write(
            f"**Structural Bias:** `{bos_bias}`"
        )

        # --------------------------------------------------------
        # LAST EVENT
        # --------------------------------------------------------

        if last_event:

            st.write(
                f"**Latest Event:** "
                f"`{last_event['event']}`"
            )

            st.write(
                f"**Candle Index:** "
                f"`{last_event['candle_index']}`"
            )

            st.write(
                f"**Price Coordinate:** "
                f"`{last_event['price']}`"
            )

        else:

            st.info(
                "No BOS or CHoCH detected yet."
            )

        # --------------------------------------------------------
        # EVENT TABLE
        # --------------------------------------------------------

        if bos_events:

            bos_df = pd.DataFrame(
                bos_events
            )

            st.write(
                "Detected BOS / CHoCH Events"
            )

            st.dataframe(
                bos_df,
                use_container_width=True,
                hide_index=True
            )
        
        # --------------------------------------------------------
        # CURRENT CANDLE
        # --------------------------------------------------------
    
        st.subheader(
            "Current Candle"
        )
    
        col1, col2, col3, col4 = st.columns(4)
    
        col1.metric(
            "Direction",
            sequence["current_direction"]
        )
    
        col2.metric(
            "Body",
            f"{sequence['body_percentage']:.1f}%"
        )
    
        col3.metric(
            "Upper Wick",
            f"{sequence['upper_wick_percentage']:.1f}%"
        )
    
        col4.metric(
            "Lower Wick",
            f"{sequence['lower_wick_percentage']:.1f}%"
        )
    
        st.write(
            f"**Detection confidence:** "
            f"{sequence['current_confidence']:.1f}%"
        )

        # --------------------------------------------------------
        # INTERPRETATION
        # --------------------------------------------------------
    
        if sequence["sequence_integrity"] >= 95:
    
            st.success(
                "The reconstructed candle sequence "
                "is internally consistent."
            )
    
        elif sequence["sequence_integrity"] >= 85:
    
            st.warning(
                "The sequence is usable, but some "
                "structural inconsistencies remain."
            )
    
        else:
    
            st.error(
                "The sequence is not reliable enough "
                "for predictive analysis."
            )
    # ========================================================
    # INTERPRETATION GUIDE
    # ========================================================

    st.header(
        "🔟 How to Read the Scores"
    )

    st.markdown(
        """
**Geometry Score**

Measures whether the detected object has a
plausible candle-like shape.

**Colour Confidence**

Measures how strongly the detected pixels support
the assigned RED or GREEN classification.

**Structure Score**

Measures how well the candle fits the surrounding
candle sequence in terms of size and spacing.

**Final Confidence**

Weighted combination of:

- Geometry: 40%
- Colour: 25%
- Structure: 25%
- Detection support: 10%

**Spacing**

- **Normal:** spacing is consistent with the local sequence.
- **Suspicious Gap:** spacing is unusual but not strong enough
  to claim a missing candle.
- **Possible Missing Candle:** a large gap is supported by
  normal neighbouring spacing.

A "Possible Missing Candle" is still a hypothesis.
It is NOT treated as an actual candle.
"""
    )


    # ========================================================
    # IMPORTANT
    # ========================================================

    st.warning(
        """
IMPORTANT:

This version does NOT generate BUY or SELL signals.

The purpose of V2.3 is still to determine whether
a screenshot can be converted into a reliable candle
sequence.

Do not use the output as a trading signal yet.

The extraction layer must be validated against
multiple independent screenshots before we build
any trading logic on top of it.
"""
    )
