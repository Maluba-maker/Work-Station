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
    "• Structural Validation • BUY / SELL SIGNAL ENGINE"
)

# ============================================================
# TOP TRADE SETUP / SIGNAL DISPLAY
# ============================================================
# The diagnostic is calculated later, after the chart and
# structure engine have run, but rendered here so it appears
# at the top of the application.
trade_setup_placeholder = st.empty()


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

            "bos_choch_bias": "UNKNOWN",
            "transition_bias": "NONE",
            "last_bos_choch": None,
            "structural_bias": "UNKNOWN",
            "structural_sequence": "NO COMPLETE NEW SEQUENCE",
            "structural_sequence_index": None,

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

    # ========================================================
    # CURRENT STRUCTURAL BIAS
    # ========================================================

    current_structure_analysis = derive_current_structural_bias(
        swing_highs,
        swing_lows,
        bos_choch_bias,
        last_bos_choch
    )

    structural_bias = current_structure_analysis["bias"]
    structural_sequence = current_structure_analysis["sequence"]
    structural_sequence_index = current_structure_analysis["sequence_index"]

    print("\n")
    print("=" * 70)
    print("CURRENT STRUCTURAL BIAS")
    print("=" * 70)
    print("Confirmed BOS bias:", bos_choch_bias)
    print(
        "Transition bias:",
        bos_choch_analysis.get("transition_bias", "NONE")
    )
    print("Last event:", last_bos_choch)
    print("Current sequence:", structural_sequence)
    print("Sequence index:", structural_sequence_index)
    print("Final current structural bias:", structural_bias)
    print("=" * 70)
    print("\n")

    # ========================================================
    # STEP 9 — STRUCTURE VALIDATION
    # ========================================================
    
    structure_validation = validate_structure(
        candles,
        swing_highs,
        swing_lows,
        bos_choch_events,
        bos_choch_bias
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

        "swing_current_structure":
            swing_current_structure,

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

        "transition_bias":
            bos_choch_analysis.get(
                "transition_bias",
                "NONE"
            ),

        "last_bos_choch":
            last_bos_choch,

        # Current structural state is based on the most recent
        # completed HH->HL / LL->LH sequence, not a lifetime
        # count of all historical swings.
        "structural_bias":
            structural_bias,

        "structural_sequence":
            structural_sequence,

        "structural_sequence_index":
            structural_sequence_index,

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
            ),
        "structure_validation":
            structure_validation
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

# Minimum pixel movement required before two swing points
# are considered structurally different.
#
# IMPORTANT:
# These are screenshot pixel coordinates.
# Smaller Y = higher market price.
# Larger Y = lower market price.
#
# A difference smaller than this is treated as essentially
# equal so screenshot noise does not create fake HH/LH/HL/LL.
#
STRUCTURE_MIN_DISTANCE = 5.0


def classify_swing_structure(
    swing_highs,
    swing_lows
):

    """
    Classify detected swing points as:

        HH = Higher High
        LH = Lower High
        HL = Higher Low
        LL = Lower Low
        EQUAL HIGH
        EQUAL LOW

    Pixel coordinates:

        Smaller Y = higher market price
        Larger Y = lower market price

    IMPORTANT:

    A small pixel difference is NOT automatically treated
    as meaningful market structure.

    Example:

        Previous high = 504
        Latest high   = 503

        Difference = 1 pixel

    With STRUCTURE_MIN_DISTANCE = 5:

        Result = EQUAL HIGH

    But:

        Previous high = 504
        Latest high   = 457

        Difference = 47 pixels

        Result = HH
    """

    # ========================================================
    # COPY ORIGINAL SWING DATA
    # ========================================================

    classified_highs = []

    classified_lows = []

    # ========================================================
    # CLASSIFY SWING HIGHS
    # ========================================================

    for i, swing in enumerate(
        swing_highs
    ):

        item = swing.copy()

        # ----------------------------------------------------
        # FIRST SWING HIGH
        # ----------------------------------------------------

        if i == 0:

            item["structure"] = (
                "INITIAL HIGH"
            )

        else:

            previous = swing_highs[
                i - 1
            ]

            current_price = float(
                swing["price"]
            )

            previous_price = float(
                previous["price"]
            )

            difference = (
                current_price
                - previous_price
            )

            # ------------------------------------------------
            # Smaller Y = higher market price
            #
            # Therefore:
            #
            # current 503
            # previous 504
            #
            # means price moved HIGHER.
            #
            # ------------------------------------------------

            if (
                difference < 0
                and
                abs(difference)
                >= STRUCTURE_MIN_DISTANCE
            ):

                item["structure"] = "HH"

            elif (
                difference > 0
                and
                abs(difference)
                >= STRUCTURE_MIN_DISTANCE
            ):

                item["structure"] = "LH"

            else:

                item["structure"] = (
                    "EQUAL HIGH"
                )

            # Store the actual pixel difference
            item["structure_distance"] = round(
                abs(difference),
                1
            )

        classified_highs.append(
            item
        )

    # ========================================================
    # CLASSIFY SWING LOWS
    # ========================================================

    for i, swing in enumerate(
        swing_lows
    ):

        item = swing.copy()

        # ----------------------------------------------------
        # FIRST SWING LOW
        # ----------------------------------------------------

        if i == 0:

            item["structure"] = (
                "INITIAL LOW"
            )

        else:

            previous = swing_lows[
                i - 1
            ]

            current_price = float(
                swing["price"]
            )

            previous_price = float(
                previous["price"]
            )

            difference = (
                current_price
                - previous_price
            )

            # ------------------------------------------------
            # Larger Y = lower market price
            #
            # Therefore:
            #
            # current 527
            # previous 654
            #
            # means price moved HIGHER.
            #
            # ------------------------------------------------

            if (
                difference < 0
                and
                abs(difference)
                >= STRUCTURE_MIN_DISTANCE
            ):

                item["structure"] = "HL"

            elif (
                difference > 0
                and
                abs(difference)
                >= STRUCTURE_MIN_DISTANCE
            ):

                item["structure"] = "LL"

            else:

                item["structure"] = (
                    "EQUAL LOW"
                )

            # Store the actual pixel difference
            item["structure_distance"] = round(
                abs(difference),
                1
            )

        classified_lows.append(
            item
        )

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
        higher_highs
        + higher_lows
    )

    bearish_score = (
        lower_highs
        + lower_lows
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

        previous_high = (
            classified_highs[-2]
        )

        latest_high = (
            classified_highs[-1]
        )

    else:

        previous_high = None
        latest_high = None

    if len(classified_lows) >= 2:

        previous_low = (
            classified_lows[-2]
        )

        latest_low = (
            classified_lows[-1]
        )

    else:

        previous_low = None
        latest_low = None

    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return {

        "swing_highs":
            classified_highs,

        "swing_lows":
            classified_lows,

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

        "previous_high":
            previous_high,

        "latest_high":
            latest_high,

        "previous_low":
            previous_low,

        "latest_low":
            latest_low,

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
# CURRENT STRUCTURAL BIAS
# ============================================================

def derive_current_structural_bias(
    swing_highs,
    swing_lows,
    bos_choch_bias="UNKNOWN",
    last_event=None
):
    """
    Derive the CURRENT market-structure bias from the most
    recent completed swing sequence.

    IMPORTANT:
        Do not determine the current bias by counting every
        HH/HL/LH/LL in the entire chart. Older structure must
        not outweigh the most recent confirmed sequence.

    A bullish sequence requires:
        HH -> HL

    A bearish sequence requires:
        LL -> LH

    If both exist, the sequence that completed most recently
    wins. A later completed swing sequence can therefore update
    a stale BOS/CHoCH state without allowing a single candle
    to flip the bias.
    """

    candidates = []

    highs = [
        s for s in swing_highs
        if s.get("structure") in ("HH", "LH")
    ]

    lows = [
        s for s in swing_lows
        if s.get("structure") in ("HL", "LL")
    ]

    # --------------------------------------------------------
    # Bullish completed sequence: HH followed by HL
    # --------------------------------------------------------

    for high in highs:

        if high.get("structure") != "HH":
            continue

        later_lows = [
            low for low in lows
            if low["index"] > high["index"]
        ]

        if later_lows:

            # Pair the HH with the FIRST confirmed low that
            # follows it. Do not skip over later swings.
            low = later_lows[0]

            if low.get("structure") == "HL":

                candidates.append({
                    "bias": "BULLISH",
                    "index": int(low["index"]),
                    "reference_index": int(high["index"]),
                    "sequence": "HH -> HL"
                })

    # --------------------------------------------------------
    # Bearish completed sequence: LL followed by LH
    # --------------------------------------------------------

    for low in lows:

        if low.get("structure") != "LL":
            continue

        later_highs = [
            high for high in highs
            if high["index"] > low["index"]
        ]

        if later_highs:

            # Pair the LL with the FIRST confirmed high that
            # follows it. Do not skip over later swings.
            high = later_highs[0]

            if high.get("structure") == "LH":

                candidates.append({
                    "bias": "BEARISH",
                    "index": int(high["index"]),
                    "reference_index": int(low["index"]),
                    "sequence": "LL -> LH"
                })

    # --------------------------------------------------------
    # No complete sequence
    # --------------------------------------------------------

    if not candidates:

        return {
            "bias": str(bos_choch_bias).upper(),
            "sequence": "NO COMPLETE NEW SEQUENCE",
            "sequence_index": None
        }

    # --------------------------------------------------------
    # MOST RECENT COMPLETED SEQUENCE WINS
    # --------------------------------------------------------

    latest = max(
        candidates,
        key=lambda item: item["index"]
    )

    sequence_index = latest["index"]

    # A completed swing sequence that formed after the last
    # BOS/CHoCH event represents newer structural information.
    if last_event is not None:

        event_index = int(
            last_event.get("candle_index", -1)
        )

        if sequence_index > event_index:

            return {
                "bias": latest["bias"],
                "sequence": latest["sequence"],
                "sequence_index": sequence_index
            }

    # Otherwise retain the event engine's confirmed bias when
    # it already has one.
    if str(bos_choch_bias).upper() in (
        "BULLISH",
        "BEARISH"
    ):

        return {
            "bias": str(bos_choch_bias).upper(),
            "sequence": latest["sequence"],
            "sequence_index": sequence_index
        }

    return {
        "bias": latest["bias"],
        "sequence": latest["sequence"],
        "sequence_index": sequence_index
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
    BOS / CHoCH STRUCTURAL ENGINE - VERSION 6

    Pixel coordinates:
        Smaller Y = higher market price
        Larger Y = lower market price

    CORE STRUCTURAL MODEL
    ---------------------

    BULLISH STRUCTURE

        HH
         ↓
        HL
         ↓
    close above HH
         ↓
    BULLISH BOS

    close below protected HL
         ↓
    BEARISH CHoCH


    BEARISH STRUCTURE

        LL
         ↓
        LH
         ↓
    close below LL
         ↓
    BEARISH BOS

    close above protected LH
         ↓
    BULLISH CHoCH


    VERSION 6 IMPROVEMENTS
    ----------------------

    1. HH/LL alone never creates BOS.

    2. BOS requires a completed retracement:
           HH -> HL -> break HH
           LL -> LH -> break LL

    3. CHoCH requires breaking the currently
       protected opposite-side swing.

    4. HL only becomes protected when it occurs
       AFTER a bullish HH reference.

    5. LH only becomes protected when it occurs
       AFTER a bearish LL reference.

    6. Equal highs/lows are ignored.

    7. A structural level can trigger only once.

    8. The current candle is allowed to trigger
       a BOS/CHoCH.

    9. Events contain the actual broken level
       and break distance.

    10. Structural state is processed strictly
        from left to right.
    """

    # ========================================================
    # RESULT
    # ========================================================

    events = []

    # ========================================================
    # BUILD COMBINED SWING LIST
    # ========================================================

    all_swings = []

    for swing in swing_highs:

        structure = swing.get(
            "structure",
            "UNKNOWN"
        )

        # Ignore unreliable classifications
        if structure in (
            "EQUAL HIGH",
            "UNKNOWN"
        ):
            continue

        all_swings.append({
            "index": int(
                swing["index"]
            ),
            "price": float(
                swing["price"]
            ),
            "x": swing.get("x"),
            "type": "HIGH",
            "structure": structure
        })

    for swing in swing_lows:

        structure = swing.get(
            "structure",
            "UNKNOWN"
        )

        # Ignore unreliable classifications
        if structure in (
            "EQUAL LOW",
            "UNKNOWN"
        ):
            continue

        all_swings.append({
            "index": int(
                swing["index"]
            ),
            "price": float(
                swing["price"]
            ),
            "x": swing.get("x"),
            "type": "LOW",
            "structure": structure
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
    # DETERMINE INITIAL STRUCTURAL BIAS
    #
    # We look for an actual structural sequence.
    #
    # Bullish:
    #     HH -> HL
    #
    # Bearish:
    #     LL -> LH
    #
    # We do NOT simply use the first HH or LL.
    # ========================================================

    bias = "UNKNOWN"

    first_hh = None
    first_ll = None

    for swing in all_swings:

        structure = swing["structure"]

        # ----------------------------------------------------
        # Bullish structure
        # ----------------------------------------------------

        if structure == "HH":

            first_hh = swing

        elif (
            structure == "HL"
            and
            first_hh is not None
            and
            swing["index"] > first_hh["index"]
        ):

            bias = "BULLISH"
            break

        # ----------------------------------------------------
        # Bearish structure
        # ----------------------------------------------------

        if structure == "LL":

            first_ll = swing

        elif (
            structure == "LH"
            and
            first_ll is not None
            and
            swing["index"] > first_ll["index"]
        ):

            bias = "BEARISH"
            break

    # ========================================================
    # FALLBACK
    #
    # Used only when a complete HH+HL or LL+LH sequence
    # cannot be found.
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

    # Preserve the initial confirmed structural direction.
    # The working "bias" variable may later flip internally
    # after a CHoCH, but that flip is only a transition until
    # a BOS confirms the new direction.
    initial_confirmed_bias = bias

    # ========================================================
    # STRUCTURAL STATE
    # ========================================================

    bullish_reference_high = None
    bullish_retracement_low = None

    bearish_reference_low = None
    bearish_retracement_high = None

    protected_high = None
    protected_low = None

    # ========================================================
    # PREVENT REPEATED BREAKS
    # ========================================================

    broken_highs = set()
    broken_lows = set()

    # ========================================================
    # EVENT CREATOR
    # ========================================================

    def add_event(
        candle_index,
        price,
        event,
        direction,
        swing_type,
        level_type,
        level_index,
        level_price
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
                int(level_index),

            "level_price":
                float(level_price),

            "break_distance":
                round(
                    abs(
                        float(price)
                        -
                        float(level_price)
                    ),
                    1
                )
        })

    # ========================================================
    # PROCESS CANDLES LEFT -> RIGHT
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
        # REGISTER ONLY CONFIRMED SWINGS
        #
        # A swing on the current candle cannot be used to
        # create an event on that same candle.
        # ====================================================

        while (
            swing_pointer
            <
            len(all_swings)
            and
            all_swings[
                swing_pointer
            ]["index"]
            <
            candle_index
        ):

            swing = all_swings[
                swing_pointer
            ]

            swing_type = swing[
                "type"
            ]

            structure = swing[
                "structure"
            ]

            # =================================================
            # HIGH SWINGS
            # =================================================

            if swing_type == "HIGH":

                # ---------------------------------------------
                # BULLISH REFERENCE HIGH
                # ---------------------------------------------

                if structure == "HH":

                    bullish_reference_high = (
                        swing
                    )

                    # A new HH starts a new bullish
                    # continuation attempt.
                    bullish_retracement_low = None

                # ---------------------------------------------
                # BEARISH RETRACEMENT HIGH
                # ---------------------------------------------

                elif structure == "LH":

                    # LH can only become a bearish
                    # retracement after a LL.
                    if (
                        bearish_reference_low
                        is not None
                        and
                        swing["index"]
                        >
                        bearish_reference_low["index"]
                    ):

                        bearish_retracement_high = (
                            swing
                        )

                        # During bearish structure this
                        # becomes the protected high.
                        if bias == "BEARISH":

                            protected_high = (
                                swing
                            )

            # =================================================
            # LOW SWINGS
            # =================================================

            elif swing_type == "LOW":

                # ---------------------------------------------
                # BEARISH REFERENCE LOW
                # ---------------------------------------------

                if structure == "LL":

                    bearish_reference_low = (
                        swing
                    )

                    # A new LL starts a new bearish
                    # continuation attempt.
                    bearish_retracement_high = None

                # ---------------------------------------------
                # BULLISH RETRACEMENT LOW
                # ---------------------------------------------

                elif structure == "HL":

                    # HL can only become a bullish
                    # retracement after an HH.
                    if (
                        bullish_reference_high
                        is not None
                        and
                        swing["index"]
                        >
                        bullish_reference_high["index"]
                    ):

                        bullish_retracement_low = (
                            swing
                        )

                        # During bullish structure this
                        # becomes the protected low.
                        if bias == "BULLISH":

                            protected_low = (
                                swing
                            )

            swing_pointer += 1

        # ====================================================
        # BULLISH STRUCTURE
        # ====================================================

        if bias == "BULLISH":

            # =================================================
            # 1. BEARISH CHoCH
            #
            # Price closes below protected HL.
            #
            # Pixel rule:
            # larger Y = lower market price
            # =================================================

            if protected_low is not None:

                level_index = int(
                    protected_low["index"]
                )

                level_price = float(
                    protected_low["price"]
                )

                broke_protected_low = (
                    close_price
                    >
                    level_price
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
                        level_index,
                        level_price
                    )

                    broken_lows.add(
                        level_index
                    )

                    # -----------------------------------------
                    # STRUCTURAL REVERSAL
                    # -----------------------------------------

                    bias = "BEARISH"

                    protected_low = None

                    bullish_reference_high = None
                    bullish_retracement_low = None

                    # Do NOT manufacture a bearish BOS.
                    # We now wait for a genuine:
                    #
                    # LL -> LH -> break LL
                    #
                    continue

            # =================================================
            # 2. BULLISH BOS
            #
            # REQUIRE:
            #
            # HH
            # ↓
            # HL
            # ↓
            # CLOSE ABOVE HH
            #
            # Pixel:
            # smaller Y = higher market price
            # =================================================

            if (
                bullish_reference_high
                is not None
                and
                bullish_retracement_low
                is not None
                and
                bullish_retracement_low["index"]
                >
                bullish_reference_high["index"]
            ):

                level_index = int(
                    bullish_reference_high["index"]
                )

                level_price = float(
                    bullish_reference_high["price"]
                )

                broke_reference_high = (
                    close_price
                    <
                    level_price
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
                        level_index,
                        level_price
                    )

                    broken_highs.add(
                        level_index
                    )

                    # -----------------------------------------
                    # HH consumed.
                    #
                    # HL remains the protected low.
                    # -----------------------------------------

                    bullish_reference_high = None

                    protected_low = (
                        bullish_retracement_low
                    )

                    continue

        # ====================================================
        # BEARISH STRUCTURE
        # ====================================================

        elif bias == "BEARISH":

            # =================================================
            # 1. BULLISH CHoCH
            #
            # Price closes above protected LH.
            #
            # Pixel rule:
            # smaller Y = higher market price
            # =================================================

            if protected_high is not None:

                level_index = int(
                    protected_high["index"]
                )

                level_price = float(
                    protected_high["price"]
                )

                broke_protected_high = (
                    close_price
                    <
                    level_price
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
                        level_index,
                        level_price
                    )

                    broken_highs.add(
                        level_index
                    )

                    # -----------------------------------------
                    # STRUCTURAL REVERSAL
                    # -----------------------------------------

                    bias = "BULLISH"

                    protected_high = None

                    bearish_reference_low = None
                    bearish_retracement_high = None

                    # Do NOT manufacture a bullish BOS.
                    #
                    # We now wait for:
                    #
                    # HH -> HL -> break HH
                    #
                    continue

            # =================================================
            # 2. BEARISH BOS
            #
            # REQUIRE:
            #
            # LL
            # ↓
            # LH
            # ↓
            # CLOSE BELOW LL
            #
            # Pixel:
            # larger Y = lower market price
            # =================================================

            if (
                bearish_reference_low
                is not None
                and
                bearish_retracement_high
                is not None
                and
                bearish_retracement_high["index"]
                >
                bearish_reference_low["index"]
            ):

                level_index = int(
                    bearish_reference_low["index"]
                )

                level_price = float(
                    bearish_reference_low["price"]
                )

                broke_reference_low = (
                    close_price
                    >
                    level_price
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
                        level_index,
                        level_price
                    )

                    broken_lows.add(
                        level_index
                    )

                    # -----------------------------------------
                    # LL consumed.
                    #
                    # LH becomes the protected high.
                    # -----------------------------------------

                    bearish_reference_low = None

                    protected_high = (
                        bearish_retracement_high
                    )

                    continue

    # ========================================================
    # SORT EVENTS
    # ========================================================

    events.sort(
        key=lambda event:
        event["candle_index"]
    )

    # ========================================================
    # LAST EVENT
    # ========================================================

    last_event = (
        events[-1]
        if events
        else None
    )

    # ========================================================
    # FINAL BIAS
    # ========================================================

    if last_event is not None:

        bias = last_event[
            "direction"
        ]

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BOS / CHoCH STRUCTURAL ANALYSIS V6")
    print("=" * 70)

    print(
        "Final bias:",
        bias
    )

    print(
        "Total events:",
        len(events)
    )

    print("-" * 70)

    for event in events:

        print(
            f"Candle {event['candle_index']} | "
            f"{event['event']} | "
            f"Level {event['level_index']} | "
            f"Level price {event['level_price']} | "
            f"Break distance {event['break_distance']}"
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
# STEP 9 — STRUCTURE VALIDATION
# ============================================================

def validate_structure(
    candles,
    swing_highs,
    swing_lows,
    bos_events,
    bos_bias
):
    """
    STRUCTURE VALIDATION ENGINE - STEP 9

    Purpose:
        Validate the structure produced by the swing
        and BOS / CHoCH engines.

    IMPORTANT:
        Candle/swing coordinates are PIXEL coordinates.

        Smaller Y = higher market price
        Larger Y = lower market price

    This function does NOT create new structure.
    It only checks whether the existing structure
    is internally consistent.
    """

    checks = []
    score_components = []

    # ========================================================
    # BASIC DATA CHECK
    # ========================================================

    if not candles:

        return {
            "validation_score": 0.0,
            "status": "NO DATA",
            "checks": [],
            "swing_high_validity": 0.0,
            "swing_low_validity": 0.0,
            "event_validity": 0.0,
            "protected_level_validity": 0.0,
            "bias_consistency": 0.0,
            "ready_for_next_step": False
        }

    # ========================================================
    # HELPER
    # ========================================================

    def add_check(
        name,
        passed,
        detail
    ):

        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail
        })

        score_components.append(
            100.0 if passed else 0.0
        )

    # ========================================================
    # 1. SWING HIGH VALIDATION
    # ========================================================

    high_checks = []

    previous_high = None

    for swing in swing_highs:

        structure = swing.get(
            "structure",
            "UNKNOWN"
        )

        if previous_high is None:

            previous_high = swing
            continue

        current_y = float(
            swing["price"]
        )

        previous_y = float(
            previous_high["price"]
        )

        valid = True

        # ----------------------------------------------------
        # HIGH STRUCTURE
        # ----------------------------------------------------

        if structure == "HH":

            # Smaller Y = higher price
            valid = current_y < previous_y

        elif structure == "LH":

            # Larger Y = lower price
            valid = current_y > previous_y

        elif structure == "EQUAL HIGH":

            valid = abs(
                current_y - previous_y
            ) <= 3.0

        # UNKNOWN is not automatically a failure
        elif structure == "UNKNOWN":

            valid = True

        high_checks.append(valid)

        previous_high = swing

    if high_checks:

        high_validity = (
            sum(high_checks)
            / len(high_checks)
            * 100
        )

    else:

        high_validity = 100.0

    add_check(
        "Swing High Classification",
        high_validity >= 90,
        f"{high_validity:.1f}% of classified "
        "swing highs are geometrically consistent."
    )

    # ========================================================
    # 2. SWING LOW VALIDATION
    # ========================================================

    low_checks = []

    previous_low = None

    for swing in swing_lows:

        structure = swing.get(
            "structure",
            "UNKNOWN"
        )

        if previous_low is None:

            previous_low = swing
            continue

        current_y = float(
            swing["price"]
        )

        previous_y = float(
            previous_low["price"]
        )

        valid = True

        # ----------------------------------------------------
        # LOW STRUCTURE
        # ----------------------------------------------------

        if structure == "HL":

            # Smaller Y = higher price
            valid = current_y < previous_y

        elif structure == "LL":

            # Larger Y = lower price
            valid = current_y > previous_y

        elif structure == "EQUAL LOW":

            valid = abs(
                current_y - previous_y
            ) <= 3.0

        elif structure == "UNKNOWN":

            valid = True

        low_checks.append(valid)

        previous_low = swing

    if low_checks:

        low_validity = (
            sum(low_checks)
            / len(low_checks)
            * 100
        )

    else:

        low_validity = 100.0

    add_check(
        "Swing Low Classification",
        low_validity >= 90,
        f"{low_validity:.1f}% of classified "
        "swing lows are geometrically consistent."
    )

    # ========================================================
    # 3. BOS / CHoCH EVENT VALIDATION
    # ========================================================

    event_checks = []

    protected_checks = []

    used_levels = set()

    for event in bos_events:

        event_type = event.get(
            "event",
            ""
        )

        direction = event.get(
            "direction",
            ""
        )

        event_price = float(
            event.get(
                "price",
                0
            )
        )

        level_price = float(
            event.get(
                "level_price",
                event_price
            )
        )

        level_index = event.get(
            "level_index"
        )

        level_type = event.get(
            "level_type",
            ""
        )

        break_distance = float(
            event.get(
                "break_distance",
                0
            )
        )

        valid_break = True

        # ----------------------------------------------------
        # BULLISH BREAK
        #
        # Price must move UP through level.
        # Pixel Y therefore becomes smaller.
        # ----------------------------------------------------

        if direction == "BULLISH":

            valid_break = (
                event_price < level_price
                and
                break_distance > 0
            )

        # ----------------------------------------------------
        # BEARISH BREAK
        #
        # Price must move DOWN through level.
        # Pixel Y therefore becomes larger.
        # ----------------------------------------------------

        elif direction == "BEARISH":

            valid_break = (
                event_price > level_price
                and
                break_distance > 0
            )

        event_checks.append(
            valid_break
        )

        # ----------------------------------------------------
        # PROTECTED LEVEL VALIDATION
        # ----------------------------------------------------

        if "CHoCH" in event_type:

            protected_valid = (
                "PROTECTED"
                in level_type.upper()
            )

            protected_checks.append(
                protected_valid
            )

        # ----------------------------------------------------
        # DUPLICATE LEVEL CHECK
        # ----------------------------------------------------

        if level_index is not None:

            if level_index in used_levels:

                valid_break = False

            used_levels.add(
                level_index
            )

    # ========================================================
    # EVENT SCORE
    # ========================================================

    if event_checks:

        event_validity = (
            sum(event_checks)
            / len(event_checks)
            * 100
        )

    else:

        event_validity = 100.0

    add_check(
        "BOS / CHoCH Break Direction",
        event_validity >= 90,
        f"{event_validity:.1f}% of structural "
        "breaks move through their level correctly."
    )

    # ========================================================
    # PROTECTED LEVEL SCORE
    # ========================================================

    if protected_checks:

        protected_validity = (
            sum(protected_checks)
            / len(protected_checks)
            * 100
        )

    else:

        protected_validity = 100.0

    add_check(
        "Protected Level Usage",
        protected_validity >= 90,
        f"{protected_validity:.1f}% of CHoCH "
        "events use a protected structural level."
    )

    # ========================================================
    # 4. BIAS CONSISTENCY
    # ========================================================

    bias_checks = []

    if bos_events:

        for event in bos_events:

            direction = event.get(
                "direction",
                "UNKNOWN"
            )

            event_type = event.get(
                "event",
                ""
            )

            # ------------------------------------------------
            # EVENT TYPE / DIRECTION MUST AGREE
            # ------------------------------------------------

            if "BULLISH" in event_type:

                bias_checks.append(
                    direction == "BULLISH"
                )

            elif "BEARISH" in event_type:

                bias_checks.append(
                    direction == "BEARISH"
                )

    if bias_checks:

        bias_consistency = (
            sum(bias_checks)
            / len(bias_checks)
            * 100
        )

    else:

        bias_consistency = 100.0

    add_check(
        "Bias / Event Consistency",
        bias_consistency >= 90,
        f"{bias_consistency:.1f}% of events "
        "agree with their stated direction."
    )

    # ========================================================
    # 5. LATEST EVENT CONSISTENCY
    # ========================================================

    latest_event_valid = True

    if bos_events:

        latest = bos_events[-1]

        latest_direction = latest.get(
            "direction",
            "UNKNOWN"
        )

        latest_event_valid = (
            latest_direction == bos_bias
        )

    add_check(
        "Latest Event / Bias",
        latest_event_valid,
        "Latest structural event agrees with "
        "the current structural bias."
        if latest_event_valid
        else
        "Latest structural event conflicts with "
        "the current structural bias."
    )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    if score_components:

        validation_score = round(
            float(
                np.mean(
                    score_components
                )
            ),
            1
        )

    else:

        validation_score = 0.0

    # ========================================================
    # FINAL STATUS
    # ========================================================

    if validation_score >= 95:

        status = "STRONG"

    elif validation_score >= 85:

        status = "USABLE"

    elif validation_score >= 70:

        status = "QUESTIONABLE"

    else:

        status = "FAILED"

    # ========================================================
    # NEXT-STEP GATE
    # ========================================================

    ready_for_next_step = (
        validation_score >= 90
        and
        high_validity >= 90
        and
        low_validity >= 90
        and
        event_validity >= 90
        and
        protected_validity >= 90
    )

    return {
        "validation_score":
            validation_score,

        "status":
            status,

        "checks":
            checks,

        "swing_high_validity":
            round(
                high_validity,
                1
            ),

        "swing_low_validity":
            round(
                low_validity,
                1
            ),

        "event_validity":
            round(
                event_validity,
                1
            ),

        "protected_level_validity":
            round(
                protected_validity,
                1
            ),

        "bias_consistency":
            round(
                bias_consistency,
                1
            ),

        "ready_for_next_step":
            ready_for_next_step
    }

# ============================================================
# STRUCTURAL CHART ANNOTATION
# ============================================================

def annotate_candles(
    image,
    candles,
    spacing_analysis=None,
    sequence_analysis=None
):

    annotated = image.copy()

    # ========================================================
    # 1. CANDLE ANNOTATIONS
    # ========================================================

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

        # ----------------------------------------------------
        # Candle bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # ----------------------------------------------------
        # Candle number
        # ----------------------------------------------------

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
    # 2. SWING STRUCTURE
    # ========================================================

    if sequence_analysis:

        swing_highs = sequence_analysis.get(
            "swing_highs",
            []
        )

        swing_lows = sequence_analysis.get(
            "swing_lows",
            []
        )

        # ----------------------------------------------------
        # SWING HIGHS
        # ----------------------------------------------------

        for swing in swing_highs:

            swing_index = int(
                swing["index"]
            )

            if swing_index >= len(candles):
                continue

            candle = candles[
                swing_index
            ]

            x = int(
                candle["x"]
                + candle["width"] / 2
            )

            y = int(
                swing["price"]
            )

            structure = swing.get(
                "structure",
                "UNKNOWN"
            )

            # Ignore unreliable classifications
            if structure in (
                "UNKNOWN",
                "EQUAL HIGH"
            ):
                continue

            # Structure label
            cv2.putText(
                annotated,
                structure,
                (
                    max(5, x - 12),
                    max(18, y - 12)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Small marker
            cv2.circle(
                annotated,
                (x, y),
                4,
                (255, 255, 255),
                -1
            )

        # ----------------------------------------------------
        # SWING LOWS
        # ----------------------------------------------------

        for swing in swing_lows:

            swing_index = int(
                swing["index"]
            )

            if swing_index >= len(candles):
                continue

            candle = candles[
                swing_index
            ]

            x = int(
                candle["x"]
                + candle["width"] / 2
            )

            y = int(
                swing["price"]
            )

            structure = swing.get(
                "structure",
                "UNKNOWN"
            )

            # Ignore unreliable classifications
            if structure in (
                "UNKNOWN",
                "EQUAL LOW"
            ):
                continue

            # Structure label
            cv2.putText(
                annotated,
                structure,
                (
                    max(5, x - 12),
                    min(
                        image.shape[0] - 10,
                        y + 22
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Small marker
            cv2.circle(
                annotated,
                (x, y),
                4,
                (255, 255, 255),
                -1
            )

    # ========================================================
    # 3. BOS / CHoCH LEVELS
    # ========================================================

    if sequence_analysis:

        bos_events = sequence_analysis.get(
            "bos_choch_events",
            []
        )

        for event in bos_events:

            candle_index = int(
                event["candle_index"]
            )

            level_index = int(
                event["level_index"]
            )

            level_price = int(
                event["level_price"]
            )

            event_name = event.get(
                "event",
                "STRUCTURE"
            )

            # ------------------------------------------------
            # Safety checks
            # ------------------------------------------------

            if candle_index >= len(candles):
                continue

            if level_index >= len(candles):
                continue

            # ------------------------------------------------
            # X positions
            # ------------------------------------------------

            event_candle = candles[
                candle_index
            ]

            level_candle = candles[
                level_index
            ]

            event_x = int(
                event_candle["x"]
                + event_candle["width"] / 2
            )

            level_x = int(
                level_candle["x"]
                + level_candle["width"] / 2
            )

            # ------------------------------------------------
            # Event colour
            # ------------------------------------------------

            if event["direction"] == "BULLISH":

                event_color = (
                    0,
                    255,
                    0
                )

            else:

                event_color = (
                    255,
                    80,
                    80
                )

            # ------------------------------------------------
            # Structural level line
            #
            # Draw from the originating swing to the
            # candle that broke the level.
            # ------------------------------------------------

            cv2.line(
                annotated,
                (
                    level_x,
                    level_price
                ),
                (
                    event_x,
                    level_price
                ),
                event_color,
                2
            )

            # ------------------------------------------------
            # Vertical marker at breaking candle
            # ------------------------------------------------

            cv2.line(
                annotated,
                (
                    event_x,
                    max(
                        0,
                        level_price - 15
                    )
                ),
                (
                    event_x,
                    min(
                        image.shape[0] - 1,
                        level_price + 15
                    )
                ),
                event_color,
                2
            )

            # ------------------------------------------------
            # Event label
            # ------------------------------------------------

            label = (
                f"{event_name}"
            )

            label_y = (
                level_price - 20
                if event["direction"] == "BULLISH"
                else
                level_price + 35
            )

            label_y = max(
                18,
                min(
                    image.shape[0] - 8,
                    label_y
                )
            )

            cv2.putText(
                annotated,
                label,
                (
                    max(
                        5,
                        event_x - 45
                    ),
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                event_color,
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Level information
            # ------------------------------------------------

            level_label = (
                f"L{level_index} "
                f"{level_price:.0f}"
            )

            cv2.putText(
                annotated,
                level_label,
                (
                    max(
                        5,
                        level_x
                    ),
                    max(
                        18,
                        level_price - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                event_color,
                1,
                cv2.LINE_AA
            )

    # ========================================================
    # 4. POSSIBLE MISSING CANDLES
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
# SIGNAL / UPLOAD LAYOUT
# ============================================================

top_signal_col, top_upload_col = st.columns(
    [1, 1],
    gap="large"
)

with top_signal_col:

    st.header(
        "1️⃣3️⃣ Trade Setup / Confluence Diagnostic"
    )

    st.subheader("🎯 Signal")

    signal_placeholder = st.empty()
    signal_details_placeholder = st.empty()


with top_upload_col:

    st.header(
        "1️⃣ Upload Chart"
    )

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
        spacing_analysis,
        sequence_analysis
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

        # ========================================================
        # STEP 9 — STRUCTURE VALIDATION
        # ========================================================
        
        st.subheader(
            "9️⃣ Structure Validation"
        )
        
        structure_validation = sequence.get(
            "structure_validation",
            None
        )
        
        if structure_validation:
        
            # ----------------------------------------------------
            # TOP METRICS
            # ----------------------------------------------------
        
            col1, col2, col3, col4 = st.columns(4)
        
            col1.metric(
                "Structure Quality",
                f"{structure_validation['validation_score']:.1f}%"
            )
        
            col2.metric(
                "Swing High Validity",
                f"{structure_validation['swing_high_validity']:.1f}%"
            )
        
            col3.metric(
                "Swing Low Validity",
                f"{structure_validation['swing_low_validity']:.1f}%"
            )
        
            col4.metric(
                "BOS / CHoCH Validity",
                f"{structure_validation['event_validity']:.1f}%"
            )
        
            # ----------------------------------------------------
            # SECOND ROW
            # ----------------------------------------------------
        
            col1, col2, col3 = st.columns(3)
        
            col1.metric(
                "Protected Levels",
                f"{structure_validation['protected_level_validity']:.1f}%"
            )
        
            col2.metric(
                "Bias Consistency",
                f"{structure_validation['bias_consistency']:.1f}%"
            )
        
            col3.metric(
                "Status",
                structure_validation["status"]
            )
        
            st.divider()
        
            # ----------------------------------------------------
            # VALIDATION CHECKS
            # ----------------------------------------------------
        
            st.write(
                "**Validation Checks**"
            )
        
            validation_df = pd.DataFrame(
                structure_validation["checks"]
            )
        
            st.dataframe(
                validation_df,
                use_container_width=True,
                hide_index=True
            )
        
            # ----------------------------------------------------
            # FINAL INTERPRETATION
            # ----------------------------------------------------
        
            if structure_validation[
                "ready_for_next_step"
            ]:
        
                st.success(
                    "Structure validation passed. "
                    "The current swing and BOS / CHoCH "
                    "structure is internally consistent "
                    "enough to proceed to the next diagnostic stage."
                )
        
            elif structure_validation[
                "validation_score"
            ] >= 85:
        
                st.warning(
                    "Structure is usable for diagnostics, "
                    "but some structural inconsistencies "
                    "remain. Do not generate trading signals yet."
                )
        
            else:
        
                st.error(
                    "Structure validation failed. "
                    "Do NOT proceed to signal generation."
                )
        
        else:
        
            st.info(
                "Structure validation is not available."
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

    # ============================================================
    # 11️⃣ MARKET STATE DIAGNOSTIC
    # ============================================================
    
    st.header(
        "Step 11 — Market State Diagnostic"
    )
    
    if "sequence_analysis" in st.session_state:
    
        sequence = st.session_state[
            "sequence_analysis"
        ]
    
        # --------------------------------------------------------
        # EXTRACT STRUCTURAL INFORMATION
        # --------------------------------------------------------
    
        structural_bias = sequence.get(
            "structural_bias",
            sequence.get("bos_choch_bias", "UNKNOWN")
        )
    
        current_structure = sequence.get(
            "swing_current_structure",
            sequence.get("current_structure", "UNKNOWN")
        )
    
        current_direction = sequence.get(
            "current_direction",
            "UNKNOWN"
        )
    
        last_event = sequence.get(
            "last_bos_choch",
            None
        )
    
        swing_highs = sequence.get(
            "swing_highs",
            []
        )
    
        swing_lows = sequence.get(
            "swing_lows",
            []
        )
    
        # --------------------------------------------------------
        # DETERMINE LATEST STRUCTURAL EVENT
        # --------------------------------------------------------
    
        if last_event:
    
            latest_event = last_event.get(
                "event",
                "NONE"
            )
    
            latest_event_index = last_event.get(
                "candle_index",
                None
            )
    
        else:
    
            latest_event = "NONE"
            latest_event_index = None
    
        # --------------------------------------------------------
        # STRUCTURE DECISION TRACE
        # --------------------------------------------------------

        st.subheader("Structure Decision Trace")

        latest_high = swing_highs[-1] if swing_highs else None
        latest_low = swing_lows[-1] if swing_lows else None

        trace_col1, trace_col2, trace_col3 = st.columns(3)

        trace_col1.write(
            f"**Latest Swing High:** `{latest_high.get('structure', 'NONE') if latest_high else 'NONE'}`"
        )

        trace_col2.write(
            f"**Latest Swing Low:** `{latest_low.get('structure', 'NONE') if latest_low else 'NONE'}`"
        )

        trace_col3.write(
            f"**Current Swing Sequence:** `{sequence.get('structural_sequence', 'UNKNOWN')}`"
        )

        st.write(
            f"**Confirmed BOS Bias:** `{sequence.get('bos_choch_bias', 'UNKNOWN')}`"
        )

        st.write(
            f"**Latest Event:** `{latest_event}`"
        )

        st.write(
            f"**Transition Bias:** `{sequence.get('transition_bias', 'NONE')}`"
        )

        st.write(
            f"**Current Structural Bias:** `{sequence.get('structural_bias', 'UNKNOWN')}`"
        )

        # --------------------------------------------------------
        # STRUCTURAL STATE
        # --------------------------------------------------------
    
        if structural_bias == "BULLISH":
    
            if latest_event == "BULLISH BOS":
    
                market_state = (
                    "BULLISH CONTINUATION"
                )
    
            elif latest_event == "BULLISH CHoCH":
    
                market_state = (
                    "BULLISH STRUCTURAL SHIFT"
                )
    
            else:
    
                market_state = (
                    "BULLISH STRUCTURE"
                )
    
        elif structural_bias == "BEARISH":
    
            if latest_event == "BEARISH BOS":
    
                market_state = (
                    "BEARISH CONTINUATION"
                )
    
            elif latest_event == "BEARISH CHoCH":
    
                market_state = (
                    "BEARISH STRUCTURAL SHIFT"
                )
    
            else:
    
                market_state = (
                    "BEARISH STRUCTURE"
                )
    
        else:
    
            market_state = (
                "UNDEFINED / MIXED"
            )
    
        # --------------------------------------------------------
        # STRUCTURAL CONFIRMATION
        # --------------------------------------------------------
    
        # IMPORTANT: confirmation must use the MOST RECENT
        # swing structure, not lifetime HH/HL/LH/LL counts.
        # Historical bullish swings must not confirm a bearish
        # current bias (and vice versa).
        latest_high_structure = (
            swing_highs[-1].get("structure", "")
            if swing_highs
            else ""
        )

        latest_low_structure = (
            swing_lows[-1].get("structure", "")
            if swing_lows
            else ""
        )

        bullish_structure = (
            latest_high_structure == "HH"
            and
            latest_low_structure == "HL"
        )

        bearish_structure = (
            latest_high_structure == "LH"
            and
            latest_low_structure == "LL"
        )
    
        if (
            structural_bias == "BULLISH"
            and bullish_structure
        ):
    
            structure_confirmation = (
                "CONFIRMED BULLISH"
            )
    
        elif (
            structural_bias == "BEARISH"
            and bearish_structure
        ):
    
            structure_confirmation = (
                "CONFIRMED BEARISH"
            )
    
        elif structural_bias == "UNKNOWN":
    
            structure_confirmation = (
                "INSUFFICIENT STRUCTURE"
            )
    
        else:
    
            structure_confirmation = (
                "STRUCTURAL CONFLICT"
            )
    
        # --------------------------------------------------------
        # CURRENT CANDLE RELATIONSHIP
        # --------------------------------------------------------
    
        if (
            structural_bias == "BULLISH"
            and
            current_direction == "GREEN"
        ):
    
            candle_alignment = (
                "ALIGNED WITH BIAS"
            )
    
        elif (
            structural_bias == "BEARISH"
            and
            current_direction == "RED"
        ):
    
            candle_alignment = (
                "ALIGNED WITH BIAS"
            )
    
        elif (
            structural_bias in (
                "BULLISH",
                "BEARISH"
            )
        ):
    
            candle_alignment = (
                "COUNTER-DIRECTION CANDLE"
            )
    
        else:
    
            candle_alignment = (
                "NO CLEAR ALIGNMENT"
            )
    
        # --------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------
    
        col1, col2, col3, col4 = st.columns(4)
    
        col1.metric(
            "Market State",
            market_state
        )
    
        col2.metric(
            "Structural Bias",
            structural_bias
        )
    
        col3.metric(
            "Structure",
            structure_confirmation
        )
    
        col4.metric(
            "Candle Alignment",
            candle_alignment
        )
    
        st.divider()
    
        # --------------------------------------------------------
        # STRUCTURAL DETAILS
        # --------------------------------------------------------
    
        st.subheader(
            "Structural Interpretation"
        )
    
        st.write(
            f"**Current Structure:** "
            f"`{current_structure}`"
        )
    
        st.write(
            f"**Latest Structural Event:** "
            f"`{latest_event}`"
        )
    
        if latest_event_index is not None:
    
            st.write(
                f"**Event Candle:** "
                f"`{latest_event_index}`"
            )
    
        st.write(
            f"**Current Candle:** "
            f"`{current_direction}`"
        )
    
        # --------------------------------------------------------
        # STRUCTURE QUALITY
        # --------------------------------------------------------
    
        structure_quality = (
            sequence.get(
                "sequence_integrity",
                0.0
            )
        )
    
        st.write(
            f"**Structure Quality:** "
            f"`{structure_quality:.1f}%`"
        )
    
        # --------------------------------------------------------
        # FINAL DIAGNOSTIC MESSAGE
        # --------------------------------------------------------
    
        if (
            structure_quality >= 95
            and
            structure_confirmation
            in (
                "CONFIRMED BULLISH",
                "CONFIRMED BEARISH"
            )
            and
            candle_alignment
            == "ALIGNED WITH BIAS"
        ):
    
            st.success(
                "The current market structure, "
                "structural bias, and current candle "
                "are aligned."
            )
    
        elif structure_quality >= 95:
    
            st.warning(
                "Structure is well reconstructed, "
                "but the current candle is not fully "
                "aligned with the prevailing structural bias."
            )
    
        else:
    
            st.error(
                "Structure quality is not strong enough "
                "for higher-level interpretation."
            )
    
    else:
    
        st.info(
            "Market state cannot be evaluated until "
            "sequence analysis is available."
        )
    
    # ============================================================
    # 1️⃣2️⃣ CURRENT CANDLE CONTEXT
    # ============================================================
    
    st.header("Step 12 — Current Candle Context")
    
    if "sequence_analysis" in st.session_state:
    
        sequence = st.session_state["sequence_analysis"]
    
        # --------------------------------------------------------
        # CURRENT CANDLE DATA
        # --------------------------------------------------------
    
        current_direction = sequence.get(
            "current_direction",
            "UNKNOWN"
        )
    
        body_percentage = sequence.get(
            "body_percentage",
            0.0
        )
    
        upper_wick_percentage = sequence.get(
            "upper_wick_percentage",
            0.0
        )
    
        lower_wick_percentage = sequence.get(
            "lower_wick_percentage",
            0.0
        )
    
        current_confidence = sequence.get(
            "current_confidence",
            0.0
        )
    
        structural_bias = sequence.get(
            "structural_bias",
            sequence.get("bos_choch_bias", "UNKNOWN")
        )
    
        # --------------------------------------------------------
        # CURRENT CANDLE STRENGTH
        # --------------------------------------------------------
    
        if body_percentage >= 70:
    
            candle_strength = "STRONG"
    
        elif body_percentage >= 40:
    
            candle_strength = "MODERATE"
    
        elif body_percentage >= 20:
    
            candle_strength = "WEAK"
    
        else:
    
            candle_strength = "VERY WEAK"
    
        # --------------------------------------------------------
        # WICK CHARACTER
        # --------------------------------------------------------
    
        if (
            upper_wick_percentage >= 40
            and
            upper_wick_percentage > lower_wick_percentage
        ):
    
            wick_character = "UPPER-WICK REJECTION"
    
        elif (
            lower_wick_percentage >= 40
            and
            lower_wick_percentage > upper_wick_percentage
        ):
    
            wick_character = "LOWER-WICK REJECTION"
    
        elif (
            upper_wick_percentage >= 30
            and
            lower_wick_percentage >= 30
        ):
    
            wick_character = "TWO-SIDED REJECTION"
    
        else:
    
            wick_character = "NO MAJOR REJECTION"
    
        # --------------------------------------------------------
        # CANDLE / STRUCTURE RELATIONSHIP
        # --------------------------------------------------------
    
        if (
            structural_bias == "BULLISH"
            and
            current_direction == "GREEN"
        ):
    
            structural_relationship = (
                "CANDLE SUPPORTS BULLISH STRUCTURE"
            )
    
        elif (
            structural_bias == "BEARISH"
            and
            current_direction == "RED"
        ):
    
            structural_relationship = (
                "CANDLE SUPPORTS BEARISH STRUCTURE"
            )
    
        elif (
            structural_bias == "BULLISH"
            and
            current_direction == "RED"
        ):
    
            structural_relationship = (
                "BEARISH CANDLE AGAINST BULLISH STRUCTURE"
            )
    
        elif (
            structural_bias == "BEARISH"
            and
            current_direction == "GREEN"
        ):
    
            structural_relationship = (
                "BULLISH CANDLE AGAINST BEARISH STRUCTURE"
            )
    
        else:
    
            structural_relationship = (
                "STRUCTURAL RELATIONSHIP UNDEFINED"
            )
    
        # --------------------------------------------------------
        # MOMENTUM INTERPRETATION
        # --------------------------------------------------------
    
        if (
            candle_strength == "STRONG"
            and
            wick_character == "NO MAJOR REJECTION"
        ):
    
            momentum_state = "STRONG DIRECTIONAL MOMENTUM"
    
        elif candle_strength == "STRONG":
    
            momentum_state = "STRONG MOVE WITH REJECTION"
    
        elif candle_strength == "MODERATE":
    
            momentum_state = "MODERATE MOMENTUM"
    
        elif candle_strength in (
            "WEAK",
            "VERY WEAK"
        ):
    
            momentum_state = "LOW MOMENTUM"
    
        else:
    
            momentum_state = "UNDEFINED"
    
        # --------------------------------------------------------
        # DISPLAY PRIMARY METRICS
        # --------------------------------------------------------
    
        col1, col2, col3, col4 = st.columns(4)
    
        col1.metric(
            "Direction",
            current_direction
        )
    
        col2.metric(
            "Candle Strength",
            candle_strength
        )
    
        col3.metric(
            "Wick Character",
            wick_character
        )
    
        col4.metric(
            "Detection Confidence",
            f"{current_confidence:.1f}%"
        )
    
        st.divider()
    
        # --------------------------------------------------------
        # CANDLE PROPORTIONS
        # --------------------------------------------------------
    
        st.subheader("Candle Composition")
    
        col1, col2, col3 = st.columns(3)
    
        col1.metric(
            "Body",
            f"{body_percentage:.1f}%"
        )
    
        col2.metric(
            "Upper Wick",
            f"{upper_wick_percentage:.1f}%"
        )
    
        col3.metric(
            "Lower Wick",
            f"{lower_wick_percentage:.1f}%"
        )
    
        st.divider()
    
        # --------------------------------------------------------
        # INTERPRETATION
        # --------------------------------------------------------
    
        st.subheader("Candle Interpretation")
    
        st.write(
            f"**Momentum:** `{momentum_state}`"
        )
    
        st.write(
            f"**Structural Relationship:** "
            f"`{structural_relationship}`"
        )
    
        # --------------------------------------------------------
        # CONTEXT WARNING / CONFIRMATION
        # --------------------------------------------------------
    
        if (
            structural_bias == "BULLISH"
            and
            current_direction == "RED"
        ):
    
            st.warning(
                "The current candle is bearish, "
                "but the validated structural bias remains "
                "bullish. The candle alone does not invalidate "
                "the bullish structure."
            )
    
        elif (
            structural_bias == "BEARISH"
            and
            current_direction == "GREEN"
        ):
    
            st.warning(
                "The current candle is bullish, "
                "but the validated structural bias remains "
                "bearish. The candle alone does not invalidate "
                "the bearish structure."
            )
    
        elif (
            structural_bias == "BULLISH"
            and
            current_direction == "GREEN"
        ):
    
            st.success(
                "The current candle is aligned with "
                "the validated bullish structure."
            )
    
        elif (
            structural_bias == "BEARISH"
            and
            current_direction == "RED"
        ):
    
            st.success(
                "The current candle is aligned with "
                "the validated bearish structure."
            )
    
        else:
    
            st.info(
                "There is not enough information to establish "
                "a candle-to-structure relationship."
            )
    
    else:
    
        st.info(
            "Current candle context cannot be evaluated "
            "until sequence analysis is available."
        )
    
    
    # ============================================================
    # STEP 13 — TRADE SETUP / CONFLUENCE DIAGNOSTIC
    # ============================================================
    with trade_setup_placeholder.container():
         def diagnose_trade_setup(
            sequence,
            current_direction,
            body_percentage,
            upper_wick_percentage,
            lower_wick_percentage,
            current_confidence
        ):
            """
            STEP 13 — RULE-BASED TRADE SETUP DIAGNOSTIC

            This stage does NOT generate a trading signal.

            It evaluates whether the currently reconstructed market
            contains enough structural and candle-level agreement
            to qualify as a potential directional setup.

            The engine deliberately separates:

                STRUCTURAL BIAS
                from
                CURRENT CANDLE DIRECTION

            A single counter-directional candle does NOT invalidate
            the structural bias.

            Returns:
                setup_direction
                structural_bias
                candle_alignment
                candle_strength
                rejection_status
                structure_status
                quality
                confluence_score
                final_status
                reasons
            """

            # ========================================================
            # NORMALISE INPUTS
            # ========================================================

            structural_bias = str(
                sequence.get(
                    "structural_bias",
                    sequence.get(
                        "bos_choch_bias",
                        sequence.get(
                            "trend",
                            "UNKNOWN"
                        )
                    )
                )
            ).upper()

            current_direction = str(
                current_direction
            ).upper()

            body_percentage = float(
                body_percentage
            )

            upper_wick_percentage = float(
                upper_wick_percentage
            )

            lower_wick_percentage = float(
                lower_wick_percentage
            )

            current_confidence = float(
                current_confidence
            )

            last_event = sequence.get(
                "last_bos_choch",
                None
            )

            current_structure = str(
                sequence.get(
                    "swing_current_structure",
                    sequence.get(
                        "structural_sequence",
                        sequence.get(
                            "current_structure",
                            "UNKNOWN"
                        )
                    )
                )
            ).upper()

            sequence_integrity = float(
                sequence.get(
                    "sequence_integrity",
                    0
                )
            )

            # ========================================================
            # DETERMINE STRUCTURAL DIRECTION
            # ========================================================

            if structural_bias == "BULLISH":

                setup_direction = "LONG"

            elif structural_bias == "BEARISH":

                setup_direction = "SHORT"

            else:

                setup_direction = "NONE"

            # ========================================================
            # STRUCTURE STATUS
            # ========================================================

            if structural_bias == "BULLISH":

                if (
                    "HIGHER HIGH" in current_structure
                    or
                    "BULLISH" in current_structure
                ):

                    structure_status = (
                        "BULLISH STRUCTURE CONFIRMED"
                    )

                else:

                    structure_status = (
                        "BULLISH BIAS — STRUCTURE DEVELOPING"
                    )

            elif structural_bias == "BEARISH":

                if (
                    "LOWER HIGH" in current_structure
                    or
                    "BEARISH" in current_structure
                ):

                    structure_status = (
                        "BEARISH STRUCTURE CONFIRMED"
                    )

                else:

                    structure_status = (
                        "BEARISH BIAS — STRUCTURE DEVELOPING"
                    )

            else:

                structure_status = (
                    "NO CONFIRMED STRUCTURAL DIRECTION"
                )

            # ========================================================
            # CANDLE ALIGNMENT
            # ========================================================

            if setup_direction == "LONG":

                if current_direction == "GREEN":

                    candle_alignment = "ALIGNED"

                elif current_direction == "RED":

                    candle_alignment = (
                        "COUNTER-DIRECTIONAL"
                    )

                else:

                    candle_alignment = "UNKNOWN"

            elif setup_direction == "SHORT":

                if current_direction == "RED":

                    candle_alignment = "ALIGNED"

                elif current_direction == "GREEN":

                    candle_alignment = (
                        "COUNTER-DIRECTIONAL"
                    )

                else:

                    candle_alignment = "UNKNOWN"

            else:

                candle_alignment = "NO STRUCTURAL DIRECTION"

            # ========================================================
            # CANDLE STRENGTH
            # ========================================================

            if body_percentage >= 70:

                candle_strength = "STRONG"

            elif body_percentage >= 45:

                candle_strength = "MODERATE"

            elif body_percentage >= 25:

                candle_strength = "WEAK"

            else:

                candle_strength = "INDECISIVE"

            # ========================================================
            # REJECTION ANALYSIS
            # ========================================================

            rejection_status = "NO MAJOR REJECTION"

            if setup_direction == "LONG":

                if (
                    upper_wick_percentage >= 45
                    and
                    upper_wick_percentage
                    >
                    lower_wick_percentage * 1.25
                ):

                    rejection_status = (
                        "BULLISH SETUP HAS UPPER-WICK REJECTION"
                    )

            elif setup_direction == "SHORT":

                if (
                    lower_wick_percentage >= 45
                    and
                    lower_wick_percentage
                    >
                    upper_wick_percentage * 1.25
                ):

                    rejection_status = (
                        "BEARISH SETUP HAS LOWER-WICK REJECTION"
                    )

            # ========================================================
            # STRUCTURAL EVENT RELATIONSHIP
            # ========================================================

            event_alignment = "NEUTRAL"

            if last_event:

                event_direction = str(
                    last_event.get(
                        "direction",
                        ""
                    )
                ).upper()

                if (
                    setup_direction == "LONG"
                    and
                    event_direction == "BULLISH"
                ):

                    event_alignment = "ALIGNED"

                elif (
                    setup_direction == "SHORT"
                    and
                    event_direction == "BEARISH"
                ):

                    event_alignment = "ALIGNED"

                elif event_direction in (
                    "BULLISH",
                    "BEARISH"
                ):

                    event_alignment = (
                        "COUNTER-DIRECTIONAL"
                    )

            # ========================================================
            # CONFLUENCE SCORE
            #
            # This is a diagnostic score only.
            #
            # It is NOT a probability of winning.
            # ========================================================

            score = 0.0

            # --------------------------------------------------------
            # Structural direction
            # --------------------------------------------------------

            if setup_direction in (
                "LONG",
                "SHORT"
            ):

                score += 30

            # --------------------------------------------------------
            # Structure confirmation
            # --------------------------------------------------------

            if (
                "CONFIRMED" in structure_status
            ):

                score += 20

            elif (
                "DEVELOPING" in structure_status
            ):

                score += 10

            # --------------------------------------------------------
            # Candle alignment
            # --------------------------------------------------------

            if candle_alignment == "ALIGNED":

                score += 20

            elif (
                candle_alignment == "COUNTER-DIRECTIONAL"
            ):

                score += 5

            # --------------------------------------------------------
            # Candle strength
            # --------------------------------------------------------

            if candle_strength == "STRONG":

                score += 15

            elif candle_strength == "MODERATE":

                score += 10

            elif candle_strength == "WEAK":

                score += 5

            # --------------------------------------------------------
            # Event alignment
            # --------------------------------------------------------

            if event_alignment == "ALIGNED":

                score += 10

            elif event_alignment == "COUNTER-DIRECTIONAL":

                score += 3

            # --------------------------------------------------------
            # Detection / sequence quality
            # --------------------------------------------------------

            if current_confidence >= 85:

                score += 3

            elif current_confidence >= 70:

                score += 2

            elif current_confidence >= 60:

                score += 1

            if sequence_integrity >= 90:

                score += 2

            elif sequence_integrity >= 75:

                score += 1

            # --------------------------------------------------------
            # Rejection penalty
            # --------------------------------------------------------

            if (
                rejection_status !=
                "NO MAJOR REJECTION"
            ):

                score -= 10

            confluence_score = round(
                clamp_score(score),
                1
            )

            # ========================================================
            # FINAL STATUS
            # ========================================================

            reasons = []

            if setup_direction == "NONE":

                final_status = "WAIT"

                reasons.append(
                    "No confirmed structural direction."
                )

            else:

                if structural_bias == "BULLISH":

                    reasons.append(
                        "Structural bias is bullish."
                    )

                elif structural_bias == "BEARISH":

                    reasons.append(
                        "Structural bias is bearish."
                    )

                if candle_alignment == "ALIGNED":

                    reasons.append(
                        "Current candle agrees with structural direction."
                    )

                elif (
                    candle_alignment ==
                    "COUNTER-DIRECTIONAL"
                ):

                    reasons.append(
                        "Current candle is counter-directional."
                    )

                if candle_strength == "STRONG":

                    reasons.append(
                        "Current candle has strong body dominance."
                    )

                elif candle_strength == "MODERATE":

                    reasons.append(
                        "Current candle has moderate body dominance."
                    )

                if (
                    rejection_status !=
                    "NO MAJOR REJECTION"
                ):

                    reasons.append(
                        rejection_status
                    )

                # ----------------------------------------------------
                # DO NOT CALL A COUNTER-DIRECTIONAL CANDLE A REVERSAL
                # ----------------------------------------------------

                if (
                    candle_alignment ==
                    "COUNTER-DIRECTIONAL"
                    and
                    confluence_score >= 55
                ):

                    final_status = (
                        "WAIT — STRUCTURE INTACT"
                    )

                elif confluence_score >= 75:

                    final_status = (
                        f"VALID {setup_direction} SETUP"
                    )

                elif confluence_score >= 55:

                    final_status = (
                        f"DEVELOPING {setup_direction} SETUP"
                    )

                else:

                    final_status = (
                        "WAIT — INSUFFICIENT CONFLUENCE"
                    )

            # ========================================================
            # RETURN
            # ========================================================

            return {

                "setup_direction":
                    setup_direction,

                "structural_bias":
                    structural_bias,

                "structure_status":
                    structure_status,

                "candle_alignment":
                    candle_alignment,

                "candle_strength":
                    candle_strength,

                "rejection_status":
                    rejection_status,

                "event_alignment":
                    event_alignment,

                "confluence_score":
                    confluence_score,

                "final_status":
                    final_status,

                "reasons":
                    reasons
            }       
    
    # ========================================================
    # STEP 13 — TRADE SETUP / CONFLUENCE DIAGNOSTIC
    # ========================================================
    
    st.header(
        "1️⃣3️⃣ Trade Setup / Confluence Diagnostic"
    )
    
    # ------------------------------------------------------------
    # RUN DIAGNOSTIC
    # ------------------------------------------------------------
    
    setup_analysis = diagnose_trade_setup(
        sequence,
    
        sequence.get(
            "current_direction",
            "UNKNOWN"
        ),
    
        sequence.get(
            "body_percentage",
            0
        ),
    
        sequence.get(
            "upper_wick_percentage",
            0
        ),
    
        sequence.get(
            "lower_wick_percentage",
            0
        ),
    
        sequence.get(
            "current_confidence",
            0
        )
    )
    
    
    # ============================================================
    # STEP 14 — ACTUAL BUY / SELL SIGNAL ENGINE
    # ============================================================
    
    def generate_signal(sequence, setup_analysis):
        """
        Convert the validated setup into an actual BUY / SELL signal.
    
        This is intentionally stricter than the setup diagnostic.
        A valid setup is not automatically a trade signal.
    
        Signal requirements:
        - confirmed structural direction
        - valid setup
        - latest structural event agrees with the direction
        - current candle agrees with the direction
        - strong current candle body
        - no major rejection
        - detection confidence >= 85%
        - sequence integrity >= 90%
        - confluence >= 75%
    
        The latest event does not have to be the current candle.
        The current candle is the entry trigger; the structural event
        establishes the directional context.
        """
    
        # --------------------------------------------------------
        # INITIALISE REASONS
        # --------------------------------------------------------
    
        reasons = []
    
        # --------------------------------------------------------
        # BASIC SETUP VALUES
        # --------------------------------------------------------
    
        direction = str(
            setup_analysis.get(
                "setup_direction",
                "NONE"
            )
        ).upper()
    
        final_status = str(
            setup_analysis.get(
                "final_status",
                "WAIT"
            )
        ).upper()
    
        event_alignment = str(
            setup_analysis.get(
                "event_alignment",
                "NEUTRAL"
            )
        ).upper()
    
        candle_alignment = str(
            setup_analysis.get(
                "candle_alignment",
                "UNKNOWN"
            )
        ).upper()
    
        candle_strength = str(
            setup_analysis.get(
                "candle_strength",
                "UNKNOWN"
            )
        ).upper()
    
        rejection_status = str(
            setup_analysis.get(
                "rejection_status",
                ""
            )
        ).upper()
    
        confidence = float(
            sequence.get(
                "current_confidence",
                0
            )
        )
    
        sequence_integrity = float(
            sequence.get(
                "sequence_integrity",
                0
            )
        )
    
        confluence_score = float(
            setup_analysis.get(
                "confluence_score",
                0
            )
        )
    
        # --------------------------------------------------------
        # LATEST STRUCTURAL EVENT
        # --------------------------------------------------------
    
        last_event = sequence.get(
            "last_bos_choch"
        ) or {}
    
        event_name = str(
            last_event.get(
                "event",
                "NONE"
            )
        ).upper()
    
        # --------------------------------------------------------
        # CURRENT CANDLE INDEX
        # --------------------------------------------------------
    
        current_index = sequence.get(
            "current_candle_index",
            sequence.get(
                "count",
                0
            ) - 1
        )
    
        # --------------------------------------------------------
        # STRUCTURAL EVENT INDEX
        # --------------------------------------------------------
    
        event_index = last_event.get(
            "candle_index",
            None
        )
    
        # --------------------------------------------------------
        # CALCULATE EVENT AGE
        # --------------------------------------------------------
    
        try:
            event_age = (
                int(current_index)
                - int(event_index)
            )
        except (
            TypeError,
            ValueError
        ):
            event_age = None
    
        # ========================================================
        # EVENT FRESHNESS
        # ========================================================
    
        MAX_SIGNAL_EVENT_AGE = 5
    
        if event_age is None:
    
            reasons.append(
                "Structural event age cannot be established."
            )
    
        elif event_age < 1:
    
            reasons.append(
                "Structural BOS occurred on the current candle; "
                "wait for confirmation on a subsequent candle."
            )
    
        elif event_age > MAX_SIGNAL_EVENT_AGE:
    
            reasons.append(
                f"Structural BOS is too old "
                f"({event_age} candles)."
            )
    
        # ========================================================
        # DIRECTION VALIDATION
        # ========================================================
    
        if direction not in (
            "LONG",
            "SHORT"
        ):
    
            reasons.append(
                "No confirmed BUY or SELL direction."
            )
    
        # ========================================================
        # SETUP VALIDATION
        # ========================================================
    
        if not final_status.startswith(
            "VALID"
        ):
    
            reasons.append(
                "The setup has not reached VALID status."
            )
    
        # ========================================================
        # STRUCTURAL EVENT ALIGNMENT
        # ========================================================
    
        if event_alignment != "ALIGNED":
    
            reasons.append(
                "The latest structural event does not "
                "support the setup direction."
            )
    
        # ========================================================
        # STRUCTURAL EVENT CONFIRMATION
        # ========================================================
        #
        # CHoCH identifies a possible structural shift.
        # It does NOT authorize an entry by itself.
        #
        # Actual BUY / SELL signals require a BOS in the
        # same direction as the setup.
        # ========================================================
    
        expected_bos = (
            "BULLISH BOS"
            if direction == "LONG"
            else "BEARISH BOS"
        )
    
        if event_name != expected_bos:
    
            reasons.append(
                f"Latest structural event is "
                f"{event_name}; "
                f"{expected_bos} is required for entry."
            )
    
        # ========================================================
        # CURRENT CANDLE ALIGNMENT
        # ========================================================
    
        if candle_alignment != "ALIGNED":
    
            reasons.append(
                "The current candle is not aligned "
                "with the setup direction."
            )
    
        # ========================================================
        # CANDLE STRENGTH
        # ========================================================
    
        if candle_strength != "STRONG":
    
            reasons.append(
                "The current candle is not strong "
                "enough to trigger an entry."
            )
    
        # ========================================================
        # WICK REJECTION
        # ========================================================
    
        if rejection_status != "NO MAJOR REJECTION":
    
            reasons.append(
                "Major wick rejection blocks the entry."
            )
    
        # ========================================================
        # DETECTION CONFIDENCE
        # ========================================================
    
        if confidence < 85:
    
            reasons.append(
                f"Detection confidence is only "
                f"{confidence:.1f}%."
            )
    
        # ========================================================
        # SEQUENCE INTEGRITY
        # ========================================================
    
        if sequence_integrity < 90:
    
            reasons.append(
                f"Sequence integrity is only "
                f"{sequence_integrity:.1f}%."
            )
    
        # ========================================================
        # CONFLUENCE SCORE
        # ========================================================
    
        if confluence_score < 75:
    
            reasons.append(
                f"Confluence is only "
                f"{confluence_score:.1f}%."
            )
    
        # ========================================================
        # NO SIGNAL
        # ========================================================
    
        if reasons:
    
            return {
                "signal": "NO SIGNAL",
    
                "trigger": "CONDITIONS NOT MET",
    
                "event": (
                    event_name
                    or "NONE"
                ),
    
                "event_age": event_age,
    
                "reasons": reasons
            }
    
        # ========================================================
        # VALID BUY / SELL SIGNAL
        # ========================================================
    
        if direction == "LONG":
    
            signal = "BUY"
    
            trigger = (
                "BULLISH CONFIRMATION"
            )
    
        else:
    
            signal = "SELL"
    
            trigger = (
                "BEARISH CONFIRMATION"
            )
    
        # ========================================================
        # SIGNAL APPROVED
        # ========================================================
    
        return {
            "signal": signal,
    
            "trigger": trigger,
    
            "event": event_name,
    
            "event_age": event_age,
    
            "reasons": [
                "Structural direction is confirmed.",
                "Latest BOS supports the direction.",
                "Current candle confirms the direction.",
                "Current candle has strong body dominance.",
                "No major wick rejection is present.",
                f"Detection confidence is {confidence:.1f}%.",
                f"Sequence integrity is {sequence_integrity:.1f}%.",
                f"Confluence score is {confluence_score:.1f}%."
            ]
        }
    
    
        # ========================================================
        # RUN SIGNAL ENGINE
        # ========================================================
        
        signal_result = generate_signal(
            sequence,
            setup_analysis
        )
        
        # ========================================================
        # SIGNAL DISPLAY
        # ========================================================

        signal_value = signal_result["signal"]

        with signal_placeholder.container():

            if signal_value == "BUY":

                st.success(
                    "🟢 BUY"
                )

            elif signal_value == "SELL":

                st.error(
                    "🔴 SELL"
                )

            else:

                st.warning(
                    "⚪ NO SIGNAL"
                )


        # ========================================================
        # SIGNAL DETAILS
        # ========================================================

        with signal_details_placeholder.container():

            signal_col1, signal_col2, signal_col3 = st.columns(3)

            signal_col1.write(
                "**Trigger:** "
                f"`{signal_result['trigger']}`"
            )

            signal_col2.write(
                "**Latest Event:** "
                f"`{signal_result['event']}`"
            )

            event_age_display = (
                f"{signal_result['event_age']} candles"
                if signal_result["event_age"] is not None
                else "— candles"
            )

            signal_col3.write(
                "**Event Age:** "
                f"`{event_age_display}`"
            )

            with st.expander(
                "🔎 Signal decision audit"
            ):

                for reason in signal_result["reasons"]:

                    st.write(
                        f"• {reason}"
                    )
        # ========================================================
        # TOP METRICS
        # ========================================================
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Setup Direction",
            setup_analysis[
                "setup_direction"
            ]
        )
        
        col2.metric(
            "Structural Bias",
            setup_analysis[
                "structural_bias"
            ]
        )
        
        col3.metric(
            "Candle Alignment",
            setup_analysis[
                "candle_alignment"
            ]
        )
        
        col4.metric(
            "Confluence Score",
            f"{setup_analysis['confluence_score']:.1f}%"
        )
        
        st.divider()
        
        # ========================================================
        # SETUP DIAGNOSTIC
        # ========================================================
        
        st.subheader(
            "Setup Diagnostic"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
        
            st.write(
                "**Structure Status:** "
                f"`{setup_analysis['structure_status']}`"
            )
        
            st.write(
                "**Candle Strength:** "
                f"`{setup_analysis['candle_strength']}`"
            )
        
            st.write(
                "**Structural Event Alignment:** "
                f"`{setup_analysis['event_alignment']}`"
            )
        
        with col2:
        
            st.write(
                "**Rejection Status:** "
                f"`{setup_analysis['rejection_status']}`"
            )
        
            st.write(
                "**Current Candle:** "
                f"`{sequence.get('current_direction', 'UNKNOWN')}`"
            )
        
            st.write(
                "**Detection Confidence:** "
                f"`{sequence.get('current_confidence', 0):.1f}%`"
            )
        
        # ========================================================
        # FINAL STATUS
        # ========================================================
        
        st.subheader(
            "Final Setup Status"
        )
        
        final_status = setup_analysis[
            "final_status"
        ]
        
        if final_status.startswith(
            "VALID"
        ):
        
            st.success(
                final_status
            )
        
        elif final_status.startswith(
            "DEVELOPING"
        ):
        
            st.warning(
                final_status
            )
        
        elif final_status.startswith(
            "WAIT"
        ):
        
            st.warning(
                final_status
            )
        
        else:
        
            st.info(
                final_status
            )
        
        # ========================================================
        # REASONING
        # ========================================================
        
        st.subheader(
            "Diagnostic Reasoning"
        )
        
        for reason in setup_analysis[
            "reasons"
        ]:
        
            st.write(
                f"• {reason}"
            )
        
        # ========================================================
        # IMPORTANT INTERPRETATION
        # ========================================================
        
        if (
            setup_analysis[
                "candle_alignment"
            ]
            ==
            "COUNTER-DIRECTIONAL"
        ):
        
            st.info(
                "The current candle is moving against the "
                "validated structural direction. This does NOT "
                "by itself invalidate the structure. A structural "
                "reversal requires a confirmed break of the "
                "protected structural level."
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
        # SIGNAL ENGINE NOTE
        # ========================================================
        
        st.caption(
            "BUY / SELL signals are generated only when the validated "
            "structural, candle, confidence and confluence gates agree."
        )
