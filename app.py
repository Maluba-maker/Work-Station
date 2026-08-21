import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Maluz Signal Engine V2.3",
    layout="wide"
)

st.title("🔹 Maluz Signal Engine V2.3")
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
# V2.4
# ============================================================

def analyze_candle_sequence(candles):

    if not candles:
        return {
            "count": count,
            "sequence_integrity": sequence_integrity,
            "ohlc_validity": ohlc_validity,
            "duplicate_centres": duplicate_centres,
            "spacing_consistency": spacing_consistency,
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows,
            "trend": trend,
            "current_structure": current_structure,
        
            "swing_highs": swing_highs,
            "swing_lows": swing_lows
        }

    # --------------------------------------------------------
    # BASIC COUNTS
    # --------------------------------------------------------

    count = len(candles)
    
    # --------------------------------------------------------
    # SWING STRUCTURE
    # --------------------------------------------------------

    swing_analysis = detect_swings(
        candles,
        lookback=2
    )

    swing_highs = swing_analysis["swing_highs"]
    swing_lows = swing_analysis["swing_lows"]
    
    # --------------------------------------------------------
    # CENTRES
    # --------------------------------------------------------

    centers = np.array([
        c["x"] + c["width"] / 2
        for c in candles
    ], dtype=float)

    # --------------------------------------------------------
    # DUPLICATE CENTRES
    # --------------------------------------------------------

    duplicate_centers = 0

    if len(centers) > 1:

        for i in range(1, len(centers)):

            if abs(centers[i] - centers[i - 1]) < 1.0:
                duplicate_centers += 1

    # --------------------------------------------------------
    # OHLC VALIDATION
    #
    # Remember:
    # These are pixel coordinates, NOT market prices.
    # --------------------------------------------------------

    valid_ohlc = 0

    for candle in candles:

        high = candle["high"]
        low = candle["low"]
        open_price = candle["open"]
        close_price = candle["close"]

        valid = (
            high <= open_price
            and high <= close_price
            and low >= open_price
            and low >= close_price
            and high <= low
        )

        if valid:
            valid_ohlc += 1

    ohlc_validity = (
        valid_ohlc / count * 100
        if count > 0
        else 0
    )

    # --------------------------------------------------------
    # SPACING CONSISTENCY
    # --------------------------------------------------------

    spacing_consistency = 0.0

    if len(centers) >= 3:

        spacing = np.diff(centers)

        median_spacing = np.median(spacing)

        if median_spacing > 0:

            deviations = np.abs(
                spacing - median_spacing
            ) / median_spacing

            average_deviation = np.mean(
                deviations
            )

            spacing_consistency = clamp_score(
                100 - (average_deviation * 100)
            )

    elif len(centers) == 2:

        spacing_consistency = 100.0

    # --------------------------------------------------------
    # MARKET STRUCTURE
    #
    # We compare each candle's HIGH and LOW
    # against the previous candle.
    #
    # This is intentionally simple for V2.4.
    # We will later replace this with swing detection.
    # --------------------------------------------------------

    higher_highs = 0
    higher_lows = 0
    lower_highs = 0
    lower_lows = 0

    for i in range(1, len(candles)):

        previous = candles[i - 1]
        current = candles[i]

        if current["high"] < previous["high"]:
            higher_highs += 1
        elif current["high"] > previous["high"]:
            lower_highs += 1

        if current["low"] > previous["low"]:
            higher_lows += 1
        elif current["low"] < previous["low"]:
            lower_lows += 1

    # --------------------------------------------------------
    # RECENT STRUCTURE
    #
    # Use last 10 candles rather than the entire chart.
    # IMPORTANT:
    # These are PIXEL coordinates.
    # Smaller Y = higher price.
    # Larger Y = lower price.
    # --------------------------------------------------------
    
    recent = candles[-10:]
    
    recent_hh = 0
    recent_hl = 0
    recent_lh = 0
    recent_ll = 0
    
    for i in range(1, len(recent)):
    
        previous = recent[i - 1]
        current = recent[i]
    
        # HIGH comparison
        # Smaller pixel Y means a higher market price.
        if current["high"] < previous["high"]:
            recent_hh += 1
    
        elif current["high"] > previous["high"]:
            recent_lh += 1
    
        # LOW comparison
        # Smaller pixel Y means a higher market price.
        if current["low"] < previous["low"]:
            recent_hl += 1
    
        elif current["low"] > previous["low"]:
            recent_ll += 1
    # --------------------------------------------------------
    # TREND CLASSIFICATION
    # --------------------------------------------------------

    bullish_score = recent_hh + recent_hl
    bearish_score = recent_lh + recent_ll

    if bullish_score >= bearish_score + 2:

        trend = "BULLISH"

    elif bearish_score >= bullish_score + 2:

        trend = "BEARISH"

    else:

        trend = "SIDEWAYS / MIXED"

    # --------------------------------------------------------
    # CURRENT CANDLE
    # --------------------------------------------------------

    current = candles[-1]

    if current["color"] == "GREEN":
        direction = "GREEN"
    else:
        direction = "RED"

    # --------------------------------------------------------
    # BODY / WICK ANALYSIS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CURRENT STRUCTURE DESCRIPTION
    # --------------------------------------------------------

    if (
        recent_hh >= 2
        and recent_hl >= 2
    ):

        current_structure = (
            "HIGHER HIGH + HIGHER LOW"
        )

    elif (
        recent_lh >= 2
        and recent_ll >= 2
    ):

        current_structure = (
            "LOWER HIGH + LOWER LOW"
        )

    elif (
        recent_hh > recent_lh
        and recent_hl > recent_ll
    ):

        current_structure = (
            "BULLISH STRUCTURE DEVELOPING"
        )

    elif (
        recent_lh > recent_hh
        and recent_ll > recent_hl
    ):

        current_structure = (
            "BEARISH STRUCTURE DEVELOPING"
        )

    else:

        current_structure = (
            "CONSOLIDATION / MIXED"
        )

    # --------------------------------------------------------
    # SEQUENCE INTEGRITY
    # --------------------------------------------------------

    integrity_components = []

    # Duplicate penalty
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

        "recent_hh":
            recent_hh,

        "recent_hl":
            recent_hl,

        "recent_lh":
            recent_lh,

        "recent_ll":
            recent_ll,

        "trend":
            trend,

        "current_structure":
            current_structure,

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
            current["confidence"]
    }

# ============================================================
# SWING STRUCTURE DETECTION
# ============================================================

def detect_swings(candles, lookback=2):
    """
    Detect confirmed swing highs and swing lows.

    IMPORTANT:
    Candle coordinates are PIXEL coordinates.

    Smaller Y = higher market price
    Larger Y = lower market price

    A swing high is a local peak.
    A swing low is a local trough.

    We allow equal neighbouring pixel values because
    screenshot reconstruction can produce flat/duplicate
    high or low coordinates.
    """

    if len(candles) < (lookback * 2 + 1):
        return {
            "swing_highs": [],
            "swing_lows": []
        }

    swing_highs = []
    swing_lows = []

    for i in range(
        lookback,
        len(candles) - lookback
    ):

        current = candles[i]

        # ====================================================
        # SWING HIGH
        # ====================================================

        current_high = current["high"]

        left_highs = [
            candles[j]["high"]
            for j in range(
                i - lookback,
                i
            )
        ]

        right_highs = [
            candles[j]["high"]
            for j in range(
                i + 1,
                i + lookback + 1
            )
        ]

        # Smaller Y = higher market price
        #
        # Allow equality because pixel reconstruction
        # can produce identical high coordinates.

        is_swing_high = (
            current_high <= min(left_highs)
            and
            current_high <= min(right_highs)
            and
            (
                current_high < max(left_highs)
                or
                current_high < max(right_highs)
            )
        )

        if is_swing_high:

            swing_highs.append({
                "index": i,
                "price": current_high,
                "x": (
                    current["x"]
                    + current["width"] / 2
                ),
                "type": "SWING HIGH"
            })

        # ====================================================
        # SWING LOW
        # ====================================================

        current_low = current["low"]

        left_lows = [
            candles[j]["low"]
            for j in range(
                i - lookback,
                i
            )
        ]

        right_lows = [
            candles[j]["low"]
            for j in range(
                i + 1,
                i + lookback + 1
            )
        ]

        # Larger Y = lower market price
        #
        # Allow equality because pixel reconstruction
        # can produce identical low coordinates.

        is_swing_low = (
            current_low >= max(left_lows)
            and
            current_low >= max(right_lows)
            and
            (
                current_low > min(left_lows)
                or
                current_low > min(right_lows)
            )
        )

        if is_swing_low:

            swing_lows.append({
                "index": i,
                "price": current_low,
                "x": (
                    current["x"]
                    + current["width"] / 2
                ),
                "type": "SWING LOW"
            })

    return {
        "swing_highs": swing_highs,
        "swing_lows": swing_lows
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
