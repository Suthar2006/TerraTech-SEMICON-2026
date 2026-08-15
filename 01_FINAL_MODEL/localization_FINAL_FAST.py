import cv2
import numpy as np
import os
import csv
import json
import math
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# ================================================================
# SEMICON INDIA HACKATHON 2026
# DAY 10 V2 - RECOVERY CONSENSUS
#
# Based on:
#   Day 9 V1
#
# Improvements:
#   1. Multiple candidate generation
#   2. Candidate score separation
#   3. Ambiguity detection
#   4. Adaptive recovery
#   5. Local verification
#   6. Edge verification
#   7. Gradient verification
#   8. Projection verification
#   9. Final candidate re-ranking
#
# Target:
#   Day 9 V1 : 97.60%
#   Day 9 V2 : >= 98.5%
# ================================================================


# ================================================================
# CONFIGURATION
# ================================================================

DATASET_DIR = Path(
    r"F:\SEMICON india Hackathon 2026\Final SEMICON\dataset\dataset1"
)

OUTPUT_DIR = Path("results")
OUTPUT_CSV = OUTPUT_DIR / "resultsrecovery_consensus.csv"

FAILED_DIR = Path("failed")
DIAGNOSTIC_DIR = Path("diagnostics")

SUCCESS_THRESHOLD = 10.0

# Candidate configuration
TOP_CANDIDATES_PER_SCALE = 8
FINAL_CANDIDATES = 20

SUPPRESSION_RADIUS_RATIO = 0.18

# Ambiguity detection
MIN_SCORE_MARGIN = 0.025
MIN_UNIQUENESS = 0.035

# Recovery thresholds
LOW_CONFIDENCE = 0.58
RECOVERY_CONFIDENCE = 0.68

# Day 10 V2: recovery consensus parameters
CONSENSUS_RADIUS = 18.0
CONSENSUS_MIN_SUPPORT = 3
CONSENSUS_SUPPORT_CAP = 8
CONSENSUS_BONUS = 0.012

# Scales
SCALES = [
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
    1.00,
    1.02,
    1.04,
    1.06,
    1.08,
    1.10
]

# Angles
ANGLES = [
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0
]

np.random.seed(2026)

# SPEED OPTIMIZATION: one OpenCV thread per worker process.
# The localization algorithm itself is unchanged.
cv2.setNumThreads(1)


# ================================================================
# DIRECTORIES
# ================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILED_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# IMAGE LOADING
# ================================================================

def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Unable to load image: {path}"
        )

    return image


# ================================================================
# NORMALIZATION
# ================================================================

def normalize_image(image):

    image = image.astype(np.float32)

    mn = np.min(image)
    mx = np.max(image)

    if mx - mn < 1e-8:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    image = (
        (image - mn)
        /
        (mx - mn)
        *
        255.0
    )

    return image.astype(np.uint8)


# ================================================================
# ROTATION + SCALE
# ================================================================

def transform_reference(
    reference,
    scale,
    angle
):

    h, w = reference.shape[:2]

    new_w = max(
        8,
        int(round(w * scale))
    )

    new_h = max(
        8,
        int(round(h * scale))
    )

    resized = cv2.resize(
        reference,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    center = (
        new_w / 2.0,
        new_h / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        resized,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return rotated


# ================================================================
# EDGE
# ================================================================

def edge_image(image):

    blur = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    return cv2.Canny(
        blur,
        40,
        120
    )


# ================================================================
# GRADIENT
# ================================================================

def gradient_image(image):

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(
        np.uint8
    )


# ================================================================
# RESIZE TWO IMAGES TO COMMON SIZE
# ================================================================

def resize_pair(a, b):

    h = min(
        a.shape[0],
        b.shape[0]
    )

    w = min(
        a.shape[1],
        b.shape[1]
    )

    if h < 4 or w < 4:
        return None, None

    a = cv2.resize(
        a,
        (w, h),
        interpolation=cv2.INTER_AREA
    )

    b = cv2.resize(
        b,
        (w, h),
        interpolation=cv2.INTER_AREA
    )

    return a, b


# ================================================================
# EDGE SCORE
# ================================================================

def edge_score(reference, patch):

    r = edge_image(reference)
    p = edge_image(patch)

    r, p = resize_pair(r, p)

    if r is None:
        return 0.0

    r = r.astype(np.float32) / 255.0
    p = p.astype(np.float32) / 255.0

    reference_energy = (
        np.sum(r) + 1e-6
    )

    overlap = np.sum(
        np.minimum(r, p)
    )

    score = (
        overlap
        /
        reference_energy
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ================================================================
# GRADIENT SCORE
# ================================================================

def gradient_score(reference, patch):

    r = gradient_image(reference)
    p = gradient_image(patch)

    r, p = resize_pair(r, p)

    if r is None:
        return 0.0

    r = r.astype(np.float32)
    p = p.astype(np.float32)

    r = (
        r - np.mean(r)
    ) / (
        np.std(r) + 1e-6
    )

    p = (
        p - np.mean(p)
    ) / (
        np.std(p) + 1e-6
    )

    correlation = np.mean(
        np.minimum(
            np.abs(r),
            np.abs(p)
        )
    )

    score = correlation / 4.0

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ================================================================
# PROJECTION SIGNATURE
# ================================================================

def projection_signature(image):

    image = (
        image.astype(np.float32)
        /
        255.0
    )

    horizontal = np.mean(
        image,
        axis=1
    )

    vertical = np.mean(
        image,
        axis=0
    )

    signature = np.concatenate(
        [horizontal, vertical]
    )

    norm = np.linalg.norm(
        signature
    )

    if norm > 1e-8:
        signature /= norm

    return signature


# ================================================================
# PROJECTION SCORE
# ================================================================

def projection_score(
    reference,
    patch
):

    r = projection_signature(
        reference
    )

    p = projection_signature(
        patch
    )

    n = min(
        len(r),
        len(p)
    )

    if n == 0:
        return 0.0

    r = r[:n]
    p = p[:n]

    score = float(
        np.dot(r, p)
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ================================================================
# PATCH EXTRACTION
# ================================================================

def get_patch(
    search,
    x,
    y,
    width,
    height
):

    h, w = search.shape[:2]

    if x < 0 or y < 0:
        return None

    if x + width > w:
        return None

    if y + height > h:
        return None

    return search[
        y:y + height,
        x:x + width
    ]


# ================================================================
# TEMPLATE MATCHING
# ================================================================

def template_matching(
    search,
    template
):

    th, tw = template.shape[:2]
    sh, sw = search.shape[:2]

    if th >= sh or tw >= sw:
        return None

    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    return result


# ================================================================
# TOP CANDIDATE EXTRACTION
# ================================================================

def get_top_candidates(
    result,
    template_width,
    template_height,
    count=8
):

    if result is None:
        return []

    work = result.copy()

    candidates = []

    radius = max(
        8,
        int(
            min(
                template_width,
                template_height
            )
            *
            SUPPRESSION_RADIUS_RATIO
        )
    )

    for _ in range(count):

        _, score, _, location = (
            cv2.minMaxLoc(work)
        )

        if score < 0:
            break

        x, y = location

        candidates.append({
            "x": int(x),
            "y": int(y),
            "raw_score": float(score)
        })

        x1 = max(
            0,
            x - radius
        )

        y1 = max(
            0,
            y - radius
        )

        x2 = min(
            work.shape[1],
            x + radius + 1
        )

        y2 = min(
            work.shape[0],
            y + radius + 1
        )

        work[
            y1:y2,
            x1:x2
        ] = -1.0

    return candidates


# ================================================================
# CANDIDATE EVALUATION
# ================================================================

def evaluate_candidate(
    search,
    template,
    candidate
):

    x = candidate["x"]
    y = candidate["y"]

    h, w = template.shape[:2]

    patch = get_patch(
        search,
        x,
        y,
        w,
        h
    )

    if patch is None:
        return None

    raw = candidate["raw_score"]

    raw_normalized = float(
        np.clip(raw, 0.0, 1.0)
    )

    e_score = edge_score(
        template,
        patch
    )

    g_score = gradient_score(
        template,
        patch
    )

    p_score = projection_score(
        template,
        patch
    )

    # ------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------

    combined = (
        0.40 * raw_normalized
        +
        0.20 * e_score
        +
        0.20 * g_score
        +
        0.20 * p_score
    )

    return {
        "x": x,
        "y": y,

        "raw_score": raw_normalized,
        "edge_score": e_score,
        "gradient_score": g_score,
        "projection_score": p_score,

        "combined_score": float(
            np.clip(
                combined,
                0.0,
                1.0
            )
        )
    }


# ================================================================
# LOCAL REFINEMENT
# ================================================================

def local_refinement(
    search,
    template,
    x,
    y,
    radius=6
):

    h, w = template.shape[:2]

    sx1 = max(
        0,
        x - radius
    )

    sy1 = max(
        0,
        y - radius
    )

    sx2 = min(
        search.shape[1],
        x + w + radius
    )

    sy2 = min(
        search.shape[0],
        y + h + radius
    )

    region = search[
        sy1:sy2,
        sx1:sx2
    ]

    if (
        region.shape[0] < h
        or
        region.shape[1] < w
    ):
        return x, y

    result = cv2.matchTemplate(
        region,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    _, _, _, loc = cv2.minMaxLoc(
        result
    )

    return (
        sx1 + loc[0],
        sy1 + loc[1]
    )


# ================================================================
# CANDIDATE SEPARATION
# ================================================================

def candidate_separation(
    candidates
):

    if len(candidates) < 2:

        return {
            "second_score": 0.0,
            "margin": 1.0,
            "uniqueness": 1.0
        }

    best = candidates[0][
        "combined_score"
    ]

    second = candidates[1][
        "combined_score"
    ]

    margin = best - second

    uniqueness = (
        margin
        /
        max(abs(best), 1e-6)
    )

    return {
        "second_score": float(second),
        "margin": float(margin),
        "uniqueness": float(
            np.clip(
                uniqueness,
                0.0,
                1.0
            )
        )
    }


# ================================================================
# RECOVERY SEARCH
#
# Recovery uses:
#   - additional scales
#   - finer scale increments
#   - stronger edge weighting
# ================================================================

def recovery_search(
    reference,
    search,
    initial_candidates
):

    recovery_candidates = []

    recovery_scales = [
        0.895,
        0.905,
        0.915,
        0.925,
        0.935,
        0.945,
        0.955,
        0.965,
        0.975,
        0.985,
        0.995,
        1.005,
        1.015,
        1.025,
        1.035,
        1.045,
        1.055,
        1.065,
        1.075,
        1.085,
        1.095,
        1.105
    ]

    recovery_angles = [
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ]

    for scale in recovery_scales:

        for angle in recovery_angles:

            template = transform_reference(
                reference,
                scale,
                angle
            )

            if (
                template.shape[0]
                >= search.shape[0]
                or
                template.shape[1]
                >= search.shape[1]
            ):
                continue

            result = template_matching(
                search,
                template
            )

            if result is None:
                continue

            raw_candidates = get_top_candidates(
                result,
                template.shape[1],
                template.shape[0],
                count=4
            )

            for candidate in raw_candidates:

                evaluated = evaluate_candidate(
                    search,
                    template,
                    candidate
                )

                if evaluated is None:
                    continue

                # ------------------------------------------------
                # Recovery scoring
                # ------------------------------------------------

                recovery_score = (
                    0.35 * evaluated["raw_score"]
                    +
                    0.25 * evaluated["edge_score"]
                    +
                    0.25 * evaluated["gradient_score"]
                    +
                    0.15 * evaluated["projection_score"]
                )

                evaluated["recovery_score"] = (
                    float(
                        np.clip(
                            recovery_score,
                            0.0,
                            1.0
                        )
                    )
                )

                evaluated["scale"] = scale
                evaluated["angle"] = angle

                recovery_candidates.append(
                    evaluated
                )

    if not recovery_candidates:
        return None

    # ------------------------------------------------------------
    # Recovery consensus reranking
    # ------------------------------------------------------------
    # A true match should recur at roughly the same location across
    # several nearby scale/angle trials. A single periodic distractor
    # can win one trial, but is less trustworthy when it has little
    # cross-trial spatial support.
    # ------------------------------------------------------------

    for candidate in recovery_candidates:

        support = 0

        for other in recovery_candidates:

            dx = candidate["x"] - other["x"]
            dy = candidate["y"] - other["y"]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance <= CONSENSUS_RADIUS:
                # Count support from distinct transforms where possible.
                if (
                    candidate["scale"] != other["scale"]
                    or candidate["angle"] != other["angle"]
                ):
                    support += 1

        candidate["consensus_support"] = min(
            support,
            CONSENSUS_SUPPORT_CAP
        )

        support_bonus = (
            CONSENSUS_BONUS
            *
            min(
                candidate["consensus_support"]
                / max(CONSENSUS_MIN_SUPPORT, 1),
                1.0
            )
        )

        candidate["consensus_score"] = float(
            np.clip(
                candidate["recovery_score"] + support_bonus,
                0.0,
                1.0
            )
        )

    recovery_candidates.sort(
        key=lambda x:
        x["consensus_score"],
        reverse=True
    )

    return recovery_candidates


# ================================================================
# FINAL LOCALIZATION
# ================================================================

def localize(
    reference,
    search
):

    reference = normalize_image(
        reference
    )

    search = normalize_image(
        search
    )

    candidates = []

    # ============================================================
    # PRIMARY SEARCH
    # ============================================================

    for scale in SCALES:

        for angle in ANGLES:

            template = transform_reference(
                reference,
                scale,
                angle
            )

            if (
                template.shape[0]
                >= search.shape[0]
                or
                template.shape[1]
                >= search.shape[1]
            ):
                continue

            result = template_matching(
                search,
                template
            )

            if result is None:
                continue

            raw_candidates = get_top_candidates(
                result,
                template.shape[1],
                template.shape[0],
                TOP_CANDIDATES_PER_SCALE
            )

            for candidate in raw_candidates:

                evaluated = evaluate_candidate(
                    search,
                    template,
                    candidate
                )

                if evaluated is None:
                    continue

                evaluated["scale"] = scale
                evaluated["angle"] = angle

                candidates.append(
                    evaluated
                )

    if not candidates:
        return None

    # ============================================================
    # SORT
    # ============================================================

    candidates.sort(
        key=lambda x:
        x["combined_score"],
        reverse=True
    )

    # ============================================================
    # REMOVE SPATIAL DUPLICATES
    # ============================================================

    filtered = []

    for candidate in candidates:

        too_close = False

        for existing in filtered:

            dx = (
                candidate["x"]
                -
                existing["x"]
            )

            dy = (
                candidate["y"]
                -
                existing["y"]
            )

            distance = math.sqrt(
                dx * dx + dy * dy
            )

            if distance < 15:

                too_close = True
                break

        if not too_close:

            filtered.append(
                candidate
            )

        if len(filtered) >= FINAL_CANDIDATES:
            break

    candidates = filtered

    # ============================================================
    # SEPARATION
    # ============================================================

    separation = candidate_separation(
        candidates
    )

    best = candidates[0]

    best_confidence = (
        best["combined_score"]
    )

    # ============================================================
    # DETERMINE IF RECOVERY IS REQUIRED
    # ============================================================

    ambiguous = (
        separation["margin"]
        <
        MIN_SCORE_MARGIN
        or
        separation["uniqueness"]
        <
        MIN_UNIQUENESS
    )

    weak_match = (
        best_confidence
        <
        LOW_CONFIDENCE
    )

    need_recovery = (
        ambiguous
        or
        weak_match
    )

    recovery_used = False

    # ============================================================
    # RECOVERY
    # ============================================================

    if need_recovery:

        recovery_used = True

        recovery_candidates = recovery_search(
            reference,
            search,
            candidates
        )

        if recovery_candidates:

            # ----------------------------------------------------
            # Combine primary and recovery candidates
            # ----------------------------------------------------

            combined_candidates = (
                candidates
                +
                recovery_candidates
            )

            # ----------------------------------------------------
            # Sort by recovery-aware score
            # ----------------------------------------------------

            for candidate in combined_candidates:

                if "recovery_score" not in candidate:

                    candidate["recovery_score"] = (
                        0.35
                        * candidate["raw_score"]
                        +
                        0.25
                        * candidate["edge_score"]
                        +
                        0.25
                        * candidate["gradient_score"]
                        +
                        0.15
                        * candidate["projection_score"]
                    )

                if "consensus_score" not in candidate:
                    candidate["consensus_score"] = candidate["recovery_score"]
                    candidate["consensus_support"] = 0

            combined_candidates.sort(
                key=lambda x:
                x["consensus_score"],
                reverse=True
            )

            # ----------------------------------------------------
            # Spatial suppression again
            # ----------------------------------------------------

            final_candidates = []

            for candidate in combined_candidates:

                duplicate = False

                for existing in final_candidates:

                    dx = (
                        candidate["x"]
                        -
                        existing["x"]
                    )

                    dy = (
                        candidate["y"]
                        -
                        existing["y"]
                    )

                    distance = math.sqrt(
                        dx * dx
                        +
                        dy * dy
                    )

                    if distance < 15:

                        duplicate = True
                        break

                if not duplicate:

                    final_candidates.append(
                        candidate
                    )

                if len(final_candidates) >= 20:
                    break

            if final_candidates:

                best = final_candidates[0]

                best_confidence = (
                    best.get(
                        "recovery_score",
                        best["combined_score"]
                    )
                )

                candidates = final_candidates

    # ============================================================
    # FINAL REFINEMENT
    # ============================================================

    final_template = transform_reference(
        reference,
        best["scale"],
        best["angle"]
    )

    refined_x, refined_y = local_refinement(
        search,
        final_template,
        best["x"],
        best["y"],
        radius=6
    )

    # ============================================================
    # FINAL SEPARATION
    # ============================================================

    if len(candidates) >= 2:

        second = candidates[1]

        second_score = second.get(
            "recovery_score",
            second["combined_score"]
        )

    else:

        second_score = 0.0

    final_margin = (
        best_confidence
        -
        second_score
    )

    final_uniqueness = (
        final_margin
        /
        max(
            abs(best_confidence),
            1e-6
        )
    )

    return {
        "x": int(refined_x),
        "y": int(refined_y),

        "raw_score": float(
            best["raw_score"]
        ),

        "edge_score": float(
            best["edge_score"]
        ),

        "gradient_score": float(
            best["gradient_score"]
        ),

        "projection_score": float(
            best["projection_score"]
        ),

        "combined_score": float(
            best["combined_score"]
        ),

        "confidence": float(
            np.clip(
                best_confidence,
                0.0,
                1.0
            )
        ),

        "scale": float(
            best["scale"]
        ),

        "angle": float(
            best["angle"]
        ),

        "second_score": float(
            second_score
        ),

        "score_margin": float(
            final_margin
        ),

        "uniqueness": float(
            np.clip(
                final_uniqueness,
                0.0,
                1.0
            )
        ),

        "candidate_count": len(
            candidates
        ),

        "recovery_used": int(
            recovery_used
        ),

        "consensus_support": int(
            best.get("consensus_support", 0)
        ),

        "consensus_score": float(
            best.get("consensus_score", best.get("recovery_score", best["combined_score"]))
        ),

        "ambiguous": int(
            ambiguous
        ),

        "weak_match": int(
            weak_match
        ),

        "top_candidates": candidates[:10]
    }


# ================================================================
# READ LABEL
# ================================================================

def read_label(pair_dir):

    label_path = pair_dir / "label.txt"

    if not label_path.exists():
        return None, None

    text = label_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    numbers = []

    for token in (
        text
        .replace(",", " ")
        .split()
    ):

        try:
            numbers.append(
                float(token)
            )
        except ValueError:
            pass

    if len(numbers) >= 2:

        return (
            int(round(numbers[0])),
            int(round(numbers[1]))
        )

    return None, None


# ================================================================
# FAILURE CLASS
# ================================================================

def classify_failure(
    error,
    confidence,
    uniqueness,
    recovery_used
):

    if error <= SUCCESS_THRESHOLD:
        return "SUCCESS"

    if error > 300:
        base = "CATASTROPHIC_WRONG_LOCATION"

    elif error > 100:
        base = "LARGE_WRONG_LOCATION"

    else:
        base = "MEDIUM_WRONG_LOCATION"

    tags = [base]

    if confidence < LOW_CONFIDENCE:
        tags.append(
            "LOW_CONFIDENCE"
        )

    if uniqueness < MIN_UNIQUENESS:
        tags.append(
            "AMBIGUOUS"
        )

    if recovery_used:
        tags.append(
            "RECOVERY_USED"
        )

    return "|".join(tags)


# ================================================================
# FIND DATASET PAIRS
# ================================================================

def find_pairs():

    if not DATASET_DIR.exists():

        print()
        print(
            "ERROR: Dataset does not exist:"
        )
        print(
            DATASET_DIR.resolve()
        )

        return []

    pairs = []

    for item in sorted(
        DATASET_DIR.iterdir()
    ):

        if not item.is_dir():
            continue

        reference = (
            item / "reference.png"
        )

        search = (
            item / "search.png"
        )

        if (
            reference.exists()
            and
            search.exists()
        ):

            pairs.append(item)

    return pairs


# ================================================================
# MAIN

# ================================================================
# PARALLEL WORKER
# ================================================================

def process_pair(task):
    """
    Process exactly one pair using the original localization pipeline.
    Only execution scheduling is changed; localization/scoring logic is unchanged.
    """
    index, pair_dir = task
    pair_name = pair_dir.name

    try:
        reference = load_gray(pair_dir / "reference.png")
        search = load_gray(pair_dir / "search.png")

        actual_x, actual_y = read_label(pair_dir)

        result = localize(reference, search)

        if result is None:
            predicted_x = -1
            predicted_y = -1
            error = -1
            confidence = 0.0
            scale = 0.0
            angle = 0.0
            raw_score = 0.0
            edge = 0.0
            gradient = 0.0
            projection = 0.0
            combined = 0.0
            second_score = 0.0
            margin = 0.0
            uniqueness = 0.0
            candidate_count = 0
            recovery_used = 0
            consensus_support = 0
            consensus_score = 0.0
            ambiguous = 1
            weak_match = 1
            failure_class = "NO_MATCH"
        else:
            predicted_x = result["x"]
            predicted_y = result["y"]

            raw_score = result["raw_score"]
            edge = result["edge_score"]
            gradient = result["gradient_score"]
            projection = result["projection_score"]
            combined = result["combined_score"]
            confidence = result["confidence"]
            scale = result["scale"]
            angle = result["angle"]
            second_score = result["second_score"]
            margin = result["score_margin"]
            uniqueness = result["uniqueness"]
            candidate_count = result["candidate_count"]
            recovery_used = result["recovery_used"]
            consensus_support = result.get("consensus_support", 0)
            consensus_score = result.get("consensus_score", combined)
            ambiguous = result["ambiguous"]
            weak_match = result["weak_match"]

            if actual_x is not None and actual_y is not None:
                error = math.sqrt(
                    (predicted_x - actual_x) ** 2 +
                    (predicted_y - actual_y) ** 2
                )
            else:
                error = -1

            failure_class = classify_failure(
                error if error >= 0 else 999999,
                confidence,
                uniqueness,
                recovery_used
            )

            # Save failed cases exactly as before.
            if error > SUCCESS_THRESHOLD:
                fail_dir = FAILED_DIR / pair_name
                fail_dir.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(fail_dir / "reference.png"), reference)
                cv2.imwrite(str(fail_dir / "search.png"), search)

                diagnostic = {
                    "pair": pair_name,
                    "actual_x": actual_x,
                    "actual_y": actual_y,
                    "predicted_x": predicted_x,
                    "predicted_y": predicted_y,
                    "error": error,
                    "confidence": confidence,
                    "scale": scale,
                    "angle": angle,
                    "raw_score": raw_score,
                    "edge_score": edge,
                    "gradient_score": gradient,
                    "projection_score": projection,
                    "combined_score": combined,
                    "second_score": second_score,
                    "score_margin": margin,
                    "uniqueness": uniqueness,
                    "candidate_count": candidate_count,
                    "recovery_used": recovery_used,
                    "ambiguous": ambiguous,
                    "weak_match": weak_match,
                    "failure_class": failure_class,
                    "top_candidates": result["top_candidates"]
                }

                with open(
                    fail_dir / "diagnostic.json",
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(diagnostic, f, indent=2)

        row = {
            "pair": pair_name,
            "actual_x": actual_x,
            "actual_y": actual_y,
            "predicted_x": predicted_x,
            "predicted_y": predicted_y,
            "error": round(error, 4),
            "success": int(
                error >= 0 and error <= SUCCESS_THRESHOLD
            ),
            "raw_score": round(raw_score, 6),
            "edge_score": round(edge, 6),
            "gradient_score": round(gradient, 6),
            "projection_score": round(projection, 6),
            "combined_score": round(combined, 6),
            "confidence": round(confidence, 6),
            "scale": scale,
            "angle": angle,
            "second_score": round(second_score, 6),
            "score_margin": round(margin, 6),
            "uniqueness": round(uniqueness, 6),
            "candidate_count": candidate_count,
            "recovery_used": recovery_used,
            "consensus_support": consensus_support,
            "consensus_score": round(consensus_score, 6),
            "ambiguous": ambiguous,
            "weak_match": weak_match,
            "failure_class": failure_class
        }

        return index, row, error

    except Exception as e:
        print(f"ERROR in {pair_name}: {e}")

        row = {
            "pair": pair_name,
            "actual_x": -1,
            "actual_y": -1,
            "predicted_x": -1,
            "predicted_y": -1,
            "error": -1,
            "success": 0,
            "raw_score": 0,
            "edge_score": 0,
            "gradient_score": 0,
            "projection_score": 0,
            "combined_score": 0,
            "confidence": 0,
            "scale": 0,
            "angle": 0,
            "second_score": 0,
            "score_margin": 0,
            "uniqueness": 0,
            "candidate_count": 0,
            "recovery_used": 0,
            "consensus_support": 0,
            "consensus_score": 0,
            "ambiguous": 1,
            "weak_match": 1,
            "failure_class": "PROCESSING_ERROR"
        }
        return index, row, -1


# ================================================================
# MAIN - PARALLEL EXECUTION
# ================================================================

def main():

    print()
    print("Dataset:", DATASET_DIR.resolve())

    pairs = find_pairs()

    print("Dataset found:", len(pairs))
    print("=" * 70)

    if not pairs:
        return

    print()
    print("Starting FAST mode...")
    print("Localization algorithm: UNCHANGED")
    print("Parallel workers only: execution is distributed across CPU cores.")
    print()

    # Safe default for Windows + OpenCV.
    # Override from command line:
    #   set FAST_WORKERS=8
    # before running the script.
    cpu_count = os.cpu_count() or 2
    workers = int(os.environ.get(
        "FAST_WORKERS",
        str(max(1, min(6, cpu_count - 1)))
    ))
    workers = max(1, min(workers, len(pairs)))

    print(f"CPU threads available : {cpu_count}")
    print(f"FAST workers          : {workers}")
    print()

    results = []
    errors = []
    successful = 0
    failed = 0

    start_time = time.time()

    tasks = [
        (index, pair_dir)
        for index, pair_dir in enumerate(pairs, start=1)
    ]

    # One task per pair. Results are collected as they finish,
    # then restored to dataset order before writing the CSV.
    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = [
            executor.submit(process_pair, task)
            for task in tasks
        ]

        completed = 0

        for future in as_completed(futures):
            index, row, error = future.result()

            results.append((index, row))

            completed += 1

            if row["success"]:
                successful += 1
            else:
                failed += 1

            if error >= 0 and math.isfinite(error):
                errors.append(error)

            if completed % 25 == 0 or completed == len(pairs):
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 1e-6)

                print(
                    f"Processed: {completed}/{len(pairs)} | "
                    f"Success: {successful} | "
                    f"Failed: {failed} | "
                    f"Rate: {rate:.2f} pair/s"
                )

    # Restore exact dataset order for the CSV.
    results.sort(key=lambda x: x[0])
    results = [row for _, row in results]

    # ============================================================
    # SAVE CSV
    # ============================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pair",
        "actual_x",
        "actual_y",
        "predicted_x",
        "predicted_y",
        "error",
        "success",
        "raw_score",
        "edge_score",
        "gradient_score",
        "projection_score",
        "combined_score",
        "confidence",
        "scale",
        "angle",
        "second_score",
        "score_margin",
        "uniqueness",
        "candidate_count",
        "recovery_used",
        "consensus_support",
        "consensus_score",
        "ambiguous",
        "weak_match",
        "failure_class"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ============================================================
    # STATISTICS
    # ============================================================

    valid_errors = [
        e for e in errors
        if math.isfinite(e)
    ]

    total = len(results)

    accuracy = (
        successful / total * 100
        if total
        else 0
    )

    average_error = (
        float(np.mean(valid_errors))
        if valid_errors
        else 0
    )

    minimum_error = (
        float(np.min(valid_errors))
        if valid_errors
        else 0
    )

    maximum_error = (
        float(np.max(valid_errors))
        if valid_errors
        else 0
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Dataset          : {total} pairs")
    print(f"Successful       : {successful}")
    print(f"Failed           : {failed}")
    print(f"Accuracy         : {accuracy:.2f}%")
    print(f"Average Error    : {average_error:.2f} px")
    print(f"Minimum Error    : {minimum_error:.2f} px")
    print(f"Maximum Error    : {maximum_error:.2f} px")

    print()

    for threshold in [1, 3, 5, 10, 20, 50, 100]:
        count = sum(
            1 for e in valid_errors
            if e <= threshold
        )

        print(
            f"<= {threshold:3d} px       : {count}"
        )

    recovery_count = sum(
        row["recovery_used"]
        for row in results
    )

    ambiguous_count = sum(
        row["ambiguous"]
        for row in results
    )

    print()
    print(f"Recovery used    : {recovery_count}")
    print(f"Ambiguous cases  : {ambiguous_count}")
    print(f"Processing time  : {elapsed:.2f} sec")
    print(f"Speed            : {total / max(elapsed, 1e-6):.2f} pair/s")

    print()
    print("Output CSV:")
    print(OUTPUT_CSV.resolve())

    print()
    print("Failed cases:")
    print(FAILED_DIR.resolve())

    print()
    print("=" * 70)
    print("Successfully Completed")
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
