"""
ocr.py — Phase 3
OCR pipeline using TrOCR-large-handwritten for text extraction.
Models are loaded ONCE at startup (globally) to avoid reload per request.
"""

import re
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ── Constants ──
MIN_HEIGHT          = 35
MIN_WIDTH_RATIO     = 0.3
GAP_TOLERANCE       = 20
LINE_GAP_TOLERANCE  = 10     # for intra-block line segmentation
MIN_LINE_HEIGHT     = 15     # minimum pixel height for a text line

# ── TrOCR model (loaded ONCE at startup) ──
print("[OCR] Loading TrOCR-large-handwritten model (first time downloads ~1.3 GB)...")
_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
_model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
_model.eval()
_device = "mps" if torch.backends.mps.is_available() else "cpu"
_model.to(_device)
print(f"[OCR] TrOCR loaded on {_device}.")

# ── Question label patterns ──
# Matches: Q1, Q.1, Q 1, q1, Ans 1, Answer 1, 1), 1., 1:, (1)
Q_LABEL_RE = re.compile(
    r'^\s*'
    r'(?:'
    r'(?:[Qq](?:uestion)?|[Aa](?:ns(?:wer)?)?)[.\s]*(\d+)'  # Q1, Ans 1, Answer 1
    r'|'
    r'\((\d+)\)'                                              # (1)
    r'|'
    r'(\d+)\s*[).:]\s*'                                       # 1) or 1. or 1:
    r')'
)


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(img):
    """Convert to binary for block detection (ink = white, bg = black)."""
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=10
    )
    return binary


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK DETECTION — two-pass gap-based approach
# ═══════════════════════════════════════════════════════════════════════════

def detect_blocks(img):
    """
    Find answer blocks via horizontal ink projection with adaptive gap splitting.

    Two-pass approach:
      1. Find all text line bands (minimal smoothing)
      2. Compute gaps between consecutive line bands
      3. Split into answer blocks where gaps are significantly larger than
         typical inter-line spacing (> 2× median gap or > MIN_BLOCK_GAP px)
    """
    binary = preprocess(img)
    h, w   = binary.shape

    # ── Pass 1: Find raw text line bands ──
    # Use small smoothing to bridge minor noise within a single line
    proj   = uniform_filter1d(np.sum(binary, axis=1).astype(float), size=5)
    thresh = proj.max() * 0.02 if proj.max() > 0 else 0

    in_line, start, line_bands = False, 0, []
    for i, val in enumerate(proj):
        if val > thresh and not in_line:
            start, in_line = i, True
        elif val <= thresh and in_line:
            line_bands.append((start, i))
            in_line = False
    if in_line:
        line_bands.append((start, h))

    # Filter out tiny noise bands
    line_bands = [(y1, y2) for y1, y2 in line_bands if y2 - y1 >= 8]

    if not line_bands:
        return []

    # ── Pass 2: Cluster line bands into answer blocks ──
    # Compute gaps between consecutive line bands
    if len(line_bands) <= 1:
        # Only one line band — treat as single block
        y1, y2 = line_bands[0]
        col_sum = np.sum(binary[y1:y2, :], axis=0)
        nz = np.where(col_sum > 0)[0]
        if len(nz) > 0:
            x1 = max(0, int(nz[0]) - 10)
            x2 = min(w, int(nz[-1]) + 10)
            return [(x1, y1, x2, y2)]
        return []

    gaps = []
    for i in range(1, len(line_bands)):
        gap = line_bands[i][0] - line_bands[i-1][1]
        gaps.append(gap)

    # Determine the split threshold:
    # Use 2× the median gap, but at least MIN_BLOCK_GAP pixels
    MIN_BLOCK_GAP = 20  # minimum gap in pixels to consider as block boundary
    median_gap = float(np.median(gaps)) if gaps else 0
    split_threshold = max(MIN_BLOCK_GAP, median_gap * 2.0)

    # Group consecutive line bands into blocks
    block_groups = [[line_bands[0]]]
    for i in range(1, len(line_bands)):
        gap = line_bands[i][0] - line_bands[i-1][1]
        if gap >= split_threshold:
            # This gap is large — start a new block
            block_groups.append([line_bands[i]])
        else:
            # Small gap — same block
            block_groups[-1].append(line_bands[i])

    # ── Build final block bounding boxes ──
    min_w = int(w * MIN_WIDTH_RATIO)
    blocks = []
    for group in block_groups:
        y1 = group[0][0]
        y2 = group[-1][1]
        if y2 - y1 < MIN_HEIGHT:
            continue
        col_sum = np.sum(binary[y1:y2, :], axis=0)
        nz = np.where(col_sum > 0)[0]
        if len(nz) == 0:
            continue
        x1 = max(0, int(nz[0]) - 10)
        x2 = min(w, int(nz[-1]) + 10)
        if x2 - x1 < min_w:
            continue
        blocks.append((x1, y1, x2, y2))

    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# LINE SEGMENTATION — split a block into individual text lines for TrOCR
# ═══════════════════════════════════════════════════════════════════════════

def segment_lines(block_img):
    """
    Split a block image into individual text lines using horizontal projection.
    Returns a list of PIL Images, one per line.
    """
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY) if len(block_img.shape) == 3 else block_img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    proj = np.sum(binary, axis=1).astype(float)
    proj = uniform_filter1d(proj, size=LINE_GAP_TOLERANCE)
    thresh = proj.max() * 0.03 if proj.max() > 0 else 0

    in_line, start, line_bands = False, 0, []
    for i, val in enumerate(proj):
        if val > thresh and not in_line:
            start, in_line = i, True
        elif val <= thresh and in_line:
            line_bands.append((start, i))
            in_line = False
    if in_line:
        line_bands.append((start, h))

    lines = []
    for (y1, y2) in line_bands:
        if y2 - y1 < MIN_LINE_HEIGHT:
            continue
        # Add a small vertical pad
        py1 = max(0, y1 - 4)
        py2 = min(h, y2 + 4)
        line_crop = block_img[py1:py2, :]
        # Convert to RGB PIL
        if len(line_crop.shape) == 3:
            pil_line = Image.fromarray(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB))
        else:
            pil_line = Image.fromarray(line_crop).convert("RGB")
        lines.append(pil_line)

    # If no lines detected, treat entire block as one line
    if not lines:
        if len(block_img.shape) == 3:
            pil_block = Image.fromarray(cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB))
        else:
            pil_block = Image.fromarray(block_img).convert("RGB")
        lines = [pil_block]

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# OCR — TrOCR per line
# ═══════════════════════════════════════════════════════════════════════════

def _trocr_recognize_line(pil_img):
    """Run TrOCR on a single line image. Returns the predicted text string."""
    pixel_values = _processor(images=pil_img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(_device)
    with torch.no_grad():
        generated_ids = _model.generate(pixel_values, max_new_tokens=128)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def ocr_block(img, x1, y1, x2, y2):
    """
    Read text from a single block using TrOCR.
    1. Crop the block from the image
    2. Segment into individual text lines
    3. Run TrOCR on each line
    4. Combine results
    """
    pad  = 8
    h, w = img.shape[:2]
    crop = img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]

    # Upscale small crops for better OCR
    ch, cw = crop.shape[:2]
    scale  = 2.5 if (ch < 100 or cw < 300) else (1.8 if ch < 200 else 1.0)
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Segment into lines
    lines = segment_lines(crop)

    # OCR each line
    line_texts = []
    for line_img in lines:
        text = _trocr_recognize_line(line_img)
        if text:
            line_texts.append(text)

    return "\n".join(line_texts).strip()


# ═══════════════════════════════════════════════════════════════════════════
# QUESTION LABEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_question_label(text):
    """
    Detect question number label at the start of a block's text.
    Returns (q_number, cleaned_text) if label found, else (None, original_text).
    """
    if not text or not text.strip():
        return None, text

    # Check only the first line for a Q-label
    lines = text.strip().split('\n')
    first_line = lines[0]

    match = Q_LABEL_RE.match(first_line)
    if match:
        q_num = int(match.group(1) or match.group(2) or match.group(3))
        cleaned_first = Q_LABEL_RE.sub('', first_line).strip()
        if cleaned_first:
            lines[0] = cleaned_first
        else:
            lines = lines[1:]
        return q_num, '\n'.join(lines).strip()

    return None, text


def _try_split_merged_block(text):
    """
    Handle edge case: single block contains multiple question answers.

    E.g., a single block might have:
        Q1. Answer to first question
        Q2. Answer to second question

    Returns a list of (q_number, text) tuples.
    """
    if not text or not text.strip():
        return []

    lines = text.strip().split('\n')
    segments = []
    current_q = None
    current_lines = []

    for line in lines:
        match = Q_LABEL_RE.match(line)
        if match:
            # Save previous segment
            if current_lines:
                segments.append((current_q, '\n'.join(current_lines).strip()))
            # Start new segment
            current_q = int(match.group(1) or match.group(2) or match.group(3))
            cleaned = Q_LABEL_RE.sub('', line).strip()
            current_lines = [cleaned] if cleaned else []
        else:
            current_lines.append(line)

    # Save last segment
    if current_lines:
        segments.append((current_q, '\n'.join(current_lines).strip()))

    return segments


def assign_questions(block_texts):
    """
    Map each block to a question number, handling edge cases:
    - Two blocks with the same Q-label → merge their text
    - One block with two Q-labels inside → split into separate questions
    - No labels found → assign sequentially (Block 1 → Q1, Block 2 → Q2)
    - Unlabeled block at start → assigned to Q1 (if no Q1 exists) or preceding Q
    """
    questions = {}  # q_num -> list of text chunks
    unlabeled_prefix = []
    last_detected_q = None

    for block_text in block_texts:
        # First, try to split if block has multiple Q-labels inside
        segments = _try_split_merged_block(block_text)

        if len(segments) > 1 and any(s[0] is not None for s in segments):
            for q_num, text in segments:
                if q_num is not None:
                    questions.setdefault(q_num, []).append(text)
                    last_detected_q = q_num
                elif last_detected_q is not None:
                    questions[last_detected_q].append(text)
                else:
                    unlabeled_prefix.append(text)
        else:
            q_num, cleaned = detect_question_label(block_text)
            if q_num is not None:
                questions.setdefault(q_num, []).append(cleaned)
                last_detected_q = q_num
            elif last_detected_q is not None:
                questions[last_detected_q].append(block_text)
            else:
                unlabeled_prefix.append(block_text)

    # If no labels were found at all, assign everything sequentially
    if not questions and unlabeled_prefix:
        return {i + 1: text for i, text in enumerate(unlabeled_prefix) if text.strip()}

    # Handle leading unlabeled blocks (before the first "Qx" label)
    if unlabeled_prefix:
        # If Q1 isn't taken, assume prefix is Q1. Otherwise, merge into the first found Q.
        target_q = 1 if 1 not in questions else min(questions.keys())
        # Insert at the beginning of the list for that question
        questions.setdefault(target_q, [])
        questions[target_q] = unlabeled_prefix + questions[target_q]

    return {q: '\n'.join(texts) for q, texts in sorted(questions.items())}


# ═══════════════════════════════════════════════════════════════════════════
# ANNOTATION — draw bounding boxes on image
# ═══════════════════════════════════════════════════════════════════════════

PALETTE = [
    (220, 60, 60), (60, 160, 220), (60, 200, 80),
    (220, 150, 40), (160, 60, 220), (40, 200, 200),
]

def annotate(img, blocks, question_map=None):
    """Draw colored boxes and labels on the image."""
    out       = img.copy()
    h, w      = out.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, w // 400)
    fscale    = max(0.55, w / 2000)

    for idx, (x1, y1, x2, y2) in enumerate(blocks):
        color          = PALETTE[idx % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label          = f"Block {idx+1}"
        (tw, th), base = cv2.getTextSize(label, font, fscale, thickness)
        ly             = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ly-th-4), (x1+tw+6, ly+base), color, cv2.FILLED)
        cv2.putText(out, label, (x1+3, ly), font, fscale, (255,255,255), thickness, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def process_image(image_path):
    """Process an image file: detect blocks, OCR, assign questions."""
    img    = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return _process_cv2_image(img)


def process_pil_image(pil_img):
    """Process a PIL Image (used for PDF pages)."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return _process_cv2_image(img)


def _process_cv2_image(img):
    """Core processing: detect blocks → OCR → assign questions → annotate."""
    blocks = detect_blocks(img)
    texts  = [ocr_block(img, *b) for b in blocks]
    question_map = assign_questions(texts)
    ann    = annotate(img, blocks, question_map)

    _, buf = cv2.imencode(".png", ann)
    return buf.tobytes(), texts, question_map