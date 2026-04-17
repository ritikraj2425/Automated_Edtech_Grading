import re
import cv2
import numpy as np
import easyocr
from PIL import Image
from scipy.ndimage import uniform_filter1d

# ── Constants ──
MIN_HEIGHT      = 40
MIN_WIDTH_RATIO = 0.3
GAP_TOLERANCE   = 20

# ── EasyOCR reader (loaded once at startup) ──
reader = easyocr.Reader(['en'], gpu=False)

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


def detect_blocks(img):
    """Find answer blocks via horizontal ink projection."""
    binary = preprocess(img)
    h, w   = binary.shape
    proj   = uniform_filter1d(np.sum(binary, axis=1).astype(float), size=GAP_TOLERANCE)
    thresh = proj.max() * 0.02

    in_block, start, bands = False, 0, []
    for i, val in enumerate(proj):
        if val > thresh and not in_block:
            start, in_block = i, True
        elif val <= thresh and in_block:
            bands.append((start, i))
            in_block = False
    if in_block:
        bands.append((start, h))

    min_w, blocks = int(w * MIN_WIDTH_RATIO), []
    for (y1, y2) in bands:
        if y2 - y1 < MIN_HEIGHT:
            continue
        col_sum = np.sum(binary[y1:y2, :], axis=0)
        nz      = np.where(col_sum > 0)[0]
        if len(nz) == 0:
            continue
        x1, x2 = max(0, int(nz[0]) - 10), min(w, int(nz[-1]) + 10)
        if x2 - x1 < min_w:
            continue
        blocks.append((x1, y1, x2, y2))
    return blocks


def ocr_block(img, x1, y1, x2, y2):
    """Read text from a single block using EasyOCR."""
    pad  = 8
    h, w = img.shape[:2]
    crop = img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
    ch, cw = crop.shape[:2]
    scale  = 2.5 if (ch < 100 or cw < 300) else (1.8 if ch < 200 else 1.0)
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # EasyOCR: detail=0 returns list of strings
    results = reader.readtext(crop, detail=0, paragraph=True)
    return "\n".join(results).strip()


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
        # Extract question number from whichever group matched
        q_num = int(match.group(1) or match.group(2) or match.group(3))
        # Remove the label from the first line
        cleaned_first = Q_LABEL_RE.sub('', first_line).strip()
        if cleaned_first:
            lines[0] = cleaned_first
        else:
            lines = lines[1:]  # entire first line was just the label
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

    Returns: dict {q_number: combined_text}
    """
    questions = {}  # q_num -> list of text chunks
    has_any_label = False

    for block_text in block_texts:
        # First, try to split if block has multiple Q-labels inside
        segments = _try_split_merged_block(block_text)

        if len(segments) > 1 and any(s[0] is not None for s in segments):
            # Block contained multiple labeled questions
            has_any_label = True
            for q_num, text in segments:
                if q_num is not None and text:
                    questions.setdefault(q_num, []).append(text)
        else:
            # Normal case: one block = one answer
            q_num, cleaned = detect_question_label(block_text)
            if q_num is not None:
                has_any_label = True
                questions.setdefault(q_num, []).append(cleaned)
            else:
                # Will be handled below if no labels found at all
                questions.setdefault(None, []).append(block_text)

    if not has_any_label:
        # Fallback: assign sequentially — Block 1 → Q1, Block 2 → Q2
        return {i + 1: text for i, text in enumerate(block_texts) if text.strip()}

    # Handle unlabeled blocks: append to previous question or assign to next available
    if None in questions:
        unlabeled = questions.pop(None)
        if questions:
            # Append unlabeled text to the last numbered question
            last_q = max(questions.keys())
            questions[last_q].extend(unlabeled)
        else:
            # No labeled questions at all — shouldn't reach here, but just in case
            for i, text in enumerate(unlabeled):
                questions[i + 1] = [text]

    # Merge duplicate Q-number texts (two blocks with same Q-label)
    return {q: '\n'.join(texts) for q, texts in sorted(questions.items())}


# ── Annotation colors ──
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