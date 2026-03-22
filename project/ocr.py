import cv2
import numpy as np
import pytesseract
from PIL import Image
from scipy.ndimage import uniform_filter1d

# Windows users: uncomment and set your path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_HEIGHT      = 40
MIN_WIDTH_RATIO = 0.3
GAP_TOLERANCE   = 20


def preprocess(img):
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
    pad  = 8
    h, w = img.shape[:2]
    crop = img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
    ch, cw = crop.shape[:2]
    scale  = 2.5 if (ch < 100 or cw < 300) else (1.8 if ch < 200 else 1.0)
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return pytesseract.image_to_string(pil, config="--oem 3 --psm 6 -l eng").strip()


PALETTE = [
    (220, 60, 60), (60, 160, 220), (60, 200, 80),
    (220, 150, 40), (160, 60, 220), (40, 200, 200),
]

def annotate(img, blocks):
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
    """
    Main entry point called by Flask.
    Returns: (annotated_img_bytes, list of extracted text strings per block)
    """
    img    = cv2.imread(image_path)
    blocks = detect_blocks(img)
    texts  = [ocr_block(img, *b) for b in blocks]
    ann    = annotate(img, blocks)

    # Encode annotated image to bytes for sending to browser
    _, buf = cv2.imencode(".png", ann)
    return buf.tobytes(), texts