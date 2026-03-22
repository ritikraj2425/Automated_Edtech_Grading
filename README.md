# AUTOMATED_EDTECH_GRADING

Upload a handwritten answer sheet, it detects each answer block, reads the text using OCR, and gives marks by comparing it with the teacher's answer.

---

## What it does

- Takes a photo/scan of a handwritten answer sheet
- Detects where each answer block is on the page
- Reads the text inside each block using Tesseract OCR
- Compares extracted text with the teacher's answer using TF-IDF cosine similarity
- Returns marks out of 10, a grade, matched/missed keywords, and the sheet with colored boxes drawn on each block

---

## Project structure

```
AUTOMATED_EDTECH_GRADING/
│
├── project/                  # main Flask app (production code)
│   ├── templates/
│   │   └── index.html        # frontend UI
│   ├── uploads/              # uploaded sheets get saved here
│   ├── app.py                # Flask server — connects everything
│   ├── ocr.py                # block detection + Tesseract OCR
│   ├── marking.py            # TF-IDF scoring logic
│   └── requirements.txt
│
├── answer_sheet_ocr.ipynb    # testing OCR pipeline (notebook version)
├── marking.ipynb             # testing marking logic (notebook version)
├── approach.md               # research notes — what I tried and why
├── sheet.png                 # sample answer sheet for testing
├── teacher_answer.txt        # sample teacher answer
├── annotated_sheet_results.json
└── presentation.html         # project presentation for teacher
```

> The `.ipynb` files outside the `project/` folder were for testing and experimenting. The actual working code is inside `project/` as `.py` files.

---

## How I approached the block detection

First I read about LayoutLM and Tesseract, but Tesseract OCR will not automatically detect "answers" as logical regions — it only detects text-level boxes like characters, words, lines, and paragraphs.

That's why I thought of using YOLO, which can automatically detect answer boxes. Instead of training YOLO from scratch, I tried using a pre-trained YOLO model through LayoutParser. These models often work reasonably well on answer sheets because answers appear as large text blocks — but the pre-trained ones weren't trained on exam sheets and training a custom one needed too much labelled data.

So I ended up using OpenCV's horizontal projection — count ink pixels per row, rows with zero ink are the gaps between answers, group the rest into blocks. No training data needed, runs instantly.

---

## Tools used

| Tool | What it does |
|---|---|
| OpenCV | Block detection, image preprocessing, drawing boxes |
| Tesseract OCR | Reads text from each detected block |
| scikit-learn (TF-IDF) | Converts answers to vectors for comparison |
| Cosine Similarity | Measures how similar student and teacher answers are |
| Flask | Web server — connects browser to Python code |
| scipy | Smoothing the projection to bridge gaps within one answer |

---

## How to run locally

**1. Install Tesseract**

```bash
# Linux / Colab
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows — download from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

**2. Install Python dependencies**

```bash
cd project
pip install -r requirements.txt
```

**3. Run the server**

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## How to use

1. Upload a photo or scan of the answer sheet (PNG or JPG)
2. Type or paste the correct answer in the text box
3. Click Evaluate
4. It shows the annotated sheet with colored blocks, marks out of 10, and which keywords were matched or missed

**For multiple questions on one sheet** — right now paste all model answers together in the text box (Q1 answer, then Q2, then Q3). The system combines all detected blocks and compares them as one. Per-question scoring is planned for the next version.

---

## Current limitations

- Tesseract struggles with very messy handwriting — words get misread
- No per-question scoring yet — whole sheet gets one combined score
- TF-IDF matches words, not meaning — paraphrased answers score lower than they should
- Only image uploads supported, no PDF yet

---

## What's planned next

**Phase 2**
- Detect question number labels ("Q1", "1)", "2.") automatically to split and score each question separately
- Accept PDF uploads
- Deploy online

**Phase 3**
- Replace TF-IDF with Sentence-BERT for meaning-aware scoring
- Batch process an entire class's answer sheets
- Generate a full report per student

---

## Requirements

```
flask
opencv-python
pillow
pytesseract
numpy
scipy
scikit-learn
```