import os
import base64
import tempfile
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path

from ocr import process_image, process_pil_image
from marking import mark_answer, mark_multiple_answers

app = Flask(__name__)

UPLOAD_FOLDER   = "uploads"
ALLOWED_EXT     = {"png", "jpg", "jpeg", "pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def get_ext(filename):
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if "sheet" not in request.files:
        return jsonify({"error": "No answer sheet uploaded"}), 400

    sheet_file = request.files["sheet"]

    if sheet_file.filename == "" or not allowed(sheet_file.filename):
        return jsonify({"error": "Please upload a valid file (PNG, JPG, or PDF)"}), 400

    # Collect teacher answers from numbered form fields (teacher_answer_1, teacher_answer_2, ...)
    teacher_answers = []
    i = 1
    while True:
        key = f"teacher_answer_{i}"
        val = request.form.get(key, "").strip()
        if not val and i > 1:
            break
        if val:
            teacher_answers.append(val)
        elif i == 1:
            # First answer is required
            return jsonify({"error": "Please provide at least one correct answer"}), 400
        i += 1

    filename  = secure_filename(sheet_file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    sheet_file.save(save_path)

    ext = get_ext(filename)

    try:
        if ext == "pdf":
            # Convert PDF pages to images and process each
            pages = convert_from_path(save_path, dpi=200)
            all_block_texts = []
            all_question_map = {}
            annotated_images = []

            for page_img in pages:
                ann_bytes, block_texts, question_map = process_pil_image(page_img)
                all_block_texts.extend(block_texts)
                # Merge question maps (offset keys to avoid collision between pages)
                offset = max(all_question_map.keys()) if all_question_map else 0
                for q_num, text in question_map.items():
                    new_key = q_num + offset if all_question_map else q_num
                    if new_key in all_question_map:
                        all_question_map[new_key] += "\n" + text
                    else:
                        all_question_map[new_key] = text
                annotated_images.append(base64.b64encode(ann_bytes).decode("utf-8"))

            block_texts  = all_block_texts
            question_map = all_question_map
            img_b64_list = annotated_images
        else:
            # Single image
            annotated_bytes, block_texts, question_map = process_image(save_path)
            img_b64_list = [base64.b64encode(annotated_bytes).decode("utf-8")]

    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    # ── Scoring ──
    multi_mode = len(teacher_answers) > 1

    if multi_mode:
        # Per-question scoring
        result = mark_multiple_answers(question_map, teacher_answers)
        return jsonify({
            "mode":           "multi",
            "images":         img_b64_list,
            "blocks":         block_texts,
            "per_question":   result["per_question"],
            "combined_marks": result["combined_marks"],
            "combined_total": result["combined_total"],
            "combined_grade": result["combined_grade"],
        })
    else:
        # Single combined scoring (backward compatible)
        student_text = "\n".join(block_texts)
        result = mark_answer(student_text, teacher_answers[0])
        return jsonify({
            "mode":       "single",
            "images":     img_b64_list,
            "blocks":     block_texts,
            "similarity": result["similarity"],
            "marks":      result["marks"],
            "total":      result["total"],
            "grade":      result["grade"],
            "matched":    result["matched"],
            "missed":     result["missed"],
        })


if __name__ == "__main__":
    app.run(debug=True)