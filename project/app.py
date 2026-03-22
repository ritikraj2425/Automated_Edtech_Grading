import os
import base64
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from ocr import process_image
from marking import mark_answer

app = Flask(__name__)

UPLOAD_FOLDER   = "uploads"
ALLOWED_EXT     = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if "sheet" not in request.files:
        return jsonify({"error": "No answer sheet uploaded"}), 400

    sheet_file    = request.files["sheet"]
    teacher_text  = request.form.get("teacher_answer", "").strip()

    if sheet_file.filename == "" or not allowed(sheet_file.filename):
        return jsonify({"error": "Please upload a valid image (png/jpg)"}), 400

    if not teacher_text:
        return jsonify({"error": "Please provide the correct answer"}), 400

    filename  = secure_filename(sheet_file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    sheet_file.save(save_path)

    try:
        annotated_bytes, block_texts = process_image(save_path)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    student_text = "\n".join(block_texts)

    result = mark_answer(student_text, teacher_text)

    img_b64 = base64.b64encode(annotated_bytes).decode("utf-8")

    return jsonify({
        "image":      img_b64,
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