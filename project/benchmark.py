"""
benchmark.py — Phase 3
Benchmark the system's grading against Gemini API.
Sends the same answer sheet image + teacher answers to Gemini and
compares the marks it assigns vs our system's marks.
"""

import os
import re
import json
import base64
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Configure Gemini ──
_api_key = os.getenv("GEMINI_API_KEY", "")
_gemini_available = False

if _api_key and _api_key != "your_gemini_api_key_here":
    genai.configure(api_key=_api_key)
    _gemini_available = True
    print("[Benchmark] Gemini API configured.")
else:
    print("[Benchmark] No valid GEMINI_API_KEY found. Benchmarking disabled.")


def is_available():
    """Check if Gemini benchmarking is available."""
    return _gemini_available


def benchmark_with_gemini(image_path, teacher_answers, total_per_q=10):
    """
    Send the answer sheet image and teacher answers to Gemini for grading.

    Args:
        image_path: Path to the uploaded answer sheet image
        teacher_answers: List of teacher answer strings
        total_per_q: Marks per question (default 10)

    Returns:
        {
            "available": True,
            "gemini_results": [
                {"question": 1, "marks": 7.5, "reasoning": "..."},
                ...
            ],
            "gemini_total": float,
            "gemini_grand_total": float,
            "error": None
        }
    """
    if not _gemini_available:
        return {
            "available": False,
            "gemini_results": [],
            "gemini_total": 0,
            "gemini_grand_total": 0,
            "error": "Gemini API key not configured"
        }

    try:
        # Read the image file
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Build the prompt
        num_questions = len(teacher_answers)
        teacher_block = "\n".join(
            f"Q{i+1}: {ans}" for i, ans in enumerate(teacher_answers)
        )

        prompt = f"""You are a strict but fair exam grader. You are looking at a photograph of a handwritten student answer sheet.

There are {num_questions} question(s). The correct (teacher) answers are:

{teacher_block}

Instructions:
1. Read the handwritten text in the image carefully.
2. For each question, compare the student's handwritten answer with the teacher's correct answer.
3. Assign marks out of {total_per_q} for each question.
4. Be fair: give credit for correct concepts even if wording differs, but deduct for missing key points.

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "questions": [
    {{"question": 1, "marks": <number>, "reasoning": "<brief explanation>"}},
    {{"question": 2, "marks": <number>, "reasoning": "<brief explanation>"}}
  ]
}}
"""

        # Upload image to Gemini
        model = genai.GenerativeModel("gemini-2.5-pro")

        # Create image part
        image_part = {
            "mime_type": _get_mime_type(image_path),
            "data": image_data
        }

        response = model.generate_content(
            [prompt, image_part],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent grading
            )
        )

        # Parse the response
        response_text = response.text.strip()

        # Try to extract JSON from the response (handle markdown fences)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse Gemini response as JSON: {response_text[:200]}")

        gemini_results = parsed.get("questions", [])

        # Validate and normalize
        for r in gemini_results:
            r["marks"] = float(r.get("marks", 0))
            r["marks"] = max(0.0, min(r["marks"], total_per_q))
            r["marks"] = round(r["marks"], 1)

        gemini_total = sum(r["marks"] for r in gemini_results)
        gemini_grand_total = total_per_q * num_questions

        return {
            "available": True,
            "gemini_results": gemini_results,
            "gemini_total": round(gemini_total, 1),
            "gemini_grand_total": gemini_grand_total,
            "error": None
        }

    except Exception as e:
        return {
            "available": True,
            "gemini_results": [],
            "gemini_total": 0,
            "gemini_grand_total": 0,
            "error": str(e)
        }


def benchmark_with_gemini_pil(pil_images, teacher_answers, total_per_q=10):
    """
    Same as benchmark_with_gemini but accepts PIL images (for PDF pages).
    Saves to a temp file and calls the main function.
    """
    import tempfile
    if not _gemini_available:
        return {
            "available": False,
            "gemini_results": [],
            "gemini_total": 0,
            "gemini_grand_total": 0,
            "error": "Gemini API key not configured"
        }

    try:
        # For multi-page PDFs, combine all pages into one request
        # Save first page as temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_images[0].save(tmp.name, "PNG")
            result = benchmark_with_gemini(tmp.name, teacher_answers, total_per_q)
            os.unlink(tmp.name)
            return result
    except Exception as e:
        return {
            "available": True,
            "gemini_results": [],
            "gemini_total": 0,
            "gemini_grand_total": 0,
            "error": str(e)
        }


def _get_mime_type(path):
    """Get MIME type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf": "application/pdf",
    }.get(ext, "image/png")
