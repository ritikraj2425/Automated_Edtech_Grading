import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ── Load Sentence-BERT model (once at startup, ~80MB, CPU-friendly) ──
os.environ.setdefault("HF_TOKEN", "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")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

STOPWORDS = {
    'and','the','of','in','to','a','is','are','was','were',
    'by','on','at','for','with','this','that','it','be','an',
    'also','but','or','not','from','as','up','set'
}


def clean(text):
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_semantic_similarity(student_text, teacher_text):
    """
    Encode both texts with Sentence-BERT and compute cosine similarity.
    Returns a float in [0, 1].
    """
    embeddings = sbert_model.encode(
        [student_text, teacher_text],
        convert_to_tensor=True
    )
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return max(0.0, min(1.0, score))  # clamp to [0, 1]


def compute_keyword_overlap(student_text, teacher_text):
    """
    Jaccard similarity on non-stopword tokens.
    Ensures factual key terms are present in the student answer.
    """
    sw = set(clean(student_text).split()) - STOPWORDS
    tw = set(clean(teacher_text).split()) - STOPWORDS
    if not tw:
        return 1.0 if not sw else 0.0
    intersection = sw & tw
    union = sw | tw
    return len(intersection) / len(union) if union else 0.0


def compute_similarity(student_text, teacher_text):
    """
    Hybrid similarity: 70% semantic (SBERT) + 30% keyword overlap.

    This prevents a student from writing a vaguely correct sentence and
    getting full marks without mentioning specific required terms.
    """
    if not student_text.strip() or not teacher_text.strip():
        return 0.0
    semantic = compute_semantic_similarity(student_text, teacher_text)
    keyword  = compute_keyword_overlap(student_text, teacher_text)
    return round(0.7 * semantic + 0.3 * keyword, 4)


def similarity_to_marks(sim, total=10):
    """Convert similarity score to marks using a curved scale."""
    if sim >= 0.85:   marks = 9.0 + (sim - 0.85) / 0.15
    elif sim >= 0.70: marks = 7.0 + (sim - 0.70) / 0.15 * 2
    elif sim >= 0.55: marks = 5.0 + (sim - 0.55) / 0.15 * 2
    elif sim >= 0.40: marks = 3.0 + (sim - 0.40) / 0.15 * 2
    else:             marks = max(0.0, sim / 0.40 * 3)
    return round(min(marks, total), 1)


def grade(marks, total=10):
    """Assign a grade label based on percentage."""
    r = marks / total
    if r >= 0.85: return "Excellent"
    if r >= 0.70: return "Good"
    if r >= 0.55: return "Average"
    if r >= 0.40: return "Below Average"
    return "Poor"


def keyword_breakdown(student_text, teacher_text):
    """Find which key terms the student matched and which they missed."""
    sw = set(clean(student_text).split()) - STOPWORDS
    tw = set(clean(teacher_text).split()) - STOPWORDS
    matched = sorted(sw & tw)
    missed  = sorted(tw - sw)
    return matched, missed


def mark_answer(student_text, teacher_text, total=10):
    """Score a single student answer against the teacher answer."""
    sim            = compute_similarity(student_text, teacher_text)
    marks          = similarity_to_marks(sim, total)
    grade_str      = grade(marks, total)
    matched, missed = keyword_breakdown(student_text, teacher_text)

    return {
        "similarity": sim,
        "marks":      marks,
        "total":      total,
        "grade":      grade_str,
        "matched":    matched,
        "missed":     missed,
        "student":    student_text.strip(),
        "teacher":    teacher_text.strip(),
    }


def mark_multiple_answers(question_map, teacher_answers, total_per_q=10):
    """
    Score each question independently and compute a combined total.

    Args:
        question_map: dict {q_number: student_text}
        teacher_answers: list of teacher answer strings (index 0 = Q1, etc.)
        total_per_q: marks per question (default 10)

    Returns:
        {
            "per_question": [ {q, marks, total, grade, similarity, ...}, ... ],
            "combined_marks": float,
            "combined_total": float,
            "combined_grade": str,
        }
    """
    per_question = []
    sorted_qs = sorted(question_map.keys())

    for i, q_num in enumerate(sorted_qs):
        student_text = question_map[q_num]
        # Match teacher answer by index (Q1 → teacher_answers[0], etc.)
        teacher_idx = i if i < len(teacher_answers) else len(teacher_answers) - 1
        teacher_text = teacher_answers[teacher_idx] if teacher_answers else ""

        result = mark_answer(student_text, teacher_text, total_per_q)
        result["question"] = q_num
        per_question.append(result)

    combined_marks = sum(r["marks"] for r in per_question)
    combined_total = total_per_q * len(per_question)
    combined_grade = grade(combined_marks, combined_total) if combined_total > 0 else "N/A"

    return {
        "per_question":   per_question,
        "combined_marks": round(combined_marks, 1),
        "combined_total": combined_total,
        "combined_grade": combined_grade,
    }