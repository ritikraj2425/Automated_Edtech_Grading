import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    'and','the','of','in','to','a','is','are','was','were',
    'by','on','at','for','with','this','that','it','be','an',
    'also','but','or','not','from','as','up','set'
}


def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_similarity(student_text, teacher_text):
    s = clean(student_text)
    t = clean(teacher_text)
    vec  = TfidfVectorizer()
    tfidf = vec.fit_transform([s, t])
    return round(float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]), 4)


def similarity_to_marks(sim, total=10):
    if sim >= 0.85:   marks = 9.0 + (sim - 0.85) / 0.15
    elif sim >= 0.70: marks = 7.0 + (sim - 0.70) / 0.15 * 2
    elif sim >= 0.55: marks = 5.0 + (sim - 0.55) / 0.15 * 2
    elif sim >= 0.40: marks = 3.0 + (sim - 0.40) / 0.15 * 2
    else:             marks = max(0.0, sim / 0.40 * 3)
    return round(min(marks, total), 1)


def grade(marks, total=10):
    r = marks / total
    if r >= 0.85: return "Excellent"
    if r >= 0.70: return "Good"
    if r >= 0.55: return "Average"
    if r >= 0.40: return "Below Average"
    return "Poor"


def keyword_breakdown(student_text, teacher_text):
    sw = set(clean(student_text).split()) - STOPWORDS
    tw = set(clean(teacher_text).split()) - STOPWORDS
    matched = sorted(sw & tw)
    missed  = sorted(tw - sw)
    return matched, missed


def mark_answer(student_text, teacher_text, total=10):
    """
    Main entry point called by Flask.
    Returns a dict with all result fields.
    """
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