First I read about layoutlm tesseract but Tesseract OCR will not automatically detect “answers” as logical regions. It only detects text-level boxes (characters, words, lines, paragraphs).
that's why I thought of using yolo which automatically detect answer boxes and instead of training yolo i used pre trained yolo model - layoutparser
These models often work reasonably well on answer sheets because answers appear as large text blocks.

