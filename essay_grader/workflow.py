from __future__ import annotations

import os
from typing import Dict, Any

from .reader import load_and_preprocess
from .ocr_extractor import OCRExtractor
from .llama_grader import grade_essay
from .utils import ensure_dir, load_json_file, save_json_file


def run_pipeline(
    image_path: str,
    configs_dir: str = "configs",
    results_dir: str = "results",
    ocr_lang: str = "en",
    api_url: str = "http://localhost:2911/v1",
    model: str = "Llama-3.1-8B-Instruct",
    ocr_mode: str = "grouped",
) -> Dict[str, Any]:
    img_bgr, img_gray = load_and_preprocess(image_path)
    # Use original image path for OCR; preprocessed array is available if needed.
    ocr_input_path = image_path
    from paddleocr import PaddleOCR as _POCR
    ocr = _POCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=ocr_lang,
    )
    result_objs = ocr.predict(input=ocr_input_path)
    ensure_dir(results_dir)
    for res in result_objs:
        try:
            res.save_to_json(results_dir)
        except Exception:
            pass

    extractor = OCRExtractor(lang=ocr_lang)
    if ocr_mode == "raw":
        raw_text = extractor.extract_full_text(image_path=ocr_input_path, results_dir=results_dir)
        student_answers = {"1": raw_text}
    else:
        student_answers = extractor.extract_answers(image_path=ocr_input_path, results_dir=results_dir)
    # Load questions and answer key
    questions = load_json_file(os.path.join(configs_dir, "questions.json"))
    answer_key = load_json_file(os.path.join(configs_dir, "answer_key.json"))

    grading: Dict[str, Any] = {}
    total_score = 0.0
    max_total_score = 0.0

    for qid, question_text in questions.items():
        key_text = answer_key.get(qid, "")
        student_text = student_answers.get(qid, "")
        max_score = 10.0  # default per question; can be customized per qid
        max_total_score += max_score

        try:
            result = grade_essay(
                question=question_text,
                answer_key=key_text,
                student_text=student_text,
                max_score=max_score,
                api_url=api_url,
                model=model,
            )
        except Exception as e:
            result = {
                "score": 0.0,
                "max_score": max_score,
                "correctness": "incorrect",
                "matched_points": [],
                "missing_points": [],
                "feedback": f"Grading failed: {e}",
            }

        grading[qid] = result
        total_score += float(result.get("score", 0.0))

    final = {
        "student_answers": student_answers,
        "grading": grading,
        "total_score": round(total_score, 2),
        "max_total_score": round(max_total_score, 2),
    }

    ensure_dir(results_dir)
    out_path = os.path.join(results_dir, "result.json")
    save_json_file(out_path, final)

    return final
