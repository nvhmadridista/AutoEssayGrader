from __future__ import annotations

import os
from typing import Dict, Any

from .reader import load_and_preprocess
from .ocr_extractor import OCRExtractor
from .vietnamese_corrector import ProtonXOfflineConfig, ProtonXOfflineCorrector
from .llama_grader import grade_essay
from .utils import ensure_dir, load_json_file, save_json_file


def _create_paddle_ocr(lang: str):
    """Create PaddleOCR with explicit resize/orientation defaults.

    Some PaddleOCR versions can default to a very small detector resize
    (e.g. limit_side_len=64 with limit_type='min'), which destroys Vietnamese
    diacritics. We set a safer max-side resize.
    """
    from paddleocr import PaddleOCR as _POCR

    # PaddleOCR pipeline interface (current package) uses explicit text_det_* args.
    try:
        return _POCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang=lang,
            text_det_limit_side_len=960,
            text_det_limit_type="max",
        )
    except (TypeError, ValueError):
        pass

    # PaddleOCR 2.x style
    return _POCR(
        use_angle_cls=True,
        lang=lang,
        det_limit_side_len=960,
        det_limit_type="max",
    )


def run_pipeline(
    image_path: str,
    configs_dir: str = "configs",
    results_dir: str = "results",
    ocr_lang: str = "vi",
    api_url: str = "http://localhost:2911/v1",
    model: str = "Llama-3.1-8B-Instruct",
    ocr_mode: str = "grouped",
    vn_corrector: str = "none",
    vn_model: str = "protonx-models/distilled-protonx-legal-tc",
    vn_top_k: int = 1,
) -> Dict[str, Any]:
    img_bgr, img_gray = load_and_preprocess(image_path)
    # Use original image path for OCR; preprocessed array is available if needed.
    ocr_input_path = image_path

    ocr = _create_paddle_ocr(ocr_lang)
    try:
        result_objs = ocr.predict(input=img_bgr)
    except Exception:
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

    if vn_corrector == "protonx_offline":
        corrector = ProtonXOfflineCorrector(ProtonXOfflineConfig(model=vn_model, top_k=vn_top_k))
        student_answers = {qid: corrector.correct(txt) for qid, txt in student_answers.items()}
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
