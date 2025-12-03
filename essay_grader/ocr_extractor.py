from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import json
import glob

class OCRExtractor:
    """Extract text from essay images using PaddleOCR and group by question."""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang

    @staticmethod
    def _is_question_header(text: str) -> bool:
        t = text.strip().lower()
        return (
            t.startswith("q") and t[1:2].isdigit()
            or t[:2].isdigit() and (t[2:3] in ".)" or t[1:2] in ").")
            or t.split()[0].rstrip(".)").isdigit()
        )

    def group_by_question(self, lines: List[str]) -> Dict[str, str]:
        """Group OCR lines into question answers using simple heuristics."""
        grouped: Dict[str, List[str]] = {}
        current_q = "1"
        grouped.setdefault(current_q, [])
        for text in lines:
            if self._is_question_header(text):
                # Extract question number token
                token = text.strip().split()[0]
                token = token.rstrip(".)").lstrip("qQ")
                if token.isdigit():
                    current_q = token
                    grouped.setdefault(current_q, [])
                    # If header contains more text after the number, keep it as part of answer
                    rest = text.strip()[len(text.strip().split()[0]) :].strip()
                    if rest:
                        grouped[current_q].append(rest)
                else:
                    grouped[current_q].append(text)
            else:
                grouped[current_q].append(text)

        # Merge lines per question
        merged: Dict[str, str] = {k: " ".join(v).strip() for k, v in grouped.items()}
        return merged

    def extract_answers(self, image_path: str, results_dir: Optional[str] = None) -> Dict[str, str]:
        """Group student answers by question, preferring saved JSON when provided.

        If `results_dir` is set and contains a JSON for `image_path`, reads rec_texts
        from that file. Otherwise, runs live PaddleOCR.predict on `image_path`.
        """
        lines: List[str] = []
        # Prefer saved JSON when available and specified
        if results_dir:
            try:
                base = Path(image_path).stem
                candidates = sorted(Path(results_dir).glob(f"{base}*_res.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                json_path = candidates[0] if candidates else None
                if json_path:
                    with json_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    rec_texts = data.get("rec_texts") or []
                    if isinstance(rec_texts, list):
                        lines = [str(t) for t in rec_texts]
            except Exception:
                lines = []

        # Fallback to live OCR if no lines were loaded
        if not lines:
            from paddleocr import PaddleOCR as _POCR
            ocr = _POCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=self._lang,
            )
            result = list(ocr.predict(input=image_path))
            for res in result:
                txts = getattr(res, "rec_texts", None)
                if isinstance(txts, list) and txts:
                    lines.extend([str(t) for t in txts])
                else:
                    try:
                        d = res.to_dict()
                        rec_texts = d.get("rec_texts")
                        if isinstance(rec_texts, list) and rec_texts:
                            lines.extend([str(t) for t in rec_texts])
                    except Exception:
                        pass

        return self.group_by_question(lines)

    def extract_full_text(self, image_path: str, results_dir: Optional[str] = None) -> str:
        """Return concatenated text, preferring saved JSON when provided.

        If `results_dir` contains a JSON for `image_path`, uses its rec_texts.
        Otherwise, runs live predict and concatenates rec_texts.
        """
        # Try saved JSON first
        if results_dir:
            try:
                base = Path(image_path).stem
                candidates = sorted(Path(results_dir).glob(f"{base}*_res.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                json_path = candidates[0] if candidates else None
                if json_path:
                    with json_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    rec_texts = data.get("rec_texts") or []
                    if isinstance(rec_texts, list):
                        return " ".join(str(t) for t in rec_texts).strip()
            except Exception:
                pass

        # Fallback to live OCR
        lines: List[str] = []
        from paddleocr import PaddleOCR as _POCR
        ocr = _POCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=self._lang,
        )
        result = list(ocr.predict(input=image_path))
        for res in result:
            txts = getattr(res, "rec_texts", None)
            if isinstance(txts, list) and txts:
                lines.extend([str(t) for t in txts])
            else:
                try:
                    d = res.to_dict()
                    rec_texts = d.get("rec_texts")
                    if isinstance(rec_texts, list) and rec_texts:
                        lines.extend([str(t) for t in rec_texts])
                except Exception:
                    pass
        return " ".join(lines).strip()