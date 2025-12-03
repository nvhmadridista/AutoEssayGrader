from __future__ import annotations

import argparse
import json

from essay_grader.workflow import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoEssayGrader CLI")
    p.add_argument("--input", required=True, help="Path to essay image (jpg/png)")
    p.add_argument("--ocr-lang", default="en", help="OCR language: en or vi")
    p.add_argument("--api-url", default="http://localhost:2911/v1", help="OpenAI-compatible base URL of llama.cpp server")
    p.add_argument("--model", default="Llama-3.1-8B-Instruct", help="Model name exposed by the local API")
    p.add_argument("--ocr-mode", choices=["grouped", "raw"], default="grouped", help="Use grouped answers by question or raw full text")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        image_path=args.input,
        ocr_lang=args.ocr_lang,
        api_url=args.api_url,
           model=args.model,
           ocr_mode=args.ocr_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
