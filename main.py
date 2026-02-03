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
    p.add_argument(
        "--vn-corrector",
        choices=["none", "protonx_offline"],
        default="none",
        help="Optional Vietnamese correction after OCR",
    )
    p.add_argument(
        "--vn-model",
        default="protonx-models/distilled-protonx-legal-tc",
        help="ProtonX offline model id",
    )
    p.add_argument(
        "--vn-top-k",
        type=int,
        default=1,
        help="Top-k candidates to generate; best candidate is used",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        image_path=args.input,
        ocr_lang=args.ocr_lang,
        api_url=args.api_url,
           model=args.model,
           ocr_mode=args.ocr_mode,
        vn_corrector=args.vn_corrector,
        vn_model=args.vn_model,
        vn_top_k=args.vn_top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
