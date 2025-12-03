from __future__ import annotations

import json
from typing import Dict, Any

from openai import OpenAI

from .utils import safe_json_loads


def grade_essay(
    question: str,
    answer_key: str,
    student_text: str,
    max_score: float,
    api_url: str = "http://localhost:2911/v1",
    model: str = "Llama-3.1-8B-Instruct",
    temperature: float = 0.0,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Call local Llama API to grade an essay answer strictly vs answer key.

    Returns a dict with the schema specified in the project requirements.
    """
    prompt = f"""
You are an automated grading system.
Grade the student's answer STRICTLY based on the answer key.
Do NOT use outside knowledge.

QUESTION:
{question}

ANSWER KEY:
{answer_key}

STUDENT ANSWER:
{student_text}

RULES:
- Score from 0 to {max_score}.
- Correctness levels: correct, partially_correct, incorrect.
- Match meaning, not wording.
- Return STRICT JSON only.

JSON FORMAT:
{{
  "score": <float>,
  "max_score": <float>,
  "correctness": "correct | partially_correct | incorrect",
  "matched_points": [...],
  "missing_points": [...],
  "feedback": "short constructive suggestion"
}}
""".strip()

    # Initialize OpenAI-compatible client (llama.cpp server)
    client = OpenAI(base_url=api_url, api_key="not-needed")

    # First attempt: request JSON mode
    try:
        response = client.chat.completions.create(
            model=model or "local-model",
            messages=[
                {"role": "system", "content": "You are a grading engine that outputs strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Retry without response_format if server doesn't support it
        response = client.chat.completions.create(
            model=model or "local-model",
            messages=[
                {"role": "system", "content": "You are a grading engine that outputs strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )

    try:
        content = response.choices[0].message.content
    except Exception:
        raise ValueError(f"Unexpected API response format: {response}")

    result = safe_json_loads(content)

    # Ensure required fields and types
    result.setdefault("max_score", max_score)
    for key in ["score", "max_score"]:
        if key in result:
            try:
                result[key] = float(result[key])
            except Exception:
                raise ValueError(f"Field '{key}' must be a float.")

    if "correctness" not in result:
        raise ValueError("Missing 'correctness' in LLM result.")
    if result["correctness"] not in {"correct", "partially_correct", "incorrect"}:
        raise ValueError("Invalid 'correctness' value.")

    # Normalize lists
    for k in ["matched_points", "missing_points"]:
        v = result.get(k, [])
        if not isinstance(v, list):
            v = [str(v)]
        result[k] = [str(x) for x in v]

    if "feedback" not in result:
        result["feedback"] = ""

    return result
