# AutoEssayGrader

A CPU-only, cross-platform (Linux/macOS/Windows) pipeline that:

- Extracts handwritten/typed essay answers from an image using PaddleOCR
- Grades answers with a local Llama-3.1-8B-Instruct API (LM Studio / Ollama / custom server)
- Compares against a provided answer key
- Returns structured JSON scoring output

## Requirements

- Python 3.10+
- CPU-only environment

## Installation

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Install PaddlePaddle CPU (if needed):

- Linux/macOS: `pip install paddlepaddle`
- Windows: `pip install paddlepaddle`

3. Install PaddleOCR:

```bash
pip install paddleocr
```

Note: On some systems you may need additional system packages for OpenCV (e.g., libgl).

## Run a Local Llama-3.1-8B-Instruct API

You can use LM Studio or Ollama or any server exposing a Chat Completions API compatible with:
`POST http://localhost:1234/v1/chat/completions`

- LM Studio:

  - Download Llama-3.1-8B-Instruct model.
  - Start the local server (Developer tab → Local Server).
  - Ensure the API endpoint is `http://localhost:1234/v1/chat/completions`.

- Ollama:
  - `ollama run llama3.1:8b-instruct` (ensure API is compatible or provide a bridge).

If you use a different port or URL, pass it via `--api-url`.

## Project Structure

```
project_root/
 ├── main.py
 ├── essay_grader/
 │    ├── __init__.py
 │    ├── reader.py          # load + preprocess image
 │    ├── ocr_extractor.py   # PaddleOCR extraction
 │    ├── llama_grader.py    # essay grading using Llama-3.1-8B
 │    ├── workflow.py        # pipeline: OCR → LLM → result JSON
 │    ├── utils.py
 ├── configs/
 │    ├── questions.json     # list of questions
 │    ├── answer_key.json    # correct sample answers
 ├── samples/
 │    ├── essay_sample.jpg   # placeholder (add your own sample image)
 ├── results/
 ├── requirements.txt
 └── README.md
```

## Usage

1. Place your essay image in `samples/` (e.g., `samples/essay_sample.jpg`).
2. Ensure `configs/questions.json` and `configs/answer_key.json` match your exam.
3. Start your local LLM API (LM Studio/Ollama/custom) and confirm endpoint.

Run the CLI:

```bash
python main.py --input samples/essay_sample.jpg \
	--ocr-lang en \
	--api-url http://localhost:1234/v1/chat/completions \
	--model Llama-3.1-8B-Instruct
```

The final JSON is printed and also saved to `results/result.json`.

## Example JSON Output

```json
{
  "student_answers": {
    "1": "...",
    "2": "...",
    "3": "..."
  },
  "grading": {
    "1": {
      "score": 7.5,
      "max_score": 10.0,
      "correctness": "partially_correct",
      "matched_points": ["mentions inequality", "notes financial crisis"],
      "missing_points": ["leadership weakness", "famine"],
      "feedback": "Add more detail on leadership and famine."
    }
  },
  "total_score": 21.0,
  "max_total_score": 30.0
}
```

## Notes

- OCR grouping uses simple heuristics (e.g., headers like `1.` or `Q2`). For complex layouts, consider custom post-processing.
- If OCR fails, the pipeline continues and returns grading errors for affected questions.
- This pipeline is CPU-only; PaddlePaddle and PaddleOCR will use CPU by default when installed without GPU.
