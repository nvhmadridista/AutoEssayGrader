from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Optional


@dataclass(frozen=True)
class ProtonXOfflineConfig:
    model: str = "protonx-models/distilled-protonx-legal-tc"
    top_k: int = 1


class ProtonXOfflineCorrector:
    def __init__(self, cfg: Optional[ProtonXOfflineConfig] = None):
        self._cfg = cfg or ProtonXOfflineConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not importlib.util.find_spec("torch"):
                raise RuntimeError(
                    "ProtonX offline correction requires PyTorch. "
                    "Install CPU PyTorch in this venv, e.g.: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cpu"
                )

            try:
                import transformers  # type: ignore

                major = int(str(transformers.__version__).split(".")[0])
                if major >= 5:
                    raise RuntimeError(
                        "Your environment has transformers>=5, which is currently incompatible with "
                        "ProtonX offline tokenizer loading (KeyError: 0). "
                        "Please downgrade: pip install -U 'transformers<5'"
                    )
            except RuntimeError:
                raise
            except Exception:
                # If transformers isn't importable, ProtonX will fail anyway.
                raise RuntimeError(
                    "ProtonX offline correction requires the 'transformers' package. "
                    "Install it with: pip install -U 'transformers<5'"
                )

            try:
                from protonx import ProtonX
            except Exception as e:
                raise RuntimeError(
                    "ProtonX is not installed. Install it with: pip install --upgrade protonx"
                ) from e
            self._client = ProtonX(mode="offline")
        return self._client

    def correct(self, text: str) -> str:
        text = text or ""
        if not text.strip():
            return text

        client = self._get_client()
        result = client.text.correct(
            input=text,
            top_k=int(self._cfg.top_k),
            model=str(self._cfg.model),
        )

        try:
            candidates = result["data"][0]["candidates"]
            if candidates:
                return str(candidates[0]["output"])
        except Exception:
            pass

        return text
