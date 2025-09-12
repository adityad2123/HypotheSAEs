"""LLM API utilities for HypotheSAEs."""

import os
import time
import ollama
import requests
from typing import Optional, Dict, Any, List

DEFAULT_MODEL = "llama3.2:1b"

def is_ollama_model(model: str):
    try:
        found_model = find_model(model)
    except ValueError:
        return False
    return found_model == model

def find_model(model_name: Optional[str]) -> str:

    models = ollama.list()
    if len(list(models)) == 0:
        raise ValueError("You do not have any models installed on Ollama, please install a model and try again.")
    model_details = {m.model: {"size": m.size, "param_size": m.details.parameter_size} for m in models['models']}
    smallest_model = min(model_details.items(), key=lambda x: x[1]["size"])

    if model_name in model_details:
        print(f"Found {model_name}, with {model_details[model_name]['param_size']} parameters")
        return model_name
    else:
        raise ValueError(f"You do not have {model_name} model installed on Ollama.")

_SUPPORTED_OLLAMA_KEYS = {
    "temperature", "seed", "stop", "stream"
}
def _to_ollama_options(
    max_tokens: int,
    kwargs: Optional[Dict[str, Any]]
) -> Dict[str, Any]:

    opts: Dict[str, Any] = {"num_predict": int(max_tokens)}
    if kwargs:
        for k, v in kwargs.items():
            if k in _SUPPORTED_OLLAMA_KEYS:
                opts[k] = v

    opts.setdefault("stream", False)
    return opts


def get_ollama_completions(
    prompts: List[str],
    model: str,
    max_tokens: int = 128,
    llm_sampling_kwargs: Optional[Dict[str, Any]] = None,
    # TODO: (future work) add a timeout parameter, because Ollama doesn't support it as its been implemented in other places
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    keep_alive: str = "30m",
    show_progress: bool = False,
) -> List[str]:
    """
    Simple 1:1 (prompt -> completion) via Ollama chat API.
    Order-preserving; returns List[str] aligned with `prompts`.
    """
    options = _to_ollama_options(max_tokens, llm_sampling_kwargs)

    out: List[str] = []
    for i, prompt in enumerate(prompts):
        attempt = 0
        while True:
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options=options,
                    keep_alive=keep_alive,
                )
                out.append(resp["message"]["content"])
                break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                attempt += 1
                if attempt >= max_retries:
                    raise
                wait = (backoff_factor ** (attempt - 1))
                if show_progress:
                    print(f"[Ollama] {type(e).__name__}: retrying in {wait:.1f}s "
                          f"(attempt {attempt}/{max_retries})")
                time.sleep(wait)

            except requests.exceptions.RequestException:
                # Non-retryable client/server error (bad request, invalid options, etc.)
                raise

    return out
