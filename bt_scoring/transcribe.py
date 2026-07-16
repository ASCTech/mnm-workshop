"""Bounded-segment ASR over the presidential-audio manifest with IBM Granite-Speech.

Reuses the load/generate pattern from ``../transcription/main.py``: librosa at
16kHz mono, a chat-template ASR prompt, greedy decoding on CUDA if available.
The one addition is chunking — our clips run minutes to tens of minutes, so we
only ever look at a bounded leading segment (default 90s) and slice that
segment into ~30s windows before calling the model, since Granite is tuned for
short utterances.

Transcripts are cached to ``transcripts/<identifier>.txt`` so a re-run skips
finished work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ASR_MODEL = "ibm-granite/granite-speech-4.1-2b-plus"
TARGET_SR = 16_000
ASR_PROMPT = "<|audio|> can you transcribe the speech into a written format?"


@dataclass
class Granite:
    processor: AutoProcessor
    model: AutoModelForSpeechSeq2Seq
    prompt_text: str
    device: str


def load_granite() -> Granite:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {ASR_MODEL} on {device} ({dtype})...", flush=True)

    processor = AutoProcessor.from_pretrained(ASR_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL, device_map=device, dtype=dtype,
    )
    model.eval()

    chat = [{"role": "user", "content": ASR_PROMPT}]
    prompt_text = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    return Granite(processor, model, prompt_text, device)


@torch.inference_mode()
def _transcribe_chunk(granite: Granite, audio_chunk) -> str:
    audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)  # [1, N]
    inputs = granite.processor(
        granite.prompt_text, audio_tensor, device=granite.device, return_tensors="pt",
    ).to(granite.device)
    outputs = granite.model.generate(
        **inputs, max_new_tokens=440, do_sample=False, num_beams=1,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return granite.processor.tokenizer.decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True,
    ).strip()


def transcribe_segment(
    granite: Granite,
    audio_path: Path,
    segment_seconds: float,
    chunk_seconds: float,
) -> str:
    """Load the first ``segment_seconds`` of ``audio_path`` and transcribe it
    in ``chunk_seconds`` windows, concatenating the per-chunk text."""
    audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True, duration=segment_seconds)
    chunk_len = int(chunk_seconds * TARGET_SR)
    min_len = int(1.0 * TARGET_SR)  # drop trailing scraps under 1s

    pieces: list[str] = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start:start + chunk_len]
        if len(chunk) < min_len:
            continue
        pieces.append(_transcribe_chunk(granite, chunk))
    return " ".join(p for p in pieces if p)


def transcribe_manifest(
    items: list[dict],
    transcripts_dir: Path,
    segment_seconds: float,
    chunk_seconds: float,
) -> tuple[dict[str, str], dict[str, str]]:
    """Transcribe every item not already cached under ``transcripts_dir``.

    Returns (identifier -> transcript text, identifier -> error message).
    """
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcripts: dict[str, str] = {}
    failures: dict[str, str] = {}

    todo = []
    for item in items:
        cache_path = transcripts_dir / f"{item['identifier']}.txt"
        if cache_path.exists():
            transcripts[item["identifier"]] = cache_path.read_text(encoding="utf-8")
        else:
            todo.append(item)

    print(f"Transcripts cached: {len(transcripts)}/{len(items)}; to run: {len(todo)}", flush=True)
    if not todo:
        return transcripts, failures

    granite = load_granite()
    for i, item in enumerate(todo):
        identifier = item["identifier"]
        audio_path = Path(item["audio_path"])
        cache_path = transcripts_dir / f"{identifier}.txt"
        try:
            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))
            text = transcribe_segment(granite, audio_path, segment_seconds, chunk_seconds)
            if not text.strip():
                raise ValueError("empty transcript")
            cache_path.write_text(text, encoding="utf-8")
            transcripts[identifier] = text
            print(f"[{i + 1}/{len(todo)}] {identifier}: {text[:80]!r}", flush=True)
        except Exception as e:  # per-item resilience: record and continue
            failures[identifier] = f"{type(e).__name__}: {e}"
            print(f"[{i + 1}/{len(todo)}] {identifier}: FAILED ({failures[identifier]})", flush=True)

    del granite
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return transcripts, failures
