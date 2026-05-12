"""Download 10 ESB common_voice clips and benchmark Granite-Speech on them.

Step 1: stream the first 10 samples from the Open ASR Leaderboard's parquet
        mirror and save them as .wav + .txt sidecars.
Step 2: run ibm-granite/granite-speech-4.1-2b-plus over each clip and report
        the Word Error Rate against the ground-truth transcript.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import soundfile as sf
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "data" / "common_voice"
NUM_SAMPLES = 50
SPLIT = "test"
TARGET_SR = 16_000

ASR_MODEL = "ibm-granite/granite-speech-4.1-2b-plus"
SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
ASR_PROMPT = "<|audio|> can you transcribe the speech into a written format?"


def download_samples(token: str) -> list[tuple[Path, str]]:
    """Stream the first NUM_SAMPLES clips; skip any that already exist on disk."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[tuple[Path, str]] = []

    wavs = [OUT_DIR / f"sample_{i:02d}.wav" for i in range(NUM_SAMPLES)]
    txts = [p.with_suffix(".txt") for p in wavs]
    if all(w.exists() and t.exists() for w, t in zip(wavs, txts)):
        print(f"Reusing {NUM_SAMPLES} cached clips in {OUT_DIR}", flush=True)
        for w, t in zip(wavs, txts):
            manifest.append((w, t.read_text(encoding="utf-8")))
        return manifest

    ds = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        "common_voice",
        split=SPLIT,
        streaming=True,
        token=token,
    )
    for i, sample in enumerate(ds.take(NUM_SAMPLES)):
        audio = sample["audio"]
        wav_path = wavs[i]
        txt_path = txts[i]
        transcript = sample.get("sentence") or sample.get("text") or ""

        sf.write(wav_path, audio["array"], audio["sampling_rate"])
        txt_path.write_text(transcript, encoding="utf-8")

        duration = len(audio["array"]) / audio["sampling_rate"]
        print(
            f"[{i + 1}/{NUM_SAMPLES}] {wav_path.name}  {duration:5.2f}s  {transcript[:60]}",
            flush=True,
        )
        manifest.append((wav_path, transcript))

    return manifest


def load_granite() -> tuple[AutoProcessor, AutoModelForSpeechSeq2Seq, str, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"\nLoading {ASR_MODEL} on {device} ({dtype})...", flush=True)

    processor = AutoProcessor.from_pretrained(ASR_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL, device_map=device, dtype=dtype,
    )
    model.eval()

    chat = [
        # {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ASR_PROMPT},
    ]
    prompt_text = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    return processor, model, prompt_text, device


@torch.inference_mode()
def transcribe_clip(
    processor: AutoProcessor,
    model: AutoModelForSpeechSeq2Seq,
    prompt_text: str,
    device: str,
    wav_path: Path,
) -> str:
    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, N]

    inputs = processor(
        prompt_text, audio_tensor, device=device, return_tensors="pt",
    ).to(device)
    outputs = model.generate(
        **inputs, max_new_tokens=2000, do_sample=False, num_beams=1,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True,
    ).strip()


def transcribe_and_score(manifest: list[tuple[Path, str]]) -> None:
    processor, model, prompt_text, device = load_granite()

    print(f"\n{'idx':>3}  {'WER':>6}", flush=True)
    wers: list[float] = []
    for i, (wav_path, reference) in enumerate(manifest):
        hypothesis = transcribe_clip(processor, model, prompt_text, device, wav_path)
        score = wer(reference, hypothesis)
        wers.append(score)
        print(f"{i:>3}  {score:6.3f}", flush=True)
        print(f"     ref: {reference}", flush=True)
        print(f"     hyp: {hypothesis}", flush=True)

    if wers:
        mean = sum(wers) / len(wers)
        print(f"\nMean WER over {len(wers)} samples: {mean:.3f}", flush=True)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ["HF_TOKEN"]

    manifest = download_samples(token)
    transcribe_and_score(manifest)


if __name__ == "__main__":
    main()
    # datasets' streaming HTTP prefetch threads can stall interpreter
    # shutdown after we stop iterating; bail out cleanly.
    os._exit(0)
