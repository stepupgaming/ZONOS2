"""End-to-end TTS generation script.

Generates audio from text using the ZONOS2 TTSLLM pipeline and saves it as a WAV file.

Usage:
    python generate_tts_audio.py --model-path ../dummy-checkpoint --text "Hello world" --output hello.wav
    python generate_tts_audio.py --use-dummy-weight --text "Hello world" --output hello.wav

Note: The dummy checkpoint produces random noise because it uses randomly initialized weights.
      For real speech, you need a trained model checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from zonos2.message.tts import TTSSamplingParams
from zonos2.tts.llm import TTSLLM


def _load_speaker_embedding_from_file(path: str, expected_dim: int):
    """Load a pre-computed speaker embedding from a .npy or .npz file."""
    import numpy as np
    import torch

    if not os.path.exists(path):
        raise FileNotFoundError(f"Speaker embedding file not found: {path}")

    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Failed to load speaker embedding from {path}: {exc}")

    arr = None
    if hasattr(loaded, "files"):
        try:
            if "emb" in loaded.files:
                arr = loaded["emb"]
            elif len(loaded.files) == 1:
                arr = loaded[loaded.files[0]]
            else:
                raise ValueError("Embedding archive must contain an 'emb' array or exactly one array.")
        finally:
            loaded.close()
    else:
        arr = loaded

    if isinstance(arr, np.ndarray) and arr.dtype.names:
        if "emb" in arr.dtype.names:
            arr = arr["emb"]
        elif len(arr.dtype.names) == 1:
            arr = arr[arr.dtype.names[0]]
        else:
            raise ValueError("Structured embedding arrays must contain an 'emb' field.")

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[-1] != expected_dim:
        raise ValueError(
            f"Speaker embedding dimension mismatch: model expects {expected_dim}, "
            f"but loaded file has {arr.shape[-1]}."
        )
    return torch.from_numpy(arr).to(dtype=torch.float32, device="cpu")


def _compute_speaker_embedding_from_audio(path: str, expected_dim: int):
    """Compute a speaker embedding from a reference audio file using Qwen3SpeakerEmbedding."""
    if expected_dim != 2048:
        raise RuntimeError(
            f"The release speaker encoder only supports 2048D embeddings; "
            f"model expects {expected_dim}D."
        )

    try:
        from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import Qwen3SpeakerEmbedding: {exc}. "
            "Ensure transformers and torchaudio are installed."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Speaker reference audio not found: {path}")

    import torch

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.T.copy())
    except Exception:
        try:
            import torchaudio
        except ImportError:
            raise RuntimeError(
                "soundfile or torchaudio is required for speaker embedding extraction. "
                "Install the project dependencies with uv."
            )
        wav, sample_rate = torchaudio.load(path)

    device = "cpu"
    print(f"  Extracting speaker embedding on {device}...")
    embedder = Qwen3SpeakerEmbedding(device=device)
    with torch.inference_mode():
        output = embedder(wav, sample_rate)

    if isinstance(output, tuple):
        candidates = [t.squeeze(0).to(dtype=torch.float32, device="cpu") for t in output]
    else:
        candidates = [output.squeeze(0).to(dtype=torch.float32, device="cpu")]

    for candidate in candidates:
        if candidate.numel() == expected_dim:
            return candidate

    raise ValueError(
        f"Speaker embedding dimension mismatch: model expects {expected_dim}, "
        f"but encoder produced {', '.join(str(c.numel()) for c in candidates)}."
    )


def _require_dac() -> None:
    """Check for the dac dependency and fail with a clear message if missing."""
    try:
        import dac  # noqa: F401
    except ImportError:
        print("[ERROR] The 'dac' (descript-audio-codec) package is required for audio decoding.")
        print("        Install the project dependencies with uv.")
        print("        Or use --no-vocoder to skip audio decoding entirely.")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZONOS2 TTS Audio Generation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../dummy-checkpoint",
        help="Path to the model checkpoint directory (default: ../dummy-checkpoint)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello world! This is a test of the ZONOS2 text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    parser.add_argument(
        "--use-dummy-weight",
        action="store_true",
        help="Use dummy weights for testing (produces random noise)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.15,
        help="Sampling temperature (default: 1.15)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=106,
        help="Top-k sampling (default: 106)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of audio tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=64,
        help="Minimum aligned audio frames before EOA can be sampled (default: 64)",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore sampled end-of-audio tokens and continue until max tokens.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2)",
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=50,
        help="Repetition window size in frames (default: 50)",
    )
    parser.add_argument(
        "--no-vocoder",
        action="store_true",
        help="Skip audio decoding and only output audio tokens (no WAV file)",
    )
    parser.add_argument(
        "--eos-decode-tail-frames",
        type=int,
        default=None,
        help=(
            "Frames to keep after aligned EOS when decoding delayed codebooks "
            "(default: n_codebooks - 1)."
        ),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Nucleus sampling probability (default: 0.0)",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.18,
        help="Minimum token probability (default: 0.18)",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="torch" if sys.platform == "win32" else "auto",
        help="Attention backend to use (default: torch on Windows, auto elsewhere)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        default=2048,
        help="Number of KV cache pages (default: 2048). Reduce this if you get 'Not enough memory for KV cache'.",
    )
    parser.add_argument(
        "--enable-cuda-graphs",
        action="store_true",
        help="Capture and replay fixed-shape CUDA graphs for decode.",
    )
    parser.add_argument(
        "--cuda-graph-bs",
        type=str,
        default="1",
        help="Comma-separated CUDA graph batch sizes to capture (default: 1).",
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=1,
        help="Maximum batch size for automatic CUDA graph capture (default: 1).",
    )
    parser.add_argument(
        "--cuda-graph-max-seq-len",
        type=int,
        default=None,
        help=(
            "Experimental maximum attention window captured inside CUDA graphs. "
            "Smaller values can reduce decode work but may break speech coherence."
        ),
    )
    parser.add_argument(
        "--allow-experimental-windowed-attention",
        action="store_true",
        help=(
            "Allow --cuda-graph-max-seq-len to be smaller than the full engine "
            "sequence length. This is a speed experiment and can reduce quality."
        ),
    )
    parser.add_argument(
        "--speaker-audio",
        type=str,
        default=None,
        help="Path to a reference audio file for voice cloning (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--speaker-embedding",
        type=str,
        default=None,
        help="Path to a pre-computed speaker embedding (.npy or .npz file)",
    )
    parser.add_argument(
        "--clean-speaker-background",
        action="store_true",
        help="Mark the speaker embedding as having a clean background",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--accurate-mode",
        dest="accurate_mode",
        action="store_true",
        default=True,
        help="Use accurate mode (default).",
    )
    mode_group.add_argument(
        "--expressive-mode",
        dest="accurate_mode",
        action="store_false",
        help="Use expressive mode instead of accurate mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en_us",
        help="Language code for text normalization (default: en_us)",
    )
    parser.add_argument(
        "--no-text-normalization",
        action="store_true",
        help="Disable text normalization before tokenization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Only require dac when the user actually wants audio decoding.
    if not args.no_vocoder:
        _require_dac()

    if args.speaker_audio and args.speaker_embedding:
        print("[ERROR] Provide either --speaker-audio or --speaker-embedding, not both.")
        sys.exit(1)

    print("=" * 60)
    print("ZONOS2 TTS Audio Generation")
    print("=" * 60)
    print(f"  Model path: {args.model_path}")
    print(f"  Text: {args.text[:60]}{'...' if len(args.text) > 60 else ''}")
    print(f"  Output: {args.output}")
    print(f"  Attention backend: {args.attention_backend}")
    if args.speaker_audio:
        print(f"  Speaker reference: {args.speaker_audio}")
    elif args.speaker_embedding:
        print(f"  Speaker embedding: {args.speaker_embedding}")
    if args.use_dummy_weight:
        print("  [WARNING] Using dummy weights - output will be random noise, not speech!")
    print()

    # Build sampling params
    sampling_params = TTSSamplingParams(
        temperature=args.temperature,
        topk=args.topk,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        ignore_eos=args.ignore_eos,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        seed=args.seed,
    )

    # Build kwargs for TTSLLM / SchedulerConfig
    try:
        cuda_graph_bs = [
            int(part.strip())
            for part in args.cuda_graph_bs.split(",")
            if part.strip()
        ]
    except ValueError:
        print(f"[ERROR] Invalid --cuda-graph-bs value: {args.cuda_graph_bs!r}")
        sys.exit(1)
    if any(bs <= 0 for bs in cuda_graph_bs):
        print("[ERROR] --cuda-graph-bs values must be positive integers.")
        sys.exit(1)
    if (
        args.cuda_graph_max_seq_len is not None
        and args.cuda_graph_max_seq_len < args.num_pages
        and not args.allow_experimental_windowed_attention
    ):
        print("[ERROR] --cuda-graph-max-seq-len smaller than --num-pages is experimental.")
        print("        It can make speech incoherent by dropping prompt/text attention context.")
        print("        Remove it for quality, or add --allow-experimental-windowed-attention for benchmarking.")
        sys.exit(1)

    llm_kwargs: dict = {
        "attention_backend": args.attention_backend,
        "num_page_override": args.num_pages,
        "disable_cuda_graphs": not args.enable_cuda_graphs,
        "cuda_graph_bs": cuda_graph_bs,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "cuda_graph_max_seq_len": args.cuda_graph_max_seq_len,
    }
    if args.use_dummy_weight:
        llm_kwargs["use_dummy_weight"] = True

    # Initialize TTSLLM
    print("Initializing TTSLLM...")
    t0 = time.perf_counter()
    try:
        tts = TTSLLM(
            model_path=args.model_path,
            decode_audio=not args.no_vocoder,
            eos_decode_tail_frames=args.eos_decode_tail_frames,
            **llm_kwargs,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize TTSLLM: {type(e).__name__}: {e}")
        sys.exit(1)
    init_time = time.perf_counter() - t0
    print(f"  Initialized in {init_time:.2f}s")
    print(f"  Device: {tts.device}")
    print(f"  n_codebooks: {tts.n_codebooks}")
    print(f"  Vocoder: {'enabled' if tts._vocoder else 'disabled'}")
    print(f"  Speaker enabled: {tts.speaker_enabled}")
    print(f"  Speaker dim: {tts.speaker_embedding_dim}")
    print()

    try:
        # Resolve speaker embedding
        speaker_embedding = None
        if args.speaker_audio or args.speaker_embedding:
            if not tts.speaker_enabled:
                print("[WARNING] Speaker reference provided, but current model does not support speaker embeddings; ignoring.")
            else:
                expected_dim = tts.speaker_embedding_dim
                try:
                    if args.speaker_audio:
                        print("Computing speaker embedding from reference audio...")
                        t0_emb = time.perf_counter()
                        speaker_embedding = _compute_speaker_embedding_from_audio(args.speaker_audio, expected_dim)
                        print(f"  Embedding computed in {time.perf_counter() - t0_emb:.2f}s")
                    else:
                        print("Loading speaker embedding from file...")
                        speaker_embedding = _load_speaker_embedding_from_file(args.speaker_embedding, expected_dim)
                        print(f"  Embedding loaded: {speaker_embedding.numel()} dimensions")
                except Exception as e:
                    print(f"[ERROR] Failed to prepare speaker embedding: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)
            print()

        # Generate audio
        print("Generating audio tokens...")
        t0 = time.perf_counter()
        try:
            result = tts.generate_one(
                args.text,
                sampling_params=sampling_params,
                decode_audio=not args.no_vocoder,
                speaker_embedding=speaker_embedding,
                clean_speaker_background=args.clean_speaker_background,
                accurate_mode=args.accurate_mode,
            )
        except Exception as e:
            print(f"[ERROR] Generation failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        gen_time = time.perf_counter() - t0
        print(f"  Generated {len(result['audio_tokens'])} frames in {gen_time:.2f}s")
        print(f"  EOS frame: {result['eos_frame']}")
        print()

        # Save or report
        if args.no_vocoder:
            print("Vocoder disabled. Audio tokens generated but not decoded.")
            print(f"  Number of frames: {len(result['audio_tokens'])}")
            return

        if result["audio"] is None or len(result["audio"]) == 0:
            print("[WARNING] No audio was generated (audio bytes are empty).")
            print("  This can happen if the model hit EOS immediately or the vocoder failed.")
            return

        audio_duration = len(result["audio"]) / (44100 * 4)  # float32 @ 44.1kHz
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Sample rate: {result['sample_rate']} Hz")
        print()

        # Save WAV
        print(f"Saving audio to {args.output}...")
        try:
            tts.save_audio(result["audio"], args.output, sample_rate=result["sample_rate"])
            print(f"  [OK] Saved {os.path.getsize(args.output)} bytes to {args.output}")
        except Exception as e:
            print(f"[ERROR] Failed to save audio: {type(e).__name__}: {e}")
            sys.exit(1)

        # Summary
        print()
        print("=" * 60)
        print("Generation complete!")
        print(f"  Frames: {len(result['audio_tokens'])}")
        print(f"  Audio: {audio_duration:.2f}s @ {result['sample_rate']} Hz")
        print(f"  File: {os.path.abspath(args.output)}")
        print(f"  Total time: {init_time + gen_time:.2f}s")
        print("=" * 60)
    finally:
        # Cleanup on all exit paths
        tts.shutdown()


if __name__ == "__main__":
    main()
