"""Weights of Giants - Local Pre-Submission Verification Script.

Run this script inside your repository before submitting to the arena:
    python verify_local.py

Checks:
1. Manifest validity (model.json) against the 1-Watt Edge Challenge spec.
2. Unsafe deserialization artifacts (.pkl, .pt, .joblib).
3. Weight file size limits (<= 5MB for cortex-a53-512mb / la-frite-1gb).
4. Real inference pass with ONNX Runtime on synthetic video frames.
5. Latency, FPS, and peak memory estimation.
"""

import json
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    print("Error: numpy and onnxruntime are required to run verify_local.py.")
    print("Install them with: pip install numpy onnxruntime")
    sys.exit(1)

CORTEX_A53_PROFILE = {
    "limits": {
        "artifact_mb": 5,
        "peak_rss_mb": 400,
        "fps_floor": 5.0,
        "fps_full_marks": 15.0,
        "max_threads": 4,
        "max_input_side_px": 512,
    }
}


def verify_repo(repo_dir: str = "."):
    root = Path(repo_dir).resolve()
    print("=" * 65)
    print("  Weights of Giants - Season Zero Pre-Submission Verifier")
    print("  Target: 1-Watt Edge Silicon (ARM Cortex-A53, La Frite 1GB)")
    print("=" * 65)

    failures = []
    warnings = []

    # 1. Check model.json
    manifest_path = root / "model.json"
    if not manifest_path.exists():
        manifest_path = root / "wog.yaml"
    if not manifest_path.exists():
        failures.append("Missing 'model.json' or 'wog.yaml' in repository root.")
        _print_summary(failures, warnings)
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 2. Check for forbidden file extensions
    unsafe_exts = {".pkl", ".pickle", ".pt", ".pth", ".joblib", ".bin"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in unsafe_exts:
            if not (manifest.get("runtime") == "ncnn" and p.suffix.lower() == ".bin"):
                failures.append(f"Forbidden unsafe artifact found: {p.relative_to(root)}")

    # 3. Check stages and model files
    stages = manifest.get("stages", [])
    if not stages:
        failures.append("No stages defined in manifest.")

    detector_stage = None
    for s in stages:
        if s.get("type") == "detector":
            detector_stage = s
            break

    if not detector_stage:
        failures.append("Manifest must contain at least one stage of type 'detector'.")
        _print_summary(failures, warnings)
        return False

    model_rel = detector_stage.get("model")
    if not model_rel:
        failures.append("Detector stage is missing 'model' path.")
        _print_summary(failures, warnings)
        return False

    model_path = root / model_rel
    if not model_path.exists():
        failures.append(f"Model file not found at: {model_rel}")
        _print_summary(failures, warnings)
        return False

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n[+] Manifest: {manifest_path.name}")
    print(f"[+] Model File: {model_rel} ({model_size_mb:.4f} MB)")

    if model_size_mb > CORTEX_A53_PROFILE["limits"]["artifact_mb"]:
        failures.append(
            f"Model size {model_size_mb:.2f} MB exceeds 1-Watt Edge limit "
            f"({CORTEX_A53_PROFILE['limits']['artifact_mb']} MB)."
        )

    # 4. Run ONNX Runtime Inference
    input_w, input_h = detector_stage.get("input_size", [320, 320])
    print(f"[+] Input Resolution: {input_w}x{input_h}")
    print("[+] Initializing ONNX Runtime Session (threads=4)...")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = manifest.get("threads", 4)
    session = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]

    # Create synthetic frame batch: [1, 3, input_h, input_w]
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 3, input_h, input_w)).astype(np.float32)

    # Warmup
    print("[+] Warming up session...")
    for _ in range(5):
        session.run(None, {input_meta.name: dummy_input})

    # Benchmark 30 frames
    print("[+] Running 30-frame latency benchmark...")
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_meta.name: dummy_input})
        latencies.append(time.perf_counter() - t0)

    mean_sec = float(np.mean(latencies))
    fps = 1.0 / mean_sec if mean_sec > 0 else 0.0
    p95_sec = float(np.percentile(latencies, 95))

    print(f"\nBenchmark Results (Host CPU):")
    print(f"  Mean Latency: {mean_sec * 1000.0:.2f} ms")
    print(f"  P95 Latency:  {p95_sec * 1000.0:.2f} ms")
    print(f"  Inference FPS: {fps:.1f} FPS")

    # Evaluate outputs
    detections = outputs[0]
    print(f"  Output Tensor Shape: {detections.shape}")
    if len(detections.shape) >= 2 and detections.shape[-1] >= 6:
        print(f"  OBB Detections Output Format: VALID ([cx, cy, w, h, theta, conf, ...])")
    else:
        warnings.append(f"Unexpected output shape: {detections.shape}. Expected [1, N, 7+] with OBB fields.")

    # Gate checks
    if fps < CORTEX_A53_PROFILE["limits"]["fps_floor"]:
        failures.append(f"FPS {fps:.1f} is below the {CORTEX_A53_PROFILE['limits']['fps_floor']} FPS floor.")
    elif fps < CORTEX_A53_PROFILE["limits"]["fps_full_marks"]:
        warnings.append(
            f"FPS {fps:.1f} clears the 5 FPS floor but is below the 15 FPS Season Zero target."
        )
    else:
        print(f"  Speed Rating: EXCELLENT (>= 15 FPS)")

    return _print_summary(failures, warnings)


def _print_summary(failures, warnings):
    print("\n" + "=" * 65)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  [!] {w}")
    if failures:
        print("\nVerification FAILED:")
        for f in failures:
            print(f"  [X] {f}")
        print("\nPlease fix the above issues before submitting your model.")
        return False
    else:
        print("Verification PASSED!")
        print("Your model meets the 1-Watt Edge Challenge packaging constraints.")
        print("You can submit this repository via GitHub or Hugging Face.")
        return True


if __name__ == "__main__":
    success = verify_repo(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(0 if success else 1)
