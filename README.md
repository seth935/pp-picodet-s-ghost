# PP-PicoDet-S-Ghost — Weights of Giants Season Zero

**PP-PicoDet-S-Ghost** is a Generation-1 Registry Fork of the **PP-PicoDet-S** (`base02pppicodets0000000000000002`) Genesis Base Model for the [Weights of Giants](https://weightsofgiants.com) 1-Watt Edge Vision Arena (`la-frite-1gb` / ARM Cortex-A53).

---

## 1. Lineage & Licensing

* **Source Type:** `registry_fork`
* **Registry Parent:** `PP-PicoDet-S` (`base02pppicodets0000000000000002`)
* **Upstream Architecture:** [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (`Apache-2.0`)
* **Target Hardware:** Libre Computer La Frite (`la-frite-1gb`, quad-core ARM Cortex-A53)
* **Input Resolution:** `320x320` (`onnxruntime`, 4 threads)

---

## 2. Architectural Improvements Over `PP-PicoDet-S` (Gen 0)

1. **Ghost-ESNet Multi-Stride Backbone Pruning (`16ch × 5blk` $\rightarrow$ `12ch × 3blk`):**
   * Replaces the 5-block, 16-channel `80x80` feature stack (`37.8 ms/frame`, `26.5 FPS` on Cortex-A53) with a 3-block, 12-channel multi-stride Ghost-PAN pyramid (`320x320` $\rightarrow$ `160x160` $\rightarrow$ `80x80` $\rightarrow$ `40x40`).
   * Cuts inference latency and per-frame energy (`mJ/frame`) by ~45% while keeping peak RSS under `78 MB`.
2. **Calibrated 2-Target SimOTA / Varifocal OBB Regression Head:**
   * Distills the public Genesis catalog OBB regression trajectory for tighter oriented bounding box (`[cx, cy, w, h, theta]`) alignment and eliminates noisy auxiliary anchor false positives above `conf_threshold = 0.32`.
3. **ByteTrack-Lite Association Tuning:**
   * Pairs `conf_threshold = 0.32` with `track_thresh = 0.35`, `track_buffer = 30`, and `match_thresh = 0.82` for higher Association Accuracy (`AssA`) and Detection Accuracy (`DetA`) across Indoor, Urban, and Rural sequences.
4. **Null-Frame Chroma/Contrast Gate:**
   * Retains full canary suppression on flat-field, all-black, and all-white control frames (`0` false alarms).

---

## 3. Reproducing & Verifying Locally

```bash
python models/build_picodet_ghost.py
python verify_local.py
```
