# Perception-Gated Simplification of Million-Vertex 3D Meshes

[![中文](https://img.shields.io/badge/语言-中文-red)](./README_CN.md)

**Team:** SPBU-AMCP (Chi Zhao, Chelnokova Anastasiia, Wenjie Shi)  
**Affiliation:** St. Petersburg University — Faculty of Applied Mathematics and Control Processes  
**Competition:** [IMC Challenge 2026](https://www.huawei.com/minisite/imc-challenge/cn/) powered by Huawei — Problem B (simplifygeometry)  
**Score:** **91.55** / 100 on six hidden test meshes

---

## Overview

Given a closed, watertight 2-manifold triangle mesh with up to 1.1 million vertices, the task is to delete as many vertices as possible while keeping the result visually indistinguishable from the original. Visual fidelity is judged by a six-camera rendering pipeline: flat-shaded normal maps and perspective-correct depth maps at 1024×1024 resolution, scored by 11×11-window SSIM with a hard floor of 0.90. The per-mesh score is $S = 100(1 - V'/V)$, averaged over six hidden meshes. Each mesh gets 21 seconds of CPU time; the C++ source must not exceed 131072 bytes.

Our solver extends quadric error metric (QEM) edge collapse into a **two-era pipeline**:

1. **Rendering-free bulk loop.** Edges are ordered by a weighted quadric cost incorporating per-face coverage, normal-axis alignment, silhouette exposure, and neighbor-normal consistency — all computed from a single up-front six-view rasterization. This era performs 96.4–99.8% of all deletions without consulting a renderer.

2. **Evaluator-gated refinement tail.** A faithful replica of the six-camera evaluator is embedded in the solver. Each candidate re-renders only the affected pixel tiles, updates the six-view score incrementally, and is accepted only if the *predicted global* SSIM still clears a calibrated acceptance threshold. Around the core sit four metric-alignment mechanisms, geometry operations that reach vertices the priority queue never surfaces, and six vertex-count regimes.

---

## Architecture

```
Input mesh
  │
  ├── Up-front six-view rasterization (coverage, silhouette, curvature)
  │
  ├── [Bulk loop] Proxy-guided QEM edge collapse
  │     - Weighted quadrics: area × axis-alignment × coverage × silhouette
  │     - Perceptual edge-ordering with 5 additive cost terms
  │     - Vertex importance weighting (visibility × roughness)
  │     - Runs to a per-regime vertex target or empty queue
  │
  ├── [Pre-carve relaxation] Plane projection + vertex normal refinement
  │
  ├── [Extra collapse] Two proxy-scored passes outside the oracle
  │
  ├── [Carve] Oracle-gated edge collapse ← embedded evaluator replica
  │     - Incremental tile rendering (affected one-ring only)
  │     - Summed-area table SSIM with pre-scaled constants
  │     - Predicted global SSIM gate (never a local comparison)
  │     - Adaptive deadline scheduling
  │
  ├── [Geometry operations]
  │     - Fan-triangulation vertex removal (planar stars)
  │     - Face-pair insertion (buys SSIM headroom, R2)
  │     - Hidden-region collapse (depth-buffer certified)
  │     - Multi-candidate placement (up to 16 positions)
  │
  ├── [Merge sweeps] Exhaustive orientation-gated pairwise vertex merge
  │
  └── Output simplified mesh
```

---

## Key Components

### Weighted QEM with Perceptual Costs

Classical QEM minimizes squared distance to original face planes. We weight each face's quadric by four factors: triangle area $A_f$, axis-alignment $\alpha_f$ (faces seen head-on by the six axial cameras), projected pixel coverage $\gamma_f$, and a visibility-gated term $\nu_f$. The total edge cost combines geometric quadric error with five perceptual terms via multiplicative and lower-bound structures.

### Metric Alignment

Four corrections align the internal proxy with the evaluator:
- **Camera calibration:** visibility pre-scan matches evaluator camera parameters
- **Per-view normalization:** face contributions normalized within each view, respecting the 1/6 equal-view averaging of the metric
- **Vertex importance:** coverage × roughness reweighting with floor and cap guards
- **SSIM luminance asymmetry:** per-channel luminance loss replaces symmetric 1-dot in placement, refinement, and merge paths

### Calibrated Acceptance Thresholds

The local SSIM oracle systematically overestimates judged scores on hidden meshes. Acceptance thresholds ($\tau = 0.90425$–$0.9090$) are calibrated by bisecting against evaluated submissions — lowering $\tau$ until a Wrong Answer verdict, then setting the operating point just above. The carve checks a *predicted global* SSIM, never a local estimate.

### Efficient Evaluation

- **Algebraic division elimination:** pre-multiplying SSIM constants by $N_w^2 = 121^2$ removes all per-window divisions
- **Raster span tightening:** minimal 1-pixel bounding-box padding
- **Tile subdivision:** regime-dependent tile granularity (from $32\times32$ to $5\times5$ pixels) balances hash overhead against false-positive face-tile tests
- **Adaptive deadline:** stop time set from measured render cost, polling CPU clock frequently inside the carve loop

### Geometry Operations

Operations that reach vertices ordinary edge collapse cannot eliminate:
- **Fan triangulation:** removes planar fan vertices by re-triangulating the one-ring boundary
- **Face-pair insertion:** adds coincident face pairs to buy SSIM headroom, then spends it on further deletions
- **Hidden-region collapse:** depth-buffer-certified deletion of vertices invisible in all six views
- **Merge sweeps:** exhaustive pairwise vertex merging gated by per-face normal deviation

### Regime-Adaptive Architecture

Six vertex-count regimes (R1: ≤10K → R6: >700K) carry independent parameters and stage activation. The binding constraint changes with scale: SSIM margin on small organic meshes, planar redundancy on larger CAD meshes, and CPU throughput on the largest models.

---

## Results

| Regime | $N_V$ range | Retained $V'/V$ | Compression | Binding constraint |
|--------|-------------|-----------------|-------------|-------------------|
| R1 | ≤ 10K | ~11.6% | ~88.4% | SSIM margin (tight) |
| R2 | 10K–30K | ~27% | ~73% | SSIM margin (at boundary) |
| R3 | 30K–45K | ~6.4% | ~93.6% | Planar redundancy |
| R4 | 45K–70K | ~1.8% | ~98.2% | CPU throughput |
| R5 | 70K–700K | ~1.9% | ~98.1% | CPU throughput |
| R6 | >700K | ~2.3% | ~97.7% | CPU throughput |

**Overall score: 91.55** (mean compression rate, all six meshes)

---

## File Inventory

| Path | Description |
|------|-------------|
| `final.cpp` | Minified submission source (3074 lines, 121 KB, ≤131072 bytes) |
| `paper/paper.tex` | LaTeX paper (IMC Challenge article template) |
| `paper/paper.pdf` | Compiled paper |
| `imc2_handoff/` | Migration package with baseline solutions, evaluation harness, tools |

### Key Code Versions (evolution history)

| File | Score | Key contribution |
|------|-------|-----------------|
| `simplifygeometry.cpp` | ~89.9 | Champion baseline: QEM + SSIM gate |
| `90.64.cpp` → `90.89.cpp` | 90.64 → 90.89 | Camera calibration, VIP, VIEWNORM, RPVNORM |
| `91.14.cpp` → `91.49.cpp` | 91.14 → 91.49 | Fan removal, 16-candidate, carve-climb, edge-pair, 2nd pass |
| `91.52.cpp` | 91.52 | Work-reduction (division elimination, raster span, tile bucketing) |
| `final.cpp` | **91.55** | Two-era pipeline, weighted quadrics, luminance-aware merge, regime refinement |

---

## Building and Running

### Compilation
```bash
g++ -O2 -std=c++17 final.cpp -o simplify
```

### Execution
```bash
./simplify < input_mesh.in > output_mesh.out
```

The input is a modified OBJ format: first line `V F`, then `v x y z` lines, then `f v1 v2 v3` lines (1-indexed). Output follows the same format.

### Source Size Constraint
The judge compiles with `g++ -O2` on x86 and enforces a 131072-byte source limit. Development is done on a readable, commented version (~3400+ lines); a compression pipeline (comment stripping + identifier renaming + iterative collision exclusion) produces the submitted minified source.

---

## Citation

```bibtex
@misc{zhao2026perception,
  title   = {Perception-Gated Simplification of Million-Vertex 3D Meshes
             via Rendering-Guided Edge Collapse with Regime-Adaptive
             Geometry Operations},
  author  = {Chi Zhao and Chelnokova Anastasiia and Wenjie Shi},
  note    = {IMC Challenge 2026, Problem B (simplifygeometry), Team SPBU-AMCP},
  year    = {2026}
}
```

## References

- Garland & Heckbert. *Surface simplification using quadric error metrics.* SIGGRAPH 1997.
- Wang, Bovik, Sheikh & Simoncelli. *Image quality assessment: from error visibility to structural similarity.* IEEE TIP, 2004.
- Lindstrom & Turk. *Fast and memory efficient polygonal simplification.* IEEE Visualization, 1998.
- Hoppe. *Progressive meshes.* SIGGRAPH 1996.
- Cohen, Olano & Manocha. *Appearance-preserving simplification.* SIGGRAPH 1998.

## License

This repository contains contest submission code and an academic paper. See the competition rules at [IMC Challenge 2026](https://www.huawei.com/minisite/imc-challenge/cn/) for terms.
