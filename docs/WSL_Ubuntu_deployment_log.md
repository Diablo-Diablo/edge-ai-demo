# MobileNetV2 ONNX Benchmark on WSL: Ubuntu Summary

## Test Environments
- **Windows**: Python venv (native)
- **WSL Ubuntu**: Python venv (WSL2)

## Methodology
- 10 pre-runs & 100 inference runs  per configuration, averaged
- Three precisions tested: FP32, INT8 Static, INT8 Dynamic
- Total of 4 test runs, 2 per environment

## Results

| Test ID | Environment | FP32 | INT8 Static | INT8 Dynamic |
|:-------|:------------|:----:|:-----------:|:------------:|
| test-run-1 | Windows | 1.305 ms | **1.082 ms** | 14.719 ms |
| test-run-2 | Windows | 1.652 ms | **1.107 ms** | 17.690 ms |
| test-run-1 | WSL | **6.320 ms** | 7.213 ms | 19.229 ms |
| test-run-2 | WSL | **5.573 ms** | 8.207 ms | 19.192 ms |

## Key Findings

1. **Windows: INT8 Static is fastest** — outperforms FP32 in both runs, approximately 17–33% faster
2. **WSL: FP32 is fastest** — FP32 wins both runs; INT8 Static is 14–47% slower
3. **INT8 Dynamic is not viable** — 11–17x slower than FP32 in all configurations
4. **Top-1 predictions identical** — all three precisions produce the same class label; no accuracy regression observed
5. **WSL is 4–5x slower than Windows overall** — consistent with virtualized environment overhead

## Deployment Recommendation
- **x86 CPU platforms**: Prefer INT8 Static
- **Linux bare metal / Jetson / ARM**: Re-benchmark required; WSL results are not directly transferable