# owt-medium — H100 run

Console record for the `medium` config. `metrics.jsonl` holds the machine-readable
per-eval rows (step, lr, train_loss, validation_loss, elapsed); this file keeps
what the log has and the JSONL does not — the config banner and the throughput /
TFLOP/s / MFU series.

Run date: 2026-09-04 · `uv run cs336_basics/train.py medium`

## Hardware

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (81,559 MiB, compute capability 9.0) |
| Driver / CUDA | 580.95.05 / 13.0 |
| torch | 2.11.0+cu130 |
| Pod | `ppi-bench` (1× H100, 8 CPU, 72Gi) |

## Configuration

| | |
|---|---|
| Run | `owt-medium` (owt, medium) |
| Model | 134,105,856 params — d_model 768 × 12 layers × 12 heads, d_ff 2048 |
| Data | 2,727,120,464 train / 66,401,098 validation tokens |
| Budget | 20,000 steps × 64 batch × 512 ctx = 655,360,000 tokens, 4.68e+17 FLOPs |
| Schedule | lr 6.00e-04 → 6.00e-05, warmup 1,000, decay to step 20,000 |
| Precision | fp32 with TF32 matmuls (`matmul_precision="high"`), `peak_flops=495e12` |
| Device | cuda, seed 42 |

## Results

| | |
|---|---|
| Final train loss | 3.3968 |
| Final validation loss | 3.4865 |
| Best validation loss | **3.4686** @ step 19,250 |
| Wall clock | 140.0 min (8,390.2 s) |
| Mean throughput | 78,014 tok/s |
| Mean compute | 55.68 TFLOP/s |
| Steady-state MFU | 11.3% |

Peak memory was 80,735 / 81,559 MiB — 99.0% of the card, ~824 MiB spare. It held
flat there for the whole run.

### Throughput ramp

Cumulative mean, so it climbs as fixed startup cost amortises. Essentially
plateaued by step ~6,000 at 55.2 TFLOP/s and crept to 55.75 by the end.

| step | tok/s | TFLOP/s | MFU |
|---:|---:|---:|---:|
| 250 | 62,313 | 44.47 | 9.0% |
| 1,000 | 73,127 | 52.19 | 10.5% |
| 2,000 | 75,719 | 54.04 | 10.9% |
| 5,000 | 77,096 | 55.02 | 11.1% |
| 10,000 | 77,820 | 55.54 | 11.2% |
| 15,000 | 77,943 | 55.63 | 11.2% |
| 20,000 | 78,111 | 55.75 | 11.3% |

### Loss trajectory

| step | train | val |
|---:|---:|---:|
| 250 | 6.4444 | 6.3703 |
| 1,000 | 4.9755 | 4.9531 |
| 2,000 | 4.3492 | 4.3540 |
| 5,000 | 3.9942 | 3.9156 |
| 10,000 | 3.6846 | 3.6852 |
| 15,000 | 3.4765 | 3.5398 |
| 19,250 | 3.5182 | **3.4686** |
| 20,000 | 3.3968 | 3.4865 |

## Compared with the earlier runs

| | tinystories-small | owt-small | owt-medium |
|---|---:|---:|---:|
| Params | 22,696,448 | 45,224,448 | **134,105,856** |
| Steps × batch × ctx | 10,000 × 128 × 256 | 10,000 × 128 × 256 | 20,000 × 64 × 512 |
| Tokens | 327,680,000 | 327,680,000 | 655,360,000 |
| Budget FLOPs | 3.69e+16 | 5.90e+16 | **4.68e+17** |
| Wall clock | 16.6 min | 21.0 min | **140.0 min** |
| Mean tok/s | 329,004 | 259,540 | 78,014 |
| Mean TFLOP/s | 37.02 | 46.74 | **55.68** |
| Steady MFU | 7.5% | 9.5% | **11.3%** |
| Best val loss | 1.3872 | 4.0141 | **3.4686** |

The trend across the three is consistent: bigger model → fewer tokens/s but more
FLOP/s, because each token carries more arithmetic and the GPU is better fed.
MFU still tops out at 11.3% against the TF32 peak.

Against `owt-small` (same corpus, directly comparable): 3x the parameters, 2x the
tokens and 2x the context take best val loss from 4.0141 to 3.4686.

## Known efficiency ceiling

MFU is limited by attention memory traffic, not by the matmuls. The hand-written
`scaled_dot_product_attention` in `cs336_basics/modules.py` materialises the full
`[B, H, T, T]` score matrix — at 64 × 12 × 512 × 512 in fp32 that is 805 MB per
layer, ~9.5 GiB across 12 layers for each such tensor the autograd graph retains.
The decomposed `softmax` (subtract-max → exp → sum → divide) saves several of
them. That is what both pins memory at 99% and leaves the tensor cores idle.

Levers, largest first: a non-materialising attention (FlashAttention-style, i.e.
Assignment 2 territory), bf16 autocast (roughly doubles peak — `peak_flops` would
need updating alongside or MFU becomes misleading), and `torch.compile` (cheapest,
fuses the softmax/SiLU/RoPE chains, but cannot fix the O(T²) materialisation).
A `medium-compiled` config now exists in `configs.py` to test the third.

## Artifacts

`checkpoints/owt-medium/` — `curve.png`, `metrics.jsonl` (80 rows, steps
250–20,000), `final.pt` (1.6 GB), and 10 step checkpoints every 2,000 steps.

## Full console log

```
run        owt-medium  (owt, medium)
model      134,105,856 params  d_model 768 x 12 layers x 12 heads
data       2,727,120,464 train / 66,401,098 validation tokens
budget     20,000 steps x 64 batch x 512 ctx = 655,360,000 tokens, 4.68e+17 FLOPs
schedule   lr 6.00e-04 -> 6.00e-05, warmup 1,000, decay to step 20,000
device     cuda  seed 42  -> checkpoints/owt-medium
  step     250/20,000  lr 1.50e-04  train 6.4444  val 6.3703     62,313 tok/s   44.47 TFLOP/s ( 9.0% MFU)  eta 173.1m
  step     500/20,000  lr 3.00e-04  train 5.6363  val 5.6398     68,966 tok/s   49.22 TFLOP/s ( 9.9% MFU)  eta 154.4m
  step     750/20,000  lr 4.50e-04  train 5.2609  val 5.2388     71,644 tok/s   51.13 TFLOP/s (10.3% MFU)  eta 146.7m
  step   1,000/20,000  lr 6.00e-04  train 4.9755  val 4.9531     73,127 tok/s   52.19 TFLOP/s (10.5% MFU)  eta 141.9m
  step   1,250/20,000  lr 6.00e-04  train 4.6583  val 4.7412     74,100 tok/s   52.88 TFLOP/s (10.7% MFU)  eta 138.2m
  step   1,500/20,000  lr 5.99e-04  train 4.5123  val 4.5422     74,795 tok/s   53.38 TFLOP/s (10.8% MFU)  eta 135.1m
  step   1,750/20,000  lr 5.98e-04  train 4.4506  val 4.4272     75,316 tok/s   53.75 TFLOP/s (10.9% MFU)  eta 132.3m
  step   2,000/20,000  lr 5.96e-04  train 4.3492  val 4.3540     75,719 tok/s   54.04 TFLOP/s (10.9% MFU)  eta 129.8m
  step   2,250/20,000  lr 5.94e-04  train 4.2770  val 4.2685     75,538 tok/s   53.91 TFLOP/s (10.9% MFU)  eta 128.3m
  step   2,500/20,000  lr 5.92e-04  train 4.2042  val 4.2394     75,843 tok/s   54.13 TFLOP/s (10.9% MFU)  eta 126.0m
  step   2,750/20,000  lr 5.89e-04  train 4.1924  val 4.1768     76,098 tok/s   54.31 TFLOP/s (11.0% MFU)  eta 123.8m
  step   3,000/20,000  lr 5.85e-04  train 4.1469  val 4.1405     76,295 tok/s   54.45 TFLOP/s (11.0% MFU)  eta 121.7m
  step   3,250/20,000  lr 5.82e-04  train 4.1916  val 4.0822     76,467 tok/s   54.57 TFLOP/s (11.0% MFU)  eta 119.6m
  step   3,500/20,000  lr 5.77e-04  train 4.0547  val 4.0500     76,625 tok/s   54.69 TFLOP/s (11.0% MFU)  eta 117.6m
  step   3,750/20,000  lr 5.73e-04  train 4.0162  val 4.0519     76,771 tok/s   54.79 TFLOP/s (11.1% MFU)  eta 115.6m
  step   4,000/20,000  lr 5.67e-04  train 4.0287  val 4.0084     76,901 tok/s   54.88 TFLOP/s (11.1% MFU)  eta 113.6m
  step   4,250/20,000  lr 5.62e-04  train 3.9758  val 3.9795     76,792 tok/s   54.81 TFLOP/s (11.1% MFU)  eta 112.0m
  step   4,500/20,000  lr 5.56e-04  train 3.7971  val 3.9656     76,901 tok/s   54.88 TFLOP/s (11.1% MFU)  eta 110.1m
  step   4,750/20,000  lr 5.50e-04  train 4.0246  val 3.9488     77,002 tok/s   54.96 TFLOP/s (11.1% MFU)  eta 108.2m
  step   5,000/20,000  lr 5.43e-04  train 3.9942  val 3.9156     77,096 tok/s   55.02 TFLOP/s (11.1% MFU)  eta 106.3m
  step   5,250/20,000  lr 5.36e-04  train 3.9449  val 3.9032     77,182 tok/s   55.08 TFLOP/s (11.1% MFU)  eta 104.4m
  step   5,500/20,000  lr 5.29e-04  train 3.9139  val 3.8971     77,259 tok/s   55.14 TFLOP/s (11.1% MFU)  eta 102.5m
  step   5,750/20,000  lr 5.21e-04  train 3.8872  val 3.8952     77,331 tok/s   55.19 TFLOP/s (11.1% MFU)  eta 100.6m
  step   6,000/20,000  lr 5.13e-04  train 3.8453  val 3.8877     77,395 tok/s   55.24 TFLOP/s (11.2% MFU)  eta  98.8m
  step   6,250/20,000  lr 5.05e-04  train 3.9168  val 3.8378     77,327 tok/s   55.19 TFLOP/s (11.1% MFU)  eta  97.1m
  step   6,500/20,000  lr 4.96e-04  train 3.6999  val 3.8297     77,370 tok/s   55.22 TFLOP/s (11.2% MFU)  eta  95.3m
  step   6,750/20,000  lr 4.87e-04  train 3.8836  val 3.8259     77,421 tok/s   55.25 TFLOP/s (11.2% MFU)  eta  93.5m
  step   7,000/20,000  lr 4.78e-04  train 3.6912  val 3.8255     77,475 tok/s   55.29 TFLOP/s (11.2% MFU)  eta  91.6m
  step   7,250/20,000  lr 4.68e-04  train 3.8145  val 3.8045     77,523 tok/s   55.33 TFLOP/s (11.2% MFU)  eta  89.8m
  step   7,500/20,000  lr 4.59e-04  train 3.8196  val 3.8023     77,570 tok/s   55.36 TFLOP/s (11.2% MFU)  eta  88.0m
  step   7,750/20,000  lr 4.49e-04  train 3.7186  val 3.7911     77,614 tok/s   55.39 TFLOP/s (11.2% MFU)  eta  86.2m
  step   8,000/20,000  lr 4.38e-04  train 3.7827  val 3.7344     77,656 tok/s   55.42 TFLOP/s (11.2% MFU)  eta  84.4m
  step   8,250/20,000  lr 4.28e-04  train 3.8115  val 3.7460     77,572 tok/s   55.36 TFLOP/s (11.2% MFU)  eta  82.7m
  step   8,500/20,000  lr 4.18e-04  train 3.7834  val 3.7392     77,614 tok/s   55.39 TFLOP/s (11.2% MFU)  eta  80.9m
  step   8,750/20,000  lr 4.07e-04  train 3.7889  val 3.7517     77,654 tok/s   55.42 TFLOP/s (11.2% MFU)  eta  79.1m
  step   9,000/20,000  lr 3.96e-04  train 3.7234  val 3.7601     77,692 tok/s   55.45 TFLOP/s (11.2% MFU)  eta  77.3m
  step   9,250/20,000  lr 3.85e-04  train 3.6580  val 3.7240     77,728 tok/s   55.47 TFLOP/s (11.2% MFU)  eta  75.5m
  step   9,500/20,000  lr 3.74e-04  train 3.6135  val 3.7098     77,761 tok/s   55.50 TFLOP/s (11.2% MFU)  eta  73.7m
  step   9,750/20,000  lr 3.63e-04  train 3.7147  val 3.7115     77,790 tok/s   55.52 TFLOP/s (11.2% MFU)  eta  72.0m
  step  10,000/20,000  lr 3.52e-04  train 3.6846  val 3.6852     77,820 tok/s   55.54 TFLOP/s (11.2% MFU)  eta  70.2m
  step  10,250/20,000  lr 3.41e-04  train 3.6758  val 3.6847     77,730 tok/s   55.48 TFLOP/s (11.2% MFU)  eta  68.5m
  step  10,500/20,000  lr 3.30e-04  train 3.6362  val 3.6508     77,760 tok/s   55.50 TFLOP/s (11.2% MFU)  eta  66.7m
  step  10,750/20,000  lr 3.19e-04  train 3.6323  val 3.6626     77,789 tok/s   55.52 TFLOP/s (11.2% MFU)  eta  64.9m
  step  11,000/20,000  lr 3.08e-04  train 3.7048  val 3.6690     77,814 tok/s   55.54 TFLOP/s (11.2% MFU)  eta  63.2m
  step  11,250/20,000  lr 2.97e-04  train 3.5724  val 3.6552     77,840 tok/s   55.55 TFLOP/s (11.2% MFU)  eta  61.4m
  step  11,500/20,000  lr 2.86e-04  train 3.6864  val 3.6583     77,864 tok/s   55.57 TFLOP/s (11.2% MFU)  eta  59.6m
  step  11,750/20,000  lr 2.75e-04  train 3.6409  val 3.6555     77,887 tok/s   55.59 TFLOP/s (11.2% MFU)  eta  57.8m
  step  12,000/20,000  lr 2.64e-04  train 3.7687  val 3.6305     77,911 tok/s   55.60 TFLOP/s (11.2% MFU)  eta  56.1m
  step  12,250/20,000  lr 2.53e-04  train 3.6375  val 3.6143     77,837 tok/s   55.55 TFLOP/s (11.2% MFU)  eta  54.4m
  step  12,500/20,000  lr 2.42e-04  train 3.5753  val 3.6064     77,851 tok/s   55.56 TFLOP/s (11.2% MFU)  eta  52.6m
  step  12,750/20,000  lr 2.32e-04  train 3.6136  val 3.6391     77,871 tok/s   55.58 TFLOP/s (11.2% MFU)  eta  50.8m
  step  13,000/20,000  lr 2.22e-04  train 3.5740  val 3.5521     77,892 tok/s   55.59 TFLOP/s (11.2% MFU)  eta  49.1m
  step  13,250/20,000  lr 2.11e-04  train 3.6212  val 3.5890     77,908 tok/s   55.60 TFLOP/s (11.2% MFU)  eta  47.3m
  step  13,500/20,000  lr 2.01e-04  train 3.5873  val 3.5878     77,922 tok/s   55.61 TFLOP/s (11.2% MFU)  eta  45.6m
  step  13,750/20,000  lr 1.92e-04  train 3.5100  val 3.5815     77,934 tok/s   55.62 TFLOP/s (11.2% MFU)  eta  43.8m
  step  14,000/20,000  lr 1.82e-04  train 3.5423  val 3.5491     77,947 tok/s   55.63 TFLOP/s (11.2% MFU)  eta  42.0m
  step  14,250/20,000  lr 1.73e-04  train 3.5456  val 3.5482     77,895 tok/s   55.59 TFLOP/s (11.2% MFU)  eta  40.3m
  step  14,500/20,000  lr 1.64e-04  train 3.5625  val 3.5287     77,910 tok/s   55.60 TFLOP/s (11.2% MFU)  eta  38.6m
  step  14,750/20,000  lr 1.55e-04  train 3.6282  val 3.5440     77,926 tok/s   55.61 TFLOP/s (11.2% MFU)  eta  36.8m
  step  15,000/20,000  lr 1.47e-04  train 3.4765  val 3.5398     77,943 tok/s   55.63 TFLOP/s (11.2% MFU)  eta  35.0m
  step  15,250/20,000  lr 1.39e-04  train 3.5084  val 3.5353     77,960 tok/s   55.64 TFLOP/s (11.2% MFU)  eta  33.3m
  step  15,500/20,000  lr 1.31e-04  train 3.5336  val 3.5434     77,976 tok/s   55.65 TFLOP/s (11.2% MFU)  eta  31.5m
  step  15,750/20,000  lr 1.24e-04  train 3.5908  val 3.5195     77,991 tok/s   55.66 TFLOP/s (11.2% MFU)  eta  29.8m
  step  16,000/20,000  lr 1.17e-04  train 3.5376  val 3.5125     78,004 tok/s   55.67 TFLOP/s (11.2% MFU)  eta  28.0m
  step  16,250/20,000  lr 1.10e-04  train 3.4864  val 3.4810     77,971 tok/s   55.65 TFLOP/s (11.2% MFU)  eta  26.3m
  step  16,500/20,000  lr 1.04e-04  train 3.5224  val 3.5153     77,985 tok/s   55.66 TFLOP/s (11.2% MFU)  eta  24.5m
  step  16,750/20,000  lr 9.81e-05  train 3.5262  val 3.4884     78,000 tok/s   55.67 TFLOP/s (11.2% MFU)  eta  22.8m
  step  17,000/20,000  lr 9.25e-05  train 3.3785  val 3.5182     78,014 tok/s   55.68 TFLOP/s (11.2% MFU)  eta  21.0m
  step  17,250/20,000  lr 8.74e-05  train 3.4242  val 3.5356     78,027 tok/s   55.69 TFLOP/s (11.2% MFU)  eta  19.2m
  step  17,500/20,000  lr 8.27e-05  train 3.3903  val 3.4790     78,040 tok/s   55.70 TFLOP/s (11.3% MFU)  eta  17.5m
  step  17,750/20,000  lr 7.85e-05  train 3.4323  val 3.5018     78,053 tok/s   55.71 TFLOP/s (11.3% MFU)  eta  15.7m
  step  18,000/20,000  lr 7.46e-05  train 3.4170  val 3.5059     78,065 tok/s   55.71 TFLOP/s (11.3% MFU)  eta  14.0m
  step  18,250/20,000  lr 7.12e-05  train 3.4716  val 3.4986     78,029 tok/s   55.69 TFLOP/s (11.3% MFU)  eta  12.2m
  step  18,500/20,000  lr 6.83e-05  train 3.4368  val 3.4712     78,042 tok/s   55.70 TFLOP/s (11.3% MFU)  eta  10.5m
  step  18,750/20,000  lr 6.57e-05  train 3.5509  val 3.4975     78,054 tok/s   55.71 TFLOP/s (11.3% MFU)  eta   8.7m
  step  19,000/20,000  lr 6.37e-05  train 3.3652  val 3.4807     78,066 tok/s   55.71 TFLOP/s (11.3% MFU)  eta   7.0m
  step  19,250/20,000  lr 6.21e-05  train 3.5182  val 3.4686     78,078 tok/s   55.72 TFLOP/s (11.3% MFU)  eta   5.2m
  step  19,500/20,000  lr 6.09e-05  train 3.5166  val 3.4748     78,089 tok/s   55.73 TFLOP/s (11.3% MFU)  eta   3.5m
  step  19,750/20,000  lr 6.02e-05  train 3.4052  val 3.5011     78,100 tok/s   55.74 TFLOP/s (11.3% MFU)  eta   1.7m
  step  20,000/20,000  lr 6.00e-05  train 3.3968  val 3.4865     78,111 tok/s   55.75 TFLOP/s (11.3% MFU)  eta   0.0m
done       140.0 min, 78,014 tok/s, 55.68 TFLOP/s
artifacts  checkpoints/owt-medium/  (curve.png, metrics.jsonl, 11 checkpoints)
```
