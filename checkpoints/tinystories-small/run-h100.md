# tinystories-small — H100 run

Console record for the `h100` config. `metrics.jsonl` holds the machine-readable
per-eval rows (step, lr, train_loss, validation_loss, elapsed); this file keeps
what the log has and the JSONL does not — the config banner and the throughput /
TFLOP/s / MFU series.

Run date: 2026-09-04 · `uv run cs336_basics/train.py h100`

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
| Run | `tinystories-small` (tinystories, small) |
| Model | 22,696,448 params — d_model 512 × 4 layers × 16 heads |
| Data | 541,229,347 train / 5,465,883 validation tokens |
| Budget | 10,000 steps × 128 batch × 256 ctx = 327,680,000 tokens, 3.69e+16 FLOPs |
| Schedule | lr 1.00e-03 → 1.00e-04, warmup 500, decay to step 10,000 |
| Device | cuda, seed 42 |

## Results

| | |
|---|---|
| Final train loss | 1.3718 |
| Final validation loss | 1.3958 |
| Best validation loss | **1.3872** @ step 8,400 |
| Wall clock | 16.6 min (993.2 s) |
| Mean throughput | 329,004 tok/s |
| Mean compute | 37.02 TFLOP/s |
| Steady-state MFU | 7.5% |

Reported MFU implies a peak of ~494 TFLOP/s (37.02 / 0.075), i.e. the H100 SXM
BF16 dense tensor-core figure.

### Throughput ramp

Throughput is reported as a cumulative mean, so it climbs as fixed startup cost
amortises and settles by roughly step 6,000.

| step | tok/s | TFLOP/s | MFU |
|---:|---:|---:|---:|
| 200 | 202,909 | 22.83 | 4.6% |
| 600 | 273,902 | 30.82 | 6.2% |
| 1,000 | 296,116 | 33.32 | 6.7% |
| 2,000 | 315,609 | 35.51 | 7.2% |
| 4,000 | 324,412 | 36.50 | 7.4% |
| 6,000 | 327,479 | 36.84 | 7.4% |
| 8,000 | 328,964 | 37.01 | 7.5% |
| 10,000 | 329,915 | 37.12 | 7.5% |

### Loss trajectory

| step | train | val |
|---:|---:|---:|
| 200 | 3.2383 | 3.1882 |
| 1,000 | 1.8631 | 1.8864 |
| 2,000 | 1.7271 | 1.6927 |
| 4,000 | 1.5189 | 1.5351 |
| 6,000 | 1.4436 | 1.4729 |
| 8,000 | 1.4347 | 1.4109 |
| 8,400 | 1.3873 | 1.3872 |
| 10,000 | 1.3718 | 1.3958 |

## Artifacts

`checkpoints/tinystories-small/` — `curve.png`, `metrics.jsonl` (50 rows,
steps 200–10,000), `final.pt`, and checkpoints at steps 2,000 / 4,000 / 6,000 /
8,000 / 10,000.

## Full console log

```
run        tinystories-small  (tinystories, small)
model      22,696,448 params  d_model 512 x 4 layers x 16 heads
data       541,229,347 train / 5,465,883 validation tokens
budget     10,000 steps x 128 batch x 256 ctx = 327,680,000 tokens, 3.69e+16 FLOPs
schedule   lr 1.00e-03 -> 1.00e-04, warmup 500, decay to step 10,000
device     cuda  seed 42  -> checkpoints/tinystories-small
  step     200/10,000  lr 4.00e-04  train 3.2383  val 3.1882    202,909 tok/s   22.83 TFLOP/s ( 4.6% MFU)  eta  26.4m
  step     400/10,000  lr 8.00e-04  train 2.4379  val 2.4555    251,267 tok/s   28.27 TFLOP/s ( 5.7% MFU)  eta  20.9m
  step     600/10,000  lr 1.00e-03  train 2.2043  val 2.1654    273,902 tok/s   30.82 TFLOP/s ( 6.2% MFU)  eta  18.7m
  step     800/10,000  lr 9.98e-04  train 1.9706  val 1.9964    287,260 tok/s   32.32 TFLOP/s ( 6.5% MFU)  eta  17.5m
  step   1,000/10,000  lr 9.94e-04  train 1.8631  val 1.8864    296,116 tok/s   33.32 TFLOP/s ( 6.7% MFU)  eta  16.6m
  step   1,200/10,000  lr 9.88e-04  train 1.8494  val 1.8271    302,346 tok/s   34.02 TFLOP/s ( 6.9% MFU)  eta  15.9m
  step   1,400/10,000  lr 9.80e-04  train 1.7936  val 1.7848    306,954 tok/s   34.54 TFLOP/s ( 7.0% MFU)  eta  15.3m
  step   1,600/10,000  lr 9.71e-04  train 1.7490  val 1.7426    310,505 tok/s   34.93 TFLOP/s ( 7.1% MFU)  eta  14.8m
  step   1,800/10,000  lr 9.59e-04  train 1.7233  val 1.7092    313,335 tok/s   35.25 TFLOP/s ( 7.1% MFU)  eta  14.3m
  step   2,000/10,000  lr 9.46e-04  train 1.7271  val 1.6927    315,609 tok/s   35.51 TFLOP/s ( 7.2% MFU)  eta  13.8m
  step   2,200/10,000  lr 9.31e-04  train 1.6291  val 1.6669    315,851 tok/s   35.54 TFLOP/s ( 7.2% MFU)  eta  13.5m
  step   2,400/10,000  lr 9.14e-04  train 1.7041  val 1.6523    317,387 tok/s   35.71 TFLOP/s ( 7.2% MFU)  eta  13.1m
  step   2,600/10,000  lr 8.96e-04  train 1.6794  val 1.6475    318,717 tok/s   35.86 TFLOP/s ( 7.2% MFU)  eta  12.7m
  step   2,800/10,000  lr 8.76e-04  train 1.5660  val 1.6118    319,868 tok/s   35.99 TFLOP/s ( 7.3% MFU)  eta  12.3m
  step   3,000/10,000  lr 8.55e-04  train 1.6154  val 1.6101    320,855 tok/s   36.10 TFLOP/s ( 7.3% MFU)  eta  11.9m
  step   3,200/10,000  lr 8.32e-04  train 1.6313  val 1.5844    321,676 tok/s   36.19 TFLOP/s ( 7.3% MFU)  eta  11.5m
  step   3,400/10,000  lr 8.08e-04  train 1.5775  val 1.5835    322,490 tok/s   36.28 TFLOP/s ( 7.3% MFU)  eta  11.2m
  step   3,600/10,000  lr 7.84e-04  train 1.5657  val 1.5744    323,199 tok/s   36.36 TFLOP/s ( 7.3% MFU)  eta  10.8m
  step   3,800/10,000  lr 7.58e-04  train 1.5382  val 1.5614    323,842 tok/s   36.44 TFLOP/s ( 7.4% MFU)  eta  10.5m
  step   4,000/10,000  lr 7.31e-04  train 1.5189  val 1.5351    324,412 tok/s   36.50 TFLOP/s ( 7.4% MFU)  eta  10.1m
  step   4,200/10,000  lr 7.03e-04  train 1.5298  val 1.5299    324,067 tok/s   36.46 TFLOP/s ( 7.4% MFU)  eta   9.8m
  step   4,400/10,000  lr 6.75e-04  train 1.5542  val 1.5258    324,534 tok/s   36.51 TFLOP/s ( 7.4% MFU)  eta   9.4m
  step   4,600/10,000  lr 6.46e-04  train 1.5340  val 1.5157    324,955 tok/s   36.56 TFLOP/s ( 7.4% MFU)  eta   9.1m
  step   4,800/10,000  lr 6.17e-04  train 1.5552  val 1.5124    325,331 tok/s   36.60 TFLOP/s ( 7.4% MFU)  eta   8.7m
  step   5,000/10,000  lr 5.87e-04  train 1.4990  val 1.5012    325,700 tok/s   36.64 TFLOP/s ( 7.4% MFU)  eta   8.4m
  step   5,200/10,000  lr 5.57e-04  train 1.5075  val 1.5124    326,055 tok/s   36.68 TFLOP/s ( 7.4% MFU)  eta   8.0m
  step   5,400/10,000  lr 5.28e-04  train 1.4793  val 1.4873    326,378 tok/s   36.72 TFLOP/s ( 7.4% MFU)  eta   7.7m
  step   5,600/10,000  lr 4.98e-04  train 1.4754  val 1.4756    326,772 tok/s   36.76 TFLOP/s ( 7.4% MFU)  eta   7.4m
  step   5,800/10,000  lr 4.69e-04  train 1.4641  val 1.4791    327,143 tok/s   36.81 TFLOP/s ( 7.4% MFU)  eta   7.0m
  step   6,000/10,000  lr 4.40e-04  train 1.4436  val 1.4729    327,479 tok/s   36.84 TFLOP/s ( 7.4% MFU)  eta   6.7m
  step   6,200/10,000  lr 4.11e-04  train 1.4828  val 1.4546    327,185 tok/s   36.81 TFLOP/s ( 7.4% MFU)  eta   6.3m
  step   6,400/10,000  lr 3.83e-04  train 1.4922  val 1.4650    327,430 tok/s   36.84 TFLOP/s ( 7.4% MFU)  eta   6.0m
  step   6,600/10,000  lr 3.56e-04  train 1.4329  val 1.4489    327,658 tok/s   36.86 TFLOP/s ( 7.4% MFU)  eta   5.7m
  step   6,800/10,000  lr 3.29e-04  train 1.4070  val 1.4369    327,892 tok/s   36.89 TFLOP/s ( 7.5% MFU)  eta   5.3m
  step   7,000/10,000  lr 3.04e-04  train 1.4037  val 1.4340    328,111 tok/s   36.92 TFLOP/s ( 7.5% MFU)  eta   5.0m
  step   7,200/10,000  lr 2.80e-04  train 1.4955  val 1.4312    328,302 tok/s   36.94 TFLOP/s ( 7.5% MFU)  eta   4.7m
  step   7,400/10,000  lr 2.56e-04  train 1.4226  val 1.4143    328,500 tok/s   36.96 TFLOP/s ( 7.5% MFU)  eta   4.3m
  step   7,600/10,000  lr 2.34e-04  train 1.4552  val 1.4204    328,641 tok/s   36.98 TFLOP/s ( 7.5% MFU)  eta   4.0m
  step   7,800/10,000  lr 2.14e-04  train 1.3680  val 1.4178    328,806 tok/s   36.99 TFLOP/s ( 7.5% MFU)  eta   3.7m
  step   8,000/10,000  lr 1.95e-04  train 1.4347  val 1.4109    328,964 tok/s   37.01 TFLOP/s ( 7.5% MFU)  eta   3.3m
  step   8,200/10,000  lr 1.77e-04  train 1.4141  val 1.4096    328,652 tok/s   36.98 TFLOP/s ( 7.5% MFU)  eta   3.0m
  step   8,400/10,000  lr 1.62e-04  train 1.3873  val 1.3872    328,812 tok/s   36.99 TFLOP/s ( 7.5% MFU)  eta   2.7m
  step   8,600/10,000  lr 1.47e-04  train 1.4101  val 1.4022    328,958 tok/s   37.01 TFLOP/s ( 7.5% MFU)  eta   2.3m
  step   8,800/10,000  lr 1.35e-04  train 1.3893  val 1.4014    329,107 tok/s   37.03 TFLOP/s ( 7.5% MFU)  eta   2.0m
  step   9,000/10,000  lr 1.24e-04  train 1.3409  val 1.3962    329,247 tok/s   37.04 TFLOP/s ( 7.5% MFU)  eta   1.7m
  step   9,200/10,000  lr 1.16e-04  train 1.3761  val 1.3894    329,383 tok/s   37.06 TFLOP/s ( 7.5% MFU)  eta   1.3m
  step   9,400/10,000  lr 1.09e-04  train 1.3826  val 1.3897    329,515 tok/s   37.07 TFLOP/s ( 7.5% MFU)  eta   1.0m
  step   9,600/10,000  lr 1.04e-04  train 1.3653  val 1.3901    329,662 tok/s   37.09 TFLOP/s ( 7.5% MFU)  eta   0.7m
  step   9,800/10,000  lr 1.01e-04  train 1.3813  val 1.3950    329,783 tok/s   37.10 TFLOP/s ( 7.5% MFU)  eta   0.3m
  step  10,000/10,000  lr 1.00e-04  train 1.3718  val 1.3958    329,915 tok/s   37.12 TFLOP/s ( 7.5% MFU)  eta   0.0m
done       16.6 min, 329,004 tok/s, 37.02 TFLOP/s
artifacts  checkpoints/tinystories-small/  (curve.png, metrics.jsonl, 6 checkpoints)
```
