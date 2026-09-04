# owt-small — H100 run

Console record for the `owt` config. `metrics.jsonl` holds the machine-readable
per-eval rows (step, lr, train_loss, validation_loss, elapsed); this file keeps
what the log has and the JSONL does not — the config banner and the throughput /
TFLOP/s / MFU series.

Run date: 2026-09-04 · `uv run cs336_basics/train.py owt`

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
| Run | `owt-small` (owt, small) |
| Model | 45,224,448 params — d_model 512 × 4 layers × 16 heads |
| Data | 2,727,120,464 train / 66,401,098 validation tokens |
| Budget | 10,000 steps × 128 batch × 256 ctx = 327,680,000 tokens, 5.90e+16 FLOPs |
| Schedule | lr 1.00e-03 → 1.00e-04, warmup 500, decay to step 10,000 |
| Device | cuda, seed 42 |

## Results

| | |
|---|---|
| Final train loss | 4.0071 |
| Final validation loss | 4.0289 |
| Best validation loss | **4.0141** @ step 9,800 |
| Wall clock | 21.0 min (1,258.6 s) |
| Mean throughput | 259,540 tok/s |
| Mean compute | 46.74 TFLOP/s |
| Steady-state MFU | 9.5% |

Reported MFU implies a peak of ~494 TFLOP/s (46.89 / 0.095), i.e. the H100 SXM
BF16 dense tensor-core figure — the same denominator as the tinystories run.

### Throughput ramp

Throughput is reported as a cumulative mean, so it climbs as fixed startup cost
amortises; unlike the tinystories run it is still inching upward at step 10,000.

| step | tok/s | TFLOP/s | MFU |
|---:|---:|---:|---:|
| 200 | 133,004 | 23.95 | 4.8% |
| 600 | 184,133 | 33.16 | 6.7% |
| 1,000 | 207,400 | 37.35 | 7.5% |
| 2,000 | 233,092 | 41.98 | 8.5% |
| 4,000 | 248,374 | 44.73 | 9.0% |
| 6,000 | 254,734 | 45.88 | 9.3% |
| 8,000 | 258,250 | 46.51 | 9.4% |
| 10,000 | 260,349 | 46.89 | 9.5% |

### Loss trajectory

| step | train | val |
|---:|---:|---:|
| 200 | 6.4082 | 6.3944 |
| 1,000 | 4.9627 | 4.9093 |
| 2,000 | 4.5977 | 4.5316 |
| 4,000 | 4.3058 | 4.2845 |
| 6,000 | 4.2074 | 4.1668 |
| 8,000 | 4.1142 | 4.0843 |
| 9,800 | 3.9233 | 4.0141 |
| 10,000 | 4.0071 | 4.0289 |

## Compared with `tinystories-small`

Same token budget and schedule; the model is ~2x larger (45.2M vs 22.7M params,
from the larger vocabulary) and the corpus is far harder.

| | tinystories-small | owt-small |
|---|---:|---:|
| Params | 22,696,448 | 45,224,448 |
| Budget FLOPs | 3.69e+16 | 5.90e+16 |
| Wall clock | 16.6 min | 21.0 min |
| Mean tok/s | 329,004 | 259,540 |
| Mean TFLOP/s | 37.02 | 46.74 |
| Steady MFU | 7.5% | 9.5% |
| Best val loss | 1.3872 | 4.0141 |

Fewer tokens/s but more FLOP/s: the larger model does more arithmetic per token,
so it uses the GPU better even though it processes tokens more slowly.

## Artifacts

`checkpoints/owt-small/` — `curve.png`, `metrics.jsonl` (50 rows, steps
200–10,000), `final.pt`, and checkpoints at steps 2,000 / 4,000 / 6,000 / 8,000
/ 10,000.

## Full console log

```
run        owt-small  (owt, small)
model      45,224,448 params  d_model 512 x 4 layers x 16 heads
data       2,727,120,464 train / 66,401,098 validation tokens
budget     10,000 steps x 128 batch x 256 ctx = 327,680,000 tokens, 5.90e+16 FLOPs
schedule   lr 1.00e-03 -> 1.00e-04, warmup 500, decay to step 10,000
device     cuda  seed 42  -> checkpoints/owt-small
  step     200/10,000  lr 4.00e-04  train 6.4082  val 6.3944    133,004 tok/s   23.95 TFLOP/s ( 4.8% MFU)  eta  40.2m
  step     400/10,000  lr 8.00e-04  train 5.6820  val 5.6723    164,460 tok/s   29.62 TFLOP/s ( 6.0% MFU)  eta  31.9m
  step     600/10,000  lr 1.00e-03  train 5.2741  val 5.3006    184,133 tok/s   33.16 TFLOP/s ( 6.7% MFU)  eta  27.9m
  step     800/10,000  lr 9.98e-04  train 5.0599  val 5.0618    197,739 tok/s   35.61 TFLOP/s ( 7.2% MFU)  eta  25.4m
  step   1,000/10,000  lr 9.94e-04  train 4.9627  val 4.9093    207,400 tok/s   37.35 TFLOP/s ( 7.5% MFU)  eta  23.7m
  step   1,200/10,000  lr 9.88e-04  train 4.8288  val 4.8136    214,986 tok/s   38.72 TFLOP/s ( 7.8% MFU)  eta  22.4m
  step   1,400/10,000  lr 9.80e-04  train 4.7935  val 4.7431    220,963 tok/s   39.79 TFLOP/s ( 8.0% MFU)  eta  21.3m
  step   1,600/10,000  lr 9.71e-04  train 4.6928  val 4.6593    225,847 tok/s   40.67 TFLOP/s ( 8.2% MFU)  eta  20.3m
  step   1,800/10,000  lr 9.59e-04  train 4.6171  val 4.6176    229,821 tok/s   41.39 TFLOP/s ( 8.4% MFU)  eta  19.5m
  step   2,000/10,000  lr 9.46e-04  train 4.5977  val 4.5316    233,092 tok/s   41.98 TFLOP/s ( 8.5% MFU)  eta  18.7m
  step   2,200/10,000  lr 9.31e-04  train 4.5043  val 4.5212    234,172 tok/s   42.17 TFLOP/s ( 8.5% MFU)  eta  18.2m
  step   2,400/10,000  lr 9.14e-04  train 4.4829  val 4.4963    236,570 tok/s   42.60 TFLOP/s ( 8.6% MFU)  eta  17.5m
  step   2,600/10,000  lr 8.96e-04  train 4.4012  val 4.4546    238,598 tok/s   42.97 TFLOP/s ( 8.7% MFU)  eta  16.9m
  step   2,800/10,000  lr 8.76e-04  train 4.3860  val 4.4233    240,470 tok/s   43.31 TFLOP/s ( 8.7% MFU)  eta  16.4m
  step   3,000/10,000  lr 8.55e-04  train 4.3989  val 4.4157    242,087 tok/s   43.60 TFLOP/s ( 8.8% MFU)  eta  15.8m
  step   3,200/10,000  lr 8.32e-04  train 4.3666  val 4.3967    243,524 tok/s   43.86 TFLOP/s ( 8.9% MFU)  eta  15.2m
  step   3,400/10,000  lr 8.08e-04  train 4.3229  val 4.3509    244,851 tok/s   44.10 TFLOP/s ( 8.9% MFU)  eta  14.7m
  step   3,600/10,000  lr 7.84e-04  train 4.3458  val 4.3276    246,097 tok/s   44.32 TFLOP/s ( 9.0% MFU)  eta  14.2m
  step   3,800/10,000  lr 7.58e-04  train 4.1877  val 4.3287    247,292 tok/s   44.54 TFLOP/s ( 9.0% MFU)  eta  13.7m
  step   4,000/10,000  lr 7.31e-04  train 4.3058  val 4.2845    248,374 tok/s   44.73 TFLOP/s ( 9.0% MFU)  eta  13.2m
  step   4,200/10,000  lr 7.03e-04  train 4.3034  val 4.2873    248,341 tok/s   44.72 TFLOP/s ( 9.0% MFU)  eta  12.8m
  step   4,400/10,000  lr 6.75e-04  train 4.3901  val 4.3018    249,283 tok/s   44.89 TFLOP/s ( 9.1% MFU)  eta  12.3m
  step   4,600/10,000  lr 6.46e-04  train 4.2744  val 4.2504    250,151 tok/s   45.05 TFLOP/s ( 9.1% MFU)  eta  11.8m
  step   4,800/10,000  lr 6.17e-04  train 4.2875  val 4.2349    250,959 tok/s   45.20 TFLOP/s ( 9.1% MFU)  eta  11.3m
  step   5,000/10,000  lr 5.87e-04  train 4.2210  val 4.2360    251,719 tok/s   45.33 TFLOP/s ( 9.2% MFU)  eta  10.8m
  step   5,200/10,000  lr 5.57e-04  train 4.2728  val 4.1994    252,417 tok/s   45.46 TFLOP/s ( 9.2% MFU)  eta  10.4m
  step   5,400/10,000  lr 5.28e-04  train 4.1888  val 4.2158    253,056 tok/s   45.57 TFLOP/s ( 9.2% MFU)  eta   9.9m
  step   5,600/10,000  lr 4.98e-04  train 4.1595  val 4.1903    253,651 tok/s   45.68 TFLOP/s ( 9.2% MFU)  eta   9.5m
  step   5,800/10,000  lr 4.69e-04  train 4.1809  val 4.1908    254,199 tok/s   45.78 TFLOP/s ( 9.2% MFU)  eta   9.0m
  step   6,000/10,000  lr 4.40e-04  train 4.2074  val 4.1668    254,734 tok/s   45.88 TFLOP/s ( 9.3% MFU)  eta   8.6m
  step   6,200/10,000  lr 4.11e-04  train 4.1458  val 4.1517    254,666 tok/s   45.86 TFLOP/s ( 9.3% MFU)  eta   8.1m
  step   6,400/10,000  lr 3.83e-04  train 4.1077  val 4.1314    255,162 tok/s   45.95 TFLOP/s ( 9.3% MFU)  eta   7.7m
  step   6,600/10,000  lr 3.56e-04  train 4.1593  val 4.1325    255,624 tok/s   46.04 TFLOP/s ( 9.3% MFU)  eta   7.3m
  step   6,800/10,000  lr 3.29e-04  train 4.0582  val 4.1191    256,066 tok/s   46.12 TFLOP/s ( 9.3% MFU)  eta   6.8m
  step   7,000/10,000  lr 3.04e-04  train 4.1461  val 4.0963    256,474 tok/s   46.19 TFLOP/s ( 9.3% MFU)  eta   6.4m
  step   7,200/10,000  lr 2.80e-04  train 4.0631  val 4.1005    256,869 tok/s   46.26 TFLOP/s ( 9.3% MFU)  eta   6.0m
  step   7,400/10,000  lr 2.56e-04  train 4.1159  val 4.0892    257,248 tok/s   46.33 TFLOP/s ( 9.4% MFU)  eta   5.5m
  step   7,600/10,000  lr 2.34e-04  train 4.0572  val 4.0967    257,606 tok/s   46.39 TFLOP/s ( 9.4% MFU)  eta   5.1m
  step   7,800/10,000  lr 2.14e-04  train 4.0681  val 4.0891    257,927 tok/s   46.45 TFLOP/s ( 9.4% MFU)  eta   4.7m
  step   8,000/10,000  lr 1.95e-04  train 4.1142  val 4.0843    258,250 tok/s   46.51 TFLOP/s ( 9.4% MFU)  eta   4.2m
  step   8,200/10,000  lr 1.77e-04  train 4.0451  val 4.0566    258,042 tok/s   46.47 TFLOP/s ( 9.4% MFU)  eta   3.8m
  step   8,400/10,000  lr 1.62e-04  train 4.0677  val 4.0429    258,345 tok/s   46.53 TFLOP/s ( 9.4% MFU)  eta   3.4m
  step   8,600/10,000  lr 1.47e-04  train 4.0714  val 4.0609    258,632 tok/s   46.58 TFLOP/s ( 9.4% MFU)  eta   3.0m
  step   8,800/10,000  lr 1.35e-04  train 4.0973  val 4.0489    258,916 tok/s   46.63 TFLOP/s ( 9.4% MFU)  eta   2.5m
  step   9,000/10,000  lr 1.24e-04  train 4.0486  val 4.0297    259,163 tok/s   46.67 TFLOP/s ( 9.4% MFU)  eta   2.1m
  step   9,200/10,000  lr 1.16e-04  train 4.0631  val 4.0586    259,425 tok/s   46.72 TFLOP/s ( 9.4% MFU)  eta   1.7m
  step   9,400/10,000  lr 1.09e-04  train 3.9905  val 4.0549    259,659 tok/s   46.76 TFLOP/s ( 9.4% MFU)  eta   1.3m
  step   9,600/10,000  lr 1.04e-04  train 4.0167  val 4.0315    259,894 tok/s   46.81 TFLOP/s ( 9.5% MFU)  eta   0.8m
  step   9,800/10,000  lr 1.01e-04  train 3.9233  val 4.0141    260,123 tok/s   46.85 TFLOP/s ( 9.5% MFU)  eta   0.4m
  step  10,000/10,000  lr 1.00e-04  train 4.0071  val 4.0289    260,349 tok/s   46.89 TFLOP/s ( 9.5% MFU)  eta   0.0m
done       21.0 min, 259,540 tok/s, 46.74 TFLOP/s
```
