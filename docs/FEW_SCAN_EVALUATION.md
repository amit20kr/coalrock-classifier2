# Few-scan evaluation: what a sensor that delivers one or a few scans gets

Scope: the v1 checkpoint, the deployed decision rule, and the 1,566 held-out test scans in `artifacts/predictions.csv` and `X_test.npy`. No retraining, no code change. Produced by replaying the exact deployed rule (flag a scan at summed gangue probability >= 0.95; REJECT if the mean gangue mass >= 0.5 or at least 4 scans are flagged; a coal verdict becomes REVIEW if the mean top fuel-class probability is below 0.60) on single scans and on small windows.

## 1. How the lab data is organized (verified on two raw files, decoded on the test scans)

- A class file holds 418 scans. In `Carbonaceous shale.csv` and `Coking coal.csv`, each has exactly 38 distinct measurement geometries and 11 granularity labels, and every geometry-granularity pair appears exactly once (38 x 11 = 418). So each file is a full grid: one scan per geometry per granularity sub-sample, with no repeats.
- Each geometry is five header numbers. Positions 2 and 4 are always 5 (the two distances, 0.5 m, fixed). Positions 1, 3 and 5 vary: values 10 to 80 (9 values, 45 dominates), 0 to 75 (11 values) and 0 to 40 (5 values). The dataset paper describes light-source zenith, detection zenith and detection azimuth; assigning those meanings to positions 1, 3 and 5 is probable but not confirmed.
- On the test set, 1,527 of 1,566 scans decode to a unique geometry (all 39 undecoded scans are rock; their values lie outside the value sets seen in the two raw files). Light-source value 45 covers 1,216 of the 1,527.
- So the sensor-relevant variables the data covers are angles. Belt-to-sensor distance variation is not covered at all.

## 2. One scan (k = 1), deployed rule

| True material | ACCEPT | REVIEW | REJECT |
|---|---|---|---|
| Coal (745 scans) | 711 | 30 | 4 |
| Rock (821 scans) | 0 | 1 | 820 |

- No rock scan is auto-accepted: 0 of 821 (95% upper bound 0.45%). The one rock scan the model calls coal (Black shale 1, scan 470) falls below the 0.60 confidence line and goes to REVIEW.
- 4.0% of coal scans go to REVIEW and 0.5% are rejected (coal lost).

### By geometry (coal scans that are not auto-accepted)

| Light-zenith position | Coal scans | REVIEW or REJECT |
|---|---|---|
| value 45 | 586 | 13 (2.2%) |
| all other values | 159 | 21 (13.2%) |

Rock auto-accepted is 0 in every slice. Off-45 slices are small (16 to 25 coal scans each), so treat the 13.2% as a signal, not a precise rate. Within the 45 group, detection azimuth makes no clear difference (1.0% to 3.4%). A specular-geometry explanation was tested and not supported (0 of 20). Mounting the sensor near the best-represented geometry reduces manual reviews about six-fold; it does not change rock safety.

## 3. A few scans of one lump (windows), pure material

Windows are drawn from one class at random from the test scans (40,000 per cell). Independent scans (rho = 0), deployed rule:

| Scans k | Coal auto-ACCEPT | Coal REVIEW | Coal REJECT (false alarm) | Coal REVIEW with stricter rule R1 |
|---|---|---|---|---|
| 1 | 95.6% | 3.98% | 0.42% | 3.93% |
| 3 | 97.5% | 2.37% | 0.10% | 2.71% |
| 5 | 98.8% | 1.14% | 0.02% | 1.63% |
| 10 | 99.9% | 0.13% | 0.00% | 1.42% |

- Pure rock windows: 0.0% auto-accepted at every k (1, 3, 5, 10, 20) and every correlation setting tested.
- More scans only helps if their errors are independent. With k = 10 and correlation between scans of the same lump:

| Correlation rho | Coal auto-ACCEPT | Coal REVIEW | Coal REJECT |
|---|---|---|---|
| 0 | 99.9% | 0.1% | 0.00% |
| 0.5 | 95.9% | 3.8% | 0.30% |
| 1 | 95.4% | 4.1% | 0.56% |

The data cannot say which correlation holds for real lumps (no sample IDs), so plan on little benefit from extra scans beyond what one scan gives.

## 4. A lump that is partly rock (k scans, g of them rock, rest coal; independent scans)

R0 is the deployed rule. R1 adds one condition: if any scan in the window has summed gangue probability >= 0.95 and the window is not already rejected, the verdict is at least REVIEW.

| k | g rock | R0 ACCEPT | R0 REVIEW | R0 REJECT | R1 ACCEPT | R1 REVIEW | R1 REJECT |
|---|---|---|---|---|---|---|---|
| 3 | 1 | 78.2% | 20.6% | 1.2% | 0.5% | 98.4% | 1.1% |
| 5 | 1 | 90.7% | 9.1% | 0.2% | 0.6% | 99.3% | 0.2% |
| 5 | 2 | 0.5% | 98.0% | 1.6% | 0.0% | 98.5% | 1.5% |
| 10 | 1 | 99.0% | 0.9% | 0.0% | 0.6% | 99.4% | 0.0% |
| 10 | 2 | 95.4% | 4.5% | 0.1% | 0.0% | 99.9% | 0.1% |
| 10 | 3 | 78.3% | 20.7% | 1.0% | 0.0% | 99.0% | 1.0% |
| 10 | 4 | 0.2% | 2.2% | 97.6% | 0.0% | 2.4% | 97.6% |

Under the deployed rule, a 3-scan window with 1 rock scan is auto-accepted 78% of the time, and a 10-scan window with 3 rock scans is auto-accepted 78% of the time. The cause is that the batch override needs at least 4 flagged scans, and fewer only produce a warning. R1 brings the auto-accept rate for all these cases to 0.6% or less.

## 5. Decision

- Single scan and pure windows: the system behaves safely. No rock scan or pure-rock window was auto-accepted in this test set.
- Mixed lumps of up to a few rock scans: the deployed rule is not safe for few-scan use. Recommended change (needs Amit's decision, then code and tests): for windows of 20 scans or fewer, any scan at or above the flag threshold forces at least REVIEW. Cost on pure coal: extra REVIEW share of +0.3 points (k = 3), +0.5 (k = 5), +1.3 (k = 10), +2.3 (k = 20), independent scans. Keep the current rule for large files (belt-segment screening), where one false flag among hundreds of coal scans would otherwise send most clean files to REVIEW.
- Mount: choose the geometry the data best supports (light-source position 45 in the header; detection angle did not matter much within it).

## 6. Limits

- Test scans come from the same samples and geometries as the training scans; these are upper bounds on field performance.
- Windows are random draws from the test scans, not real lumps. Geometry was not held fixed inside a window.
- 821 rock scans bound what "0 auto-accepted" can claim (95% upper bound 0.45% per scan).
- Not covered: distance changes, moisture, dust, belt vibration, new seams or mines, scans without the 5 header values (those go to REVIEW).
- Positions 1, 3 and 5 are probable, not confirmed, geometry meanings.
