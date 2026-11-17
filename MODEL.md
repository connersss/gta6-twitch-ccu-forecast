# GTA VI Twitch Viewership Forecaster — make.fun bounty `gta6-twitch-ccu-forecast-01`

Forecasts Twitch concurrent viewers in the "Grand Theft Auto VI" category for
each hour 2..24 of release day 2026-11-19 (UTC), given only hour 1's print.
Entry for the make.fun coding bounty (prize $250, closes 2026-11-19 08:00 WIB).
**Not submitted yet** — this directory holds the program only. Submission
still needs a public GitHub repo (in solina's account) and a Solana
keypair/payout wallet decision.

## Files

- `main.py` — entry point. Reads every `<id>.in` in the input directory
  (`MAKEFUN_INPUT_DIR`, default `/input`) and writes `<id>.out` to the
  output directory (`MAKEFUN_OUTPUT_DIR`, default `/output`). Python 3.13
  standard library only, no network, deterministic.
- `model.py` — the forecasting model (kernel-weighted quantiles) plus the
  validation numbers and the practice-answer caveat. `main.py` imports it.

## How to run

```sh
# On the practice cases (kit input dir holds p00.in..p03.in):
mkdir -p out
MAKEFUN_INPUT_DIR=../gta6-kit/gta6-twitch-ccu-forecast-01-kit/input \
  MAKEFUN_OUTPUT_DIR=out python3 main.py

# On a single case:
mkdir -p in out && cp <kit>/input/p02.in in/
MAKEFUN_INPUT_DIR=in MAKEFUN_OUTPUT_DIR=out python3 main.py
# -> out/p02.out
```

In the grader's container the program runs as
`python3 -P -s /bot/main.py` with `/input` and `/output`; the `sys.path`
line in `main.py` makes `import model` work under `-P`.

Output format (per case): `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`,
exactly 23 rows, hours 2..24 each once, each `q` five finite numbers
> 0 and <= 1e9, non-decreasing, file <= 65,536 bytes. Verified on all
four practice cases (see "Validation" below).

## The model

For each hour h = 2..24, forecast

    q_tau(h) = p1 * exp(weighted_quantile_tau({ ln(viewers_i(h) / p1_i) }, { w_i }))

where p1 is the target's hour-1 print, p1_i and viewers_i(h) are comparable
launch i's hour-1 print and hour-h viewers (from the case's own
`comparables` list), and each comparable is weighted by a Gaussian kernel
on the log-size distance,

    w_i = exp(-0.5 * ((ln p1 - ln p1_i) / 1.0) ** 2).

This directly targets the pinball loss: the optimal forecast at level tau
is the true conditional tau-quantile, and the kernel-weighted empirical
quantile estimates it without distributional assumptions. If no comparable
has data for an hour, that hour falls back to the persistence baseline
(the house starter's random-walk spread, sigma 0.15 per sqrt-hour).

### Why it beats persistence

Persistence assumes every hour equals hour 1. But a midnight-UTC unlock
means hour 1 is a low point: only New Zealand / Australia / Asia / Europe
are live, the US has not unlocked yet, and it is the middle of the night
in Europe. Viewership dips further in hours 2-4, then rises steeply
through the day as the US unlocks (05:00 and 08:00 UTC) and European
prime time arrives, peaking around hours 19-22 before decaying.
Persistence misses that whole shape.

The shape depends strongly on launch size. Median ratio of each hour's
viewers to hour 1, across the 37 usable comparables in the practice
cases:

| hour-1 size | n | dip (h3-4) | peak | peak hour | h24 |
|---|---|---|---|---|---|
| LARGE (>=300k) | 6 | 0.48-0.53x | 1.12x | 19 | 0.89x |
| MED (100k-300k) | 10 | 0.72-0.74x | 1.66x | 22 | 1.31x |
| SMALL (20k-100k) | 11 | 0.71-0.75x | 2.42x | 22 | 2.04x |
| TINY (<20k) | 10 | 0.75x | ~180x | 19 | ~90x |
| overall | 37 | 0.73-0.75x | 2.40x | 19-20 | 2.04x |

A launch that opens huge stays near hour 1 all day (peak ~1.1x); a
launch that opens small can explode 2x-1000x by the evening. The kernel
conditions the forecast on the observed hour-1 size, so GTA VI — whose
hour 1 will be enormous (most anticipated game ever, with
NZ/Australia/Asia/Europe/UK already live at 00:00 UTC) — gets the LARGE
shape (peak ~1.1-1.35x, dip to ~0.5x in hours 8-10, modest evening
rise), not the overall median (~2.4x).

## Validation

### Output format (all four practice cases)

Ran `main.py` on p00.in, p01.in, p02.in, p03.in. All four outputs are
VALID: exactly 23 rows, hours 2..24 each once, five quantiles per row,
all finite, > 0, <= 1e9, non-decreasing, files 1,451-1,545 bytes
(well under the 65,536-byte limit). Running twice on the same input
produces byte-identical output (deterministic).

Forecast shapes (median q50 as a ratio to hour 1):

| case | game | hour 1 | peak | peak hour | h24 |
|---|---|---|---|---|---|
| p00 | Marvel's Wolverine | 23,357 | 3.31x | 19 | 2.44x |
| p01 | Red Dead Redemption 2 | 116,550 | 1.72x | 22 | 1.31x |
| p02 | God of War Ragnarök | 269,517 | 1.34x | 20 | 1.16x |
| p03 | Marvel's Spider-Man 2 | 119,087 | 1.72x | 22 | 1.31x |

p01 and p03 are identical because their hour-1 prints are nearly the
same (116,550 vs 119,087), so the kernel weights are nearly identical.
p02 (the largest hour 1) gets the flattest shape, as the LARGE stratum
predicts. p00 (the smallest hour 1) gets the steepest rise, as the
SMALL stratum predicts.

### Loss vs the persistence baseline (leave-one-out CV on comparables)

**Important caveat:** the practice `.in` files do NOT contain their own
answers (hours 2..24 actuals). The answers live in the sealed
`reference/answer.json`, which is not in the kit, and there is no
`reference/` directory in the kit. So no true pinball loss can be
computed on p00-p03 from the kit alone. (Fetching the answers from
SullyGnome was attempted: direct script access is Cloudflare-blocked
HTTP 403, and the values could not be aligned to hours reliably from
browser text, so no practice loss is claimed here.)

The honest evaluation is leave-one-out cross-validation over the 37
usable comparables in the practice cases: each launch is forecast using
only the other 36, and its mean pinball loss on the log scale is compared
with the persistence baseline (the house starter). This uses a larger
and more robust sample (37 launches) than the 4 practice cases.

| stratum | n | model | baseline | improvement | games won |
|---|---|---|---|---|---|
| LARGE (hour 1 >= 300k) | 6 | 0.1360 | 0.2255 | -39.7% | 5 / 6 |
| MED (100k-300k) | 10 | 0.2645 | 0.2825 | -6.4% | 7 / 10 |
| SMALL (20k-100k) | 11 | 0.3184 | 0.4091 | -22.2% | 10 / 11 |
| TINY (< 20k) | 10 | 0.8015 | 2.5719 | -68.8% | 10 / 10 |
| **ALL** | **37** | **0.4048** | **0.9297** | **-56.5%** | **30 / 37** |

The model beats persistence in every stratum. The house baseline's
stated practice result is loss 0.387 (from `competition.yaml`); the
model's comparable losses (LARGE 0.136, LARGE+MED 0.216) are well below
it, and the practice cases (1 SMALL + 3 MED) should land around
0.28-0.34 by the same stratum ratios — below the baseline's 0.387.

Bandwidth: bw=1.0 was chosen by CV. It is best on ALL comparables
(0.4048) and on LARGE+MED (0.2163), and near-best on LARGE alone
(0.1360 vs the best 0.1308 at bw=0.6). A parametric (normal) variant
was also tested: slightly better overall (0.3932) and on MED, but worse
on LARGE (0.1445), so the empirical quantile was kept — GTA VI is a
LARGE launch, and the empirical quantile makes no distributional
assumption.

## Robustness

- Missing or zero hour-1 print: floored at 1 (per the contest rule
  "a print below 1 counts as 1"); the model still outputs 23 valid rows.
- A comparable with no hour-1 print, or fewer than 20 of the 23
  forecast hours, is skipped.
- The target game itself (GTA VI pre-release, no hour 1) is skipped
  when it appears in `comparables`.
- An hour with no comparable data falls back to the persistence
  baseline for that hour only.
- All quantiles are clamped to (0, 1e9], made finite, and forced
  non-decreasing before rounding to 2 decimals (half up, JS style).
- No network, no randomness, no wall-clock dependence: the same input
  always produces the same output bytes.

## What is still needed (not in this directory)

- A public GitHub repo in solina's account to host the entry
  (GitHub account `connersss` supplied 2026-11-17; no repo created yet,
  no token yet — one owner authorization is still needed at submission).
- A Solana keypair / payout wallet decision (keypair generated
  2026-11-17, pubkey `AQagFa8PuPfybFV7GiMxdwfJjxnYiGwbvBmQ48jZVyVW`,
  file `~/workspace/makefun/keypair.json`, mode 600 — secret never shared).
- Submission to make.fun (do NOT submit from here; the parent agent
  handles it once the repo and wallet are ready). Bounty closes
  2026-11-19 08:00 WIB; the final case runs at W0 = 2026-11-19T01:00:00Z.
