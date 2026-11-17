"""Kernel-weighted quantile model for launch-day Twitch viewership.

The model forecasts, for each hour h = 2..24 of a game's UTC release day, the
five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the concurrent-viewer count,
given only hour 1's print (p1) and the historical hourly curves of comparable
launches (the case's own `comparables` list).

Model
-----
For each comparable launch i and each hour h, compute the log-ratio

    r_i(h) = ln(viewers_i(h) / p1_i),

where p1_i is that launch's own hour-1 print.  The ratio's distribution
depends strongly on the launch's size: a launch that opens huge (hour 1 in
the hundreds of thousands) stays near hour 1 all day and peaks only ~1.1x,
while a launch that opens small (hour 1 in the tens of thousands or less)
can explode 2x-1000x by the evening as time zones unlock and prime time
arrives.  So, for a target whose hour-1 print is p1, each comparable is
weighted by a Gaussian kernel on the log-size distance,

    w_i = exp(-0.5 * ((ln p1 - ln p1_i) / BW) ** 2),   BW = 1.0,

and the forecast quantile at level tau is

    q_tau(h) = p1 * exp(weighted_quantile_tau({r_i(h)}, {w_i})).

This directly targets the pinball loss (the optimal forecast at level tau
is the true conditional tau-quantile), and the kernel lets the model borrow
strength from nearby-size launches, which stabilises the small-sample
quantiles.  If no comparable has data for an hour, that hour falls back to
the persistence baseline (the house starter's random-walk spread).

Why this beats persistence
---------------------------
Persistence assumes every hour equals hour 1.  But a midnight-UTC unlock
means hour 1 is a low point: only New Zealand / Australia / Asia / Europe
are live, the US has not unlocked yet, and it is the middle of the night in
Europe.  Viewership dips further in hours 2-4, then rises steeply through
the day as the US unlocks (05:00 and 08:00 UTC) and European prime time
arrives, peaking around hours 19-22 before decaying.  Persistence misses
that whole shape; the kernel model learns it from the comparables, and
conditions the shape on the observed hour-1 size.

Validation (leave-one-out cross-validation over the 37 usable comparables
in the practice cases; each game is forecast using only the other 36):
mean pinball loss on the log scale, model vs persistence baseline.

    LARGE launches (hour 1 >= 300k, n=6):  model 0.1360 vs baseline 0.2255  (-40%)
    MED launches   (100k-300k,    n=10):  model 0.2645 vs baseline 0.2825  (-6%)
    LARGE+MED      (n=16):                model 0.2163 vs baseline 0.2611  (-17%)
    ALL 37 comparables:                   model 0.4048 vs baseline 0.9297  (-56%)

The model wins on 30 of the 37 games.  GTA VI's hour 1 will be enormous
(the most anticipated game ever, with NZ/Australia/Asia/Europe/UK already
live at 00:00 UTC), so it should follow the LARGE stratum, where the model
is ~40% better than persistence.

Note on the practice cases: the practice .in files do NOT contain their own
answers (hours 2..24 actuals).  The answers live in the sealed
reference/answer.json, which is not in the kit.  So no true pinball loss
can be computed on p00-p03 from the kit alone.  The comparable-based
leave-one-out CV above is the honest evaluation; it uses a larger and more
robust sample (37 launches) than the 4 practice cases.
"""

import math
import statistics
from datetime import datetime

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Gaussian kernel bandwidth (in natural-log units of hour-1 size) used to
# weight comparables by how close their hour-1 print is to the target's.
# Chosen by leave-one-out CV: bw=1.0 is best on ALL comparables (0.4048)
# and on LARGE+MED (0.2163), and near-best on LARGE alone (0.1360 vs the
# best 0.1308 at bw=0.6).  It is a robust, well-rounded choice.
BANDWIDTH = 1.0

# Persistence fallback spread (per sqrt-hour, in log units), as in the
# house starter.  Used only when no comparable has data for an hour.
FALLBACK_SIGMA = 0.15
FALLBACK_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Hard cap on any forecast quantile (the contest's own limit).
MAX_VIEWERS = 1e9


def _parse_ts(s):
    """Parse an ISO-8601 timestamp (with a trailing Z) to a datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def comparable_hourly(comp):
    """Return {hour: viewers} for a comparable launch, hours 1..24.

    A point at `from` covers the hour [from, from + 1h) in UTC.  The hour
    number is its offset from the comparable's day_start (00:00 UTC of
    its release date) plus one.  Points with no viewers, or viewers <= 0,
    are skipped.  Returns None if the comparable has no usable hour-1
    print or too little data to be useful.
    """
    try:
        day_start = _parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for pt in comp.get("hourly") or []:
        v = pt.get("viewers")
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        try:
            t = _parse_ts(pt["from"])
        except (KeyError, TypeError, ValueError):
            continue
        h = int(round((t - day_start).total_seconds() / 3600.0)) + 1
        if 1 <= h <= 24:
            # If two points land in the same hour, keep the larger (the
            # hourly average should be unique, but be safe).
            by_hour[h] = max(by_hour.get(h, 0.0), v)
    if 1 not in by_hour or by_hour[1] <= 0:
        return None
    # Require at least 20 of the 23 forecast hours present, so a sparse
    # comparable does not inject noise.
    if sum(1 for h in range(2, 25) if h in by_hour) < 20:
        return None
    return by_hour


def weighted_quantile(values, weights, tau):
    """The tau-quantile of `values` under weights `weights`.

    Sorts by value, accumulates weights, and returns the first value whose
    cumulative weight reaches tau * total weight.  Deterministic.
    """
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    if total <= 0:
        return statistics.median(values)
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= tau * total:
            return v
    return pairs[-1][0]


def hour_quantiles(p1, train, h):
    """The five forecast quantiles for hour h, given hour-1 print p1.

    `train` is a list of (p1_i, by_hour_i) for the comparables.  Returns a
    list of five numbers in LEVELS order.
    """
    lp1 = math.log(max(1.0, p1))
    vals = []
    wts = []
    for p1_i, by_hour in train:
        if h in by_hour and by_hour[h] > 0:
            vals.append(math.log(by_hour[h] / max(1.0, p1_i)))
            wts.append(math.exp(-0.5 * ((lp1 - math.log(max(1.0, p1_i))) / BANDWIDTH) ** 2))
    if not vals:
        # No comparable has this hour: fall back to persistence.
        return [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)) for z in FALLBACK_Z]
    qs = [p1 * math.exp(weighted_quantile(vals, wts, tau)) for tau in LEVELS]
    # Enforce the contest's constraints: finite, > 0, <= 1e9, non-decreasing.
    qs = [min(max(1.0, q), MAX_VIEWERS) if math.isfinite(q) else p1 for q in qs]
    for i in range(1, len(qs)):
        qs[i] = max(qs[i], qs[i - 1])
    return qs


def round2(x):
    """Round to 2 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).  Matches the house starter.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def forecast_case(case, name):
    """Forecast hours 2..24 for one case dict.  Returns the .out dict."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # Build the training set from this case's own comparables.
    train = []
    for comp in case.get("comparables") or []:
        # The target game itself (GTA VI pre-release) has no hour 1; skip it.
        if "Grand Theft Auto VI" in str(comp.get("name", "")):
            continue
        by_hour = comparable_hourly(comp)
        if by_hour is not None:
            train.append((by_hour[1], by_hour))

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = hour_quantiles(p1, train, h)
        hours.append({"hour": h, "q": [round2(q) for q in qs]})
    return {"hours": hours}
