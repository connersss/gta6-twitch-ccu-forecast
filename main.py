"""gta6-twitch-ccu-forecast-01 entry: kernel-weighted quantile forecaster.

Forecasts Twitch concurrent viewers in the "Grand Theft Auto VI" category
for each hour 2..24 of release day 2026-11-19 (UTC), given only hour 1's
print and the historical hourly curves of comparable launches.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.  -P
keeps this file's directory OFF sys.path, so the sys.path line below puts
it back first, so `import model` finds the model.py shipped beside this
file; -s drops user site-packages.

What it does:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history",
    "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until,
    viewers, channels, reads} and `forecast` lists the hours to forecast,
    2..24;
  - write <id>.out to the output directory:
    {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
    exactly 23 rows, hours 2..24 each once, no other key, each q five
    finite numbers above 0 and at most 1e9 (viewers; fractions allowed),
    non-decreasing; at most 65,536 bytes.  Anything else is an invalid
    case, scored as if every quantile equalled hour 1's print, and an
    entry with an invalid case takes no prize.

Robustness: one bad case must not take the whole run down.  Each case is
processed on its own: a malformed file, a model failure, or a forecast
that fails the output contract falls back to the persistence output
(every quantile equals hour 1's print -- exactly how the grader scores
an invalid case).  When only the case's forecast list is wrong or
incomplete, the 23 rows are rebuilt straight from the model instead of
falling back.  Every .out is validated against the contract and the byte
cap before it is written, and written atomically, so a crash cannot leave
a partial file.  Fallbacks are reported on stderr; a clean run prints
nothing.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The model (see model.py): for each hour, the forecast quantiles are
p1 * exp(weighted quantile of the comparables' log-ratios
ln(viewers_h / p1_i)), with each comparable weighted by a Gaussian kernel
on how close its own hour-1 print is to the target's hour-1 print.  This
learns the launch-day shape (early-morning dip, steep rise as the US
unlocks, evening peak around hours 19-22, then decay) and conditions it
on the observed hour-1 size -- large launches stay near hour 1, small
launches explode.  It beats the persistence baseline by ~40% on large
launches and ~56% overall in leave-one-out cross-validation over the 37
comparable launches (see model.py and README.md).
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The contest's own limits on every quantile, from the case contract.
MAX_VIEWERS = 1e9

# The contest caps each .out file at 65,536 bytes.
MAX_BYTES = 65536


def hour1_print(case):
    """Hour 1's viewer count, or None when the case has no usable print."""
    for k in case.get("known") or []:
        if isinstance(k, dict) and k.get("hour") == 1:
            try:
                v = float(k["viewers"])
            except (KeyError, TypeError, ValueError):
                return None
            return v if v > 0 and math.isfinite(v) else None
    return None


def fallback_forecast(case):
    """Persistence output for a case the model cannot handle.

    Every quantile equals hour 1's print, which is exactly how the grader
    scores an invalid case -- so this is the best output available for
    such a case, and it keeps one bad file from taking the whole run
    down with it.  Falls back to 1 viewer when even hour 1 is unreadable.
    """
    p1 = hour1_print(case)
    if p1 is None:
        p1 = 1.0
    p1 = min(max(1.0, p1), MAX_VIEWERS)
    return {"hours": [{"hour": h, "q": [p1] * 5} for h in range(2, 25)]}


def valid_forecast(forecast):
    """True when a forecast dict meets the contest's output contract."""
    if not isinstance(forecast, dict):
        return False
    hours = forecast.get("hours")
    if not isinstance(hours, list) or len(hours) != 23:
        return False
    if [r.get("hour") for r in hours if isinstance(r, dict)] != list(range(2, 25)):
        return False
    for row in hours:
        q = row.get("q")
        if not isinstance(q, list) or len(q) != 5:
            return False
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 < x <= MAX_VIEWERS for x in q):
            return False
        if q != sorted(q):
            return False
    return True


def write_output(path, forecast):
    """Write one .out file atomically, so a crash cannot leave a partial file."""
    text = json.dumps(forecast, separators=(",", ":")) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def rebuild_forecast(case):
    """Forecast hours 2..24 straight from the model, ignoring the case's
    own forecast list.

    Used when the model's normal output fails validation only because the
    case asked for a wrong or incomplete set of hours: the per-hour
    quantiles are still good, so rebuild the rows instead of throwing the
    model forecast away for persistence.  Returns the same numbers
    forecast_case would return for a well-formed case.
    """
    p1 = hour1_print(case)
    if p1 is None:
        raise ValueError("no hour-1 print")
    train = []
    for comp in case.get("comparables") or []:
        # The target game itself (GTA VI pre-release) has no hour 1; skip it.
        if "Grand Theft Auto VI" in str(comp.get("name", "")):
            continue
        by_hour = model.comparable_hourly(comp)
        if by_hour is not None:
            train.append((by_hour[1], by_hour))
    hours = []
    for h in range(2, 25):
        qs = model.hour_quantiles(p1, train, h)
        hours.append({"hour": h, "q": [model.round2(q) for q in qs]})
    return {"hours": hours}


def process_case(path):
    """Forecast one case file, falling back to persistence on any failure."""
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Unreadable or malformed JSON: nothing to condition on.
        print(f"{path.name}: unreadable, writing persistence fallback", file=sys.stderr)
        return fallback_forecast({})
    try:
        forecast = model.forecast_case(case, path.name)
    except Exception:
        print(f"{path.name}: model failed, writing persistence fallback", file=sys.stderr)
        return fallback_forecast(case)
    if not valid_forecast(forecast):
        # The model ran, but its output does not meet the contract --
        # usually because the case's forecast list is wrong or
        # incomplete.  Rebuild the 23 rows from the model before giving
        # up on it entirely.
        try:
            rebuilt = rebuild_forecast(case)
        except Exception:
            rebuilt = None
        if rebuilt is not None and valid_forecast(rebuilt):
            print(f"{path.name}: forecast list invalid, rebuilt hours 2..24 from model", file=sys.stderr)
            forecast = rebuilt
        else:
            print(f"{path.name}: forecast failed validation, writing persistence fallback", file=sys.stderr)
            return fallback_forecast(case)
    if len((json.dumps(forecast, separators=(",", ":")) + "\n").encode("utf-8")) > MAX_BYTES:
        print(f"{path.name}: forecast over {MAX_BYTES} bytes, writing persistence fallback", file=sys.stderr)
        return fallback_forecast(case)
    return forecast


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        forecast = process_case(path)
        write_output(OUTPUT_DIR / f"{path.stem}.out", forecast)


if __name__ == "__main__":
    main()
