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
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = model.forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
