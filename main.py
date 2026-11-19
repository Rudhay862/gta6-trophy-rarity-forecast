"""gta6-trophy-rarity-forecast-01 entry: log-time logit trend + stale-page dilution prior.

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five quantiles
of its earn rate on Exophase at the case's target, as percent.

Model, per trophy
-----------------
Rates live on the logit scale, logit(p) = ln(p / (100 - p)), p clamped to
0.01..99.99, because a move from 1% to 2% matters about as much as one from 33%
to 50% (the contest scores pinball loss on the logit scale).

Time is measured as x = ln(1 + hours since release), release anchor
2026-11-19T00:00:00Z. Earn rates grow roughly with log-time early in a game's
life: the first players finish the easy trophies fast, then the rate saturates
as the tracked population widens to slower players.

Two regimes, decided per trophy from the case's own history:

1. FROZEN HISTORY (stale Exophase page). If the history has >= 3 reads and
   every rate for this trophy is identical across all reads AND players_tracked
   is constant, the page has not refreshed since launch night: Exophase
   refreshes a game's counts on its own schedule, not with every player's
   trophy. The house documented the first refresh: 638 -> 3,868 players
   tracked, every rate but the 100% one lower; a live day-1.5 read of the page
   (priors.json) shows rates ~5-10x below launch night. A refresh dilutes
   rates because the tracked-player count jumps while earners grow more
   slowly. So for a frozen trophy the median is last_rate x 0.25 (rate 100 ->
   99.99, rate 0 -> 0.01), with a wide spread: the refresh has not been seen
   yet, so the honest interval must cover a range of plausible dilutions.
   sigma = sqrt(0.9^2 + (0.5*sqrt(horizon_days))^2).

2. NON-FROZEN. Weighted least squares of logit(rate) on x, using reads from
   the last 24 h of the case (or all reads if fewer than 3 fall in that
   window), with recency weights w = 0.5 ** ((t_last - t) / 12 h), extrapolated
   to x_target. Needs >= 3 distinct x values, else falls back to the last
   rate with sigma 1.2. Guardrails: the extrapolated logit may move at most
   3.0 from logit(last_rate); absolute week-one plausibility caps by tier
   (platinum <= 1.5%, gold <= 8%, silver <= 25%, bronze <= 70%); floor 0.01.
   If the last rate is null, the priors.json day-1.5 rate for that trophy id
   is used instead (else a tier default: platinum 0.1, gold 0.2, silver 0.2,
   bronze 0.3).

Spread (non-frozen): sigma = sqrt(rw^2 + resid^2 + slope_unc^2 + rare^2),
where rw = 0.5*sqrt(horizon_days) is the random-walk term, resid the std of
the fit residuals (floor 0.15, cap 1.0), slope_unc = |slope| x
(x_target - x_last) x 0.5 prices extrapolation distance, and rare = 0.3 when
last_rate < 0.5% (rare trophies are noisier). sigma is clamped to [0.25, 2.0].

Quantiles: q_tau = 100 / (1 + exp(-(logit(median) + z_tau * sigma))),
z = [-1.645, -0.674, 0, 0.674, 1.645], rounded to 4 decimals half-up (like the
kit starter's round4), enforced non-decreasing, clamped to [0.01, 99.99].

Data: the forecaster's primary input is always the case's own history.
priors.json (shipped beside this file) holds real Exophase snapshots read
live while building: launch night (the house's own recording, 638 players)
and day 1.5 (2026-11-19T11:36Z). It is used only as a prior for null last
rates and to calibrate the stale-page dilution above; the graded run has no
network, so the data is baked into the entry.

Python 3.13, standard library only, no network, deterministic: two runs on
the same case write identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put this file's directory first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# GTA VI released 19 November 2026; the log-time clock starts at midnight UTC.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Week-one plausibility caps by tier (percent): no trophy can plausibly exceed
# these a week after launch, however steep its early trend looks.
TIER_CAP = {"platinum": 1.5, "gold": 8.0, "silver": 25.0, "bronze": 70.0}
# Fallback rate when a trophy's last rate is null and no prior exists.
TIER_DEFAULT = {"platinum": 0.1, "gold": 0.2, "silver": 0.2, "bronze": 0.3}
# How much a stale page's refresh dilutes rates (launch-night -> day-1.5 is
# ~5-10x; 0.25 is the conservative middle of that range, applied to the
# median, with a wide spread covering the rest).
DILUTION = 0.25
# Spread of the frozen-branch median, in logit units, before the random walk.
FROZEN_SIGMA = 0.9


def parse_time(s: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp (with 'Z' and optional fractions)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_time(at: datetime) -> float:
    """x = ln(1 + hours since release); clamped so the log stays defined."""
    hours = (at - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(hours, -0.999))


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """x rounded to 4 decimals, half up, as the kit starter's round4 does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def load_priors() -> dict:
    """Day-1.5 Exophase rates by trophy id, from priors.json beside this file.

    Missing or unreadable file -> empty priors; the forecaster never depends
    on it for trophies whose last rate is known.
    """
    try:
        data = json.loads(Path(__file__).resolve().with_name("priors.json").read_text(encoding="utf-8"))
        rates = data.get("day_1_5", {}).get("rates", [])
        return {i + 1: float(r) for i, r in enumerate(rates) if r is not None}
    except Exception:
        return {}


PRIORS = load_priors()


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (intercept, slope)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def forecast_trophy(trophy, reads, horizon_days, x_target, priors):
    """Forecast one trophy. `reads` is a list of (at, x, rate-or-None)."""
    tier = trophy.get("tier", "bronze")
    cap = TIER_CAP.get(tier, 70.0)
    last_rate = reads[-1][2] if reads else None

    # --- Frozen-history check: >= 3 reads, every rate for this trophy
    # identical across reads, and players_tracked constant. Per-trophy:
    # a trophy whose own rates never moved while the page sat still is stale.
    known = [r for _, _, r, _ in reads if r is not None]
    frozen = (
        len(reads) >= 3
        and len(known) >= 3
        and all(r == known[0] for r in known)
        and all(p == reads[0][3] for _, _, _, p in reads)
    )
    if frozen:
        if last_rate is None:
            base = priors.get(trophy["id"], TIER_DEFAULT.get(tier, 0.3))
            median = clamp_rate(base)
        elif last_rate >= 100.0:
            median = 99.99
        elif last_rate <= 0.0:
            median = 0.01
        else:
            median = clamp_rate(last_rate * DILUTION)
        sigma = math.sqrt(FROZEN_SIGMA ** 2 + (0.5 * math.sqrt(horizon_days)) ** 2)
        sigma = min(2.0, max(0.25, sigma))
        return [round4(min(99.99, max(0.01, inv_logit(logit(median) + z * sigma)))) for z in Z]

    # --- Non-frozen: log-time logit trend on the last 24 h of reads.
    t_last = reads[-1][0]
    window = [r for r in reads if (t_last - r[0]).total_seconds() <= 24 * 3600]
    if len(window) < 3:
        window = list(reads)
    pts = [(x, logit(r)) for _, x, r, _ in window if r is not None]
    xs = sorted({x for x, _ in pts})
    if last_rate is None:
        base = priors.get(trophy["id"], TIER_DEFAULT.get(tier, 0.3))
        median = clamp_rate(base)
        sigma = 1.2
        return [round4(min(99.99, max(0.01, inv_logit(logit(median) + z * sigma)))) for z in Z]
    if len(pts) < 3 or len(xs) < 3:
        # Too little movement to fit a line: hold the last rate, wide spread.
        median = clamp_rate(last_rate)
        sigma = 1.2
        return [round4(min(99.99, max(0.01, inv_logit(logit(median) + z * sigma)))) for z in Z]

    # Recency weights, aligned with pts (same non-null filter as pts).
    ws = [0.5 ** ((t_last - at).total_seconds() / (12 * 3600)) for at, x, r, _ in window if r is not None]
    a, b = weighted_line([x for x, _ in pts], [y for _, y in pts], ws)
    x_last = reads[-1][1]
    lg_med = a + b * x_target
    # Guardrail: the extrapolated logit may move at most 3.0 from the last rate.
    lg_last = logit(last_rate)
    lg_med = min(lg_last + 3.0, max(lg_last - 3.0, lg_med))
    median = clamp_rate(inv_logit(lg_med))
    median = min(median, cap)
    median = max(median, 0.01)

    # Spread: random walk + fit residuals + slope extrapolation + rare-trophy term.
    rw = 0.5 * math.sqrt(horizon_days)
    resid_sq = [(y - (a + b * x)) ** 2 for x, y in pts]
    n = len(resid_sq)
    var = sum(resid_sq) / (n - 1) if n > 1 else 0.0
    resid = min(1.0, max(0.15, math.sqrt(var)))
    slope_unc = abs(b) * (x_target - x_last) * 0.5
    rare = 0.3 if last_rate < 0.5 else 0.0
    sigma = math.sqrt(rw ** 2 + resid ** 2 + slope_unc ** 2 + rare ** 2)
    sigma = min(2.0, max(0.25, sigma))

    qs = [round4(min(99.99, max(0.01, inv_logit(logit(median) + z * sigma)))) for z in Z]
    # Enforce non-decreasing (rounding can tie, never invert, but be safe).
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = log_time(target)

    # (at, x, rate, players_tracked) per read, oldest first.
    reads_all = []
    for read in history:
        at = parse_time(read["at"])
        reads_all.append((at, log_time(at), read.get("players_tracked")))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        reads = []
        for j, read in enumerate(history):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            reads.append((reads_all[j][0], reads_all[j][1], rate, reads_all[j][2]))
        rows.append({"id": trophy["id"], "q": forecast_trophy(trophy, reads, horizon_days, x_target, PRIORS)})
    return {"trophies": rows}


def main() -> None:
    # mkdir only: never remove the output directory (the grader user cannot
    # recreate /output if it is gone).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
