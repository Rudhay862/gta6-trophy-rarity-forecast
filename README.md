# GTA VI Trophy Rarity Forecaster — make.fun `gta6-trophy-rarity-forecast-01`

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five
quantiles of its earn rate on Exophase at the case's target, as percent.

## Model

Rates live on the logit scale, `logit(p) = ln(p / (100 - p))`, because the
contest scores pinball loss on that scale. Time is `x = ln(1 + hours since
release)`, release anchor 2026-11-19T00:00:00Z: earn rates grow roughly with
log-time early in a game's life, then saturate.

Two regimes, decided per trophy from the case's own history:

1. **Frozen history (stale Exophase page).** If the history has ≥ 3 reads and
   every rate for a trophy is identical across all reads AND `players_tracked`
   is constant, the page has not refreshed since launch night. The house
   documented the first refresh: 638 → 3,868 players tracked, every rate but
   the 100% one lower; a live day-1.5 read of the page (see `priors.json`)
   shows rates ~5–10× below launch night. A refresh dilutes rates because
   the tracked-player count jumps while earners grow more slowly. So a frozen
   trophy's median is `last_rate × 0.25` (100% → 99.99, 0% → 0.01), with a
   wide spread: `sigma = sqrt(0.9² + (0.5·sqrt(horizon_days))²)`.

2. **Non-frozen.** Weighted least squares of `logit(rate)` on `x`, using reads
   from the last 24 h (or all reads if fewer than 3 fall in that window),
   recency weights `w = 0.5 ** ((t_last − t) / 12 h)`, extrapolated to the
   target. Needs ≥ 3 distinct x values, else it holds the last rate with
   sigma 1.2. Guardrails: the extrapolated logit may move at most 3.0 from
   `logit(last_rate)`; week-one plausibility caps by tier (platinum ≤ 1.5%,
   gold ≤ 8%, silver ≤ 25%, bronze ≤ 70%); floor 0.01. If the last rate is
   null, the `priors.json` day-1.5 rate for that trophy id is used (else a
   tier default: platinum 0.1, gold 0.2, silver 0.2, bronze 0.3).

Spread (non-frozen): `sigma = sqrt(rw² + resid² + slope_unc² + rare²)` —
random walk `0.5·sqrt(horizon_days)`, fit-residual std (floor 0.15, cap 1.0),
slope-uncertainty `|slope| × (x_target − x_last) × 0.5`, and `0.3` when the
last rate is below 0.5%. Sigma is clamped to [0.25, 2.0].

Quantiles: `q_τ = 100 / (1 + exp(−(logit(median) + z_τ·sigma)))`,
`z = [−1.645, −0.674, 0, 0.674, 1.645]`, rounded to 4 decimals half-up,
enforced non-decreasing, clamped to [0.01, 99.99].

## Data sources

- **The case's own history** is the primary input — every read the house made
  of Exophase's GTA VI trophy page before the case's checkpoint.
- **`priors.json`** — real Exophase snapshots read live while building (the
  graded run has no network, so the data is baked into the entry): launch
  night (the house's own recording, 638 players tracked) and day 1.5
  (2026-11-19T11:36Z). Used only as a prior for null last rates and to
  calibrate the stale-page dilution above.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic: two runs on
the same case write identical bytes. Reads every `*.in` in the input
directory and writes `/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR/<id>.out`):
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## Practice results (kit cases p00/p01)

- Valid: 55 rows each, ids 1–55 once, quantiles monotone in [0, 100],
  2,767 bytes each, byte-identical across two runs, ~0.07 s for both cases.
- Pinball loss vs the kit starter on synthetic answers: dilution ×0.25 —
  **0.181 vs 1.279**; day-1.5 priors — **0.243 vs 1.508** (starter 0.025 only
  in the no-refresh world, which the documented refresh rules out).
  Typical miss on the dilution answer 0.0002 points, 90% coverage 0.982.
- Sanity: platinum (id 1) median 0.1575% < chapter bronzes (~0.275%);
  the 100% trophy (id 2) median ~99.99; frozen medians below last rates.
