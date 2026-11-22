# Neural SDE Asset Modeling: A Leakage-Aware Research Benchmark

This Python research package compares a **neural conditional stochastic differential equation**, a constant-volatility **geometric Brownian motion (GBM)** baseline, and a **Heston-inspired variance filter**. It forecasts one-step daily log-return distributions and evaluates them with held-out predictive log likelihood, 5% Value-at-Risk coverage, and probability integral transform diagnostics.

> **Model classification:** The neural model represents `d log S_t = mu_theta(x_t) dt + sigma_theta(x_t) dW_t`, where `x_t` is a causal vector of past returns and realized-volatility features. Training minimizes the *Euler Gaussian approximation* to the daily transition density. It is **not** an exact Heston likelihood, an end-to-end neural latent diffusion, a risk-neutral options calibrator, or a guarantee of trading performance. The comparator called `Heston-style filter` is a variance-reverting observed-data proxy, **not a fitted CIR/Heston latent variance SDE**.

## Install and test

Python 3.10+:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
neural-sde --days 900 --epochs 35 --hidden 32 --output results
```

Fast CI smoke run:

```bash
neural-sde --days 220 --epochs 2 --hidden 8 --output smoke
```

For your own data, supply a CSV with ordered `date,close` rows containing **positive, point-in-time adjusted** prices:

```bash
neural-sde --csv daily_adjusted_close.csv --epochs 60 --output real-results
```

Output files: `metrics.json`, `predictions.csv`, `diagnostics.png`.

## Research protocol

1. **Simulator.** Full-truncation Euler with correlated Brownian shocks produces synthetic daily closes from a known Heston dynamics. Only the prices are exposed to fitted models; latent variance is retained only for simulation diagnostics.
2. **Causal features.** At date t, lagged return, realized volatility, trailing average return, and short-horizon realized volatility use returns strictly before the prediction target. The normalization mean and standard deviation are estimated on training samples only.
3. **Chronological splits.** 60% train, 20% validation, and 20% test, without shuffling the ordering or training on test data.
4. **Models.** GBM estimates constant daily drift and volatility. The Heston-inspired observed variance filter fits a mean-reverting exponential estimate from training returns and updates after each realized observation. A small PyTorch network models conditional daily return mean and standard deviation using Gaussian Euler transitions.
5. **Evaluation.** All methods use identical held-out dates. Compare predictive Gaussian NLL (lower is better), return RMSE, volatility-forecast MSE, nominal 5% left-tail exceedance, and uniformity of probability integral transforms. The 20-day terminal GBM Monte Carlo distribution is illustrative only.

## Synthetic experiment

Run with `--days 900 --epochs 35 --hidden 32 --seed 42`:

| Model | Held-out Gaussian NLL | Return RMSE | 5% VaR exceedance |
| --- | ---: | ---: | ---: |
| GBM | -3.339416 | 0.0078903 | 1.69% |
| Heston-style filter | -3.448829 | 0.0078903 | 5.08% |
| Neural SDE | -3.407632 | 0.0078881 | 1.69% |

In this synthetic experiment, the Heston-style filter leads on **test log likelihood** and its 5% VaR exceedance is closest to nominal. The neural model is slightly better on mean-return RMSE and variance-forecast MSE, but **does not beat the simpler volatility filter overall**. Synthetic experiments do not show real-market pricing or hedging performance.

## Limitations and next research stages

- The chosen features are discrete observed states, not fully specified latent diffusion dynamics. Neural model paths are not simulated autoregressively here; adding that, more sophisticated solvers, exact path constraints, and model calibration would be separate work.
- All forecast densities use a Gaussian approximation. Heavy tails, jumps, liquidity shocks, structural breaks, stochastic correlation and nonstationarity require separate treatment.
- The Heston volatility simulator includes leverage correlation; its parameters are **not** inferred from observed option prices. Results do not represent arbitrage-free option prices.
- PIT and VaR quality need larger samples, uncertainty intervals and multiple market periods for conclusive inference.
- No live execution, transaction costs, portfolio returns, or investment decisions are modeled.

## Repository layout

- `src/neural_sde/data.py`: Heston simulator, strict price loader, causal features.
- `src/neural_sde/models.py`: GBM, variance filter, trainable neural conditional SDE.
- `src/neural_sde/research.py`: reproducible chronological benchmark, diagnostics.
- `src/neural_sde/cli.py`: command-line experiments and plots.
- `tests/test_research.py`: 15 simulation, causality, shape, calibration, and reproducibility tests.
- `.github/workflows/tests.yml`: CI for Python 3.10 and 3.12.

MIT licensed.