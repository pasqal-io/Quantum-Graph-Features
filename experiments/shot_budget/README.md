# Shot budget for the hardware run (App. D.3)

```bash
uv run python experiments/shot_budget/run.py                         # formula + empirical v_max
uv run python experiments/shot_budget/run.py --pulses fm --epsilon 0.01   # original notebook setting
```

`shots_clt(N) = ceil(v_max / eps^2 * k_N^2)`, `k_N = Phi^-1(1 - delta / (N (N + 1)))`
(Bonferroni over `M = N(N+1)/2` correlations), `v_max = 0.06`, `eps = 0.03`, `delta = 0.05`.
For N = 18: `k_18 = 3.622`, `66.67 k_18^2 = 874.57`, i.e. 875 shots.
The QPU runs used `shots_clt(N)` per graph: 734 (N = 10) to 983 (N = 28).

Empirical per-shot variance of the connected-correlation estimators (bootstrap, 200
resamples, max over pairs and times, per graph), from the emulated MUTAG shots:

| pulse | median | range |
|---|---|---|
| AFM | 0.0749 | 0.0670 - 0.0844 |
| FM | 0.0777 | 0.0697 - 0.0924 |

The port reproduces the original notebook (FM, eps = 0.01) to 5e-7 in `sigma2_max` and
exactly in every CLT/Bernstein budget (`expected/reference_fm_correlation.json`). Note that
`v_max = 0.06` is below these empirical maxima (by ~25%).
