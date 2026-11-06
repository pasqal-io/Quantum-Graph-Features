# Emulator-QPU concordance at t4 (App. D.4, Fig. 5)

Reproduced from the stored raw counts (`data/raw/emulation/mutag/afm.npz`,
`data/raw/qpu/mutag_afm/t4_715ns.npz`); the QPU itself cannot be re-run from here.

QPU timing: the register is rescaled so that Omega = 0.3 J_max fits the device amplitude,
and pulse durations are stretched accordingly (`t_QPU = t_emu * scale^6`). On hardware
the sequences were padded with a leading delay to a total of 6000 ns; the effective
pulse is unchanged (at most 4000 ns of emulation-equivalent time). Readout correction
uses (eps, eps') = (0.025, 0.1) for P(1|0) and P(0|1).

```bash
uv run python experiments/qpu_vs_emulation/run.py
```

Outputs `results/qpu_vs_emulation/fig5.{json,png}` (expected: `expected/fig5.json`).

| quantity | value |
|---|---|
| one-body Spearman per graph, median / mean | 0.921 / 0.910 |
| two-body Spearman per graph, median / mean | 0.870 / 0.861 |
| pooled one-body / two-body | 0.927 / 0.863 |

Both QPU and emulation counts go through the readout-corrected weighted jackknife
(eps = P(1|0) = 0.025, eps' = P(0|1) = 0.1): two-atom correction off the diagonal,
single-atom correction of the occupation on the diagonal (`C_ii = p(1-p)`). Entries are
rounded to 1e-12 before ranking so that atoms with identical statistics stay exactly tied.

Hardware protocol (`src/qgf/generate/qpu.py`), checked against the stored registers:
register scaled by `s = (0.3 C6 / (r_min^6 0.99 max_amp))^(1/6)` = 1.0474 for all graphs,
pulse `t_QPU = floor_4ns(t_emu s^6)` (715 ns -> 940 ns; 715 s^6 = 943.8-944.0 so a few
graphs sit at the floor boundary with float32 positions), leading delay up to 6 us, shots
`shots_clt(N)` (App. D.3).

Submission / retrieval (credentials from `PASQAL_USERNAME`, `PASQAL_PASSWORD`,
`PASQAL_PROJECT_ID`; batch id on the command line):

```bash
uv run --extra generate python experiments/qpu_vs_emulation/submit.py submit --t-emu 715 --dry-run
uv run --extra generate python experiments/qpu_vs_emulation/submit.py fetch --batch-id <id> \
    --time-key 940 --out data/raw/my_run.npz
```
