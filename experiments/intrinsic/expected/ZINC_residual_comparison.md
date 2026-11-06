# ZINC — residual PE comparison

Mean ± standard error over 5 folds. A winner is declared only when the ±1 SE bars
of the two best columns are disjoint, otherwise *statistical tie*. Cycle benchmarks
leak the ring encoding (*leak*) for the R+G columns.

*Note: `RRWP+L+AFM+FM` is `RRWP_FM_AFM` (RRWP+FM+AFM); no separate Laplacian block exists for it.*

| Benchmark | Metric | RRWP+L | RRWP+L+AFM+FM | RRWP+L+R+G | RRWP+L+R+G+AFM+FM | RRWP+L+R+G+Both res. | Winner |
|---|---|---|---|---|---|---|---|
| Degree | MAE↓ | 0.070 ± 0.011 | 0.013 ± 0.002 | 0.008 ± 0.003 | 0.009 ± 0.002 | 0.008 ± 0.003 | statistical tie |
| Betweenness | MAE↓ | 0.048 ± 0.001 | 0.033 ± 0.000 | 0.020 ± 0.001 | 0.018 ± 0.000 | 0.017 ± 0.001 | statistical tie |
| Bridge Node | Acc↑ | 0.950 ± 0.002 | 0.980 ± 0.001 | 0.992 ± 0.000 | 0.995 ± 0.000 | **0.996 ± 0.000** | RRWP+L+R+G+Both res. |
| Eccentricity | MAE↓ | 1.081 ± 0.018 | 1.466 ± 0.011 | 0.632 ± 0.006 | 0.648 ± 0.015 | 0.631 ± 0.003 | statistical tie |
| Min Cycle (Node) | MAE↓ | 0.603 ± 0.012 | **0.278 ± 0.006** | *leak* | *leak* | *leak* | RRWP+L+AFM+FM |
| Link Pred. | Acc↑ | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | statistical tie |
| Edge Betw. | MAE↓ | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | **0.003 ± 0.000** | RRWP+L+R+G+Both res. |
| Min Cycle (Edge) | MAE↓ | 0.102 ± 0.005 | **0.072 ± 0.003** | *leak* | *leak* | *leak* | RRWP+L+AFM+FM |
| Bridge Edge | Acc↑ | 0.994 ± 0.000 | 0.995 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | statistical tie |
| Local Conn. | MAE↓ | 0.074 ± 0.002 | 0.051 ± 0.001 | 0.028 ± 0.000 | 0.024 ± 0.001 | 0.023 ± 0.001 | statistical tie |
| Num Cycles | MAE↓ | 0.059 ± 0.003 | **0.047 ± 0.002** | *leak* | *leak* | *leak* | RRWP+L+AFM+FM |
| Diameter | MAE↓ | 0.473 ± 0.022 | 0.323 ± 0.007 | 0.295 ± 0.010 | 0.290 ± 0.010 | 0.280 ± 0.010 | statistical tie |
| Density | MAE↓ | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 | statistical tie |
| Wiener Index | MAE↓ | 0.096 ± 0.004 | 0.072 ± 0.003 | 0.059 ± 0.001 | 0.062 ± 0.005 | 0.065 ± 0.007 | statistical tie |
| Alg. Connect. | MAE↓ | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.000 | statistical tie |
