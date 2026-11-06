"""Time one AFM emulation (1000 shots, 20 times) per backend: emu-sv and emu-mps.

uv run --extra generate python experiments/generation/benchmark_emulation.py [--cpu]
"""

import argparse
import time

import torch

from qgf.data import load
from qgf.generate.emulate import emulate
from qgf.generate.sequences import EVALUATION_TIMES_NS, quench_sequence, register_from_positions


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="*", default=[12, 17, 20, 24, 28])
    args = parser.parse_args(argv)
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    graphs, _ = load("mutag_afm")
    for n in args.sizes:
        data = next(g for g in graphs if int(g.num_nodes) == n)
        seq = quench_sequence(register_from_positions(data.pos), "afm")
        t0 = time.time()
        emulate(seq, EVALUATION_TIMES_NS, use_gpu=not args.cpu)
        backend = "emu-sv" if n <= 24 else "emu-mps"
        print(
            f"N={n} {backend} {'cpu' if args.cpu else 'gpu'}: {time.time() - t0:.1f} s", flush=True
        )


if __name__ == "__main__":
    main()
