"""Benchmark kantorovich vs POT (Python Optimal Transport).

Produces a markdown table and optional CSV for plotting.
Run: python benchmarks/bench_vs_pot.py
"""

import numpy as np
import time
import sys
import os

# Ensure kantorovich is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kantorovich as kt

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("POT not installed. Run: pip install POT")
    print("Benchmarking kantorovich only.\n")


def benchmark_fn(fn, warmup=2, runs=10):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)

    return np.median(times)


def bench_sinkhorn():
    print("=" * 70)
    print("SINKHORN BENCHMARK")
    print("=" * 70)
    print(f"{'N':>6} | {'kantorovich':>12} | {'POT':>12} | {'Speedup':>8}")
    print("-" * 50)

    np.random.seed(42)
    reg = 1.0

    for n in [100, 200, 500, 1000, 2000, 5000]:
        x = np.random.randn(n, 5)
        y = np.random.randn(n, 5)
        a = np.ones(n) / n
        b = np.ones(n) / n

        M_k = kt.cost_matrix(x, y, metric="sqeuclidean")

        runs = 5 if n >= 5000 else 10
        t_k = benchmark_fn(lambda: kt.sinkhorn_solve(a, b, M_k, reg=reg), runs=runs)

        if HAS_POT:
            M_p = ot.dist(x, y, metric="sqeuclidean")
            t_p = benchmark_fn(lambda: ot.sinkhorn(a, b, M_p, reg=reg), runs=runs)
            ratio = t_p / t_k
            print(f"{n:6d} | {t_k:10.1f}ms | {t_p:10.1f}ms | {ratio:7.2f}x")
        else:
            print(f"{n:6d} | {t_k:10.1f}ms | {'N/A':>12} | {'N/A':>8}")


def bench_sliced():
    print()
    print("=" * 70)
    print("SLICED WASSERSTEIN BENCHMARK")
    print("=" * 70)
    print(f"{'N':>8} | {'d':>3} | {'Proj':>5} | {'Time':>10}")
    print("-" * 40)

    np.random.seed(42)

    configs = [
        (1000, 5, 50),
        (1000, 10, 100),
        (10000, 5, 50),
        (50000, 5, 50),
        (100000, 5, 50),
    ]

    for n, d, proj in configs:
        x = np.random.randn(n, d)
        y = np.random.randn(n, d) + 1.0
        a = np.ones(n) / n
        b = np.ones(n) / n

        runs = 3 if n >= 50000 else 5
        t = benchmark_fn(
            lambda: kt.sliced_wasserstein_solve(x, y, a, b, n_projections=proj, p=2.0),
            runs=runs,
        )
        print(f"{n:8d} | {d:3d} | {proj:5d} | {t:8.1f}ms")


def bench_cost_matrix():
    print()
    print("=" * 70)
    print("COST MATRIX BENCHMARK")
    print("=" * 70)
    print(f"{'N':>6} | {'kantorovich':>12} | {'POT':>12} | {'Speedup':>8}")
    print("-" * 50)

    np.random.seed(42)

    for n in [100, 500, 1000, 2000, 5000]:
        x = np.random.randn(n, 5)
        y = np.random.randn(n, 5)

        runs = 5 if n >= 5000 else 10
        t_k = benchmark_fn(
            lambda: kt.cost_matrix(x, y, metric="sqeuclidean"), runs=runs
        )

        if HAS_POT:
            t_p = benchmark_fn(
                lambda: ot.dist(x, y, metric="sqeuclidean"), runs=runs
            )
            ratio = t_p / t_k
            print(f"{n:6d} | {t_k:10.1f}ms | {t_p:10.1f}ms | {ratio:7.2f}x")
        else:
            print(f"{n:6d} | {t_k:10.1f}ms | {'N/A':>12} | {'N/A':>8}")


def bench_1d():
    print()
    print("=" * 70)
    print("1D EXACT OT BENCHMARK")
    print("=" * 70)
    print(f"{'N':>8} | {'kantorovich':>12}")
    print("-" * 25)

    np.random.seed(42)

    for n in [1000, 10000, 100000, 1000000]:
        x_a = np.sort(np.random.randn(n))
        x_b = np.sort(np.random.randn(n)) + 1.0
        a = np.ones(n) / n
        b = np.ones(n) / n

        runs = 3 if n >= 100000 else 10
        t = benchmark_fn(lambda: kt.emd_1d(x_a, a, x_b, b, p=2.0), runs=runs)
        print(f"{n:8d} | {t:10.1f}ms")


if __name__ == "__main__":
    bench_sinkhorn()
    bench_cost_matrix()
    bench_sliced()
    bench_1d()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
