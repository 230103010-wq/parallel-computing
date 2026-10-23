import os, time, statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

A, B = 2.5, 1.0
CORES = os.cpu_count()
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

def run_serial(x):
    return A * x + B

def thread_kernel(args):
    x, out, lo, hi = args
    np.multiply(x[lo:hi], A, out=out[lo:hi])
    out[lo:hi] += B

def run_threads(pool, x):
    n = len(x)
    out = np.empty_like(x)
    step = -(-n // CORES)
    tasks = [(x, out, i, min(i + step, n)) for i in range(0, n, step)]
    list(pool.map(thread_kernel, tasks))
    return out

def proc_kernel(chunk):
    return A * chunk + B

def run_procs(pool, x):
    chunks = np.array_split(x, CORES)
    return np.concatenate(list(pool.map(proc_kernel, chunks)))

def bench(fn, n):
    reps = max(5, min(200, int(2_000_000 / n)))
    fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)

def measure(n, runner, pool):
    x = np.random.rand(n)
    ts = bench(lambda: run_serial(x), n)
    tp = bench(lambda: runner(pool, x), n)
    return ts, tp

def find_crossover(name, runner, pool):
    print(f"\n=== {name} ({CORES} workers) ===")
    print(f"{'N':>12} {'serial, s':>12} {'parallel, s':>12} {'speedup':>9}  verdict")
    res = []
    for n in SIZES:
        ts, tp = measure(n, runner, pool)
        res.append((n, ts, tp))
        print(f"{n:>12,} {ts:>12.6f} {tp:>12.6f} {ts/tp:>9.2f}x  "
              f"{'PARALLEL faster' if tp < ts else 'serial faster'}")

    for i in range(1, len(res)):
        if res[i-1][2] >= res[i-1][1] and res[i][2] < res[i][1]:
            lo, hi = res[i-1][0], res[i][0]
            for _ in range(10):
                mid = int((lo * hi) ** 0.5)
                if mid in (lo, hi):
                    break
                ts, tp = measure(mid, runner, pool)
                if tp < ts: hi = mid
                else:       lo = mid
            print(f"\n>>> {name}: parallel became FASTER than serial for N in "
                  f"({lo:,} .. {hi:,}]")
            print(f"    slower at N={lo:,}, faster at N={hi:,}")
            return
    if res[0][2] < res[0][1]:
        print(f"\n>>> {name}: parallel is already faster at N={SIZES[0]:,}")
    else:
        print(f"\n>>> {name}: parallel did NOT beat serial up to N={SIZES[-1]:,}")

if __name__ == "__main__":
    print(f"CPU cores: {CORES}, numpy {np.__version__}")
    with ThreadPoolExecutor(CORES) as tpool:
        find_crossover("Threads", run_threads, tpool)
    with ProcessPoolExecutor(CORES) as ppool:
        list(ppool.map(abs, range(CORES)))
        find_crossover("Processes", run_procs, ppool)