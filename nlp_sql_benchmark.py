"""
nlp_sql_benchmark.py
====================
Measures the speedup of the optimized NL->SQL data pipeline against the
original (pre-optimization) implementation.

WHY THIS MEASURES WHAT IT MEASURES
----------------------------------
The NL->SQL request path is:

    prompt
      -> extract_schema_context()        (DB round-trip, now cached)
      -> ollama.chat(llama3.2)           (LLM inference: ~2-30s, non-deterministic)
      -> validate + extract SQL
      -> execute SQL on a DB connection  (new connect() per call -> now pooled)
      -> serialize rows

The optimizations made were strictly in the DATA layer:
  1. extract_schema_context() is now @lru_cache'd (was a full DB round-trip
     on every single request).
  2. All psycopg2.connect() calls now use a ThreadedConnectionPool (was a
     fresh TCP handshake + PostgreSQL session per request).
  3. pgvector type registration happens once per pooled connection.

The LLM inference step is non-deterministic and dominated by model/hardware,
NOT by anything we changed. Including it would drown our optimization in noise
and produce an unreproducible number. So this benchmark isolates the data-layer
work we actually optimized: the schema-context fetch + the per-query DB
connection + a representative SELECT execution.

This is an apples-to-apples comparison of the OLD per-request DB pattern vs the
NEW pooled+cached pattern, run against the same live PostgreSQL instance.

USAGE
-----
    # 1. Make sure PostgreSQL is up and the DB has data (run seed_test_data.py).
    # 2. Run directly (imports main.py; the server itself need NOT be running):
    python nlp_sql_benchmark.py

    # Optionally tune iteration count:
    python nlp_sql_benchmark.py --iterations 100
"""

import argparse
import time
import statistics

import psycopg2
from pgvector.psycopg2 import register_vector

# Import the application module so we exercise the REAL optimized code paths.
import main


# A representative SELECT that the NL->SQL stage typically produces. We execute
# the SAME statement in both baseline and optimized runs so the only difference
# measured is the connection + schema-context handling, not query complexity.
REPRESENTATIVE_SQL = "SELECT scene_id, frame_timestamp FROM spatial_scene_store LIMIT 50;"


# ---------------------------------------------------------------------------
# BASELINE: faithfully reproduces the ORIGINAL pre-optimization behavior.
#   - a brand-new psycopg2.connect() per "query"
#   - register_vector() on every new connection
#   - a fresh, uncached schema-context DB round-trip every call
# ---------------------------------------------------------------------------

def _baseline_schema_context():
    """Original extract_schema_context(): a full DB round-trip, no caching."""
    conn = psycopg2.connect(**main.DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_name = 'spatial_scene_store'
        ORDER BY ordinal_position;
    """)
    cursor.fetchall()
    cursor.close()
    conn.close()


def baseline_pipeline_step():
    """One full data-layer pass using the ORIGINAL per-request pattern."""
    # 1. Uncached schema introspection (new connection, full round-trip).
    _baseline_schema_context()

    # 2. Fresh connection for query execution + per-connection vector register.
    conn = psycopg2.connect(**main.DB_PARAMS)
    try:
        conn.set_session(readonly=True)
        register_vector(conn)
        cursor = conn.cursor()
        cursor.execute(REPRESENTATIVE_SQL)
        cursor.fetchall()
        cursor.close()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# OPTIMIZED: uses the actual functions/pool from main.py.
#   - extract_schema_context() is @lru_cache'd
#   - get_conn()/put_conn() reuse pooled, vector-registered connections
# ---------------------------------------------------------------------------

def optimized_pipeline_step():
    """One full data-layer pass using the NEW pooled + cached pattern."""
    # 1. Cached schema context (DB hit only on the very first call, ever).
    main.extract_schema_context()

    # 2. Pooled, already-vector-registered connection for query execution.
    conn = main.get_conn(register_vec=True)
    try:
        # set_session() requires no open transaction; register_vector() may have
        # opened one on first checkout, so roll back to a clean state first.
        conn.rollback()
        conn.set_session(readonly=True)
        cursor = conn.cursor()
        cursor.execute(REPRESENTATIVE_SQL)
        cursor.fetchall()
        cursor.close()
    finally:
        main.put_conn(conn)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_runs(fn, iterations, warmup):
    """Run fn `iterations` times, return per-call latencies in ms.
    The first `warmup` calls are discarded (cold-start, cache priming)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


def summarize(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "mean": statistics.mean(s),
        "p50": s[int(n * 0.50)],
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "min": s[0],
        "max": s[-1],
    }


def fmt_row(label, st):
    return (f"  {label:<10} mean={st['mean']:8.3f}ms  p50={st['p50']:8.3f}ms  "
            f"p95={st['p95']:8.3f}ms  p99={st['p99']:8.3f}ms  "
            f"min={st['min']:8.3f}ms  max={st['max']:8.3f}ms")


def main_benchmark(iterations, warmup):
    print("====== NL->SQL DATA-PIPELINE OPTIMIZATION BENCHMARK ======")
    print(f"Iterations: {iterations}  (warmup discarded: {warmup})")
    print("Scope: schema-context fetch + per-query DB connection + representative SELECT")
    print("Excluded: LLM inference (non-deterministic; not part of the optimization)\n")

    # Pre-flight: confirm DB connectivity and prime the optimized pool/cache.
    try:
        main.init_pool()
        main.init_db()
    except Exception as e:
        print(f"[FATAL] Could not initialize DB/pool: {e}")
        print("        Is PostgreSQL running and is 'spatial_vector_db' reachable?")
        return

    # --- Baseline ---
    print("[1/2] Measuring BASELINE (original per-request connect + uncached schema)...")
    try:
        baseline = time_runs(baseline_pipeline_step, iterations, warmup)
    except Exception as e:
        print(f"[FATAL] Baseline run failed: {e}")
        return

    # --- Optimized ---
    print("[2/2] Measuring OPTIMIZED (pooled connections + lru_cached schema)...")
    try:
        optimized = time_runs(optimized_pipeline_step, iterations, warmup)
    except Exception as e:
        print(f"[FATAL] Optimized run failed: {e}")
        return

    base_st = summarize(baseline)
    opt_st = summarize(optimized)

    print("\n[ RESULTS ]")
    print(fmt_row("BASELINE", base_st))
    print(fmt_row("OPTIMIZED", opt_st))

    speedup_mean = base_st["mean"] / opt_st["mean"] if opt_st["mean"] > 0 else float("inf")
    speedup_p99 = base_st["p99"] / opt_st["p99"] if opt_st["p99"] > 0 else float("inf")

    print("\n[ SPEEDUP ]")
    print(f"  • Mean latency : {base_st['mean']:.3f}ms -> {opt_st['mean']:.3f}ms  "
          f"= {speedup_mean:.1f}x faster")
    print(f"  • P99 latency  : {base_st['p99']:.3f}ms -> {opt_st['p99']:.3f}ms  "
          f"= {speedup_p99:.1f}x faster")

    print("\n[ RESUME LINE — copy/paste ]")
    print(f'  "Optimized a local NL-to-SQL telemetry pipeline\'s data layer via '
          f'connection pooling and schema-context caching, cutting per-query '
          f'overhead from {base_st["mean"]:.1f}ms to {opt_st["mean"]:.2f}ms '
          f'({speedup_mean:.0f}x throughput improvement)."')
    print("\nNote: speedup reflects the data-layer work that was actually optimized,")
    print("measured against the same PostgreSQL instance. LLM inference is excluded")
    print("because it is unchanged and non-deterministic.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=50,
                    help="number of timed iterations per variant (default 50)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="warmup iterations to discard (default 5)")
    args = ap.parse_args()
    main_benchmark(args.iterations, args.warmup)
