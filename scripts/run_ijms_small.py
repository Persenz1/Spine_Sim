"""Prepare, resume or inspect the prescribed IJMS small-array campaign."""
import os

# The runner parallelizes cases; avoid multiplying BLAS threads per worker.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from spine_sim.small_array_campaign import main

if __name__ == "__main__":
    raise SystemExit(main())
