"""Reject friction scans derived from the obsolete fixed-holder-Z campaign."""

from __future__ import annotations


MESSAGE = """
M2 friction campaign derivation is disabled.

The low/medium/high scans were based on the obsolete fixed-holder-Z model.
Friction sensitivity must be rebuilt on the continuous-preload dynamic model
after dynamic contact parameters and time-step convergence are closed.

See docs/m2/M2_DYNAMIC_CONSTANT_PRELOAD_SPEC.md.
""".strip()


def main() -> int:
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
