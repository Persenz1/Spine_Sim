"""Reject the obsolete fixed-holder-Z M2 round-one campaign.

The previous materializer converted 15 terrain seeds into fixed-height paths.
That load boundary is no longer the M2 production model, so retaining a
working generator would make it too easy to create scientifically invalid
screening results.
"""

from __future__ import annotations


MESSAGE = """
M2 round-one campaign generation is disabled.

The former script prescribed holder Z after a one-time preload.  M2 now
requires continuous external preload and time-domain holder/contact dynamics.
Build a new dynamic campaign only after the holder mass, damping, contact
parameters, and production time-step convergence gates are frozen.

See:
  docs/m2/M2_DYNAMIC_CONSTANT_PRELOAD_SPEC.md
  docs/m2/M2_to_M3_handoff.md
  docs/m2/M2_DYNAMIC_TEST_REPORT.md
""".strip()


def main() -> int:
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
