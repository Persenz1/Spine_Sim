$ErrorActionPreference = "Stop"

throw @"
This offline runner is disabled because its campaign prescribes a fixed holder Z.
M2 now requires continuous external preload and time-domain contact dynamics.
Do not resume the old results/m2_round1 campaign.

Read docs/m2/M2_DYNAMIC_CONSTANT_PRELOAD_SPEC.md and close the calibration and
time-step convergence gates before creating a new formal screening campaign.
"@
