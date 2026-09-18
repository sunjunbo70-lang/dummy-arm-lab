# Contributing

Thanks for looking. This project has a few rules that are stricter than usual, because
it drives real hardware and because unreproducible robot results are expensive.

## The one rule that matters most

**No change may raise a claim's evidence level without attaching the experiment that
earned it.**

| Level | Meaning |
| :---: | --- |
| L1 | Software test, no hardware attached |
| L2 | Controller feedback reached the target |
| L3 | Physically witnessed by an operator |

If your PR moves a row in [`docs/STATUS.md`](docs/STATUS.md) from L1 to L2 or L3, it must
also add a directory under `experiments/` containing the run record. Firmware reporting
that it arrived is L2, not L3 — the controller readout is not end-effector accuracy.

## Safety boundaries

These are enforced socially, and where possible in code. See [`AGENTS.md`](AGENTS.md).

1. Do not remove or weaken the gates in `core.load_profile`. They exist because
   uncalibrated parameters mean the software's understanding of the world is unverified,
   and a wrong sign silently drives the arm the wrong way.
2. Do not flip a `*_verified` flag in a committed profile. Calibration results go in a
   new file with provenance, never by editing an existing one.
3. Do not delete or rewrite anything under `experiments/`, including failure logs.
   Failed runs are kept on purpose — a repo where only successes survive cannot be audited.
4. `SerialRobot.send_action` raising is intentional. Autonomous execution over the legacy
   USB path is not enabled until the per-axis freshness work (M2) and calibration (M3)
   are complete.

## Before opening a PR

```bash
pip install -r requirements/dev.txt
python -m unittest discover -s tests -v
```

Tests that need DummyStudio-derived data skip cleanly; that is expected, not a failure.
`requirements/dev.txt` adds `jsonschema`, which turns on declarative validation of device
profiles — without it the hand-written checks still run, but you get one error at a time
instead of all of them.

## Conventions

- Configuration lives in `configs/`, never hardcoded in source. Device identity included.
- New data files follow [`docs/DATA_FILES.md`](docs/DATA_FILES.md): split by *who writes
  it, how often it changes, and who gets hurt if it is overwritten* — not by whether the
  contents look similar.
- Experiment directories follow `experiments/_template`. One directory per run, with
  commands, seed, environment and outcome. Do not overwrite the baseline.
- Docs are the deliverable, not an afterthought. A behaviour change that is not reflected
  in `docs/` is incomplete.

## Scope

Issues about the Dummy V2 hardware itself, the upstream firmware, or DummyStudio belong
upstream — see [`THIRD_PARTY.md`](THIRD_PARTY.md) for where each source came from.
