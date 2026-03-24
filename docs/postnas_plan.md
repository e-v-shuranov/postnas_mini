# PostNAS sandbox plan

This repository now supports a tiny PostNAS-style attention sandbox controlled entirely by environment variables.

## Goals

- Preserve baseline behavior by default (`POSTNAS_ENABLE=0`).
- Keep the existing attention path as the `full` backend.
- Add one optional backend: local causal sliding-window attention.
- Make layer replacement explicit and 0-based.

## Env controls

- `POSTNAS_ENABLE=0|1`
- `POSTNAS_BACKEND=full|local`
- `POSTNAS_REPLACE_LAYERS="comma,separated,0based,indices"`
- `POSTNAS_WINDOW=128`

## Behavior

- With `POSTNAS_ENABLE=0`, all layers use the original full attention.
- With `POSTNAS_ENABLE=1`, only layers listed in `POSTNAS_REPLACE_LAYERS` use `POSTNAS_BACKEND`.
- Empty `POSTNAS_REPLACE_LAYERS` means no layer replacement.
- Layer indices are interpreted as 0-based and validated against `NUM_LAYERS`.

## Included run preset

- `configs/postnas_local_last3.env` enables local attention on the final three layers for the default 9-layer model (`6,7,8`) with `POSTNAS_WINDOW=128`.
- `scripts/run_postnas_local_last3.sh` sources that preset and runs `train_gpt.py`.
