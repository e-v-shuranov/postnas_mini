from __future__ import annotations


def build_layer_replace_mask(num_layers: int, selected_indices: tuple[int, ...]) -> tuple[bool, ...]:
    mask = [False] * num_layers
    for idx in selected_indices:
        if idx >= num_layers:
            raise ValueError(
                f"POSTNAS_REPLACE_LAYERS index {idx} is out of range for NUM_LAYERS={num_layers}"
            )
        mask[idx] = True
    return tuple(mask)
