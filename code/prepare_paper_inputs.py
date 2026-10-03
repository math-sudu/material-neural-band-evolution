"""Recover the paper's matched synthetic inputs from the saved experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mixed_pinn import ROOT, specimen


RUNS = {
    "common_start": "synthetic_state_heads",
    "control": "synthetic_state_heads_continued",
    "virtual_work": "synthetic_state_heads_virtual",
}
OUTPUT = ROOT / "results/paper_evidence"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def load_case():
    data = specimen("m-1-1", frame_count=12)
    source_frames = np.r_[data["frames"], [-1, -1, -1]]
    peak = data["forces"].max()
    data["forces"] = np.r_[data["forces"], np.array([.5, .15, .65])*peak]
    data["frames"] = np.arange(len(data["forces"]))
    data["reference_frame"] = 0
    data["observed_valid"] = np.concatenate(
        [data["observed_valid"], np.repeat(data["observed_valid"][-1:], 3, axis=0)])
    metadata, fields = {}, {}
    for role, name in RUNS.items():
        directory = ROOT / "results/mixed_pinn" / name
        metadata[role] = read_json(directory / "training.json")
        with np.load(directory / "training_fields.npz") as archive:
            fields[role] = {key: archive[key] for key in archive.files}
        np.testing.assert_array_equal(fields[role]["frames"], data["frames"])
        np.testing.assert_array_equal(fields[role]["forces_N"], data["forces"])
        np.testing.assert_array_equal(fields[role]["points_mm"], data["points_mm"])
        np.testing.assert_array_equal(fields[role]["observed_mm"], fields["common_start"]["observed_mm"])
        np.testing.assert_array_equal(fields[role]["cell_centers_mm"], fields["common_start"]["cell_centers_mm"])
        assert metadata[role]["geometry"] == json.loads(json.dumps(data["geometry"].__dict__))
    data["observed"] = fields["common_start"]["observed_mm"]
    data["sid"] = "synthetic_m-1-1_geometry"
    first, second = metadata["control"]["settings"], metadata["virtual_work"]["settings"]
    differences = {key: [first.get(key), second.get(key)] for key in first.keys() | second.keys()
                   if first.get(key) != second.get(key)}
    assert set(differences) == {"output", "observation_virtual_weight"}, differences
    assert first["warm_start"] == "results/mixed_pinn/synthetic_state_heads/checkpoint.pt"
    return data, source_frames, metadata, fields, differences


def main():
    data, source_frames, metadata, fields, differences = load_case()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    s, n = data["local_s_mm"], data["local_n_mm"]
    fit = data["observed_valid"] & ((abs(s) <= 4.) & (abs(n) >= 4.))
    holdout = data["observed_valid"] & ((abs(s) >= 8.) & (abs(n) >= 4.))
    blocks = np.floor(np.column_stack([s, n])/4).astype(int)
    weights = np.zeros(fit.shape)
    for k, mask in enumerate(fit):
        _, inverse, counts = np.unique(blocks[mask], axis=0, return_inverse=True, return_counts=True)
        weights[k, mask] = 1./counts[inverse]
        weights[k] /= 3.*weights[k].sum()
    stages = ["loaded_reference"] + ["rising_load"]*10 + ["maximum_load", "unloading", "low_load_unloading", "reloading"]
    pd.DataFrame({
        "state": data["frames"], "stage": stages,
        "source_frame": [int(x) if x >= 0 else None for x in source_frames],
        "force_N": data["forces"], "valid_points": data["observed_valid"].sum(1),
        "training_points": fit.sum(1), "holdout_points": holdout.sum(1),
        "load_origin": ["retained_experimental_load"]*12 + ["synthetic_half_maximum", "synthetic_15_percent_maximum", "synthetic_65_percent_maximum"],
        "displacement_origin": "known_material_simulation",
    }).to_csv(OUTPUT / "history_inputs.csv", index=False)
    pd.DataFrame({"point": np.arange(len(s)), "x_mm": data["points_mm"][:, 0],
                  "y_mm": data["points_mm"][:, 1], "z_mm": data["points_mm"][:, 2],
                  "s_mm": s, "n_mm": n, "training": fit[0], "holdout": holdout[0],
                  "weight_per_component": weights[0]}).to_csv(OUTPUT / "observation_points.csv", index=False)
    np.savez_compressed(OUTPUT / "observation_inputs.npz", points_mm=data["points_mm"],
                        local_s_mm=s, local_n_mm=n, observed_valid=data["observed_valid"],
                        training=fit, holdout=holdout, weight_per_component=weights,
                        source_frames=source_frames, forces_N=data["forces"],
                        observed_mm=data["observed"])
    truth = read_json(ROOT / "results/mixed_pinn/synthetic_state_heads_virtual/synthetic_truth.json")
    for role in ("control", "virtual_work"):
        other = read_json(ROOT / "results/mixed_pinn" / RUNS[role] / "synthetic_truth.json")
        assert truth == other
    summary = {
        "producer": "python code/prepare_paper_inputs.py",
        "source_specimen": "m-1-1", "run_roles": RUNS,
        "settings_differences": differences,
        "geometry": metadata["virtual_work"]["geometry"],
        "mesh": metadata["virtual_work"]["mesh"],
        "state_count": len(data["forces"]), "reference_state": 0,
        "source_reference_frame": int(source_frames[0]),
        "reference_force_N": float(data["forces"][0]),
        "sample_points": len(s), "training_points_by_state": fit.sum(1).tolist(),
        "holdout_points_by_state": holdout.sum(1).tolist(),
        "unscored_points_by_state": (data["observed_valid"] & ~(fit | holdout)).sum(1).tolist(),
        "training_weight_sum_over_all_components": (3*weights.sum(1)).tolist(),
        "known_band_parameters": {"young_MPa": 240., "shear_MPa": 100., "poisson": .2,
                                   "residual_MPa": .12, "friction": .16, "dilation": .035},
        "known_host_parameters": {"young_MPa": 13550., "poisson": .28},
        "nonlocal_parameters": {"radius_mm": truth["material"]["radius_mm"],
                                "mix": truth["material"]["nonlocal_mix"]},
        "learned_parameters": ["log_excess", "log_history_scale", "rate_logits", "weight_logits"],
        "truth_material": truth["material"],
        "input_reconciliation": "All three saved runs match reconstructed forces, state identities, registered points, synthetic observations and band-cell positions. The two continuation settings differ only in output location and observation virtual weight.",
    }
    write_json(OUTPUT / "input_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ["state_count", "sample_points", "source_reference_frame", "reference_force_N", "settings_differences"]}))


if __name__ == "__main__":
    main()
