"""Separate material-recovery error from the synthetic discretization mismatch.

The truth is replayed on the identification discretization with no fitting.
This is a numerical diagnosis of the PINN's known-material experiment.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from run_mixed_pinn import ROOT, specimen, make_core, Observations, window_metrics
from band_material import BandMaterial, averaging_matrix
from condensed_band import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    output = ROOT / args.result
    record = json.loads((output / "training.json").read_text(encoding="utf-8"))
    truth = json.loads((output / "synthetic_truth.json").read_text(encoding="utf-8"))
    settings = record["settings"]
    data = specimen("m-1-1", settings["frames"], settings["end_fraction"],
                    settings.get("surface_directory", "data/processed/surface_history"),
                    settings.get("observation_layout", "central"))
    with np.load(output / "training_fields.npz") as saved:
        data["observed"] = saved["observed_mm"].copy()
        data["frames"] = saved["frames"].copy()
        data["forces"] = saved["forces_N"].copy()
    added = len(data["forces"])-len(data["observed_valid"])
    data["observed_valid"] = np.concatenate((data["observed_valid"],
                                              np.repeat(data["observed_valid"][-1:], added, axis=0)))
    core = make_core(data, settings["mesh_size"], settings["band_size"])
    material = BandMaterial.from_export(truth["material"])
    u, stress, kappa, diag = replay(core, material, data["forces"])
    observer = Observations(core, data, "cpu")
    with torch.no_grad():
        predicted = observer.predict(torch.as_tensor(u), torch.as_tensor(data["forces"]))
        metrics = observer.metrics(predicted, data["frames"], data["forces"],
                                   alignment="fit_and_holdout_quotient")
    pd.DataFrame(metrics).to_csv(output / "known_material_metrics.csv", index=False)
    pd.DataFrame(window_metrics(data, predicted.numpy(), core.Q)).to_csv(
        output / "known_material_windows.csv", index=False)
    np.savez_compressed(output / "known_material_fields.npz", predicted_mm=predicted.numpy(),
                        u_mm=u, stress_MPa=stress, kappa=kappa)
    learned = BandMaterial.from_export(json.loads((output / "material.json").read_text()))
    average = averaging_matrix(core.centers, core.volumes, material.radius_mm)
    driver = material.nonlocal_mix*(kappa @ average.T)+(1.-material.nonlocal_mix)*kappa
    history = torch.linspace(float(driver.min()), float(driver.max()), 101)
    diagnostic_points = torch.tensor([0., truth["maximum_kappa"], float(driver.max())])
    with torch.no_grad():
        known_curve, fitted_curve = material.cohesion(history), learned.cohesion(history)
        known_points, fitted_points = material.cohesion(diagnostic_points), learned.cohesion(diagnostic_points)
    pd.DataFrame({"history": history.numpy(), "true_cohesion_MPa": known_curve.numpy(),
                  "identified_cohesion_MPa": fitted_curve.numpy()}).to_csv(
        output / "cohesion_curves.csv", index=False)
    result = {
        "role": "Known material replay on the identification mesh, without fitting; estimates discretization contribution to observation error",
        "curve_coordinate": "Cohesion-network input chi over the range actually visited by the known material on the identification mesh",
        "truth_mesh": truth["mesh"], "identification_mesh": core.model.mesh_info,
        "mean_holdout_quotient_rmse_um": float(np.mean([r["holdout_rmse_um"] for r in metrics])),
        "true_initial_cohesion_MPa": float(known_points[0]),
        "identified_initial_cohesion_MPa": float(fitted_points[0]),
        "true_cohesion_at_truth_maximum_kappa_MPa": float(known_points[1]),
        "identified_cohesion_at_truth_maximum_kappa_MPa": float(fitted_points[1]),
        "known_material_minimum_driver": float(driver.min()),
        "known_material_maximum_driver": float(driver.max()),
        "true_cohesion_at_known_maximum_driver_MPa": float(known_points[2]),
        "identified_cohesion_at_known_maximum_driver_MPa": float(fitted_points[2]),
        "truth_maximum_kappa": truth["maximum_kappa"],
        "same_material_identification_mesh_maximum_kappa": float(kappa.max()),
        "steps": diag,
    }
    (output / "known_material_replay.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print({k:v for k,v in result.items() if k not in {"steps", "truth_mesh", "identification_mesh"}}, flush=True)


if __name__ == "__main__":
    main()
