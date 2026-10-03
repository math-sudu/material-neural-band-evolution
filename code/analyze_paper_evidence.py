"""Extract and reconcile the saved common-start virtual-work comparison.

No optimization is performed. The saved displacements, mixed stresses and
exported materials are used to re-evaluate the implemented losses and metrics.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from band_material import BandMaterial
from prepare_observations import rigid_matrix
from prepare_paper_inputs import ROOT, OUTPUT, RUNS, load_case, read_json, write_json
from run_mixed_pinn import Observations, make_core, window_metrics


def assert_values(actual, expected):
    # These tolerances compare repeated arithmetic on the same saved inputs;
    # they are not physical prediction-accuracy requirements.
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-9)


def compare_table(actual, source):
    saved = pd.read_csv(source)
    assert len(actual) == len(saved), source
    for key in saved:
        if pd.api.types.is_numeric_dtype(saved[key]):
            assert_values(actual[key], saved[key])
        else:
            assert actual[key].tolist() == saved[key].tolist(), (source, key)


def evaluate_fields(prediction, data, masks, *, alignment):
    rigid = rigid_matrix(data["points_mm"])
    residual = prediction.reshape(data["observed"].shape)-data["observed"]
    rows = []
    for k in range(1, len(residual)):
        fit = np.flatnonzero(np.repeat(masks["training"][k], 3))
        holdout = np.flatnonzero(np.repeat(masks["holdout"][k], 3))
        error = residual[k].ravel()
        if alignment == "training_region":
            weights = np.repeat(masks["weight_per_component"][k, masks["training"][k]], 3)
            amplitude = np.linalg.lstsq(np.sqrt(weights)[:, None]*rigid[fit],
                                        np.sqrt(weights)*error[fit], rcond=None)[0]
        else:
            index = np.r_[fit, holdout]
            amplitude = np.linalg.lstsq(rigid[index], error[index], rcond=None)[0]
        aligned = error-rigid@amplitude
        rms = lambda x: float(1000*np.sqrt(np.mean(x*x)))
        row = {"frame": k, "force_N": data["forces"][k], "alignment": alignment,
               "raw_rmse_um": rms(error[holdout]), "holdout_rmse_um": rms(aligned[holdout]),
               "fit_rmse_um": rms(aligned[fit])}
        for j, axis in enumerate("xyz"):
            row[f"holdout_{axis}_rmse_um"] = rms(aligned[holdout[j::3]])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    data, _, metadata, fields, _ = load_case()
    with np.load(OUTPUT / "observation_inputs.npz") as archive:
        masks = {key: archive[key] for key in archive.files}
    core = make_core(data, 10., 3.)
    np.testing.assert_allclose(core.centers, fields["common_start"]["cell_centers_mm"], rtol=0, atol=1e-12)
    assert core.model.mesh_info == metadata["virtual_work"]["mesh"]
    obs = Observations(core, data, "cpu")
    truth = read_json(ROOT / "results/mixed_pinn/synthetic_state_heads_virtual/synthetic_truth.json")
    truth_material = BandMaterial.from_export(truth["material"])
    reference = truth_material.elastic_matrix().detach()
    stiffness = torch.tensor(core.elastic_stiffness(reference.numpy()))
    factor = torch.linalg.cholesky(stiffness)
    force = torch.tensor(data["forces"])
    g = torch.tensor(core.g)
    energy = force[-1]**2*(g @ torch.cholesky_solve(g[:, None], factor)[:, 0])
    history = pd.read_csv(OUTPUT / "history_inputs.csv")
    all_metrics, all_windows, material_rows, summaries = [], [], [], {}
    display_fields = {"truth_mm": data["observed"], "points_mm": data["points_mm"],
                      "local_s_mm": data["local_s_mm"], "local_n_mm": data["local_n_mm"],
                      "forces_N": data["forces"], "valid": data["observed_valid"],
                      "training": masks["training"], "holdout": masks["holdout"]}
    curve_output = None
    for role in ["control", "virtual_work"]:
        directory = ROOT / "results/mixed_pinn" / RUNS[role]
        with np.load(directory / "identification_replay_fields.npz") as archive:
            replay = {key: archive[key] for key in archive.files}
        for key in ["frames", "observed_mm", "points_mm", "cell_centers_mm"]:
            np.testing.assert_array_equal(replay[key], fields[role][key])
        np.testing.assert_array_equal(replay["force_N"], data["forces"])
        replay_record = read_json(directory / "identification_replay.json")
        assert replay_record["field_networks_used"] is False
        material = BandMaterial.from_export(read_json(directory / "material.json"))
        band = core.tensors(material)
        with torch.no_grad():
            u = torch.tensor(fields[role]["u_mm"])
            sigma = torch.tensor(fields[role]["stress_MPa"])
            material_stress, kappa, diagnostics = material.history(band.strain(u), band.average, 20)
            assert_values(kappa.numpy(), fields[role]["kappa"])
            residual = band.residual(u, sigma, force)
            weak = (residual*torch.cholesky_solve(residual.T, factor).T).sum()/energy/len(force)
            difference = sigma-material_stress
            constitutive = torch.einsum("tei,ij,tej,e->", difference,
                                        torch.linalg.inv(reference), difference, band.volumes)/energy/len(force)
            predicted = obs.predict(u, force)
            assert_values(predicted.numpy(), fields[role]["predicted_mm"])
            observation = obs.loss(predicted, .002)
            material_residual = band.residual(u, material_stress, force)
            defect = torch.cholesky_solve(material_residual.T, factor).T
            virtual = obs.residual_loss(obs.predict(defect, torch.zeros_like(force)), .002)
            assert_values([float(observation), float(weak), float(constitutive)], metadata[role]["loss_terms"])
            if role == "virtual_work":
                assert_values(float(virtual), metadata[role]["observation_virtual_loss"])
        for field_name, values, alignment, source in [
            ("identified", fields[role]["predicted_mm"], "training_region", "training"),
            ("frozen", replay["predicted_mm"], "fit_and_holdout_quotient", "identification_replay")]:
            metrics = evaluate_fields(values, data, masks, alignment=alignment)
            compare_table(metrics, directory / f"{source}_metrics.csv")
            if field_name == "frozen":
                frozen_metrics = metrics.copy()
            metrics = metrics.rename(columns={"frame": "state"})
            metrics["run"], metrics["field"] = role, field_name
            all_metrics.append(metrics.merge(history[["state", "stage"]], on="state"))
            windows = pd.DataFrame(window_metrics(data, values, core.Q))
            compare_table(windows, directory / f"{source}_windows.csv")
            windows = windows.rename(columns={"frame": "state"})
            windows["run"], windows["field"] = role, field_name
            all_windows.append(windows.merge(history[["state", "stage"]], on="state"))
            display_fields[f"{role}_{field_name}_mm"] = values.reshape(data["observed"].shape)
        valid = data["observed_valid"].copy()
        valid[0] = False
        gap = (replay["predicted_mm"]-fields[role]["predicted_mm"]).reshape(data["observed"].shape)
        pooled_gap = float(1000*np.sqrt(np.mean(gap[valid]**2)))
        assert_values(pooled_gap, replay_record["identification_field_gap_rmse_um"])
        quotient = float(frozen_metrics.holdout_rmse_um.mean())
        raw = float(frozen_metrics.raw_rmse_um.mean())
        assert_values(quotient, replay_record["mean_holdout_quotient_rmse_um"])
        assert_values(raw, replay_record["mean_raw_rmse_um"])
        curves = pd.read_csv(directory / "cohesion_curves.csv")
        chi = torch.tensor(curves.history.to_numpy())
        with torch.no_grad():
            assert_values(curves.true_cohesion_MPa, truth_material.cohesion(chi).numpy())
            assert_values(curves.identified_cohesion_MPa, material.cohesion(chi).numpy())
        if curve_output is None:
            curve_output = curves[["history", "true_cohesion_MPa"]].rename(columns={"history": "chi"})
        else:
            np.testing.assert_array_equal(curve_output.chi, curves.history)
            np.testing.assert_array_equal(curve_output.true_cohesion_MPa, curves.true_cohesion_MPa)
        curve_output[f"{role}_cohesion_MPa"] = curves.identified_cohesion_MPa
        errors = curves.identified_cohesion_MPa-curves.true_cohesion_MPa
        summaries[role] = {
            "curve_max_absolute_error_MPa": float(errors.abs().max()),
            "curve_max_error_at_chi": float(curves.history.iloc[int(errors.abs().argmax())]),
            "initial_cohesion_MPa": float(material.cohesion(torch.tensor(0.)).detach()),
            "upper_drive_cohesion_MPa": float(curves.identified_cohesion_MPa.iloc[-1]),
            "frozen_holdout_quotient_mean_rmse_um": quotient,
            "frozen_holdout_raw_mean_rmse_um": raw,
            "identification_frozen_all_valid_pooled_rmse_um": pooled_gap,
            "observation_loss": float(observation), "weak_loss": float(weak),
            "constitutive_loss": float(constitutive), "virtual_loss": float(virtual),
        }
        for name, value in summaries[role].items():
            material_rows.append({"run": role, "quantity": name, "value": value})
    curve_output.to_csv(OUTPUT / "cohesion_comparison.csv", index=False, lineterminator="\n")
    pd.concat(all_metrics, ignore_index=True).to_csv(OUTPUT / "response_metrics.csv", index=False, lineterminator="\n")
    pd.concat(all_windows, ignore_index=True).to_csv(OUTPUT / "window_histories.csv", index=False, lineterminator="\n")
    pd.DataFrame(material_rows).to_csv(OUTPUT / "comparison_summary.csv", index=False, lineterminator="\n")
    np.savez_compressed(OUTPUT / "response_fields.npz", **display_fields)
    improvements = {key: 100*(1-summaries["virtual_work"][key]/summaries["control"][key])
                    for key in ["curve_max_absolute_error_MPa", "frozen_holdout_quotient_mean_rmse_um",
                                "frozen_holdout_raw_mean_rmse_um", "identification_frozen_all_valid_pooled_rmse_um"]}
    report = {"producer": "python code/analyze_paper_evidence.py", "runs": summaries,
              "reductions_percent": improvements, "curve_samples": len(curve_output),
              "curve_interval": [float(curve_output.chi.min()), float(curve_output.chi.max())],
              "evaluated_nonreference_states": 14, "window_stations_mm": [-8., 0., 8.],
              "representative_field_states": {"maximum_load": 11, "low_load_unloading": 13, "reloading": 14},
              "reference_energy_N_mm": float(energy),
              "physics_normalization": "common final-load elastic energy, as implemented when the saved runs were produced",
              "source_reconciliation": "All saved metric and window rows, sampled cohesion values, material histories, training losses and reported aggregate response metrics reproduced from their saved inputs.",
              "scope": "No training or nonlinear frozen-equilibrium rerun; material integration at saved identified displacements only."}
    write_json(OUTPUT / "evidence_summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
