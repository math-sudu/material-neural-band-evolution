"""Train a mixed interpolated VPINN and replay its exported band material.

The displacement MLP is evaluated on band vertices; a P1 interpolant supplies
its strain and shared interface trace.  A second MLP supplies independent stress
corrections to an elastic lifting.  These fields enter weak equilibrium and
constitutive consistency directly.  Replay discards both field networks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from band_forward import BandForward, BandGeometry
from band_material import BandMaterial
from process_material import ProcessMaterial, elastic_stress
from condensed_band import CondensedBand, EquilibriumProjection, ElasticReferenceSolve, replay, save_accepted_state
from prepare_observations import rigid_matrix

ROOT = Path(__file__).resolve().parents[1]


def specimen(sid, frame_count=12, end_fraction=1.,
             surface_directory="data/processed/surface_history", observation_layout="central"):
    """Load the published geometry, locations and rising-load inputs."""
    if sid != "m-1-1" or frame_count != 12 or end_fraction != 1.:
        raise ValueError("The published case uses m-1-1 and twelve rising-load states")
    if observation_layout != "central":
        raise ValueError("The published comparison uses the central observation layout")
    case = json.loads((ROOT / "inputs/case.json").read_text(encoding="utf-8"))
    points = np.asarray(case["points_mm"])
    frames = np.asarray(case["source_frames"], dtype=int)
    return dict(
        sid=sid, observation_layout=observation_layout,
        points_mm=points, local_s_mm=np.asarray(case["local_s_mm"]),
        local_n_mm=np.asarray(case["local_n_mm"]),
        frames=frames, forces=np.asarray(case["forces_N"]),
        reference_frame=int(frames[0]),
        observed=np.zeros((len(frames), len(points), 3)),
        observed_valid=np.asarray(case["valid"], dtype=bool),
        geometry=BandGeometry(**case["geometry"]),
    )


def make_core(data, mesh_size=12., band_size=4., process_radius_mm=0.):
    return CondensedBand(BandForward(data["geometry"], mesh_size_mm=mesh_size,
                                     band_mesh_size_mm=band_size), process_radius_mm)


class MLP(nn.Module):
    def __init__(self, outputs, width=48, inputs=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width),
                                 nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class MixedFields(nn.Module):
    def __init__(self, core, band, forces, path, C_ref, width=48, fit_host=False,
                 initial_host_scale=1., constitutive_stress=False, state_heads=False,
                 local_features=False):
        super().__init__()
        self.state_heads = state_heads
        self.state_count = len(forces)
        self.local_features = local_features
        outputs = self.state_count if state_heads else 1
        base_inputs = 3 if state_heads else 4
        self.displacement = MLP(3*outputs, width, inputs=base_inputs+3*local_features)
        self.stress = (None if constitutive_stress else
                       MLP(6*outputs, width, inputs=base_inputs+4*local_features))
        self.host_log_scale = nn.Parameter(torch.tensor(math.log(initial_host_scale)), requires_grad=fit_host)
        device, dtype = forces.device, forces.dtype
        tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        center = np.asarray(core.model.geometry.center_mm)
        scales = np.array([core.model.geometry.band_length_mm/2+core.process_radius_mm,
                           max(1.5, core.process_radius_mm), 25.])
        def inputs(points, material_mask=None):
            local = (points-center) @ core.Q.T
            xyz = tensor(local / scales)
            original = torch.cat((xyz[None].expand(len(path), -1, -1),
                                  path[:, None, None].expand(-1, len(points), -1)), -1)
            if not local_features:
                return original
            # Geometry-scale features retain the thin-band transition when the
            # nonlinear domain grows. P1 nodal sharing still enforces continuity.
            normal = np.tanh(local[:, 1]/(core.model.geometry.band_width_mm/2))
            extra = np.column_stack((local[:, 0]/(core.model.geometry.band_length_mm/2),
                                     normal, np.abs(normal)))
            if material_mask is not None:
                # Only the element stress trial sees a material indicator;
                # displacement remains single-valued at shared interface nodes.
                extra = np.column_stack((extra, material_mask.astype(float)))
            return torch.cat((original, tensor(extra)[None].expand(len(path), -1, -1)), -1)
        self.register_buffer("nodes_input", inputs(core.model.mesh.p.T[core.nodes]))
        self.register_buffer("centers_input", inputs(core.centers, core.is_band))
        self.register_buffer("free_node_dofs", torch.as_tensor(core.free_node_dofs, device=device), persistent=False)
        stiffness = core.elastic_stiffness(C_ref.detach().cpu().numpy(), initial_host_scale)
        unit = np.linalg.solve(stiffness, core.g)
        # Fixed elastic lifting only initializes the neural trial.  It is never
        # re-solved as a hidden inverse forward pass when material parameters move.
        self.register_buffer("lifting", forces[:, None]*tensor(unit))
        # A loaded state can retain plastic displacement after unloading to zero.
        # State heads use the attained load scale, not the instantaneous load.
        amplitude = torch.cummax(forces.abs(), dim=0).values if state_heads else forces
        self.register_buffer("amplitude", amplitude / forces.abs().max())
        self.displacement_scale_mm = .1
        self.stress_scale_MPa = 10.

    def forward(self):
        amp = self.amplitude[:, None, None]
        def evaluate(network, inputs, components):
            if self.state_heads:
                spatial = torch.cat((inputs[0, :, :3], inputs[0, :, 4:]), -1)
                return network(spatial).reshape(-1, self.state_count, components).transpose(0, 1)
            return network(inputs)
        correction = (amp*evaluate(self.displacement, self.nodes_input, 3)*self.displacement_scale_mm).flatten(1)
        u = self.lifting + correction[:, self.free_node_dofs]
        stress_correction = (None if self.stress is None else
                             amp*evaluate(self.stress, self.centers_input, 6)*self.stress_scale_MPa)
        return u, stress_correction

    def warm_start(self, saved):
        """Preserve the trial when adding features or selecting a load prefix."""
        old_count = len(saved["lifting"])
        prefix_scale = None
        if old_count != self.state_count:
            if not self.state_heads or self.state_count > old_count:
                raise ValueError("Changed history length requires a prefix of saved state heads")
            torch.testing.assert_close(self.lifting, saved["lifting"][:self.state_count])
            active = self.amplitude != 0.
            prefix_scale = torch.ones_like(self.amplitude)
            prefix_scale[active] = saved["amplitude"][:self.state_count][active]/self.amplitude[active]
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "host_log_scale" and not parameter.requires_grad:
                    continue
                source = saved[name]
                if prefix_scale is not None and name in {
                        "displacement.net.6.weight", "displacement.net.6.bias",
                        "stress.net.6.weight", "stress.net.6.bias"}:
                    components = 3 if name.startswith("displacement") else 6
                    heads = source.reshape(old_count, components, *source.shape[1:])[:self.state_count]
                    source = (heads*prefix_scale.reshape((-1,)+(1,)*(heads.ndim-1))).reshape_as(parameter)
                if (self.local_features and name in {"displacement.net.0.weight", "stress.net.0.weight"}
                        and source.shape[1] == (3 if self.state_heads else 4)):
                    parameter.zero_()
                    parameter[:, :source.shape[1]].copy_(source)
                else:
                    parameter.copy_(source)


class Observations:
    def __init__(self, core, data, device):
        dtype = torch.get_default_dtype()
        tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        surface = core.model.observation_operator(data["points_mm"])
        H, h = core.observer(surface)
        self.H, self.h = tensor(H), tensor(h)
        self.observed = tensor(data["observed"].reshape(len(data["forces"]), -1))
        self.rigid = tensor(rigid_matrix(data["points_mm"]))
        self.conditioning_rigid = self.rigid
        self.train, self.holdout, self.weights, self.aligners = [], [], [], []
        local_s, local_n = data["local_s_mm"], data["local_n_mm"]
        region = np.abs(local_n) >= 4.
        fit_region = region & (np.abs(local_s) <= 4.)
        test_region = region & (np.abs(local_s) >= 8.)
        if data.get("observation_layout", "central") in {"central_and_outer", "central_and_outer_deformation"}:
            fit_region |= np.abs(local_n) >= 13.
            test_region &= np.abs(local_n) < 10.
            if data["observation_layout"] == "central_and_outer_deformation":
                # Central motion fixes the common rigid alignment. Each outer
                # block contributes only its own deformation, using the same
                # points and weights as central_and_outer.
                outer = np.abs(local_n) >= 13.
                positive = tensor(np.repeat(outer & (local_n > 0.), 3))[:, None]
                negative = tensor(np.repeat(outer & (local_n < 0.), 3))[:, None]
                self.conditioning_rigid = torch.cat((self.rigid, positive*self.rigid,
                                                     negative*self.rigid), -1)
        elif data.get("observation_layout") in {"outer_host", "outer_host_deformation"}:
            fit_region = np.abs(local_n) >= 13.
            test_region = region & (np.abs(local_n) < 10.)
            if data["observation_layout"] == "outer_host_deformation":
                # Relative rigid motion between rock blocks also contains the
                # band response. Condition the host on each side's deformation.
                side = tensor(np.repeat(local_n > 0., 3))[:, None]
                self.conditioning_rigid = torch.cat((side*self.rigid, (1.-side)*self.rigid), -1)
        # Equal contribution from occupied 4-mm spatial blocks, not from a
        # presumed independent sample count at the 0.25-mm DIC grid pitch.
        blocks = np.floor(np.column_stack((local_s, local_n))/4).astype(int)
        for valid in data["observed_valid"]:
            fit = valid & fit_region
            test = valid & test_region
            fit_index = np.flatnonzero(np.repeat(fit, 3))
            self.train.append(torch.as_tensor(fit_index, device=device))
            self.holdout.append(torch.as_tensor(np.flatnonzero(np.repeat(test, 3)), device=device))
            if fit.sum() < 6 or test.sum() < 2:
                raise ValueError("Insufficient valid observations in the buffered spatial split")
            _, inverse, counts = np.unique(blocks[fit], axis=0, return_inverse=True, return_counts=True)
            weights = np.repeat(1./counts[inverse], 3)
            weights = weights/weights.sum()
            self.weights.append(tensor(weights))
            rigid = self.conditioning_rigid[fit_index]
            weighted = rigid*torch.sqrt(tensor(weights))[:, None]
            self.aligners.append(torch.linalg.pinv(weighted)*torch.sqrt(tensor(weights))[None])
        self.projection_max_mm = float(surface.projection_distance_mm.max())

    def predict(self, u, forces, host_scale=1.):
        total = u @ self.H.T + forces[:, None]*self.h / host_scale
        return total-total[:1]

    def loss(self, predicted, scale_mm):
        return self.residual_loss(predicted-self.observed, scale_mm)

    def residual_loss(self, residuals, scale_mm):
        """Training-only weighted quotient norm, also used for virtual work."""
        values = []
        for k in range(1, len(residuals)):
            index = self.train[k]
            residual = residuals[k]
            residual = residual-self.conditioning_rigid @ (self.aligners[k] @ residual[index])
            values.append((residual[index].square()*self.weights[k]).sum()/scale_mm**2)
        return torch.stack(values).mean()

    def metrics(self, predicted, frames, forces, *, alignment="training_region"):
        rows = []
        for k in range(1, len(predicted)):
            residual = predicted[k]-self.observed[k]
            rigid = self.rigid
            alignment_label = alignment
            if alignment == "training_region":
                nuisance = self.aligners[k] @ residual[self.train[k]]
                rigid = self.conditioning_rigid
                if rigid.shape[1] == 12:
                    alignment_label = "training_region_separate_side_rigid"
                elif rigid.shape[1] == 18:
                    alignment_label = "central_rigid_and_outer_side_deformation"
            elif alignment == "fit_and_holdout_quotient":
                index = torch.cat((self.train[k], self.holdout[k]))
                nuisance = torch.linalg.lstsq(self.rigid[index], residual[index]).solution
            else:
                nuisance = torch.zeros(6, dtype=residual.dtype, device=residual.device)
            aligned = residual-rigid@nuisance
            row = {"frame": int(frames[k]), "force_N": float(forces[k]), "alignment": alignment_label,
                   "raw_rmse_um": 1000*float(residual[self.holdout[k]].square().mean().sqrt()),
                   "holdout_rmse_um": 1000*float(aligned[self.holdout[k]].square().mean().sqrt()),
                   "fit_rmse_um": 1000*float(aligned[self.train[k]].square().mean().sqrt())}
            for j, label in enumerate("xyz"):
                row[f"holdout_{label}_rmse_um"] = 1000*float(aligned[self.holdout[k][j::3]].square().mean().sqrt())
            rows.append(row)
        return rows


def window_metrics(data, predicted, Q):
    """Three-dimensional affine offsets on the identical sampled point set.

    The affine basis contains x,y,z and annihilates all rigid modes.  These
    offsets are distinct from the source paper's side means and the earlier
    two-dimensional affine gauges.  They mix band and surrounding-rock motion.
    """
    prediction = np.asarray(predicted).reshape(data["observed"].shape)
    points = data["points_mm"]
    coordinates = points-points.mean(axis=0)
    axes = Q.copy()
    axes[1] *= -1  # Align with the sign of the recorded local_n_mm coordinate.
    rows = []
    for k in range(1, len(prediction)):
        for station in [-8., 0., 8.]:
            keep = (data["observed_valid"][k] & (np.abs(data["local_s_mm"]-station)<4.)
                    & (np.abs(data["local_n_mm"])>=4.) & (np.abs(data["local_n_mm"])<11.))
            X = np.column_stack((np.ones(keep.sum()), coordinates[keep],
                                 .5*np.sign(data["local_n_mm"][keep])))
            if min(np.sum(keep & (data["local_n_mm"]>0)),
                   np.sum(keep & (data["local_n_mm"]<0))) < 5:
                continue
            functional = np.linalg.pinv(X)[-1]
            observed = (functional @ data["observed"][k, keep]) @ axes.T
            estimated = (functional @ prediction[k, keep]) @ axes.T
            row = {"frame": int(data["frames"][k]), "force_N": float(data["forces"][k]),
                   "station_mm": station, "points": int(keep.sum())}
            for j, name in enumerate(["tangential", "normal", "depth"]):
                row[f"observed_{name}_um"] = 1000*float(observed[j])
                row[f"predicted_{name}_um"] = 1000*float(estimated[j])
            rows.append(row)
    return rows


def physics_energy_scales(forces, unit_compliance, normalization):
    """Fixed load-work scales; unloading retains the attained loading scale."""
    if normalization == "final_load":
        amplitude = forces[-1].abs().expand_as(forces)
    elif normalization == "attained_load":
        amplitude = torch.cummax(forces.abs(), dim=0).values
        nonzero = amplitude[amplitude > 0.]
        if not len(nonzero):
            raise ValueError("Physics normalization requires a nonzero applied load")
        # A virgin zero state uses the first loaded state as its unit. Later
        # zero-force unloading states retain their own attained load scale.
        amplitude = torch.where(amplitude > 0., amplitude, nonzero[0])
    else:
        raise ValueError(f"Unsupported physics normalization: {normalization}")
    if torch.any(amplitude == 0.):
        raise ValueError("Final-load normalization requires a nonzero final load")
    return (amplitude.square()*unit_compliance).detach()


def train(core, data, material, out, args, initial_host_scale=1.):
    device = args.device
    material.to(device)
    if args.freeze_band:
        (material.band if isinstance(material, ProcessMaterial) else material).requires_grad_(False)
    band = core.tensors(material, device)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)
    forces = tensor(data["forces"])
    frame = np.asarray(data["frames"])
    path = tensor((frame-frame[0])/max(float(frame[-1]-frame[0]), 1.))
    C_ref = material.elastic_matrix(initial_host_scale).detach()
    stiffness = tensor(core.elastic_stiffness(C_ref.cpu().numpy(), initial_host_scale))
    factor = torch.linalg.cholesky(stiffness)
    compliance = torch.linalg.inv(C_ref)
    stress_factor = torch.linalg.cholesky(compliance)
    unit_reference = torch.cholesky_solve(band.g[:, None], factor)[:, 0]
    energy_scale = physics_energy_scales(forces, band.g @ unit_reference,
                                        args.physics_normalization)
    fields = MixedFields(core, band, forces, path, C_ref, args.width, args.fit_host,
                         initial_host_scale, args.constitutive_stress, args.state_heads,
                         args.local_features).to(device)
    projection = (EquilibriumProjection(core, band, C_ref, initial_host_scale)
                  if args.equilibrated_fields else None)
    if args.eliminate_stress and (projection is None or projection.has_exterior):
        raise ValueError("Stress elimination requires projected fields on the full nonlinear domain")
    if args.elastic_reference and not args.eliminate_stress:
        raise ValueError("Elastic reference anchoring requires full-domain stress elimination")
    lifting_stress = elastic_stress(band.strain(fields.lifting), C_ref)
    obs = Observations(core, data, device)
    optimizer = torch.optim.Adam([{"params": fields.parameters(), "lr": args.learning_rate},
                                  {"params": material.parameters(), "lr": args.learning_rate*.2}])
    start = time.perf_counter()
    history = []
    constraint_multipliers = None
    constraint_history = []
    def objective(physics_weight):
        u, stress_correction = fields()
        host_scale = fields.host_log_scale.exp()
        if projection is not None:
            # Independent stress trial: using C_ref*B*u here would cancel the
            # displacement network algebraically under this projection.
            u, stress = projection(u, lifting_stress+stress_correction, forces, host_scale)
        if args.elastic_reference:
            loaded_reference = ElasticReferenceSolve.apply(
                material.elastic_matrix(host_scale), host_scale, forces[0], core)
            # DIC measures these increments. Anchor the unobserved total offset
            # without changing the trial's relative displacement at any state.
            u = u-u[:1]+loaded_reference[None]
        strains = band.strain(u)
        mat_stress, kappa, diagnostics = material.history(strains, band.average, args.return_iterations, host_scale)
        if args.elastic_reference:
            reference_plasticity = float(kappa[0, core.is_band].detach().max())
            reference_damage = float(diagnostics["damage"][0].detach().max())
            if reference_plasticity > 0. or reference_damage > 0.:
                raise RuntimeError("Loaded reference left the elastic branch: "
                                   f"band plastic shear={reference_plasticity:g}, host damage={reference_damage:g}. "
                                   "A nonlinear loaded-reference solve is required for this material.")
        if args.eliminate_stress:
            # Keep the same projected displacement trial and its warm start.
            # Eliminate only the independent admissible stress at fixed u.
            stress = projection.optimal_stress(mat_stress, forces)
        if projection is None:
            stress = (mat_stress if stress_correction is None else
                      elastic_stress(strains, material.elastic_matrix(host_scale)) + stress_correction)
        residual = band.residual(u, stress, forces, host_scale)
        weak_vector = torch.linalg.solve_triangular(factor, residual.T, upper=False).T
        weak_vector = weak_vector/torch.sqrt(energy_scale[:, None]*len(forces))
        weak = weak_vector.square().sum()
        constitutive_difference = (torch.einsum("tei,eij->tej", stress-mat_stress, stress_factor)
                                   if stress_factor.ndim == 3 else (stress-mat_stress)@stress_factor)
        constitutive_vector = constitutive_difference*torch.sqrt(
            band.volumes[None, :, None]/(energy_scale[:, None, None]*len(forces)))
        constitutive = constitutive_vector.square().sum()
        constraint = torch.cat((weak_vector.flatten(), constitutive_vector.flatten()))
        predicted = obs.predict(u, forces, host_scale)
        observation = obs.loss(predicted, args.observation_scale_mm)
        loss = observation + physics_weight*(weak+constitutive)
        if args.observation_virtual_weight:
            # These are linear combinations of the existing nodal tests,
            # weighted in measured displacement directions. They introduce
            # neither hidden observations nor a nonlinear forward solve.
            material_residual = band.residual(u, mat_stress, forces, host_scale)
            defect = torch.cholesky_solve(material_residual.T, factor).T
            visible_defect = obs.predict(defect, torch.zeros_like(forces), host_scale)
            virtual = obs.residual_loss(visible_defect, args.observation_scale_mm)
            loss = loss + args.observation_virtual_weight*virtual
            diagnostics["observation_virtual_loss"] = virtual
        if constraint_multipliers is not None:
            loss = loss + constraint_multipliers @ constraint
        return loss, (observation, weak, constitutive), (u, stress, kappa, diagnostics, predicted), constraint
    checkpoint = out / "checkpoint.pt"
    def save_checkpoint(**details):
        torch.save({"fields": fields.state_dict(), "material": material.state_dict(),
                    "history": history, "constraint_multipliers": constraint_multipliers,
                    "constraint_history": constraint_history,
                    "material_substeps": material.integration_substeps,
                    "physics_normalization": args.physics_normalization, **details}, checkpoint)
    if args.warm_start:
        saved = torch.load(ROOT / args.warm_start, map_location=device, weights_only=True)
        if args.material_substeps is None and not args.warm_start_fields_only:
            material.integration_substeps = saved.get("material_substeps", material.integration_substeps)
        fields.warm_start(saved["fields"])
        if not args.warm_start_fields_only:
            material.load_state_dict(saved["material"])
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        saved_substeps = saved.get("material_substeps", 1)
        if args.material_substeps is not None and args.material_substeps != saved_substeps:
            raise ValueError("Changed material integration requires --warm-start to reset history constraints")
        material.integration_substeps = saved_substeps
        if (saved.get("constraint_multipliers") is not None and
                saved.get("physics_normalization", "final_load") != args.physics_normalization):
            raise ValueError("Changed constraint units require --warm-start, which resets the multipliers")
        fields.load_state_dict(saved["fields"])
        material.load_state_dict(saved["material"])
        history = saved["history"]
        constraint_multipliers = saved.get("constraint_multipliers")
        constraint_history = saved.get("constraint_history", [])
    args.material_substeps = material.integration_substeps
    if args.probe_objective:
        # A stopped line search can reflect a constitutive event discontinuity,
        # not a stationary solution. Probe the actual saved objective in its
        # descent direction; no parameter or training output is overwritten.
        loss, _, (_, _, _, diag, _), _ = objective(args.physics_weight)
        loss.backward()
        parameters = [p for p in list(fields.parameters())+list(material.parameters())
                      if p.requires_grad and p.grad is not None]
        gradient_scale = max(float(p.grad.abs().max()) for p in parameters)
        directions = [-p.grad.detach()/gradient_scale for p in parameters]
        originals = [p.detach().clone() for p in parameters]
        derivative = sum(float((p.grad*d).sum()) for p, d in zip(parameters, directions))
        def birth(diagnostics):
            damaged = diagnostics["damage"] > 0.
            return torch.where(damaged.any(0), damaged.to(torch.int64).argmax(0), -1)
        baseline_birth = birth(diag)
        normal = diag.get("crack_normal")
        rows = []
        with torch.no_grad():
            for step in [-1e-4, -1e-6, -1e-8, 0., 1e-8, 1e-6, 1e-4]:
                for p, value, direction in zip(parameters, originals, directions):
                    p.copy_(value+step*direction)
                value, _, (_, _, _, perturbed, _), _ = objective(args.physics_weight)
                row = {"step": step, "objective": float(value),
                       "linear_prediction": float(loss.detach())+step*derivative,
                       "first_damage_step_changes": int((birth(perturbed) != baseline_birth).sum())}
                if normal is not None:
                    dots = (normal[-1]*perturbed["crack_normal"][-1]).sum(-1).abs()
                    active = (normal[-1].square().sum(-1) > 0.) & (perturbed["crack_normal"][-1].square().sum(-1) > 0.)
                    row["max_crack_rotation_deg"] = float(torch.rad2deg(torch.acos(dots[active].clamp(0., 1.))).max())
                if "crack_fabric" in diag:
                    row["max_crack_fabric_change"] = float(torch.linalg.matrix_norm(
                        perturbed["crack_fabric"][-1]-diag["crack_fabric"][-1]).max())
                rows.append(row)
            for p, value in zip(parameters, originals):
                p.copy_(value)
        record = {"objective": float(loss.detach()), "gradient_max_abs": gradient_scale,
                  "directional_derivative": derivative, "rows": rows,
                  "host_crack_orientation": material.export().get("host_crack_orientation")}
        (out / args.probe_objective).write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
        return None
    if args.material_block_steps:
        field_gradients = {name: p.requires_grad for name, p in fields.named_parameters()}
        for p in fields.parameters():
            p.requires_grad_(False)
        block = [p for p in material.parameters() if p.requires_grad]
        if args.fit_host:
            fields.host_log_scale.requires_grad_(True)
            block.append(fields.host_log_scale)
        material_optimizer = torch.optim.LBFGS(block, max_iter=args.material_block_steps,
                                                line_search_fn="strong_wolfe", tolerance_change=1e-11)
        def material_closure():
            material_optimizer.zero_grad()
            loss = objective(args.physics_weight)[0]
            loss.backward()
            return loss
        material_optimizer.step(material_closure)
        for name, p in fields.named_parameters():
            p.requires_grad_(field_gradients[name])
        print("material_block", {name: float(p.detach().exp().cpu())
                                  for name, p in material.named_parameters() if name.startswith("log_")}, flush=True)
    offset = len(history)
    for step in range(args.steps):
        optimizer.zero_grad()
        loss, terms, _, _ = objective(args.physics_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite mixed PINN loss")
        loss.backward()
        optimizer.step()
        history.append([offset+step, *[float(v.detach()) for v in terms]])
        if step % 100 == 0 or step == args.steps-1:
            print(json.dumps({"step": offset+step, "observation": history[-1][1],
                              "weak": history[-1][2], "constitutive": history[-1][3],
                              "elapsed_s": round(time.perf_counter()-start, 1)}), flush=True)
            save_checkpoint()
    lbfgs_history = []
    if args.lbfgs_steps:
        parameters = list(fields.parameters()) + list(material.parameters())
        for cycle in range(args.constraint_updates+1):
            lbfgs = torch.optim.LBFGS(parameters, max_iter=args.lbfgs_steps, history_size=30,
                                      line_search_fn="strong_wolfe", tolerance_grad=1e-8,
                                      tolerance_change=1e-11)
            calls = [0]
            def closure():
                lbfgs.zero_grad()
                loss, terms, _, _ = objective(args.physics_weight)
                loss.backward()
                calls[0] += 1
                # A difficult constitutive trial can fail during a later line
                # search. Preserve the last fully evaluated field and material
                # so that repairing its solver does not discard completed work.
                save_checkpoint(checkpoint_role="successful_objective_evaluation",
                                lbfgs_cycle=cycle, lbfgs_evaluations=calls[0])
                interval = (5 if isinstance(material, ProcessMaterial)
                            and material.host_crack_coupling == "shared_stress" else 25)
                if calls[0] % interval == 0:
                    print("lbfgs", cycle, calls[0], [float(v.detach()) for v in terms], flush=True)
                return loss
            lbfgs.step(closure)
            optimizer_state = lbfgs.state[parameters[0]]
            lbfgs_history.append({"cycle": cycle, "iterations": optimizer_state["n_iter"],
                                  "function_evaluations": optimizer_state["func_evals"],
                                  "configured_max_iterations": args.lbfgs_steps})
            print("lbfgs_result", lbfgs_history[-1], flush=True)
            if args.constraint_updates:
                with torch.no_grad():
                    _, terms, _, constraint = objective(args.physics_weight)
                    constraint_history.append([len(constraint_history), *[float(v) for v in terms]])
                    if cycle < args.constraint_updates:
                        if constraint_multipliers is None:
                            constraint_multipliers = torch.zeros_like(constraint)
                        constraint_multipliers += 2*args.physics_weight*constraint
                save_checkpoint()
                print("constraint_cycle", constraint_history[-1], flush=True)
    with torch.no_grad():
        _, terms, (u, stress, kappa, diagnostics, prediction), _ = objective(args.physics_weight)
        metrics = obs.metrics(prediction, data["frames"], data["forces"])
    material_record = material.export()
    material_record["host_young_MPa"] = core.model.host_young_MPa * float(fields.host_log_scale.detach().exp().cpu())
    (out / "material.json").write_text(json.dumps(material_record, indent=2)+"\n", encoding="utf-8")
    save_checkpoint()
    np.savez_compressed(out / "training_fields.npz", frames=data["frames"], forces_N=data["forces"],
                        u_mm=u.detach().cpu().numpy(), stress_MPa=stress.detach().cpu().numpy(),
                        kappa=kappa.detach().cpu().numpy(), predicted_mm=prediction.detach().cpu().numpy(),
                        observed_mm=data["observed"], points_mm=data["points_mm"],
                        cell_is_band=core.is_band, cell_centers_mm=core.centers,
                        damage=(diagnostics["damage"].detach().cpu().numpy() if "damage" in diagnostics else np.zeros_like(kappa.cpu())))
    pd.DataFrame(metrics).to_csv(out / "training_metrics.csv", index=False)
    pd.DataFrame(window_metrics(data, prediction.detach().cpu().numpy(), core.Q)).to_csv(
        out / "training_windows.csv", index=False)
    pd.DataFrame(history, columns=["step", "observation", "weak", "constitutive"]).to_csv(out / "training_loss.csv", index=False)
    metadata = {"mode": args.mode, "sid": data["sid"], "reference_frame": data["reference_frame"],
                "frames": data["frames"].tolist(), "force_N": data["forces"].tolist(),
                "mesh": core.model.mesh_info, "geometry": core.model.geometry.__dict__,
                "field_representation": ("P1-interpolated neural displacement; closest equilibrated material stress"
                                         if args.eliminate_stress else
                                         "P1-interpolated neural displacement; stress from the material return"
                                         if args.constitutive_stress else
                                         "P1-interpolated neural displacement; neural element stress correction"),
                "equilibrium_constraint": ("linear projection of the neural pair onto all condensed nodal tests"
                                            if projection is not None else "weak residual penalty"),
                "path_representation": ("shared spatial networks with independent heads at ordered load states; attained-load amplitude"
                                        if args.state_heads else "smooth frame-coordinate input; instantaneous-load amplitude"),
                "loaded_reference": ("current-material elastic equilibrium; implicit parameter gradients; "
                                     "plasticity and damage checked at every evaluation"
                                     if args.elastic_reference else "neural total field with soft constitutive consistency"),
                "spatial_fit": {"central": "abs(s)<=4 mm, abs(n)>=4 mm",
                                "central_and_outer": "abs(s)<=4 mm, abs(n)>=4 mm; plus abs(n)>=13 mm",
                                "central_and_outer_deformation": "same central_and_outer points; extra rigid modes only on each outer block",
                                "outer_host": "abs(n)>=13 mm",
                                "outer_host_deformation": "abs(n)>=13 mm; independent rigid modes removed on each side"}[data["observation_layout"]],
                "spatial_holdout": {"central": "abs(s)>=8 mm, abs(n)>=4 mm",
                                    "central_and_outer": "abs(s)>=8 mm, 4<=abs(n)<10 mm",
                                    "central_and_outer_deformation": "abs(s)>=8 mm, 4<=abs(n)<10 mm",
                                    "outer_host": "4<=abs(n)<10 mm",
                                    "outer_host_deformation": "4<=abs(n)<10 mm"}[data["observation_layout"]],
                "fixed_band_source": args.initial_material if args.freeze_band else None,
                "material": material_record, "settings": vars(args),
                "loss_terms": [float(v.detach()) for v in terms],
                "physics_energy_scales_N_mm": energy_scale.cpu().tolist(),
                "observation_virtual_loss": (float(diagnostics["observation_virtual_loss"].detach())
                                             if args.observation_virtual_weight else None),
                "constraint_history": constraint_history,
                "lbfgs_history": lbfgs_history,
                "max_return_error_MPa": float(diagnostics["consistency_MPa"].abs().max().cpu()),
                "min_plastic_work_MPa": float(diagnostics["min_substep_dissipation_MPa"].min().cpu()),
                "maximum_kappa": float(kappa.max().cpu()),
                "maximum_host_damage": (float(diagnostics["damage"].max().cpu()) if "damage" in diagnostics else 0.),
                "holdout_mean_rmse_um": float(np.mean([r["holdout_rmse_um"] for r in metrics])),
                "elapsed_s": time.perf_counter()-start}
    (out / "training.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print("training_result", json.dumps({k:metadata[k] for k in ["loss_terms", "max_return_error_MPa", "maximum_kappa", "holdout_mean_rmse_um"]}), flush=True)
    return material_record


def replay_specimen(core, data, exported, out, label, replay_options=None):
    material = (ProcessMaterial.from_export(exported, core.is_band) if core.process_radius_mm > 0
                else BandMaterial.from_export(exported))
    host_scale = exported["host_young_MPa"] / core.model.host_young_MPa
    replay_options = {"progress": True, **(replay_options or {})}
    integration = []
    def accept(displacement, stress, state, record):
        integration.append((displacement.copy(), stress.copy(), state[1].copy(), dict(record)))
        save_accepted_state(out / f"{label}_accepted_state.npz", displacement, stress, state, record, material)
        temporary = out / f"{label}_integration_fields.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,
                                u_mm=np.stack([step[0] for step in integration]),
                                stress_MPa=np.stack([step[1] for step in integration]),
                                kappa=np.stack([step[2] for step in integration]),
                                force_N=[step[3]["force_N"] for step in integration],
                                target=[step[3]["target"] for step in integration])
        temporary.replace(out / f"{label}_integration_fields.npz")
    u, stress, kappa, diag = replay(core, material, data["forces"], host_scale=host_scale,
                                   on_step=accept, **replay_options)
    observer = Observations(core, data, "cpu")
    with torch.no_grad():
        prediction = observer.predict(torch.as_tensor(u), torch.as_tensor(data["forces"]), host_scale)
        # Test response is used here only for explicitly labelled quotient-space
        # diagnostics.  It never changes the material, boundary, or replay field.
        metrics = observer.metrics(prediction, data["frames"], data["forces"], alignment="fit_and_holdout_quotient")
    pd.DataFrame(metrics).to_csv(out / f"{label}_metrics.csv", index=False)
    pd.DataFrame(window_metrics(data, prediction.numpy(), core.Q)).to_csv(
        out / f"{label}_windows.csv", index=False)
    np.savez_compressed(out / f"{label}_fields.npz", frames=data["frames"], force_N=data["forces"],
                        u_mm=u, stress_MPa=stress, kappa=kappa, predicted_mm=prediction.numpy(),
                        observed_mm=data["observed"], points_mm=data["points_mm"],
                        cell_is_band=core.is_band, cell_centers_mm=core.centers)
    record = {"sid": data["sid"], "reference_frame": data["reference_frame"],
              "geometry": core.model.geometry.__dict__, "mesh": core.model.mesh_info,
              "material_input": "material.json", "field_networks_used": False,
              "integration_fields": f"{label}_integration_fields.npz",
              "conditioning_observations": ("Specimen outer-host DIC, abs(n)>=13 mm; host parameters fitted with the band frozen; "
                                             + data["observation_layout"]
                                             if label == "conditioned_replay" else None),
              "replay_options": replay_options or {},
              "steps": diag, "mean_holdout_quotient_rmse_um": float(np.mean([x["holdout_rmse_um"] for x in metrics])),
              "mean_raw_rmse_um": float(np.mean([x["raw_rmse_um"] for x in metrics]))}
    if label in {"identification_replay", "conditioned_replay"}:
        with np.load(out / "training_fields.npz") as identified:
            difference = prediction.numpy().reshape(data["observed"].shape)-identified["predicted_mm"].reshape(data["observed"].shape)
        valid = data["observed_valid"].copy()
        valid[0] = False
        record["identification_field_gap_rmse_um"] = 1000*float(np.sqrt(np.mean(difference[valid]**2)))
    (out / f"{label}.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(label, record["mean_holdout_quotient_rmse_um"], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "synthetic-monotonic", "real"], default="synthetic")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lbfgs-steps", type=int, default=150)
    parser.add_argument("--constraint-updates", type=int, default=0,
                        help="Augmented-Lagrangian multiplier updates between LBFGS blocks")
    parser.add_argument("--material-block-steps", type=int, default=0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--physics-weight", type=float, default=10.)
    parser.add_argument("--physics-normalization", choices=["final_load", "attained_load"],
                        default="final_load", help="Reference load-work scale of each history constraint")
    parser.add_argument("--observation-scale-mm", type=float, default=.02)
    parser.add_argument("--return-iterations", type=int, default=20)
    parser.add_argument("--material-substeps", type=int, default=None,
                        help="Constitutive substeps per straight strain interval; inherited from the material/checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU worker threads")
    parser.add_argument("--mesh-size", type=float, default=12.)
    parser.add_argument("--band-size", type=float, default=4.)
    parser.add_argument("--end-fraction", type=float, default=1.)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fit-host", action="store_true")
    parser.add_argument("--freeze-band", action="store_true",
                        help="Keep the supplied band parameters fixed while fitting specimen host parameters and fields")
    parser.add_argument("--sid", default="m-1-1")
    parser.add_argument("--host-degradation", choices=["isotropic", "strain_spectral", "oriented_contact"], default=None)
    parser.add_argument("--host-crack-initiation", choices=["sample", "damage_weighted"], default=None)
    parser.add_argument("--host-crack-coupling", choices=["parallel", "shared_stress"], default=None)
    parser.add_argument("--host-crack-direction", choices=["local", "nonlocal_tensile", "angular_tensile"], default=None)
    parser.add_argument("--host-direction-order", type=int, default=None,
                        help="Lebedev order for the angular crack-family distribution")
    parser.add_argument("--initial-material", default=None)
    parser.add_argument("--constitutive-stress", action="store_true")
    parser.add_argument("--equilibrated-fields", action="store_true")
    parser.add_argument("--eliminate-stress", action="store_true",
                        help="Minimize over admissible stress at each neural displacement; full domain only")
    parser.add_argument("--elastic-reference", action="store_true",
                        help="Anchor total displacement at the loaded reference using the current elastic material")
    parser.add_argument("--state-heads", action="store_true",
                        help="Separate field outputs at each load state, coupled by the constitutive history")
    parser.add_argument("--local-features", action="store_true",
                        help="Add band-scale coordinates and an element-stress material indicator")
    parser.add_argument("--observation-virtual-weight", type=float, default=0.,
                        help="Weight material-equilibrium tests in training-observation directions")
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--warm-start-fields-only", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--replay-device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for frozen constitutive updates; Newton-Krylov remains on the CPU")
    parser.add_argument("--replay-max-subdivisions", type=int, default=0,
                        help="Load bisections after a failed equilibrium or coupled contact trial")
    parser.add_argument("--probe-objective", default=None,
                        help="Write a saved-objective directional diagnostic to this JSON filename and exit")
    parser.add_argument("--surface-directory", default="data/processed/surface_history")
    parser.add_argument("--observation-layout", choices=["central", "central_and_outer", "central_and_outer_deformation",
                                                        "outer_host", "outer_host_deformation"], default="central")
    parser.add_argument("--process-radius", type=float, default=0.)
    args = parser.parse_args()
    if args.material_substeps is not None and args.material_substeps < 1:
        parser.error("--material-substeps must be positive")
    if args.equilibrated_fields and args.constitutive_stress:
        raise ValueError("Pair projection requires an independent stress trial; omit --constitutive-stress")
    if args.constraint_updates and not args.lbfgs_steps:
        raise ValueError("Constraint multiplier updates require nonzero --lbfgs-steps")
    if args.freeze_band and not args.initial_material:
        raise ValueError("Frozen-band conditioning requires --initial-material")
    if args.freeze_band and args.warm_start and not args.warm_start_fields_only:
        raise ValueError("Frozen-band conditioning must retain its supplied material; use --warm-start-fields-only")
    if args.host_degradation and not args.process_radius:
        raise ValueError("Host damage requires a nonzero --process-radius")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    out = ROOT / (args.output or f"results/mixed_pinn/{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    data = specimen(args.sid, args.frames, args.end_fraction,
                    args.surface_directory, args.observation_layout)
    core = make_core(data, args.mesh_size, args.band_size, args.process_radius)
    print("core", core.model.mesh_info, "local_dofs", len(core.dofs), flush=True)
    material = BandMaterial(young=140., shear=65., cohesion=.8, residual=.15, friction=.2)
    initial_host_scale = 1.
    if args.initial_material:
        initial = json.loads((ROOT / args.initial_material).read_text(encoding="utf-8"))
        if args.host_degradation:
            if args.host_degradation == "oriented_contact" and initial.get("host_degradation") != "oriented_contact":
                initial["host_crack_initiation"] = "damage_weighted"
            initial["host_degradation"] = args.host_degradation
        if args.host_crack_initiation:
            initial["host_crack_initiation"] = args.host_crack_initiation
        if args.host_crack_coupling:
            initial["host_crack_coupling"] = args.host_crack_coupling
        if args.host_crack_direction:
            initial["host_crack_direction"] = args.host_crack_direction
        if args.host_direction_order:
            initial["host_direction_order"] = args.host_direction_order
        material = (ProcessMaterial.from_export(initial, core.is_band) if args.process_radius > 0
                    else BandMaterial.from_export(initial))
        initial_host_scale = initial["host_young_MPa"] / core.model.host_young_MPa
    elif args.process_radius > 0:
        material = ProcessMaterial(material, core.is_band,
                                   host_degradation=args.host_degradation or "isotropic",
                                   host_crack_initiation=args.host_crack_initiation or "damage_weighted",
                                   host_crack_direction=args.host_crack_direction or "local",
                                   host_direction_order=args.host_direction_order or 7,
                                   host_crack_coupling=args.host_crack_coupling or "parallel")
    if args.mode.startswith("synthetic"):
        if args.process_radius > 0:
            raise ValueError("The synthetic band-only protocol has no rock-damage truth")
        truth = BandMaterial(young=240., shear=100., cohesion=.65, residual=.12,
                             friction=.16, dilation=.035, history_scale=.025)
        # A monotonic path can hide irreversible shear in elastic compliance.
        # Synthetic unloading directly tests the missing history information;
        # the actual experimental data are not presented as cyclic tests.
        if args.mode == "synthetic":
            maximum = data["forces"].max()
            data["forces"] = np.r_[data["forces"], [.5*maximum, .15*maximum, .65*maximum]]
            data["frames"] = np.arange(len(data["forces"]))
            data["reference_frame"] = 0
            data["observed_valid"] = np.concatenate((data["observed_valid"],
                                                      np.repeat(data["observed_valid"][-1:], 3, axis=0)))
            material = BandMaterial(young=240., shear=100., cohesion=.3, residual=.12,
                                    friction=.16, dilation=.035, history_scale=.04)
            for name, parameter in material.named_parameters():
                parameter.requires_grad_(name in {"log_excess", "log_history_scale",
                                                 "rate_logits", "weight_logits"})
        data["sid"] = "synthetic_m-1-1_geometry"
        truth_core = make_core(data, 10., 3.)
        true_u, _, true_kappa, truth_diagnostics = replay(truth_core, truth, data["forces"])
        H, h = truth_core.observer(truth_core.model.observation_operator(data["points_mm"]))
        true_prediction = true_u @ H.T + data["forces"][:, None]*h
        data["observed"] = (true_prediction-true_prediction[:1]).reshape(-1, len(data["points_mm"]), 3)
        (out / "synthetic_truth.json").write_text(json.dumps({"material":truth.export(),
                    "mesh":truth_core.model.mesh_info, "forces_N":data["forces"].tolist(),
                    "maximum_kappa":float(true_kappa.max()), "steps":truth_diagnostics,
                    "known_parameters":(["young", "shear", "residual", "friction", "dilation"]
                                         if args.mode == "synthetic" else []),
                    "identified_object":("positive cohesion evolution network" if args.mode == "synthetic"
                                         else "elastic, friction and positive cohesion parameters"),
                    "path":("synthetic loading, unloading and reloading; not experimental cycle evidence"
                            if args.mode == "synthetic" else "synthetic monotonic loading")}, indent=2)+"\n", encoding="utf-8")
    if args.material_substeps is not None:
        material.integration_substeps = args.material_substeps
    if args.replay_only:
        exported = json.loads((out / "material.json").read_text())
        if args.material_substeps is not None and args.material_substeps != exported.get("integration_substeps", 1):
            raise ValueError("Frozen replay must use the exported material integration_substeps")
    else:
        exported = train(core, data, material, out, args, initial_host_scale)
    if args.probe_objective:
        return
    replay_specimen(core, data, exported, out, "conditioned_replay" if args.freeze_band and
                    args.observation_layout in {"outer_host", "outer_host_deformation"} else "identification_replay",
                    {"device": args.replay_device, "max_subdivisions": args.replay_max_subdivisions})
    if args.mode == "real" and args.sid != "m-1-2" and not args.freeze_band:
        test = specimen("m-1-2", args.frames, args.end_fraction,
                        args.surface_directory, args.observation_layout)
        test_core = make_core(test, args.mesh_size, args.band_size, args.process_radius)
        replay_specimen(test_core, test, exported, out, "replicate_replay",
                        {"device": args.replay_device, "max_subdivisions": args.replay_max_subdivisions})


if __name__ == "__main__":
    main()
