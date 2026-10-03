"""Differentiable weak-plane plasticity with an over-nonlocal shear history.

Local engineering order is ss, nn, zz, sn, nz, sz.  Stress is tension-positive;
the two shear entries are physical shear stresses (strains are engineering).
The positive two-unit cohesion network consumes only the history variable.
The averaging kernel and its physical radius belong to the exported material.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist


def averaging_matrix(points, volumes, radius_mm):
    """Compact quartic kernel, volume weighted and normalized at boundaries."""
    if radius_mm <= 0:
        raise ValueError("The nonlocal radius must be positive")
    distance = cdist(points, points) / radius_mm
    weights = np.maximum(1. - distance ** 2, 0.) ** 2 * volumes[None, :]
    return weights / weights.sum(axis=1, keepdims=True)


class BandMaterial(nn.Module):
    """Nonassociated shear return; tensile normal response remains elastic.

    The flow dilates with beta <= mu in both pressure branches. Friction uses
    the positive part of the returned pressure. Switching the dilation off at
    zero trial pressure would make the finite-step stress update discontinuous.
    This extension is not a tensile damage or opening law.
    """
    def __init__(self, young=200., shear=100., cohesion=1., residual=.15,
                 friction=.35, dilation=.04, history_scale=.03,
                 radius_mm=6., nonlocal_mix=2., poisson=.2, integration_substeps=1):
        super().__init__()
        if not (young > 0 and shear > 0 and cohesion > residual > 0
                and 0 < dilation < friction and history_scale > 0
                and radius_mm > 0 and nonlocal_mix > 1):
            raise ValueError("Invalid weak-plane material parameters")
        self.log_young = nn.Parameter(torch.tensor(math.log(young)))
        self.log_shear = nn.Parameter(torch.tensor(math.log(shear)))
        self.log_residual = nn.Parameter(torch.tensor(math.log(residual)))
        self.log_excess = nn.Parameter(torch.tensor(math.log(cohesion - residual)))
        self.log_friction = nn.Parameter(torch.tensor(math.log(friction)))
        self.dilation_logit = nn.Parameter(torch.tensor(math.log(dilation / (friction-dilation))))
        self.log_history_scale = nn.Parameter(torch.tensor(math.log(history_scale)))
        self.rate_logits = nn.Parameter(torch.tensor([-.4, .4]))
        self.weight_logits = nn.Parameter(torch.zeros(2))
        self.radius_mm = float(radius_mm)
        self.nonlocal_mix = float(nonlocal_mix)
        self.poisson = float(poisson)
        self.integration_substeps = integration_substeps

    def elastic_matrix(self, host_scale=1.):
        young, shear = self.log_young.exp(), self.log_shear.exp()
        nu = self.poisson
        lame = young * nu / ((1. + nu) * (1. - 2. * nu))
        mu = young / (2. * (1. + nu))
        normal = torch.ones((3, 3), device=young.device, dtype=young.dtype) * lame
        normal = normal + torch.eye(3, device=young.device, dtype=young.dtype) * 2 * mu
        zero = torch.zeros_like(normal)
        tangential = torch.diag(torch.stack((shear, shear, mu)))
        return torch.cat((torch.cat((normal, zero), 1), torch.cat((zero, tangential), 1)), 0)

    def cohesion(self, history):
        rates = self.rate_logits.exp()
        weights = self.weight_logits.softmax(0)
        value = (2. * torch.sigmoid(-history[..., None] * rates / self.log_history_scale.exp())
                 * weights).sum(-1)
        return self.log_residual.exp() + self.log_excess.exp() * value

    def cohesion_slope(self, history):
        rates = self.rate_logits.exp()/self.log_history_scale.exp()
        sigmoid = torch.sigmoid(-history[..., None]*rates)
        return -self.log_excess.exp()*(2.*sigmoid*(1.-sigmoid)*rates*self.weight_logits.softmax(0)).sum(-1)

    def initial_state(self, strain):
        return torch.zeros_like(strain), torch.zeros_like(strain[..., 0])

    def update(self, strain, state, average, iterations=16, host_scale=1.):
        """Coupled backward-Euler return with an implicit history derivative.

        Active-set Newton solves the plastic increments. Differentiation uses
        the converged return Jacobian, not a truncated fixed-point trajectory.
        A failed return is exposed before its stress enters a training loss.
        """
        plastic_previous, kappa_previous = state
        C = self.elastic_matrix()
        trial = (strain - plastic_previous) @ C.T
        shear_trial = trial[..., [3, 4]]
        norm = torch.linalg.vector_norm(shear_trial, dim=-1)
        direction = shear_trial / norm.clamp_min(torch.finfo(strain.dtype).eps)[..., None]
        pressure = -trial[..., 1]
        mu = self.log_friction.exp()
        beta = mu * self.dilation_logit.sigmoid()
        denominator = C[3, 3] + mu * C[1, 1] * beta
        def radial_return(strength):
            cohesive = torch.relu((norm-strength)/C[3, 3])
            compressed = torch.relu((norm-strength-mu*pressure)/denominator)
            branch = pressure+C[1, 1]*beta*cohesive > 0.
            return (torch.where(branch, compressed, cohesive),
                    torch.where(branch, denominator, C[3, 3]))
        mix = self.nonlocal_mix
        identity = torch.eye(len(norm), dtype=norm.dtype, device=norm.device)
        mixing = mix*average+(1.-mix)*identity
        def equation(increment):
            driver = mixing @ (kappa_previous+increment)
            target, branch_modulus = radial_return(self.cohesion(driver))
            slope = -self.cohesion_slope(driver)/branch_modulus*(target>0.)
            return increment-target, identity-slope[:, None]*mixing
        with torch.no_grad():
            upper = norm/C[3, 3]
            trial_increment = radial_return(self.cohesion(mixing @ kappa_previous))[0]
            # A steep softening law can have several algebraic roots. Once the
            # trial state yields, approach its dissipative branch from the
            # admissible upper increment; inactive trial points start at zero.
            increment = torch.where(trial_increment>0., upper, torch.zeros_like(norm))
            for _ in range(iterations):
                residual, jacobian = equation(increment)
                error = (residual*denominator).abs().max()
                if float(error) < 1e-10:
                    break
                active = (increment>0.) | (residual<0.)
                newton_step = torch.zeros_like(increment)
                newton_step[active] = torch.linalg.solve(jacobian[active][:, active], -residual[active])
                merit = residual.square().sum()
                fraction = 1.
                for _ in range(16):
                    candidate = (increment+fraction*newton_step).clamp_min(0.).minimum(upper)
                    if equation(candidate)[0].square().sum() < merit:
                        break
                    fraction *= .5
                else:
                    raise RuntimeError(f"Nonlocal return Newton line search failed: {float(error):g} MPa")
                increment = candidate
            residual, jacobian = equation(increment)
            error = float((residual*denominator).abs().max())
            if error > 1e-8:
                raise RuntimeError(f"Nonlocal material return did not converge: {error:g} MPa")
            active = increment>0.
        # At the solved root, d(delta)/d(theta) = -J^-1 partial_theta F.
        # The detached Jacobian is sufficient for this first derivative; all
        # dependencies through trial stress and previous history remain live.
        if torch.is_grad_enabled() and active.any():
            residual, _ = equation(increment)
            correction = torch.linalg.solve(jacobian[active][:, active], residual[active])
            implicit_increment = increment.clone()
            implicit_increment[active] = increment[active]-correction
            increment = implicit_increment
        kappa = kappa_previous + increment
        flow = torch.zeros_like(strain)
        flow[..., 1] = beta
        flow[..., 3:5] = direction
        plastic_increment = increment[..., None] * flow
        plastic = plastic_previous + plastic_increment
        stress = (strain - plastic) @ C.T
        driver = mix * (average @ kappa) + (1. - mix) * kappa
        strength = self.cohesion(driver)
        target = radial_return(strength)[0]
        consistency = (increment - target) * denominator
        dissipation = (stress * plastic_increment).sum(-1)
        return stress, (plastic, kappa), {
            "consistency_MPa": consistency, "dissipation_MPa": dissipation,
            "pressure_MPa": -stress[..., 1], "cohesion_MPa": strength,
            "plastic_increment": increment, "history_driver": driver,
        }

    def integrate_interval(self, previous_strain, strain, state, average, iterations=16,
                           host_scale=1., *, substeps=None, on_substep=None):
        """Integrate the straight strain segment without detaching its history.

        Only the interval endpoint is an observation/equilibrium state. Work is
        accumulated over returns; endpoint pressure, damage and fabric retain
        their endpoint meaning. The optional callback exposes each incoming
        state to the approximate contact preconditioner.
        """
        count = self.integration_substeps if substeps is None else substeps
        if not isinstance(count, int) or count < 1:
            raise ValueError("Material integration_substeps must be a positive integer")
        work = error = minimum = increment = None
        for step in range(1, count+1):
            current = strain if step == count else previous_strain+(strain-previous_strain)*(step/count)
            incoming = state
            stress, state, diag = self.update(current, state, average, iterations, host_scale)
            if on_substep is not None:
                on_substep(incoming)
            dissipation = diag["dissipation_MPa"]
            work = dissipation if work is None else work+dissipation
            minimum = dissipation if minimum is None else torch.minimum(minimum, dissipation)
            residual = diag["consistency_MPa"].abs()
            error = residual if error is None else torch.maximum(error, residual)
            if "plastic_increment" in diag:
                increment = diag["plastic_increment"] if increment is None else increment+diag["plastic_increment"]
        diag = {**diag, "dissipation_MPa": work, "consistency_MPa": error,
                "min_substep_dissipation_MPa": minimum}
        if increment is not None:
            diag["plastic_increment"] = increment
        return stress, state, diag

    def history(self, strains, average, iterations=16, host_scale=1.):
        state = self.initial_state(strains[0])
        previous_strain = torch.zeros_like(strains[0])
        stresses, kappas, diagnostics = [], [], []
        for strain in strains:
            stress, state, diag = self.integrate_interval(
                previous_strain, strain, state, average, iterations, host_scale)
            previous_strain = strain
            stresses.append(stress)
            kappas.append(state[1])
            diagnostics.append(diag)
        return torch.stack(stresses), torch.stack(kappas), {
            key: torch.stack([d[key] for d in diagnostics]) for key in diagnostics[0]
        }

    def export(self):
        return {"kind": "nonlocal_weak_plane_v2", "poisson": self.poisson,
                "integration_substeps": self.integration_substeps,
                "return_scheme": "active-set Newton with implicit differentiation; yielded points start at the upper admissible increment",
                "radius_mm": self.radius_mm, "nonlocal_mix": self.nonlocal_mix,
                "state_dict": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()}}

    @classmethod
    def from_export(cls, data):
        if data["kind"] != "nonlocal_weak_plane_v2":
            raise ValueError("Unsupported material definition")
        model = cls(poisson=data["poisson"], radius_mm=data["radius_mm"],
                    nonlocal_mix=data["nonlocal_mix"], integration_substeps=data.get("integration_substeps", 1))
        model.load_state_dict({k: torch.as_tensor(v, dtype=model.log_young.dtype)
                               for k, v in data["state_dict"].items()})
        return model
