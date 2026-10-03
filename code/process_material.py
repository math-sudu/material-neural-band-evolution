"""Finite-band plasticity plus a separate nonlocal tensile-damage rock region.

The rock uses a Rankine equivalent strain, maximum-history update and scalar
exponential damage. Its equivalent strain is averaged within the rock process
domain; it is not averaged together with filling plastic strain. The optional
strain-spectral energy split preserves the negative-strain energy during damage.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from scipy.integrate import lebedev_rule

from band_material import BandMaterial
from crack_contact import MaximumPrincipalDirection, contact_update, damage_family_update
from coupled_cracks import coupled_family_update


def elastic_stress(strains, matrices):
    if matrices.ndim == 2:
        return strains @ matrices.T
    return torch.einsum("...ej,eij->...ei", strains, matrices)


class PositiveSpectralPart(torch.autograd.Function):
    """Symmetric matrix positive part with a finite tangent at repeated roots.

    The divided-difference derivative acts on the matrix, so equal positive or
    negative eigenvalues do not require differentiating individual eigenvectors.
    At a zero root we select the zero one-sided derivative.
    """
    @staticmethod
    def forward(ctx, matrix):
        values, vectors = torch.linalg.eigh(matrix)
        positive = values.clamp_min(0.)
        ctx.save_for_backward(values, vectors, positive)
        return (vectors*positive[..., None, :]) @ vectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, gradient):
        values, vectors, positive = ctx.saved_tensors
        left, right = values[..., :, None], values[..., None, :]
        same_sign = (left > 0.) == (right > 0.)
        denominator = torch.where(same_sign, torch.ones_like(left-right), left-right)
        divided = (positive[..., :, None]-positive[..., None, :])/denominator
        tangent = torch.where(same_sign, (left > 0.).to(values.dtype), divided)
        local = vectors.transpose(-1, -2) @ (.5*(gradient+gradient.transpose(-1, -2))) @ vectors
        return vectors @ (tangent*local) @ vectors.transpose(-1, -2)


class ProcessMaterial(nn.Module):
    def __init__(self, band, is_band, young=13550., poisson=.28, threshold=.0003,
                 softening=.003, radius_mm=6., host_degradation="isotropic", host_friction=.6,
                 host_crack_initiation="damage_weighted", host_crack_direction="local",
                 host_direction_order=7, host_crack_coupling="parallel", integration_substeps=1):
        super().__init__()
        self.band = band
        self.register_buffer("is_band", torch.as_tensor(is_band, dtype=torch.bool), persistent=False)
        self.log_threshold = nn.Parameter(torch.tensor(math.log(threshold)))
        self.log_softening = nn.Parameter(torch.tensor(math.log(softening)))
        self.young = float(young)
        self.poisson = float(poisson)
        self.radius_mm = float(radius_mm)
        self.integration_substeps = integration_substeps
        if host_degradation not in {"isotropic", "strain_spectral", "oriented_contact"}:
            raise ValueError(f"Unsupported host degradation: {host_degradation}")
        self.host_degradation = host_degradation
        if host_crack_initiation not in {"sample", "damage_weighted"}:
            raise ValueError(f"Unsupported crack initiation rule: {host_crack_initiation}")
        self.host_crack_initiation = host_crack_initiation
        if host_crack_coupling not in {"parallel", "shared_stress"}:
            raise ValueError(f"Unsupported crack coupling: {host_crack_coupling}")
        if host_crack_coupling == "shared_stress" and (
                host_degradation != "oriented_contact" or host_crack_initiation != "damage_weighted"):
            raise ValueError("Shared-stress coupling requires damage-weighted oriented contact")
        self.host_crack_coupling = host_crack_coupling
        if host_crack_direction not in {"local", "nonlocal_tensile", "angular_tensile"}:
            raise ValueError(f"Unsupported crack direction: {host_crack_direction}")
        self.host_crack_direction = host_crack_direction
        self.host_direction_order = int(host_direction_order)
        if host_crack_direction == "angular_tensile":
            if host_crack_initiation != "damage_weighted":
                raise ValueError("Angular directions require damage-weighted crack families")
            points, weights = lebedev_rule(self.host_direction_order)
            if (weights <= 0.).any():
                raise ValueError("Crack-family quadrature requires positive weights")
            # A plane and its antipode have identical contact response. Retain
            # one representative and combine their solid-angle weights.
            first_nonzero = (points != 0.).argmax(axis=0)
            keep = points[first_nonzero, range(points.shape[1])] > 0.
            self.register_buffer("direction_points", self.log_threshold.new_tensor(points[:, keep].T.copy()), persistent=False)
            self.register_buffer("direction_weights", self.log_threshold.new_tensor(2*weights[keep]/(4*math.pi)), persistent=False)
        if host_degradation == "oriented_contact":
            self.log_host_friction = nn.Parameter(torch.tensor(math.log(host_friction)))

    def host_matrix(self, host_scale=1.):
        scale = self.log_threshold.new_tensor(self.young)*host_scale
        nu = self.poisson
        lame, mu = scale*nu/((1+nu)*(1-2*nu)), scale/(2*(1+nu))
        unit = torch.eye(6, device=scale.device, dtype=scale.dtype)
        normal = torch.zeros_like(unit)
        normal[:3, :3] = 1.
        diagonal = scale.new_tensor([2., 2., 2., 1., 1., 1.])
        return normal*lame + unit*diagonal*mu

    def elastic_matrix(self, host_scale=1.):
        return torch.where(self.is_band[:, None, None], self.band.elastic_matrix()[None],
                           self.host_matrix(host_scale)[None])

    def initial_state(self, strain):
        state = (torch.zeros_like(strain), torch.zeros_like(strain[..., 0]))
        if self.host_degradation == "oriented_contact":
            if self.host_crack_initiation == "sample":
                state += (torch.zeros_like(strain[..., :3]),)
            else:
                host = strain[~self.is_band]
                width = 4 if self.host_crack_coupling == "shared_stress" else 6
                state += (host.new_empty((0, len(host), 3)), host.new_empty((0, len(host), width)),
                          host.new_empty((0, len(host))))
        return state

    def damage(self, history):
        threshold = self.log_threshold.exp()
        kappa = history.clamp_min(threshold)
        return 1.-threshold/kappa*torch.exp(-(kappa-threshold)/self.log_softening.exp())

    def host_response(self, strain, damage, host_scale=1.):
        """Return stress and the energy conjugate to increasing scalar damage."""
        if self.host_degradation == "oriented_contact":
            raise ValueError("Oriented contact requires the crack history; use update")
        matrix = self.host_matrix(host_scale)
        effective = strain @ matrix.T
        if self.host_degradation == "isotropic":
            return (1.-damage[..., None])*effective, .5*(strain*effective).sum(-1)
        # Engineering shear strain is twice the tensor off-diagonal entry.
        tensor = torch.stack((torch.stack((strain[..., 0], strain[..., 3]/2, strain[..., 5]/2), -1),
                              torch.stack((strain[..., 3]/2, strain[..., 1], strain[..., 4]/2), -1),
                              torch.stack((strain[..., 5]/2, strain[..., 4]/2, strain[..., 2]), -1)), -2)
        positive = PositiveSpectralPart.apply(tensor)
        trace_positive = strain[..., :3].sum(-1).clamp_min(0.)
        lame, mu = matrix[0, 1], matrix[3, 3]
        positive_stress = (2*mu*positive + lame*trace_positive[..., None, None]
                           *torch.eye(3, dtype=strain.dtype, device=strain.device))
        positive_voigt = torch.stack((positive_stress[..., 0, 0], positive_stress[..., 1, 1],
                                      positive_stress[..., 2, 2], positive_stress[..., 0, 1],
                                      positive_stress[..., 1, 2], positive_stress[..., 0, 2]), -1)
        release = .5*lame*trace_positive.square() + mu*positive.square().sum((-2, -1))
        return effective-damage[..., None]*positive_voigt, release

    def update(self, strain, state, average, iterations=16, host_scale=1.):
        plastic, history = state[:2]
        band_mask, host_mask = self.is_band, ~self.is_band
        band_stress, band_state, band_diag = self.band.update(
            strain[band_mask], (plastic[band_mask], history[band_mask]), average[0], iterations)
        host_strain = strain[host_mask]
        effective = host_strain @ self.host_matrix(host_scale).T
        # Engineering order ss,nn,zz,sn,nz,sz; effective stress has physical shear.
        stress_tensor = torch.stack((effective[:, [0, 3, 5]], effective[:, [3, 1, 4]],
                                     effective[:, [5, 4, 2]]), dim=-2)
        equivalent = torch.relu(torch.linalg.eigvalsh(stress_tensor)[:, -1])/(self.young*host_scale)
        averaged_equivalent = average[1] @ equivalent
        host_history = torch.maximum(history[host_mask], averaged_equivalent)
        damage = self.damage(host_history)
        if self.host_degradation == "oriented_contact":
            direction_tensor = stress_tensor
            if self.host_crack_direction in {"nonlocal_tensile", "angular_tensile"}:
                # Damage is driven by this same host neighborhood. Preserve
                # its tensile directions before choosing a birth normal; a
                # local repeated root need not select an arbitrary crack plane.
                positive = PositiveSpectralPart.apply(stress_tensor)
                direction_tensor = (average[1] @ positive.flatten(1)).reshape_as(positive)
            if self.host_crack_initiation == "damage_weighted":
                update_families = (coupled_family_update if self.host_crack_coupling == "shared_stress"
                                   else damage_family_update)
                host_stress, host_plastic, host_state, host_diag = update_families(
                    host_strain, state[2:], damage, self.damage(history[host_mask]),
                    self.host_matrix(host_scale), self.log_host_friction.exp(), direction_tensor,
                    ((self.direction_points, self.direction_weights)
                     if self.host_crack_direction == "angular_tensile" else None))
            else:
                normal = state[2][host_mask].clone()
                initiate = (normal.square().sum(-1) == 0.) & (damage > 0.)
                if initiate.any():
                    normal[initiate] = MaximumPrincipalDirection.apply(direction_tensor[initiate])
                host_stress, host_plastic, host_diag = contact_update(
                    host_strain, plastic[host_mask], normal, damage, self.damage(history[host_mask]),
                    self.host_matrix(host_scale), self.log_host_friction.exp())
        else:
            host_stress, release_energy = self.host_response(host_strain, damage, host_scale)
        stresses = torch.zeros_like(strain)
        stresses[band_mask], stresses[host_mask] = band_stress, host_stress
        next_plastic, next_history = plastic.clone(), history.clone()
        next_plastic[band_mask] = band_state[0]
        next_history[band_mask], next_history[host_mask] = band_state[1], host_history
        consistency = torch.zeros_like(history)
        consistency[band_mask] = band_diag["consistency_MPa"]
        dissipation = torch.zeros_like(history)
        dissipation[band_mask] = band_diag["dissipation_MPa"]
        next_state = (next_plastic, next_history)
        if self.host_degradation == "oriented_contact":
            next_plastic[host_mask] = host_plastic
            if self.host_crack_initiation == "damage_weighted":
                next_state += host_state
            else:
                next_normal = state[2].clone()
                next_normal[host_mask] = normal
                next_state += (next_normal,)
            consistency[host_mask] = host_diag["consistency_MPa"]
            dissipation[host_mask] = host_diag["dissipation_MPa"]
        else:
            dissipation[host_mask] = release_energy*(damage-self.damage(history[host_mask]))
        total_damage = torch.zeros_like(history)
        total_damage[host_mask] = damage
        diagnostics = {
            "consistency_MPa": consistency, "dissipation_MPa": dissipation,
            "pressure_MPa": -stresses[:, 1], "damage": total_damage,
        }
        if self.host_degradation == "oriented_contact":
            fabric = strain.new_zeros((len(strain), 3, 3))
            if self.host_crack_initiation == "damage_weighted":
                fabric[host_mask] = host_diag["crack_fabric"]
            else:
                diagnostics["crack_normal"] = next_normal
                fabric[host_mask] = damage[:, None, None]*normal[:, :, None]*normal[:, None, :]
            diagnostics["crack_fabric"] = fabric
        return stresses, next_state, diagnostics

    integrate_interval = BandMaterial.integrate_interval
    history = BandMaterial.history

    def export(self):
        data = {"kind": "band_plasticity_host_damage_v1", "band": self.band.export(),
                "integration_substeps": self.integration_substeps,
                "host_reference_young_MPa": self.young, "host_poisson": self.poisson,
                "threshold_strain": float(self.log_threshold.detach().exp()),
                "softening_strain": float(self.log_softening.detach().exp()),
                "radius_mm": self.radius_mm,
                "host_degradation": self.host_degradation,
                "host_driver": "nonlocal positive maximum effective principal stress / host Young modulus",
                "history_channels": "band: accumulated plastic shear; host: maximum nonlocal equivalent strain"}
        if self.host_degradation == "oriented_contact":
            data["host_friction"] = float(self.log_host_friction.detach().exp())
            data["host_crack_initiation"] = self.host_crack_initiation
            data["host_crack_direction"] = self.host_crack_direction
            data["host_crack_coupling"] = self.host_crack_coupling
            source = ("nonlocal positive effective-stress tensor" if self.host_crack_direction == "nonlocal_tensile"
                      else "local effective-stress tensor")
            data["host_crack_orientation"] = (f"maximum-principal directions of the {source}; "
                                               + ("fixed families weighted by nonlocal damage increments"
                                                  if self.host_crack_initiation == "damage_weighted" else
                                                  "fixed at first damage"))
            if self.host_crack_direction == "angular_tensile":
                data["host_direction_order"] = self.host_direction_order
                data["host_crack_orientation"] = (
                    "spherical directions mapped by the nonlocal positive effective-stress tensor; "
                    "birth weights proportional to squared mapped lengths; fixed family histories")
            data["host_shear_state"] = ("damage-weighted contact-family shear strains; each family has frictional force tau0-G*b"
                                        if self.host_crack_initiation == "damage_weighted" else
                                        "crack shear strain sym(b tensor n); frictional force tau0-G*b/d")
            if self.host_crack_coupling == "shared_stress":
                data["host_shear_state"] = (
                    "fixed-family opening and tangent shear vectors; additive weighted crack strain; "
                    "shared traction with backstress (1-d)*G*b")
                data["host_return_scheme"] = "primal semismooth Newton-GMRES with implicit history gradients"
                data["host_dissipation_diagnostic"] = (
                    "spring-energy decrease at fixed incoming crack state plus end-force friction work; "
                    "normal relaxation is excluded")
        return data

    @classmethod
    def from_export(cls, data, is_band):
        if data["kind"] != "band_plasticity_host_damage_v1":
            raise ValueError("Unsupported process-zone material")
        return cls(BandMaterial.from_export(data["band"]), is_band,
                   data["host_reference_young_MPa"], data["host_poisson"],
                   data["threshold_strain"], data["softening_strain"], data["radius_mm"],
                   data.get("host_degradation", "isotropic"), data.get("host_friction", .6),
                   data.get("host_crack_initiation", "sample"), data.get("host_crack_direction", "local"),
                   data.get("host_direction_order", 7), data.get("host_crack_coupling", "parallel"),
                   data.get("integration_substeps", 1))
