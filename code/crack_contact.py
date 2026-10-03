"""Fixed-plane unilateral opening and frictional crack shear in 3D.

Engineering strain order is xx, yy, zz, xy, yz, xz. The crack shear vector b
is tangent to its stored normal n; its strain is sym(b tensor n). Damage adds
compliance in these crack modes, not in the intact plane. See pinn_design.md.
"""
from __future__ import annotations

import torch


def strain_tensor(strain):
    return torch.stack((strain[..., [0, 3, 5]]*strain.new_tensor([1., .5, .5]),
                        strain[..., [3, 1, 4]]*strain.new_tensor([.5, 1., .5]),
                        strain[..., [5, 4, 2]]*strain.new_tensor([.5, .5, 1.])), -2)


def stress_tensor(stress):
    return torch.stack((stress[..., [0, 3, 5]], stress[..., [3, 1, 4]],
                        stress[..., [5, 4, 2]]), -2)


def stress_vector(tensor):
    return torch.stack((tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2],
                        tensor[..., 0, 1], tensor[..., 1, 2], tensor[..., 0, 2]), -1)


class MaximumPrincipalDirection(torch.autograd.Function):
    """Largest-eigenvector derivative without divisions between lower roots.

    The selected largest root must be simple. A repeated lower pair, as in
    uniaxial tension, does not invalidate the derivative of the largest vector.
    """
    @staticmethod
    def forward(ctx, matrix):
        values, vectors = torch.linalg.eigh(matrix)
        gaps = values[..., -1, None]-values[..., :-1]
        tolerance = 64*torch.finfo(matrix.dtype).eps*values.abs().amax(-1).clamp_min(1.)
        if torch.any(gaps[..., -1] <= tolerance):
            raise RuntimeError("Crack initiation has no unique maximum principal direction")
        ctx.save_for_backward(vectors, gaps)
        return vectors[..., -1]

    @staticmethod
    def backward(ctx, gradient):
        vectors, gaps = ctx.saved_tensors
        normal = vectors[..., -1]
        lower = vectors[..., :-1]
        coefficients = (lower.transpose(-1, -2) @ gradient[..., None])[..., 0]/gaps
        tangent = (lower @ coefficients[..., None])[..., 0]
        derivative = tangent[..., :, None]*normal[..., None, :]
        return .5*(derivative+derivative.transpose(-1, -2))


def contact_update(strain, plastic, normal, damage, previous_damage, matrix, friction):
    """Eliminate crack opening and radially return its two shear components.

    For a fixed plane the shear energy is -tau0.b + G |b|^2/(2 d), added to
    intact elastic energy; the Coulomb force is tau0-G*b/d. The return is
    expressed in slip units so the virgin d=0 state requires no division by d.
    """
    effective = stress_tensor(strain @ matrix.T)
    traction = (effective @ normal[..., None])[..., 0]
    normal_stress = (traction*normal).sum(-1)
    shear = traction-normal_stress[..., None]*normal
    mu, lame = matrix[3, 3], matrix[0, 1]
    modulus = lame+2*mu
    old_slip = 2*(strain_tensor(plastic) @ normal[..., None])[..., 0]
    old_slip = old_slip-(old_slip*normal).sum(-1, keepdim=True)*normal
    trial = damage[..., None]*shear/mu-old_slip
    trial_norm = torch.linalg.vector_norm(trial, dim=-1)
    pressure = (-normal_stress).clamp_min(0.)
    radius = damage*friction*pressure/mu
    active = trial_norm > radius
    denominator = torch.where(active, trial_norm, torch.ones_like(trial_norm))
    increment = torch.where(active[..., None],
                            (1.-radius/denominator)[..., None]*trial, torch.zeros_like(trial))
    slip = old_slip+increment
    slip_strain = .5*(slip[..., :, None]*normal[..., None, :]
                     +normal[..., :, None]*slip[..., None, :])
    opening = damage*normal_stress.clamp_min(0.)/modulus
    opening_tensor = (lame*torch.eye(3, dtype=strain.dtype, device=strain.device)
                      +2*mu*normal[..., :, None]*normal[..., None, :])
    stress = effective-opening[..., None, None]*opening_tensor-2*mu*slip_strain
    next_plastic = stress_vector(slip_strain)*strain.new_tensor([1., 1., 1., 2., 2., 2.])
    # Damage grows before frictional return. This is the exact energy decrease
    # at fixed strain/slip for that damage substep, plus end-force slip work.
    old_safe = torch.where(previous_damage > 0., previous_damage, torch.ones_like(previous_damage))
    safe = torch.where(damage > 0., damage, torch.ones_like(damage))
    damage_work = .5*mu*old_slip.square().sum(-1)*(1./old_safe-1./safe)
    damage_work += .5*(damage-previous_damage)*normal_stress.clamp_min(0.).square()/modulus
    friction_work = friction*pressure*torch.linalg.vector_norm(increment, dim=-1)
    force = shear-mu*slip/safe[..., None]
    excess = (torch.linalg.vector_norm(force, dim=-1)-friction*pressure).clamp_min(0.)
    excess = torch.where(damage > 0., excess, torch.zeros_like(excess))
    return stress_vector(stress), next_plastic, {
        "dissipation_MPa": damage_work+friction_work,
        "consistency_MPa": excess,
        "crack_slip": torch.linalg.vector_norm(slip, dim=-1),
    }


def tensor_direction_families(tensor, increment, directions, quadrature_weights):
    """Push spherical directions through a tensile tensor without eigenvectors.

    Birth weights are proportional to |T v|^2. Thus the weighted orientation
    tensor is exactly increment*T^2/tr(T^2) for a second-order spherical rule.
    A vanishing mapped direction has zero weight; repeated eigenvalues require
    no arbitrary selection of a principal basis.
    """
    mapped = torch.einsum("eij,qj->qei", tensor, directions)
    square = mapped.square().sum(-1)
    occupied = square > 0.
    length = torch.sqrt(torch.where(occupied, square, torch.ones_like(square)))
    normals = mapped/length[..., None]
    mass = quadrature_weights[:, None]*square
    total = mass.sum(0)
    if torch.any((increment > 0.) & (total == 0.)):
        raise RuntimeError("Growing damage has no tensile orientation support")
    denominator = torch.where(total > 0., total, torch.ones_like(total))
    return normals, increment[None]*mass/denominator[None]


def damage_family_update(strain, state, damage, previous_damage, matrix, friction,
                         direction_tensor=None, quadrature=None):
    """Quadrature over fixed crack families born with positive damage increments.

    Each family is a fully cracked contact branch; its weight is the damage
    increment at birth. Old weights and directions never change. The intact
    branch has weight 1-d, so vanishing early damage cannot rotate later cracks.
    """
    normals, slips, weights = state
    increment = damage-previous_damage
    effective = strain @ matrix.T
    tensor = stress_tensor(effective)
    source = tensor if direction_tensor is None else direction_tensor
    if quadrature is None:
        normal = torch.zeros_like(strain[..., :3])
        growing = increment > 0.
        if growing.any():
            normal[growing] = MaximumPrincipalDirection.apply(source[growing])
        born_normals, born_weights = normal[None], increment[None]
    else:
        born_normals, born_weights = tensor_direction_families(source, increment, *quadrature)
    normals = torch.cat((normals, born_normals), 0)
    old_slips = torch.cat((slips, strain.new_zeros((*born_weights.shape, 6))), 0)
    weights = torch.cat((weights, born_weights), 0)
    full_damage = torch.ones_like(weights)
    branches, slips, diagnostics = contact_update(
        strain[None], old_slips, normals, full_damage, full_damage, matrix, friction)
    stress = (1.-damage[:, None])*effective+(weights[..., None]*branches).sum(0)
    plastic = (weights[..., None]*slips).sum(0)
    normal_stress = torch.einsum("qei,eij,qej->qe", born_normals, tensor, born_normals)
    release = (.5*born_weights*normal_stress.clamp_min(0.).square()/matrix[0, 0]).sum(0)
    fabric = torch.einsum("fe,fei,fej->eij", weights, normals, normals)
    return stress, plastic, (normals, slips, weights), {
        "consistency_MPa": diagnostics["consistency_MPa"].amax(0),
        "dissipation_MPa": release+(weights*diagnostics["dissipation_MPa"]).sum(0),
        "crack_fabric": fabric,
    }
