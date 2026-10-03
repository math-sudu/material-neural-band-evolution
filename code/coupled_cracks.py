"""Shared-stress crack families with additive opening and shear strains.

The return solves normal complementarity and Coulomb friction together. Its
primal equations remain defined at complete damage. A six-component Schur
preconditioner accelerates matrix-free Newton/GMRES; its diagonal shift changes
only the linear preconditioner, never the material equations.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.optimize import least_squares, root

from crack_contact import MaximumPrincipalDirection, tensor_direction_families


class CrackReturnError(RuntimeError):
    """A trial strain has not obtained a converged coupled contact return."""

    def __init__(self, message, local_problem=None):
        super().__init__(message)
        self.local_problem = local_problem


def crack_maps(normals):
    """Work-conjugate maps from [opening, tangential vector] to Voigt strain."""
    x, y, z = normals.unbind(-1)
    zero = torch.zeros_like(x)
    traction = torch.stack((torch.stack((x, zero, zero, y, zero, z), -1),
                            torch.stack((zero, y, zero, x, z, zero), -1),
                            torch.stack((zero, zero, z, zero, y, x), -1)), -2)
    normal = (normals[..., :, None]*traction).sum(-2)
    tangent = traction-normals[..., :, None]*normal[..., None, :]
    return torch.cat((normal[..., None, :], tangent), -2)


def return_equation(value, strain, normals, weights, previous, damage, matrix, friction,
                    linearize=False):
    maps = crack_maps(normals)
    crack_strain = torch.einsum("fe,fei,feij->ej", weights, value, maps)
    stress = (strain-crack_strain) @ matrix.T
    local = torch.einsum("feij,ej->fei", maps, stress)
    modulus, shear = matrix[0, 0], matrix[3, 3]
    projector = torch.eye(3, dtype=strain.dtype, device=strain.device)-normals[..., :, None]*normals[..., None, :]
    tangent_value = (projector @ value[..., 1:, None])[..., 0]
    argument = damage[None]*value[..., 0]+local[..., 0]/modulus
    trial = damage[None, :, None]*tangent_value-previous[..., 1:]+local[..., 1:]/shear
    length = torch.linalg.vector_norm(trial, dim=-1)
    radius = friction*(-local[..., 0]).clamp_min(0.)/shear
    sliding = length > radius
    safe = torch.where(sliding, length, torch.ones_like(length))
    direction = trial/safe[..., None]
    returned = torch.where(sliding[..., None], (1.-radius/safe)[..., None]*trial,
                           torch.zeros_like(trial))
    residual = torch.cat(((value[..., 0]-argument.clamp_min(0.))[..., None],
                          value[..., 1:]-previous[..., 1:]-returned), -1)
    occupied = weights > 0.
    residual = torch.where(occupied[..., None], residual, value)
    if not linearize:
        return residual, stress
    opening = (argument > 0.) & occupied
    sliding = sliding & occupied
    ratio = torch.where(sliding, radius/safe, torch.zeros_like(radius))
    derivative = torch.where(sliding[..., None, None],
                             (1.-ratio)[..., None, None]*projector
                             +ratio[..., None, None]*direction[..., :, None]*direction[..., None, :],
                             torch.zeros_like(projector))
    diagonal = torch.zeros((*weights.shape, 4, 4), dtype=strain.dtype, device=strain.device)
    diagonal[..., 0, 0] = 1.-damage[None]*opening
    diagonal[..., 1:, 1:] = torch.eye(3, dtype=strain.dtype, device=strain.device)-damage[None, :, None, None]*derivative
    coupling = torch.zeros((*weights.shape, 4, 6), dtype=strain.dtype, device=strain.device)
    coupling[..., 0, :] = -(opening[..., None]*maps[..., 0, :])/modulus
    coupling[..., 1:, :] = -(derivative @ maps[..., 1:, :])/shear
    pressure_active = sliding & (local[..., 0] < 0.)
    coupling[..., 1:, :] -= (friction/shear*pressure_active[..., None, None]
                             *direction[..., :, None]*maps[..., None, 0, :])
    reduction = weights[..., None, None]*(matrix @ maps.transpose(-1, -2))
    return residual, stress, diagonal, coupling, reduction


def family_dot(left, right):
    return (left*right).sum((0, 2))


def gmres(operator, rhs, precondition, tolerance=1e-8, max_iterations=40):
    """Batched right-preconditioned GMRES, one independent solve per cell."""
    full_scale = family_dot(rhs, rhs).sqrt()
    cells = torch.where(full_scale > 0.)[0]
    solution = torch.zeros_like(rhs)
    solved = full_scale == 0.
    if not len(cells):
        return solution, solved
    scale = full_scale[cells]
    vectors = [rhs[:, cells]/scale[None, :, None]]
    basis = []
    hessenberg = rhs.new_zeros((len(cells), max_iterations+1, max_iterations))
    cosines, sines = [], []
    transformed = rhs.new_zeros((len(cells), max_iterations+1))
    transformed[:, 0] = scale
    for j in range(max_iterations):
        basis.append(precondition(vectors[j], cells))
        following = operator(basis[-1], cells)
        # Reorthogonalization matters for the semidefinite complete-damage limit.
        for _ in range(2):
            for i in range(j+1):
                coefficient = family_dot(vectors[i], following)
                hessenberg[:, i, j] += coefficient
                following -= coefficient[None, :, None]*vectors[i]
        norm = family_dot(following, following).sqrt()
        hessenberg[:, j+1, j] = norm
        vectors.append(following/norm.clamp_min(torch.finfo(rhs.dtype).tiny)[None, :, None])
        for i in range(j):
            first, second = hessenberg[:, i, j].clone(), hessenberg[:, i+1, j].clone()
            hessenberg[:, i, j] = cosines[i]*first+sines[i]*second
            hessenberg[:, i+1, j] = -sines[i]*first+cosines[i]*second
        first, second = hessenberg[:, j, j].clone(), hessenberg[:, j+1, j].clone()
        length = torch.hypot(first, second)
        active = length > 0.
        denominator = torch.where(active, length, torch.ones_like(length))
        cosine = torch.where(active, first/denominator, torch.ones_like(length))
        sine = second/denominator
        cosines.append(cosine)
        sines.append(sine)
        hessenberg[:, j, j] = length
        hessenberg[:, j+1, j] = 0.
        transformed[:, j+1] = -sine*transformed[:, j]
        transformed[:, j] *= cosine
        converged = transformed[:, j+1].abs() <= tolerance*scale
        if bool(converged.any()) or j+1 == max_iterations:
            triangular = hessenberg[:, :j+1, :j+1].clone()
            diagonal = triangular.diagonal(dim1=-2, dim2=-1)
            diagonal[diagonal == 0.] = 1.
            coefficients = torch.linalg.solve_triangular(
                triangular, transformed[:, :j+1, None], upper=True)[..., 0]
            candidate = sum(coefficients[:, i][None, :, None]*basis[i] for i in range(j+1))
            difference = operator(candidate, cells)-rhs[:, cells]
            error = family_dot(difference, difference).sqrt()
            complete = torch.isfinite(error) & (error <= (10*tolerance)*scale+1e-14)
            solution[:, cells] = candidate
            solved[cells] = complete
            keep = ~complete
            if not bool(keep.any()):
                return solution, solved
            # Finished cells leave the Arnoldi basis, rather than continuing
            # reductions and normalization of an already exhausted subspace.
            cells, scale = cells[keep], scale[keep]
            hessenberg, transformed = hessenberg[keep], transformed[keep]
            vectors = [v[:, keep] for v in vectors]
            basis = [v[:, keep] for v in basis]
            cosines = [v[keep] for v in cosines]
            sines = [v[keep] for v in sines]
    return solution, solved


def linear_solver(diagonal, coupling, reduction, transpose=False):
    # The shift regularizes the preconditioner only. GMRES uses the unshifted J.
    shift = 1e-5
    shifted_shear = diagonal[..., 1:, 1:]+shift*torch.eye(3, dtype=diagonal.dtype, device=diagonal.device)
    first, second, third = shifted_shear.unbind(-2)
    cofactors = torch.stack((torch.linalg.cross(second, third), torch.linalg.cross(third, first),
                             torch.linalg.cross(first, second)), -2)
    determinant = (first*cofactors[..., 0, :]).sum(-1)
    inverse = torch.zeros_like(diagonal)
    inverse[..., 0, 0] = 1./(diagonal[..., 0, 0]+shift)
    inverse[..., 1:, 1:] = cofactors.transpose(-1, -2)/determinant[..., None, None]
    left = inverse @ coupling
    right = reduction @ inverse
    schur = torch.eye(6, dtype=diagonal.dtype, device=diagonal.device)-(reduction @ left).sum(0)
    def operator(value, cells=None):
        local_diagonal = diagonal if cells is None else diagonal[:, cells]
        local_coupling = coupling if cells is None else coupling[:, cells]
        local_reduction = reduction if cells is None else reduction[:, cells]
        if transpose:
            aggregate = torch.einsum("feij,fei->ej", local_coupling, value)
            correction = torch.einsum("feij,ei->fej", local_reduction, aggregate)
        else:
            aggregate = torch.einsum("feij,fej->ei", local_reduction, value)
            correction = torch.einsum("feij,ej->fei", local_coupling, aggregate)
        return (local_diagonal @ value[..., None])[..., 0]-correction
    def precondition(value, cells=None):
        local_inverse = inverse if cells is None else inverse[:, cells]
        local_left = left if cells is None else left[:, cells]
        local_right = right if cells is None else right[:, cells]
        local_schur = schur if cells is None else schur[cells]
        base = (local_inverse @ value[..., None])[..., 0]
        if transpose:
            aggregate = torch.einsum("feij,fei->ej", local_left, value)
            solved = torch.linalg.solve(local_schur.transpose(-1, -2), aggregate[..., None])[..., 0]
            return base+torch.einsum("feij,ei->fej", local_right, solved)
        aggregate = torch.einsum("feij,fej->ei", local_right, value)
        solved = torch.linalg.solve(local_schur, aggregate[..., None])[..., 0]
        return base+torch.einsum("feij,ej->fei", local_left, solved)
    return operator, precondition


def stress_equation(stress, remaining, strain, normals, weights, previous, matrix, friction, maps,
                    linearize=False):
    """Six shared-stress equations after positive-hardening elimination."""
    compliance = torch.linalg.inv(matrix)
    traction = torch.einsum("feij,ej->fei", maps, stress)
    trial = traction[..., 1:]/matrix[3, 3]-remaining[None, :, None]*previous[..., 1:]
    length = torch.linalg.vector_norm(trial, dim=-1)
    radius = friction*(-traction[..., 0]).clamp_min(0.)/matrix[3, 3]
    active = length > radius
    safe = torch.where(active, length, torch.ones_like(length))
    shear = remaining[None, :, None]*previous[..., 1:]+torch.where(
        active[..., None], (1.-radius/safe)[..., None]*trial, torch.zeros_like(trial))
    scaled = torch.cat(((traction[..., 0].clamp_min(0.)/matrix[0, 0])[..., None], shear), -1)
    residual = (remaining[:, None]*(stress @ compliance.T-strain)
                +torch.einsum("fe,fei,feij->ej", weights, scaled, maps))
    if not linearize:
        return residual, scaled
    direction = trial/safe[..., None]
    ratio = radius/safe
    projector = torch.eye(3, dtype=stress.dtype, device=stress.device)-normals[..., :, None]*normals[..., None, :]
    derivative = torch.where(active[..., None, None],
        (1.-ratio)[..., None, None]*projector
        +ratio[..., None, None]*direction[..., :, None]*direction[..., None, :], torch.zeros_like(projector))
    tangent = torch.empty_like(maps)
    tangent[..., 0, :] = (traction[..., 0] > 0.)[..., None]*maps[..., 0, :]/matrix[0, 0]
    tangent[..., 1:, :] = (derivative @ maps[..., 1:, :])/matrix[3, 3]
    tangent[..., 1:, :] += (friction/matrix[3, 3]*(active & (traction[..., 0] < 0.))[..., None, None]
                           *direction[..., :, None]*maps[..., None, 0, :])
    jacobian = remaining[:, None, None]*compliance+(weights[..., None, None]
                                                  *(maps.transpose(-1, -2) @ tangent)).sum(0)
    return residual, scaled, jacobian


def stress_newton(value, arguments):
    """Solve positive-hardening cells in six dimensions before primal refinement."""
    strain, normals, weights, previous, damage, matrix, friction = arguments
    cells = torch.where(damage < 1.)[0]
    if not len(cells):
        return value
    strain, normals, weights, previous = strain[cells], normals[:, cells], weights[:, cells], previous[:, cells]
    remaining = 1.-damage[cells]
    maps = crack_maps(normals)
    stress = (strain-torch.einsum("fe,fei,feij->ej", weights, previous, maps)) @ matrix.T
    for _ in range(20):
        residual, scaled, jacobian = stress_equation(stress, remaining, strain, normals, weights,
                                                     previous, matrix, friction, maps, True)
        done = residual.abs().amax(-1)*matrix[0, 0]/remaining < 1e-8
        if bool(done.any()):
            recovered = scaled[:, done]/remaining[None, done, None]
            value[:, cells[done]] = torch.where(weights[:, done, None] > 0., recovered, 0.)
        active = ~done
        if not bool(active.any()):
            break
        step, info = torch.linalg.solve_ex(jacobian, -residual[..., None])
        step = step[..., 0]
        active &= info == 0
        fraction = torch.ones_like(remaining)
        accepted = ~active
        candidate = stress.clone()
        merit = residual.square().sum(-1)
        for _ in range(20):
            trial = stress+fraction[:, None]*step
            trial_residual = stress_equation(trial, remaining, strain, normals, weights,
                                             previous, matrix, friction, maps)[0]
            improved = trial_residual.square().sum(-1) < merit
            candidate = torch.where((improved & ~accepted)[:, None], trial, candidate)
            accepted |= improved
            if bool(accepted.all()):
                break
            fraction = torch.where(accepted, fraction, fraction*.5)
        keep = active & accepted
        if not bool(keep.any()):
            break
        cells, remaining, strain, stress = cells[keep], remaining[keep], strain[keep], candidate[keep]
        normals, weights, previous, maps = normals[:, keep], weights[:, keep], previous[:, keep], maps[:, keep]
    return value


def hybrid_refine(value, arguments, cells):
    """Trust-region refinement for contact-cone changes that stall Newton.

    With positive hardening, eliminate crack strains and solve six common
    stresses. Complete damage retains the primal equations. Every reconstructed
    state is checked in those equations before being accepted.
    """
    strain, normals, weights, previous, damage, matrix, friction = arguments
    for cell in cells.tolist():
        occupied = weights[:, cell] > 0.
        local = tuple(item.detach().cpu() for item in (
            strain[cell:cell+1], normals[occupied, cell:cell+1], weights[occupied, cell:cell+1],
            previous[occupied, cell:cell+1], damage[cell:cell+1], matrix, friction))
        shape = local[3].shape
        def equation(vector):
            return return_equation(torch.as_tensor(vector).reshape(shape), *local)[0].flatten().numpy()
        def jacobian(vector):
            _, _, diagonal, coupling, reduction = return_equation(
                torch.as_tensor(vector).reshape(shape), *local, linearize=True)
            return (block_diag(*diagonal[:, 0].numpy())
                    -coupling[:, 0].reshape(-1, 6).numpy()
                    @ reduction[:, 0].permute(1, 0, 2).reshape(6, -1).numpy())
        initial = value[occupied, cell].detach().cpu().numpy().ravel()
        error = abs(equation(initial)).max()*float(matrix[0, 0])
        if error > 1e-7 and float(local[4][0]) < 1.:
            local_strain, local_normals, local_weights, local_previous, local_damage, local_matrix, local_friction = local
            maps = crack_maps(local_normals)
            remaining = 1.-local_damage
            def dual(vector, linearize=False):
                return stress_equation(torch.as_tensor(vector).reshape(1, 6), remaining,
                    local_strain, local_normals, local_weights, local_previous,
                    local_matrix, local_friction, maps, linearize)
            # Near complete damage, the eliminated strain residual can be
            # tiny while the reconstructed primal return is still inaccurate.
            # Give MINPACK residuals and Jacobians the same stress units used
            # by the actual return check; the constitutive root is unchanged.
            def stress_residual(vector):
                return dual(vector)[0].flatten().numpy()*float(local_matrix[0, 0]/remaining[0])
            def stress_jacobian(vector):
                return dual(vector, True)[2][0].numpy()*float(local_matrix[0, 0]/remaining[0])
            stress = return_equation(torch.as_tensor(initial).reshape(shape), *local)[1]
            elastic = local_strain @ local_matrix.T
            pressure = ((elastic[:, :3].square().sum()+2*elastic[:, 3:].square().sum())/3).sqrt()
            closed = torch.zeros_like(stress)
            closed[:, :3] = -pressure
            # The closed compressive cone can be disconnected from an opening
            # trial's Newton basin. Its seed has the elastic stress tensor norm.
            for seed in (stress, elastic, torch.zeros_like(stress), closed):
                for method in ("hybr", "lm"):
                    solved = root(stress_residual, seed.flatten().numpy(), jac=stress_jacobian,
                                  method=method, options={"xtol": 1e-11})
                    recovered = (dual(solved.x)[1]/remaining[None, :, None]).flatten().numpy()
                    recovered_error = abs(equation(recovered)).max()*float(matrix[0, 0])
                    if recovered_error < error:
                        initial, error = recovered, recovered_error
                    if error <= 1e-7:
                        break
                if error <= 1e-7:
                    break
            if error > 1e-7:
                # A frozen-equilibrium trial can cross several contact cones.
                # Stronger springs initialize the target root; intermediate
                # roots are never committed as material history.
                target_remaining = remaining.clone()
                remaining = torch.ones_like(remaining)
                seed = local_strain @ local_matrix.T
                while True:
                    solved = root(stress_residual, seed.flatten().numpy(), jac=stress_jacobian,
                                  method="hybr", options={"xtol": 1e-11})
                    seed = torch.as_tensor(solved.x).reshape(1, 6)
                    if bool((remaining == target_remaining).all()):
                        break
                    remaining = (remaining*.35).maximum(target_remaining)
                recovered = (dual(solved.x)[1]/remaining[None, :, None]).flatten().numpy()
                recovered_error = abs(equation(recovered)).max()*float(matrix[0, 0])
                if recovered_error < error:
                    initial, error = recovered, recovered_error
        if error > 1e-7:
            solved = root(equation, initial, jac=jacobian, method="hybr", options={"xtol": 1e-10})
            initial = solved.x
            error = abs(equation(initial)).max()*float(matrix[0, 0])
        if error > 1e-7 and float(local[4][0]) < 1.:
            # Primal refinement can enter the correct contact cone without
            # converging all family strains. Its actual shared stress then
            # initializes a residual least-squares trust region in six stresses.
            # Acceptance still uses the original family return equations.
            stress = return_equation(torch.as_tensor(initial).reshape(shape), *local)[1]
            solved = least_squares(stress_residual, stress.flatten().numpy(), jac=stress_jacobian,
                                   x_scale="jac", xtol=1e-11, ftol=1e-11, gtol=1e-11)
            recovered = (dual(solved.x)[1]/remaining[None, :, None]).flatten().numpy()
            recovered_error = abs(equation(recovered)).max()*float(matrix[0, 0])
            if recovered_error < error:
                initial, error = recovered, recovered_error
        if error > 1e-7 and float(local[4][0]) < 1.:
            # Direct stress seeds can stall across frictional contact cones.
            # Follow roots from zero friction at the same strain,
            # damage and incoming history; only the target-friction root is
            # eligible for the original return-equation check below.
            target_friction = local_friction
            local_friction = target_friction*0.
            solved = root(stress_residual, np.zeros(6), jac=stress_jacobian,
                          method="hybr", options={"xtol": 1e-11})
            seed = solved.x
            fraction, step = 0., .1
            if np.linalg.norm(stress_residual(seed), ord=np.inf) <= 1e-7:
                while fraction < 1. and step > np.finfo(float).eps:
                    trial_fraction = min(1., fraction+step)
                    local_friction = target_friction*trial_fraction
                    solved = root(stress_residual, seed, jac=stress_jacobian,
                                  method="hybr", options={"xtol": 1e-11})
                    if np.linalg.norm(stress_residual(solved.x), ord=np.inf) <= 1e-7:
                        seed, fraction = solved.x, trial_fraction
                        step = min(.1, step*1.5)
                    else:
                        step *= .5
            local_friction = target_friction
            if fraction == 1.:
                recovered = (dual(seed)[1]/remaining[None, :, None]).flatten().numpy()
                recovered_error = abs(equation(recovered)).max()*float(matrix[0, 0])
                if recovered_error < error:
                    initial, error = recovered, recovered_error
        if error > 1e-7:
            packet = dict(zip(("strain", "normals", "weights", "previous", "damage", "matrix", "friction"), local))
            packet.update(value=torch.as_tensor(initial.reshape(shape)).clone(), cell_index=cell)
            raise CrackReturnError(
                f"Coupled-crack trust-region return failed in cell {cell}: {error:g} MPa; {solved.message}", packet)
        value[occupied, cell] = torch.as_tensor(initial.reshape(-1, 4), device=value.device)
    return value


class CoupledReturn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, strain, normals, weights, previous, damage, matrix, friction):
        value = previous.clone()
        arguments = (strain, normals, weights, previous, damage, matrix, friction)
        modulus = max(float(matrix[0, 0]), float(matrix[3, 3]))
        value = stress_newton(value, arguments)
        initial_error = return_equation(value, *arguments)[0].abs().amax((0, 2))*modulus
        cells = torch.where(initial_error >= 1e-8)[0]
        for _ in range(20):
            if not len(cells):
                break
            current = value[:, cells]
            local_arguments = (strain[cells], normals[:, cells], weights[:, cells], previous[:, cells],
                               damage[cells], matrix, friction)
            residual, _, diagonal, coupling, reduction = return_equation(current, *local_arguments, linearize=True)
            error = residual.abs().amax((0, 2))*modulus
            if float(error.max()) < 1e-8:
                break
            operator, precondition = linear_solver(diagonal, coupling, reduction)
            increment, linear_solved = gmres(operator, -residual, precondition, tolerance=1e-5)
            if not bool(linear_solved.all()):
                current = hybrid_refine(current, local_arguments, torch.where(~linear_solved)[0])
                value[:, cells] = current
                cells = cells[linear_solved]
                if not len(cells):
                    break
                continue
            merit = family_dot(residual, residual)
            fraction = torch.ones_like(merit)
            finished = error < 1e-8
            candidate = current.clone()
            for _ in range(22):
                trial = current+fraction[None, :, None]*increment
                trial_residual = return_equation(trial, *local_arguments)[0]
                accepted = (family_dot(trial_residual, trial_residual) < merit) | finished
                candidate = torch.where((accepted & ~finished)[None, :, None], trial, candidate)
                finished |= accepted
                if bool(finished.all()):
                    break
                fraction = torch.where(finished, fraction, fraction*.5)
            if not bool(finished.all()):
                # At a changing contact cone, a generalized Newton step need
                # not descend the merit. Use its actual gradient to globalize.
                aggregate = torch.einsum("feij,fei->ej", coupling, residual)
                gradient = ((diagonal @ residual[..., None])[..., 0]
                            -torch.einsum("feij,ei->fej", reduction, aggregate))
                tangent = operator(gradient)
                fraction = family_dot(gradient, gradient)/family_dot(tangent, tangent).clamp_min(1e-30)
                for _ in range(40):
                    trial = current-fraction[None, :, None]*gradient
                    trial_residual = return_equation(trial, *local_arguments)[0]
                    accepted = (family_dot(trial_residual, trial_residual) < merit) | finished
                    candidate = torch.where((accepted & ~finished)[None, :, None], trial, candidate)
                    finished |= accepted
                    if bool(finished.all()):
                        break
                    fraction = torch.where(finished, fraction, fraction*.5)
                if not bool(finished.all()):
                    candidate = hybrid_refine(candidate, local_arguments, torch.where(~finished)[0])
            value[:, cells] = candidate
            cells = cells[error >= 1e-8]
        error = return_equation(value, *arguments)[0].abs().amax((0, 2))*modulus
        if bool((error > 1e-7).any()):
            value = hybrid_refine(value, arguments, torch.where(error > 1e-7)[0])
        ctx.save_for_backward(value, *arguments)
        return value

    @staticmethod
    def backward(ctx, gradient):
        value, *arguments = ctx.saved_tensors
        with torch.no_grad():
            _, _, diagonal, coupling, reduction = return_equation(value, *arguments, linearize=True)
            operator, precondition = linear_solver(diagonal, coupling, reduction, transpose=True)
            adjoint, solved = gmres(operator, gradient, precondition)
            for cell in torch.where(~solved)[0].tolist():
                full = (torch.block_diag(*diagonal[:, cell].cpu().unbind(0))
                        -coupling[:, cell].cpu().reshape(-1, 6)
                        @ reduction[:, cell].cpu().permute(1, 0, 2).reshape(6, -1))
                target = gradient[:, cell].cpu().flatten()
                exact = torch.linalg.lstsq(full.T, target, driver="gelsd").solution
                if not torch.allclose(full.T @ exact, target, rtol=1e-7, atol=1e-10):
                    raise RuntimeError("Coupled-crack history adjoint is not solvable at the returned state")
                adjoint[:, cell] = exact.reshape_as(adjoint[:, cell]).to(adjoint.device)
        with torch.enable_grad():
            differentiable = [item.detach().requires_grad_(True) for item in arguments]
            residual = return_equation(value.detach(), *differentiable)[0]
            derivatives = torch.autograd.grad(residual, differentiable, -adjoint, allow_unused=True)
        return tuple(derivatives)


def coupled_family_update(strain, state, damage, previous_damage, matrix, friction,
                          direction_tensor, quadrature=None):
    normals, previous, weights = state
    increment = damage-previous_damage
    if quadrature is None:
        normal = torch.zeros_like(strain[:, :3])
        growing = increment > 0.
        if bool(growing.any()):
            normal[growing] = MaximumPrincipalDirection.apply(direction_tensor[growing])
        born_normals, born_weights = normal[None], increment[None]
    else:
        born_normals, born_weights = tensor_direction_families(direction_tensor, increment, *quadrature)
    normals = torch.cat((normals, born_normals))
    previous = torch.cat((previous, strain.new_zeros((*born_weights.shape, 4))))
    weights = torch.cat((weights, born_weights))
    value = CoupledReturn.apply(strain, normals, weights, previous, damage, matrix,
                                torch.as_tensor(friction, dtype=strain.dtype, device=strain.device))
    maps = crack_maps(normals)
    crack_strain = torch.einsum("fe,fei,feij->ej", weights, value, maps)
    stress = (strain-crack_strain) @ matrix.T
    plastic = torch.einsum("fe,fei,feij->ej", weights, value[..., 1:], maps[..., 1:, :])
    # These reported checks are not objective terms. Keep their unused graphs
    # out of the differentiable history; stress and state retain all gradients.
    with torch.no_grad():
        residual = return_equation(value, strain, normals, weights, previous, damage, matrix, friction)[0]
        traction = torch.einsum("fej,ej->fe", maps[..., 0, :], stress)
        friction_work = (weights*friction*(-traction).clamp_min(0.)
                         *torch.linalg.vector_norm(value[..., 1:]-previous[..., 1:], dim=-1)).sum(0)
        old_spring = matrix[0, 0]*previous[..., 0].square()+matrix[3, 3]*previous[..., 1:].square().sum(-1)
        damage_work = .5*increment*(weights*old_spring).sum(0)
        fabric = torch.einsum("fe,fei,fej->eij", weights, normals, normals)
    return stress, plastic, (normals, value, weights), {
        "consistency_MPa": residual.abs().amax((0, 2))*matrix[0, 0],
        "dissipation_MPa": damage_work+friction_work,
        "crack_fabric": fabric,
    }
