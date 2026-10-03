"""Conforming P1 neural trial, elastic Schur complement and material replay.

Network values at band nodes define a continuous piecewise-affine trial field.
The same trace is used on both sides of each interface triangle, including its
interior.  Every band nodal test, including nonzero interface traces, enters
weak equilibrium.  All operators use N, mm and MPa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu, lgmres, LinearOperator
from scipy.optimize import root, KrylovJacobian, OptimizeResult
import torch
from skfem import asm
from skfem.models.elasticity import linear_elasticity, lame_parameters

from band_material import averaging_matrix
from process_material import elastic_stress
from coupled_cracks import CrackReturnError, crack_maps, stress_equation


class CondensedBand:
    def __init__(self, model, process_radius_mm=0.):
        self.model = model
        self.process_radius_mm = float(process_radius_mm)
        nonlinear = model.is_band.copy()
        if process_radius_mm > 0:
            centers = model.mesh.p[:, model.mesh.t].mean(axis=1).T
            local = centers-np.asarray(model.geometry.center_mm)
            s, n = local@model.geometry.tangent, local@model.geometry.normal
            distance = np.sqrt((np.abs(s)-model.geometry.band_length_mm/2)**2+n**2)
            nonlinear |= distance < process_radius_mm
        self.cells = model.mesh.t[:, nonlinear].T
        self.is_band = model.is_band[nonlinear]
        self.nodes = np.unique(self.cells)
        node_dofs = model.basis.nodal_dofs[:, self.nodes].T.ravel()
        self.free_node_dofs = np.flatnonzero(~np.isin(node_dofs, model.constrained))
        self.dofs = node_dofs[self.free_node_dofs]
        host_nodes = np.unique(model.mesh.t[:, ~nonlinear])
        self.interface_nodes = np.intersect1d(self.nodes, host_nodes)
        self.interface = np.flatnonzero(np.isin(self.nodes, host_nodes))
        host_free = np.intersect1d(model.free, model.basis.nodal_dofs[:, host_nodes].ravel())
        self.internal = np.setdiff1d(host_free, self.dofs)
        K = model.K_host
        if process_radius_mm > 0:
            outside = model.basis.with_elements(np.flatnonzero(~nonlinear))
            K = asm(linear_elasticity(*lame_parameters(model.host_young_MPa, model.host_poisson)), outside).tocsr()
        self.Kib = K[self.internal][:, self.dofs].tocsc()
        if len(self.internal):
            self.lu = splu(K[self.internal][:, self.internal].tocsc())
            self.reconstruction = -self.lu.solve(self.Kib.toarray())
            self.load_internal = self.lu.solve(model.unit_force[self.internal])
        else:
            self.reconstruction = np.empty((0, len(self.dofs)))
            self.load_internal = np.empty(0)
        self.load_compliance = float(model.unit_force[self.internal] @ self.load_internal)
        self.S = K[self.dofs][:, self.dofs].toarray() + self.Kib.T @ self.reconstruction
        self.S = .5 * (self.S + self.S.T)
        self.g = model.unit_force[self.dofs] - self.Kib.T @ self.load_internal
        self.local_cells = np.searchsorted(self.nodes, self.cells)
        xyz = model.mesh.p.T[self.cells]
        affine = np.concatenate((np.ones((len(xyz), 4, 1)), xyz), axis=2)
        self.gradients = np.linalg.inv(affine)[:, 1:, :].transpose(0, 2, 1)
        self.volumes = np.abs(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) / 6.
        self.centers = xyz.mean(axis=1)
        # Rows of Q map global vectors into the local s,n,z coordinates.
        self.Q = np.stack((model.geometry.tangent, model.geometry.normal, [0., 0., 1.]))
        grad_local = self.gradients @ self.Q.T
        self.B = np.zeros((len(xyz), 6, 12))
        for a in range(4):
            for j in range(3):
                v = self.Q[:, j]
                d = grad_local[:, a]
                self.B[:, :3, 3*a+j] = d * v
                self.B[:, 3, 3*a+j] = v[0]*d[:, 1] + v[1]*d[:, 0]
                self.B[:, 4, 3*a+j] = v[1]*d[:, 2] + v[2]*d[:, 1]
                self.B[:, 5, 3*a+j] = v[0]*d[:, 2] + v[2]*d[:, 0]
        self.cell_dofs = (3*self.local_cells[..., None] + np.arange(3)).reshape(-1, 12)
        free_index = np.full(len(node_dofs), -1, dtype=int)
        free_index[self.free_node_dofs] = np.arange(len(self.dofs))
        self.cell_dofs = free_index[self.cell_dofs]
        # Essential values are zero. Zero their strain columns before using a
        # valid scatter index; they add no force or neural degree of freedom.
        constrained = self.cell_dofs < 0
        self.B *= (~constrained)[:, None, :]
        self.cell_dofs[constrained] = 0
        rows = np.repeat(np.arange(6*len(xyz)), 12)
        cols = np.repeat(self.cell_dofs[:, None, :], 6, axis=1).ravel()
        self.strain_operator = coo_matrix((self.B.ravel(), (rows, cols)),
                                         shape=(6*len(xyz), len(self.dofs))).tocsr()

    def elastic_stiffness(self, C, host_scale=1.):
        matrices = np.broadcast_to(C, (len(self.cells), 6, 6))
        element = np.einsum("eai,eab,ebj,e->eij", self.B, matrices, self.B, self.volumes)
        rows = np.broadcast_to(self.cell_dofs[:, :, None], element.shape).ravel()
        cols = np.broadcast_to(self.cell_dofs[:, None, :], element.shape).ravel()
        return host_scale*self.S + coo_matrix((element.ravel(), (rows, cols)),
                                  shape=self.S.shape).toarray()

    def reconstruct(self, band, force, host_scale=1.):
        out = np.zeros(self.model.basis.N)
        out[self.dofs] = band
        out[self.internal] = self.reconstruction @ band + force * self.load_internal / host_scale
        return out

    def end_shortening(self, band, force, host_scale=1.):
        """Area-mean top compression, conjugate to the uniform traction load."""
        return np.asarray(band) @ self.g + np.asarray(force)*self.load_compliance/host_scale

    def observer(self, surface_operator):
        H = surface_operator.matrix
        return (H[:, self.dofs].toarray() + H[:, self.internal] @ self.reconstruction,
                np.asarray(H[:, self.internal] @ self.load_internal))

    def tensors(self, material, device="cpu"):
        return TorchBand(self, material, device)


class TorchBand:
    def __init__(self, core, material, device):
        self.core = core
        dtype = next(material.parameters()).dtype
        tensor = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        self.S, self.g = tensor(core.S), tensor(core.g)
        self.B, self.volumes = tensor(core.B), tensor(core.volumes)
        self.cells = torch.as_tensor(core.cell_dofs, device=device)
        if core.process_radius_mm > 0:
            self.average = tuple(tensor(averaging_matrix(core.centers[mask], core.volumes[mask], radius))
                                 for mask, radius in ((core.is_band, material.band.radius_mm),
                                                      (~core.is_band, material.radius_mm)))
        else:
            self.average = tensor(averaging_matrix(core.centers, core.volumes, material.radius_mm))

    def strain(self, u):
        return torch.einsum("eij,...ej->...ei", self.B, u[..., self.cells])

    def internal_force(self, stress):
        element = torch.einsum("eij,...ei,e->...ej", self.B, stress, self.volumes)
        shape = (*stress.shape[:-2], self.S.shape[0])
        out = torch.zeros(shape, dtype=stress.dtype, device=stress.device)
        return out.index_add(-1, self.cells.ravel(), element.flatten(-2))

    def residual(self, u, stress, forces, host_scale=1.):
        return host_scale*(u @ self.S.T) + self.internal_force(stress) - forces[..., None] * self.g


class EquilibriumProjection:
    """Project a neural trial pair onto every condensed weak equilibrium test.

    A fixed positive elastic metric supplies the correction. It is independent
    of the learned history law and is not a nonlinear material equilibrium solve.
    Both the displacement trace and element stress are corrected together.
    """
    def __init__(self, core, band, reference_matrix, host_scale=1.):
        self.band = band
        self.C = reference_matrix.detach()
        self.local_stiffness = torch.as_tensor(
            core.elastic_stiffness(self.C.cpu().numpy(), 0.),
            dtype=self.C.dtype, device=self.C.device)
        self.host_scale = float(host_scale)
        self.has_exterior = bool(np.any(core.S))
        self.factor = torch.linalg.cholesky(self.local_stiffness + self.host_scale*band.S)

    def __call__(self, u, stress, forces, host_scale=1.):
        residual = self.band.residual(u, stress, forces, host_scale)
        if not self.has_exterior:
            factor = self.factor
        elif isinstance(host_scale, torch.Tensor) and host_scale.requires_grad:
            factor = torch.linalg.cholesky(self.local_stiffness + host_scale*self.band.S)
        elif float(host_scale) == self.host_scale:
            factor = self.factor
        else:
            factor = torch.linalg.cholesky(self.local_stiffness + host_scale*self.band.S)
        correction = -torch.cholesky_solve(residual.T, factor).T
        return u+correction, stress+elastic_stress(self.band.strain(correction), self.C)

    def optimal_stress(self, material_stress, forces):
        """Minimize stress compatibility at fixed full-domain displacement.

        The reference compliance is fixed. This linear elimination of the
        admissible auxiliary stress leaves the nonlinear displacement and
        history to the neural optimizer.
        """
        if self.has_exterior:
            raise ValueError("Stress elimination requires the full nonlinear domain")
        residual = self.band.internal_force(material_stress)-forces[:, None]*self.band.g
        correction = -torch.cholesky_solve(residual.T, self.factor).T
        return material_stress+elastic_stress(self.band.strain(correction), self.C)


class ElasticReferenceSolve(torch.autograd.Function):
    """Current-material loaded reference with an implicit elastic derivative.

    A sparse factorization handles this single initial state. Subsequent states
    remain neural unknowns. The caller must verify that the returned reference
    does not activate plasticity or damage in the actual material update.
    """
    @staticmethod
    def forward(ctx, matrices, host_scale, force, core):
        C = np.broadcast_to(matrices.detach().cpu().numpy(), (len(core.cells), 6, 6))
        element = np.einsum("eai,eab,ebj,e->eij", core.B, C, core.B, core.volumes)
        rows = np.broadcast_to(core.cell_dofs[:, :, None], element.shape).ravel()
        cols = np.broadcast_to(core.cell_dofs[:, None, :], element.shape).ravel()
        stiffness = coo_matrix((element.ravel(), (rows, cols)), shape=core.S.shape).tocsc()
        stiffness = stiffness + float(host_scale.detach().cpu())*csc_matrix(core.S)
        ctx.factor = splu(stiffness)
        displacement = ctx.factor.solve(float(force.detach().cpu())*core.g)
        ctx.core, ctx.displacement = core, displacement
        ctx.save_for_backward(matrices)
        return torch.as_tensor(displacement, dtype=matrices.dtype, device=matrices.device)

    @staticmethod
    def backward(ctx, gradient):
        matrices, = ctx.saved_tensors
        core, displacement = ctx.core, ctx.displacement
        adjoint = ctx.factor.solve(gradient.detach().cpu().numpy(), trans="T")
        strain = np.einsum("eij,ej->ei", core.B, displacement[core.cell_dofs])
        adjoint_strain = np.einsum("eij,ej->ei", core.B, adjoint[core.cell_dofs])
        derivative = -core.volumes[:, None, None]*adjoint_strain[:, :, None]*strain[:, None, :]
        if matrices.ndim == 2:
            derivative = derivative.sum(0)
        tensor = lambda value: torch.as_tensor(value, dtype=matrices.dtype, device=matrices.device)
        return (tensor(derivative), tensor(-adjoint @ core.S @ displacement),
                tensor(adjoint @ core.g), None)


class FrozenMaterialPreconditioner(LinearOperator):
    """Sparse current-contact tangent, holding damage and birth geometry fixed.

    Only this approximate inverse regularizes vanishing crack springs. The
    nonlinear residual continues to use the exported material without a floor.
    """
    def __init__(self, core, material, band, previous, host_scale, reference,
                 control_vector=None, control_load=0., force_scale=1., previous_strain=None):
        self.core, self.material, self.band, self.previous = core, material, band, previous
        self.host_scale, self.reference = host_scale, reference
        self.previous_strain = (band.strain(torch.zeros(len(core.dofs), dtype=band.B.dtype, device=band.B.device))
                                if previous_strain is None else previous_strain)
        self.control_vector, self.control_load, self.force_scale = control_vector, control_load, force_scale
        size = len(core.dofs)+(control_vector is not None)
        super().__init__(dtype=np.dtype(float), shape=(size, size))

    def setup(self, value, residual, function):
        self.update(value, residual)

    def update(self, value, residual):
        displacement = value if self.control_vector is None else value[:-1]
        material, band = self.material, self.band
        with torch.no_grad():
            strain = band.strain(torch.as_tensor(displacement, dtype=band.B.dtype, device=band.B.device))
            incoming = self.previous
            def remember(previous):
                nonlocal incoming
                incoming = previous
            stress, state, _ = material.integrate_interval(
                self.previous_strain, strain, self.previous, band.average, 36, self.host_scale,
                on_substep=remember)
            host = ~material.is_band
            normals, cracks, weights = state[2:]
            previous = torch.cat((incoming[3], torch.zeros_like(cracks[len(incoming[3]):])))
            remaining = (1.-material.damage(state[1][host])).clamp_min(1e-3)
            matrix = material.host_matrix(self.host_scale)
            jacobian = stress_equation(stress[host], remaining, strain[host], normals, weights,
                                       previous, matrix, material.log_host_friction.exp(),
                                       crack_maps(normals), True)[2]
            tangent = torch.linalg.solve(jacobian, remaining[:, None, None]*torch.eye(
                6, dtype=matrix.dtype, device=matrix.device))
            matrices = material.elastic_matrix(self.host_scale).cpu().numpy().copy()
            matrices[host.cpu().numpy()] = tangent.cpu().numpy()
        stiffness = self.core.elastic_stiffness(matrices, self.host_scale)
        self.factor = splu(csc_matrix(stiffness))
        if self.control_vector is not None:
            self.unit = self.factor.solve(self.core.g)
            self.compliance = self.control_vector @ self.unit+self.control_load

    def _matvec(self, value):
        if self.control_vector is None:
            return self.factor.solve(self.reference @ value)
        force = value[-1]/self.force_scale
        raw = self.reference @ value[:-1]-self.core.g*force
        constraint = self.control_vector @ value[:-1]+self.control_load*force
        displacement = self.factor.solve(raw)
        force = (constraint-self.control_vector @ displacement)/self.compliance
        return np.r_[displacement+self.unit*force, self.force_scale*force]


def contact_equilibrium_root(equilibrium, initial, tolerance, max_iterations, *, callback=None,
                             preconditioner=None):
    """Newton-Krylov with rejected contact returns kept out of the line search."""
    value = initial.copy()
    residual = equilibrium(value)
    jacobian = KrylovJacobian(inner_M=preconditioner)
    jacobian.setup(value.copy(), residual, equilibrium)
    forcing = 1e-3
    for iteration in range(max_iterations+1):
        if np.max(np.abs(residual)) <= tolerance:
            return OptimizeResult(x=value, success=True, nit=iteration,
                                  message="Converged with valid contact returns")
        if iteration == max_iterations:
            break
        norm = np.linalg.norm(residual)
        # KrylovJacobian passes tol as LGMRES rtol. The forcing ratio is
        # dimensionless; multiplying it by a residual in mm oversolves the
        # linear equation and makes its effort depend on the chosen units.
        accepted = False
        for refresh in (False, True):
            if refresh:
                # Contact switches can invalidate the recycled subspace. A
                # fresh solve gets its own Krylov restarts before a nonlinear
                # load subdivision is allowed to change the material history.
                jacobian.method_kw["outer_v"].clear()
                direction, _ = lgmres(jacobian.op, -residual, rtol=min(forcing, .1),
                                      atol=0., maxiter=20, M=preconditioner)
            else:
                direction = -jacobian.solve(residual, tol=forcing)
            tangent = jacobian.matvec(direction)
            linear_ratio = np.linalg.norm(tangent+residual)/norm
            slope = float(residual @ tangent/norm**2)
            if linear_ratio >= 1. or slope >= 0.:
                continue
            fraction = 1.
            for _ in range(30):
                candidate = value+fraction*direction
                try:
                    following = equilibrium(candidate)
                except CrackReturnError:
                    fraction *= .5
                    continue
                next_norm = np.linalg.norm(following)
                if next_norm <= (1.-1e-4*fraction)*norm:
                    accepted = True
                    break
                fraction *= .5
            if accepted:
                break
        if not accepted:
            if callback is not None:
                callback({"iteration": iteration+1, "fraction": 0., "forcing": forcing,
                          "maximum_correction_mm": float(np.max(np.abs(residual))),
                          "linear_relative_residual": float(linear_ratio),
                          "relative_merit_slope": slope, "refreshed": refresh})
            return OptimizeResult(x=value, success=False, nit=iteration+1,
                                  message="No valid decreasing contact-equilibrium step")
        if callback is not None:
            callback({"iteration": iteration+1, "fraction": fraction, "forcing": forcing,
                      "maximum_correction_mm": float(np.max(np.abs(following))),
                      "residual_ratio": float(next_norm/norm),
                      "linear_relative_residual": float(linear_ratio),
                      "relative_merit_slope": slope, "refreshed": refresh})
        value, residual = candidate, following
        jacobian.update(value.copy(), residual)
        forcing = min(.9999, .9*(next_norm/norm)**2)
    return OptimizeResult(x=value, success=False, nit=max_iterations,
                          message="Contact equilibrium reached the maximum iteration count")


def save_accepted_state(path, displacement, stress, state, record, material):
    """Replace a complete accepted checkpoint after its compressed write finishes."""
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, u_mm=displacement, stress_MPa=stress,
            plastic_strain=state[0], kappa=state[1],
            **({"crack_normal": state[2]} if len(state) == 3 else {}),
            **({"crack_normals": state[2],
                ("crack_opening_shear" if material.host_crack_coupling == "shared_stress"
                 else "crack_slips"): state[3], "crack_weights": state[4]} if len(state) == 5 else {}),
            force_N=record["force_N"], end_shortening_mm=record["end_shortening_mm"],
            control=record["control"], control_target=record["target"],
            material_substeps=material.integration_substeps)
    temporary.replace(path)


def replay(core, material, targets, *, displacement_tolerance_mm=2e-6,
           max_iterations=100, return_iterations=36, host_scale=1.,
           max_subdivisions=0, progress=False, control="force", on_step=None, device="cpu",
           initial=None, displacement_functional=None, initial_predictor=None,
           current_contact_preconditioner=False):
    """Independent nodal equilibrium solve with the frozen exported material.

    Newton-Krylov measures the residual on a normal-elastic displacement scale.
    History is committed only at convergence; failure exposes load and correction.
    Each residual uses the material's interval integration from previous_u to
    the trial endpoint. Constitutive substeps are not equilibrium subdivisions.
    Optional bisection inserts integration states, not observation frames.
    The tolerance is a forward-solver setting in displacement units.
    With control='end_shortening', targets are area-mean top compression in mm
    and the force is an additional unknown. The traction shape stays unchanged;
    this does not impose a rigid platen or prescribe measured machine motion.
    on_step receives every accepted integration state, including subdivisions.
    initial=(displacement, complete_material_state, force) resumes an accepted
    state of this same material and domain; targets then contain only new states.
    With control='linear_displacement', displacement_functional=(q,h) prescribes
    q @ u + h * force. This can follow a localized motion through end snap-back.
    initial_predictor=(du/dtarget,dforce/dtarget) carries the last accepted secant
    into a resumed displacement-controlled segment.
    current_contact_preconditioner enables the approximate current-contact
    inverse for shared-stress material; the residual and history stay unchanged.
    """
    if control not in {"force", "end_shortening", "linear_displacement"}:
        raise ValueError("Unsupported replay control")
    if control == "linear_displacement" and displacement_functional is None:
        raise ValueError("Linear displacement control requires its nodal and load coefficients")
    material = material.to(device)
    band = core.tensors(material, device)
    C = material.elastic_matrix(host_scale).detach().cpu().numpy()
    # The elastic shear modulus can be weakly identified once the whole band
    # yields.  Its arbitrarily large value must not hide force errors behind a
    # nearly zero elastic correction.  Use the isotropic normal-elastic scale
    # for preconditioning; actual stresses still use the complete material C.
    C_reference = C.copy()
    C_reference[..., 3, 3] = C_reference[..., 4, 4] = C_reference[..., 5, 5]
    reference_stiffness = core.elastic_stiffness(C_reference, host_scale)
    factor = cho_factor(reference_stiffness)
    unit_displacement = cho_solve(factor, core.g)
    if control == "linear_displacement":
        control_vector, control_load = displacement_functional
    else:
        control_vector = core.g
        control_load = float(core.end_shortening(np.zeros_like(unit_displacement), 1., host_scale))
    control_value = lambda displacement, force: control_vector @ displacement+control_load*force
    compliance = float(control_value(unit_displacement, 1.))
    if initial is None:
        u = np.zeros(len(core.dofs))
        state = material.initial_state(band.strain(torch.as_tensor(u, device=device)))
        previous_force = 0.
    else:
        u, state, previous_force = initial
        u = np.asarray(u).copy()
        state = tuple(torch.as_tensor(value, dtype=torch.as_tensor(u).dtype, device=device) for value in state)
    previous_target = (previous_force if control == "force"
                       else float(control_value(u, previous_force)))
    displacements, stresses, kappas, diagnostics = [], [], [], []
    predictor = initial_predictor

    def advance(previous_u, previous_state, previous_force, previous_target, target, depth):
        nonlocal predictor
        previous_strain = band.strain(torch.as_tensor(previous_u, device=device))
        def unpack(value):
            if control == "force":
                return value, float(target)
            return value[:-1], float(value[-1]/compliance)

        def equilibrium(value):
            displacement, force = unpack(value)
            displacement_tensor = torch.as_tensor(displacement, device=device)
            strain = band.strain(displacement_tensor)
            stress = material.integrate_interval(previous_strain, strain, previous_state,
                                                  band.average, return_iterations, host_scale)[0]
            residual = band.residual(displacement_tensor, stress, torch.as_tensor(force, device=device),
                                     host_scale).cpu().numpy()
            correction = cho_solve(factor, residual)
            if control == "force":
                return correction
            constraint = control_value(displacement, force)-target
            # Inverse of the fixed bordered elastic operator. Scale the force
            # coordinate to mm so both blocks share the displacement tolerance.
            force_correction = (constraint-control_vector @ correction)/compliance
            return np.r_[correction+unit_displacement*force_correction,
                         compliance*force_correction]

        initial = previous_u
        if control != "force":
            increment = target-previous_target
            if predictor is None:
                force_increment = increment/compliance
                trial_u, trial_force = previous_u+unit_displacement*force_increment, previous_force+force_increment
            else:
                # Continue the actual accepted branch, including a falling
                # force and decreasing end shortening at increasing local slip.
                trial_u = previous_u+increment*predictor[0]
                trial_force = previous_force+increment*predictor[1]
            initial = np.r_[trial_u, compliance*trial_force]
        def subdivide(reason):
            if depth >= max_subdivisions:
                raise RuntimeError(f"Replay failed at {control}={target:g} after {previous_force:g} N: {reason}")
            middle_target = .5*(previous_target+target)
            if progress:
                print(f"replay_subdivide {control}: {previous_target:g} -> {middle_target:g} -> {target:g}", flush=True)
            middle_u, _, middle_state, left = advance(previous_u, previous_state, previous_force, previous_target, middle_target, depth+1)
            final_u, final_stress, final_state, right = advance(middle_u, middle_state, left["force_N"], middle_target, target, depth+1)
            right["substeps"] += left["substeps"]
            right["iterations"] += left["iterations"]
            right["dissipation_N_mm"] += left["dissipation_N_mm"]
            right["min_dissipation_MPa"] = min(left["min_dissipation_MPa"], right["min_dissipation_MPa"])
            right["correction_mm"] = max(left["correction_mm"], right["correction_mm"])
            right["return_error_MPa"] = max(left["return_error_MPa"], right["return_error_MPa"])
            return final_u, final_stress, final_state, right

        try:
            if getattr(material, "host_crack_coupling", None) == "shared_stress":
                def trace_iteration(row):
                    if row["iteration"] % 10 == 0 or row["fraction"] == 0.:
                        print(f"replay_iteration {control}={target:g}; iteration {row['iteration']}; "
                              f"correction {row['maximum_correction_mm']:g} mm", flush=True)
                solution = contact_equilibrium_root(equilibrium, initial,
                                                    displacement_tolerance_mm, max_iterations,
                                                    callback=trace_iteration if progress else None,
                                                    preconditioner=FrozenMaterialPreconditioner(
                                                        core, material, band, previous_state, host_scale,
                                                        reference_stiffness,
                                                        None if control == "force" else control_vector,
                                                        control_load, compliance, previous_strain)
                                                    if current_contact_preconditioner else None)
            else:
                solution = root(equilibrium, initial, method="krylov", options={
                    "fatol": displacement_tolerance_mm, "maxiter": max_iterations})
            next_u, force = unpack(solution.x)
            correction = equilibrium(solution.x)
        except CrackReturnError as exc:
            return subdivide(str(exc))
        if not solution.success:
            return subdivide(f"force {force:g} N, correction {abs(correction).max():g} mm; {solution.message}")

        stress, new_state, diag = material.integrate_interval(
            previous_strain, band.strain(torch.as_tensor(next_u, device=device)), previous_state,
            band.average, return_iterations, host_scale)
        error = float(diag["consistency_MPa"].abs().max())
        if error > 1e-5:
            raise RuntimeError(f"Unconverged material return at {force:g} N: {error:g} MPa")
        record = {"force_N": float(force), "iterations": int(solution.nit), "substeps": 1,
                  "material_substeps": material.integration_substeps,
                  "end_shortening_mm": float(core.end_shortening(next_u, force, host_scale)),
                  "control_displacement_mm": float(control_value(next_u, force)),
                  "control": control, "target": float(target), "subdivision_depth": depth,
                  "correction_mm": float(abs(correction).max()),
                  "return_error_MPa": error,
                  "dissipation_N_mm": float((diag["dissipation_MPa"]*band.volumes).sum()),
                  "min_dissipation_MPa": float(diag["min_substep_dissipation_MPa"].min()),
                  "tensile_volume_fraction": float((band.volumes*(diag["pressure_MPa"]<0)).sum()/band.volumes.sum())}
        if on_step is not None:
            on_step(next_u, stress.cpu().numpy(), tuple(value.cpu().numpy() for value in new_state), dict(record))
        if control != "force" and target != previous_target:
            predictor = ((next_u-previous_u)/(target-previous_target),
                         (force-previous_force)/(target-previous_target))
        return next_u, stress, new_state, record

    with torch.no_grad():
        for target in targets:
            u, stress, state, record = advance(u, state, previous_force, previous_target, target, 0)
            previous_force = record["force_N"]
            previous_target = float(target)
            displacements.append(u.copy())
            stresses.append(stress.cpu().numpy().copy())
            kappas.append(state[1].cpu().numpy().copy())
            diagnostics.append(record)
            if progress:
                print(f"replay_force {previous_force:g} N; shortening {record['end_shortening_mm']:g} mm; accepted substeps {record['substeps']}", flush=True)
    return np.stack(displacements), np.stack(stresses), np.stack(kappas), diagnostics
