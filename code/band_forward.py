"""Three-dimensional secant elastic transfer for a cylinder with a finite band.

Model coordinates are x (horizontal), y (loading axis, bottom at zero), and z
(depth).  The 30 x 3 mm rectangle in x-y is extruded through the cylinder in z.
Its tangent makes ``angle_to_load_deg`` with +y.  The finite band is bonded to
the host; there is no zero-thickness jump, contact, damage, or history update.

``band_young_MPa`` is the common local Young modulus in the s, n, z directions,
with the fixed band Poisson ratio.  ``band_shear_MPa`` independently specifies
G_sn.  The other two shear moduli remain E_b / (2 (1 + nu_b)).  Both coefficients
are positive, making the local small-strain elastic energy positive definite.
It is isotropic exactly when G_sn = E_b / (2 (1 + nu_b)).  A load-dependent
choice of these coefficients is a secant-response diagnostic, not a validated
constitutive history law.

Gmsh constructs conforming tetrahedra; scikit-fem assembles three constant
stiffness matrices.  Each solve uses a sparse LU factorization.  Sensitivities
are exact linear-system derivatives with respect to natural-log coefficients.
No source project is modified and no mesh or solver output is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import argparse
import json
import math
import time

import gmsh
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTetP1, ElementVector, FacetBasis
from skfem import LinearForm, MeshTet, asm
from skfem.helpers import sym_grad
from skfem.models.elasticity import lame_parameters, linear_elasticity


@dataclass(frozen=True)
class BandGeometry:
    radius_mm: float = 25.0
    height_mm: float = 100.0
    band_length_mm: float = 30.0
    band_width_mm: float = 3.0
    angle_to_load_deg: float = 45.0
    center_mm: tuple[float, float, float] = (0.0, 50.0, 0.0)

    @property
    def tangent(self) -> np.ndarray:
        theta = math.radians(90.0 - self.angle_to_load_deg)
        return np.array([math.cos(theta), math.sin(theta), 0.0])

    @property
    def normal(self) -> np.ndarray:
        sx, sy, _ = self.tangent
        return np.array([-sy, sx, 0.0])


@dataclass
class BandSolution:
    displacement_mm: np.ndarray  # (n_nodes, 3), model coordinates
    sensitivity_mm: dict[str, np.ndarray]  # du / d log(parameter), same shape
    force_N: float  # applied compressive load magnitude
    support_reaction_N: np.ndarray  # vector sum of all constrained reactions
    force_balance_relative: float
    free_residual_relative: float
    strain_energy_N_mm: float
    external_work_N_mm: float  # u.T f; equals twice strain energy
    parameters: dict[str, float]


@dataclass
class SurfaceObservation:
    """P1 interpolation at the nearest point of the triangulated outer surface.

    The original cylinder is curved and its linear-element boundary is faceted.
    Projection distances are returned for the caller's registration/mesh checks;
    no distance screen, scaling, or extrapolation is silently applied.
    """

    matrix: csr_matrix  # maps node-major flattened displacements to point-major
    points_mm: np.ndarray
    projected_points_mm: np.ndarray
    projection_distance_mm: np.ndarray

    def evaluate(self, solution: BandSolution) -> np.ndarray:
        return np.asarray(self.matrix @ solution.displacement_mm.ravel()).reshape(-1, 3)

    def sensitivities(self, solution: BandSolution) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(self.matrix @ values.ravel()).reshape(-1, 3)
            for name, values in solution.sensitivity_mm.items()
        }


def _build_mesh(geometry: BandGeometry, mesh_size_mm: float,
                band_mesh_size_mm: float) -> tuple[MeshTet, np.ndarray]:
    if gmsh.isInitialized():
        raise RuntimeError("BandForward requires an unused Gmsh session")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("finite_band_cylinder")
        occ = gmsh.model.occ
        r, h = geometry.radius_mm, geometry.height_mm
        cylinder = occ.addCylinder(0.0, 0.0, 0.0, 0.0, h, 0.0, r)
        length, width = geometry.band_length_mm, geometry.band_width_mm
        # The slab covers the entire cylinder depth even when its center is
        # registered away from z=0.  Its z location does not create a hidden tip.
        depth = 2.0 * (r + abs(geometry.center_mm[2]) + mesh_size_mm)
        slab = occ.addBox(-length / 2.0, -width / 2.0, -depth / 2.0,
                          length, width, depth)
        theta = math.radians(90.0 - geometry.angle_to_load_deg)
        occ.rotate([(3, slab)], 0., 0., 0., 0., 0., 1., theta)
        occ.translate([(3, slab)], *geometry.center_mm)
        pieces, mapping = occ.fragment([(3, cylinder)], [(3, slab)])
        cylinder_parts = {tag for dim, tag in mapping[0] if dim == 3}
        slab_parts = {tag for dim, tag in mapping[1] if dim == 3}
        band_parts = cylinder_parts & slab_parts
        if not band_parts or band_parts == cylinder_parts:
            raise ValueError("The band must intersect a proper subvolume of the cylinder")
        outside = [(dim, tag) for dim, tag in pieces if dim == 3 and tag not in cylinder_parts]
        if outside:
            occ.remove(outside, recursive=True)
        occ.synchronize()

        # Mesh size varies with distance to band faces; all tetrahedra share
        # conforming nodes on the host/band boundary.
        faces = sorted({abs(tag) for volume in band_parts
                        for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)
                        if dim == 2})
        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "FacesList", faces)
        field.setNumber(distance, "Sampling", 60)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", band_mesh_size_mm)
        field.setNumber(threshold, "SizeMax", mesh_size_mm)
        field.setNumber(threshold, "DistMin", geometry.band_width_mm / 2.0)
        field.setNumber(threshold, "DistMax", 1.5 * mesh_size_mm)
        field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        node_index = {int(tag): i for i, tag in enumerate(node_tags)}
        cells, is_band = [], []
        for volume in sorted(cylinder_parts):
            types, tags, connectivity = gmsh.model.mesh.getElements(3, volume)
            for element_type, element_tags, nodes in zip(types, tags, connectivity):
                if int(element_type) != 4:
                    raise RuntimeError(f"Expected linear tetrahedra, got type {element_type}")
                cells.extend([[node_index[int(v)] for v in row]
                              for row in nodes.reshape(-1, 4)])
                is_band.extend([volume in band_parts] * len(element_tags))
        mesh = MeshTet(np.asarray(xyz).reshape(-1, 3).T,
                       np.asarray(cells, dtype=np.int64).T)
        return mesh, np.asarray(is_band, dtype=bool)
    finally:
        gmsh.finalize()


class BandForward:
    """Reusable stiffness and surface-observation operators for one geometry."""

    def __init__(self, geometry: BandGeometry | None = None, *,
                 host_young_MPa: float = 13550.0, host_poisson: float = 0.28,
                 band_poisson: float = 0.20, mesh_size_mm: float = 9.0,
                 band_mesh_size_mm: float = 2.0):
        self.geometry = geometry or BandGeometry()
        self.host_young_MPa = float(host_young_MPa)
        self.host_poisson = float(host_poisson)
        self.band_poisson = float(band_poisson)
        if host_young_MPa <= 0.0 or not (-1.0 < host_poisson < 0.5):
            raise ValueError("Host elasticity must be positive definite")
        if not (-1.0 < band_poisson < 0.5):
            raise ValueError("Band Poisson ratio must be in (-1, 0.5)")
        if min(mesh_size_mm, band_mesh_size_mm, self.geometry.radius_mm,
               self.geometry.height_mm, self.geometry.band_length_mm,
               self.geometry.band_width_mm) <= 0.0:
            raise ValueError("Geometry and mesh lengths must be positive")
        self.mesh, self.is_band = _build_mesh(self.geometry, mesh_size_mm, band_mesh_size_mm)
        self.basis = Basis(self.mesh, ElementVector(ElementTetP1()))
        host_basis = self.basis.with_elements(np.flatnonzero(~self.is_band))
        band_basis = self.basis.with_elements(np.flatnonzero(self.is_band))
        self.K_host = asm(linear_elasticity(*lame_parameters(host_young_MPa, host_poisson)),
                          host_basis).tocsr()
        tangent, normal = self.geometry.tangent, self.geometry.normal

        @BilinearForm
        def shear_sn(u, v, w):
            eu, ev = sym_grad(u), sym_grad(v)
            usn = np.einsum("i,ij...,j->...", tangent, eu, normal)
            vsn = np.einsum("i,ij...,j->...", tangent, ev, normal)
            return 4.0 * usn * vsn

        self.K_band_shear = asm(shear_sn, band_basis).tocsr()
        mu_unit = 1.0 / (2.0 * (1.0 + band_poisson))
        band_isotropic = asm(linear_elasticity(*lame_parameters(1.0, band_poisson)), band_basis)
        self.K_band_young = (band_isotropic - mu_unit * self.K_band_shear).tocsr()
        self._set_boundary_and_load()
        self._make_surface_triangles()
        self.mesh_info = {
            "nodes": int(self.mesh.nvertices), "tetrahedra": int(self.mesh.nelements),
            "band_tetrahedra": int(self.is_band.sum()),
            "degrees_of_freedom": int(self.basis.N),
            "top_area_mm2": float(self.top_area_mm2),
            "nominal_area_mm2": math.pi * self.geometry.radius_mm ** 2,
            "mesh_size_mm": float(mesh_size_mm),
            "band_mesh_size_mm": float(band_mesh_size_mm),
        }

    def _set_boundary_and_load(self) -> None:
        p, nd = self.mesh.p, self.basis.nodal_dofs
        bottom = np.flatnonzero(np.isclose(p[1], 0.0, atol=1e-7))
        top_facets = self.mesh.facets_satisfying(
            lambda x: np.isclose(x[1], self.geometry.height_mm, atol=1e-7),
            boundaries_only=True)
        if not bottom.size or not top_facets.size:
            raise RuntimeError("Missing cylinder end surfaces")
        pin = bottom[np.argmin(p[0, bottom] ** 2 + p[2, bottom] ** 2)]
        second = bottom[np.argmax(p[0, bottom])]
        # Bottom roller plus three scalar pins remove only the remaining rigid
        # x/z translations and rotation about y.  No lateral end restraint.
        self.constrained = np.unique(np.r_[nd[1, bottom], nd[[0, 2], pin], nd[2, second]])
        self.free = np.setdiff1d(np.arange(self.basis.N), self.constrained)
        self.bottom_nodes = bottom
        fb = FacetBasis(self.mesh, self.basis.elem, facets=top_facets)

        @LinearForm
        def unit_pressure(v, w):
            return -v[1]

        pressure_load = asm(unit_pressure, fb)
        self.top_area_mm2 = -float(pressure_load[nd[1]].sum())
        self.unit_force = pressure_load / self.top_area_mm2

    def solve(self, force_N: float, band_young_MPa: float,
              band_shear_MPa: float, *, host_scale: float = 1.0,
              with_sensitivities: bool = True) -> BandSolution:
        """Solve a force-controlled compression state; coefficients use MPa.

        ``force_N`` is a prescribed positive compressive force.  Equilibrium
        makes the support reaction equal that force: reaction agreement is a
        numerical check, not an independent force-prediction validation.
        """
        if min(band_young_MPa, band_shear_MPa, host_scale) <= 0.0:
            raise ValueError("Elastic coefficients and host_scale must be positive")
        if not np.isfinite([force_N, band_young_MPa, band_shear_MPa, host_scale]).all():
            raise ValueError("Load and elastic coefficients must be finite")
        components = {
            "log_band_young_MPa": band_young_MPa * self.K_band_young,
            "log_band_shear_MPa": band_shear_MPa * self.K_band_shear,
            "log_host_scale": host_scale * self.K_host,
        }
        stiffness = sum(components.values()).tocsr()
        lu = splu(stiffness[self.free][:, self.free].tocsc())
        f = float(force_N) * self.unit_force
        u = np.zeros(self.basis.N)
        u[self.free] = lu.solve(f[self.free])
        residual = stiffness @ u - f
        nd = self.basis.nodal_dofs
        sensitivities = {}
        if with_sensitivities:
            rhs = -np.column_stack([matrix @ u for matrix in components.values()])
            derivatives = np.zeros((self.basis.N, len(components)))
            derivatives[self.free] = lu.solve(rhs[self.free])
            sensitivities = {key: derivatives[nd, j].T
                             for j, key in enumerate(components)}
        support = np.zeros(self.basis.N)
        support[self.constrained] = residual[self.constrained]
        reaction = support[nd].sum(axis=1)
        applied = f[nd].sum(axis=1)
        force_scale = max(abs(float(force_N)), np.finfo(float).tiny)
        return BandSolution(
            displacement_mm=u[nd].T,
            sensitivity_mm=sensitivities,
            force_N=float(force_N), support_reaction_N=reaction,
            force_balance_relative=float(np.linalg.norm(applied + reaction) / force_scale),
            free_residual_relative=float(np.linalg.norm(residual[self.free]) / force_scale),
            strain_energy_N_mm=float(0.5 * u @ (stiffness @ u)),
            external_work_N_mm=float(u @ f),
            parameters={"band_young_MPa": float(band_young_MPa),
                        "band_shear_MPa": float(band_shear_MPa),
                        "host_scale": float(host_scale)},
        )

    def _make_surface_triangles(self) -> None:
        facets = self.mesh.boundary_facets()
        self._surface_nodes = self.mesh.facets[:, facets].T
        self._triangles = self.mesh.p.T[self._surface_nodes]

    def observation_operator(self, points_mm: np.ndarray, *,
                             max_projection_mm: float | None = None) -> SurfaceObservation:
        """Interpolate all three components at registered surface coordinates.

        The nearest triangle/point search is exhaustive over the mesh surface,
        so it also works at curved faces and band tips without element-finder
        extrapolation.  The operator can be reused for every parameter/load.
        """
        points = np.asarray(points_mm, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_mm must be a finite (n_points, 3) array")
        triangles = self._triangles
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, ac = b - a, c - a
        d00 = np.einsum("ij,ij->i", ab, ab)
        d01 = np.einsum("ij,ij->i", ab, ac)
        d11 = np.einsum("ij,ij->i", ac, ac)
        denom = d00 * d11 - d01 * d01
        if np.any(denom <= 0.0):
            raise RuntimeError("Degenerate surface triangle")
        rows, columns, values = [], [], []
        projected, distances = [], []
        for i, point in enumerate(points):
            ap = point - a
            d20 = np.einsum("ij,ij->i", ap, ab)
            d21 = np.einsum("ij,ij->i", ap, ac)
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            weights = np.column_stack((1.0 - v - w, v, w))
            plane_point = np.einsum("fi,fij->fj", weights, triangles)
            best_distance = np.linalg.norm(plane_point - point, axis=1)
            best_distance[np.any(weights < 0.0, axis=1)] = np.inf
            best_point = plane_point.copy()
            for first, second in ((0, 1), (1, 2), (2, 0)):
                start = triangles[:, first]
                edge = triangles[:, second] - start
                position = np.clip(np.einsum("ij,ij->i", point - start, edge)
                                   / np.einsum("ij,ij->i", edge, edge), 0.0, 1.0)
                edge_point = start + position[:, None] * edge
                edge_distance = np.linalg.norm(edge_point - point, axis=1)
                improve = edge_distance < best_distance
                best_distance[improve] = edge_distance[improve]
                best_point[improve] = edge_point[improve]
                weights[improve] = 0.0
                weights[improve, first] = 1.0 - position[improve]
                weights[improve, second] = position[improve]
            face = int(np.argmin(best_distance))
            projected.append(best_point[face])
            distances.append(best_distance[face])
            for component in range(3):
                for node, weight in zip(self._surface_nodes[face], weights[face]):
                    rows.append(3 * i + component)
                    columns.append(3 * int(node) + component)
                    values.append(weight)
        distances_array = np.asarray(distances)
        if max_projection_mm is not None and np.any(distances_array > max_projection_mm):
            raise ValueError(f"Surface projection exceeds {max_projection_mm:g} mm; "
                             f"maximum is {distances_array.max():.6g} mm")
        matrix = coo_matrix((values, (rows, columns)),
                            shape=(3 * len(points), 3 * self.mesh.nvertices)).tocsr()
        return SurfaceObservation(matrix, points.copy(), np.asarray(projected), distances_array)


def _diagnostic() -> Mapping[str, object]:
    start = time.perf_counter()
    model = BandForward()
    construction = time.perf_counter() - start
    start = time.perf_counter()
    solution = model.solve(20000., 100., 40.)
    solve_time = time.perf_counter() - start
    points = np.array([[0., 50., 25.], [-10., 40., math.sqrt(525.)],
                       [10., 60., math.sqrt(525.)]])
    observer = model.observation_operator(points)
    return {"mesh": model.mesh_info, "construction_s": construction,
            "solve_with_three_sensitivities_s": solve_time,
            "support_reaction_N": solution.support_reaction_N.tolist(),
            "force_balance_relative": solution.force_balance_relative,
            "free_residual_relative": solution.free_residual_relative,
            "energy_work_relative": abs(2. * solution.strain_energy_N_mm
                                         / solution.external_work_N_mm - 1.),
            "projection_distance_mm": observer.projection_distance_mm.tolist(),
            "displacement_mm": observer.evaluate(solution).tolist()}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(_diagnostic(), indent=2))
