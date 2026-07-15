"""Mesh and boundary-surface helpers."""

import numpy as np
from scipy import linalg as dense_linalg
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from . import _core


def _csr_matrix(data):
    matrix = sparse.coo_matrix(
        (data["data"], (data["rows"], data["cols"])),
        shape=tuple(data["shape"]),
    )
    return matrix.tocsr()


class BoundarySurface:
    """A named or tagged simplicial boundary submesh."""

    def __init__(self, impl):
        self._impl = impl

    @property
    def dimension(self):
        return self._impl.dimension

    @property
    def num_nodes(self):
        return self._impl.num_nodes

    @property
    def num_elements(self):
        return self._impl.num_elements

    @property
    def mesh_node_ids(self):
        return self._impl.mesh_node_ids

    @property
    def coordinates(self):
        return self._impl.coordinates

    @property
    def elements(self):
        return self._impl.elements

    @property
    def rim_node_ids(self):
        """Local node ids on the boundary of this boundary surface."""

        return self._impl.rim_node_ids

    @property
    def rim_mesh_node_ids(self):
        """Volume-mesh node ids on the boundary-surface rim."""

        return self._impl.rim_mesh_node_ids

    @property
    def interior_node_ids(self):
        mask = np.ones(self.num_nodes, dtype=bool)
        mask[self.rim_node_ids] = False
        return np.flatnonzero(mask)

    def laplacian_matrices(self):
        """Return P1 surface stiffness, mass, and constant-load data."""

        data = self._impl.scalar_matrix_data()
        stiffness = _csr_matrix(data["stiffness"])
        mass = _csr_matrix(data["mass"])
        return stiffness, mass, data["load"]

    def poisson_profile(self, normalize=True):
        """Solve ``-surface_laplacian(phi) = 1`` with zero rim values.

        When ``normalize`` is true, the returned profile has unit surface
        integral and can be multiplied directly by a volumetric flow rate.
        """

        stiffness, _, load = self.laplacian_matrices()
        if self.rim_node_ids.size == 0:
            raise ValueError("boundary surface has no rim for Dirichlet values")
        interior = self.interior_node_ids
        if interior.size == 0:
            raise ValueError(
                "boundary surface has no interior nodes; refine the boundary mesh"
            )

        profile = np.zeros(self.num_nodes)
        reduced = stiffness[interior][:, interior]
        profile[interior] = sparse_linalg.spsolve(reduced, load[interior])

        if normalize:
            integral = float(load @ profile)
            if not np.isfinite(integral) or integral <= 0.0:
                raise RuntimeError("Poisson profile has a non-positive integral")
            profile /= integral
        return profile

    def laplacian_modes(self, count):
        """Return the lowest Dirichlet surface-Laplacian eigenpairs.

        The mode matrix has shape ``(num_nodes, count)``.  Its columns vanish
        on rim nodes and are orthonormal in the surface mass inner product.
        """

        if count <= 0:
            raise ValueError("count must be positive")
        if self.rim_node_ids.size == 0:
            raise ValueError("boundary surface has no rim for Dirichlet modes")

        stiffness, mass, _ = self.laplacian_matrices()
        interior = self.interior_node_ids
        if count > interior.size:
            raise ValueError(
                f"requested {count} modes but only {interior.size} interior nodes exist"
            )

        reduced_stiffness = stiffness[interior][:, interior]
        reduced_mass = mass[interior][:, interior]

        if count == interior.size or interior.size <= 16:
            eigenvalues, reduced_modes = dense_linalg.eigh(
                reduced_stiffness.toarray(),
                reduced_mass.toarray(),
                subset_by_index=(0, count - 1),
            )
        else:
            eigenvalues, reduced_modes = sparse_linalg.eigsh(
                reduced_stiffness,
                k=count,
                M=reduced_mass,
                sigma=0.0,
                which="LM",
            )
            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            reduced_modes = reduced_modes[:, order]

        modes = np.zeros((self.num_nodes, count))
        modes[interior, :] = reduced_modes
        return eigenvalues, modes


class Mesh:
    """An unstructured finite-element mesh owned by the femx engine."""

    def __init__(self, impl):
        self._impl = impl

    @classmethod
    def read(cls, path):
        return cls(_core.Mesh.read(str(path)))

    @property
    def dimension(self):
        return self._impl.dimension

    @property
    def num_nodes(self):
        return self._impl.num_nodes

    @property
    def num_elements(self):
        return self._impl.num_elements

    @property
    def coordinates(self):
        return self._impl.coordinates

    @property
    def physical_names(self):
        return self._impl.physical_names

    def boundary(self, selector):
        return BoundarySurface(self._impl.boundary(selector))
