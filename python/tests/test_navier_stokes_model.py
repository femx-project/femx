import unittest
from pathlib import Path

import numpy as np

import femx


ROOT = Path(__file__).resolve().parents[2]
MESH_FILE = ROOT / "data" / "meshes" / "2d_rectangle.msh"


class NavierStokesModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = femx.NavierStokesModel(
            MESH_FILE,
            num_steps=4,
            dt=0.05,
            rho=1.2,
            mu=0.03,
        )

    def test_model_metadata(self):
        self.assertEqual(self.model.num_steps, 4)
        self.assertEqual(self.model.dt, 0.05)
        self.assertEqual(self.model.rho, 1.2)
        self.assertEqual(self.model.mu, 0.03)
        self.assertEqual(self.model.mesh.dimension, 2)

        dims = self.model.residual.dims()
        self.assertEqual(dims.num_steps, self.model.num_steps)
        self.assertEqual(dims.num_states, self.model.num_states)
        self.assertEqual(dims.num_param, 0)

    def test_velocity_dof_queries(self):
        self.assertEqual(
            self.model.velocity_dofs.size,
            self.model.mesh.num_nodes * self.model.mesh.dimension,
        )

        inlet_by_name = self.model.velocity_boundary_dofs("inlet")
        inlet_by_tag = self.model.velocity_boundary_dofs(4)
        np.testing.assert_array_equal(inlet_by_name, inlet_by_tag)
        self.assertEqual(inlet_by_name.size, 2 * 21)

    def test_invalid_boundary_selector(self):
        with self.assertRaises(RuntimeError):
            self.model.velocity_boundary_dofs(True)


if __name__ == "__main__":
    unittest.main()
