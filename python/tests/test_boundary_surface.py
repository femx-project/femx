import unittest
from pathlib import Path

import numpy as np

import femx


ROOT = Path(__file__).resolve().parents[2]
MESH_FILE = ROOT / "data" / "meshes" / "2d_rectangle.msh"


class BoundarySurfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = femx.Mesh.read(MESH_FILE)
        cls.inlet = cls.mesh.boundary("inlet")

    def test_mesh_metadata(self):
        self.assertEqual(self.mesh.dimension, 2)
        self.assertGreater(self.mesh.num_nodes, 0)
        self.assertEqual(self.mesh.coordinates.shape, (self.mesh.num_nodes, 3))
        self.assertIn(
            {"dimension": 1, "tag": 4, "name": "inlet"},
            self.mesh.physical_names,
        )

    def test_boundary_extraction(self):
        self.assertEqual(self.inlet.dimension, 1)
        self.assertEqual(self.inlet.num_nodes, 21)
        self.assertEqual(self.inlet.num_elements, 20)
        self.assertEqual(self.inlet.elements.shape, (20, 2))
        self.assertEqual(self.inlet.rim_node_ids.size, 2)

        by_tag = self.mesh.boundary(4)
        np.testing.assert_array_equal(by_tag.mesh_node_ids, self.inlet.mesh_node_ids)

    def test_laplacian_matrices(self):
        stiffness, mass, load = self.inlet.laplacian_matrices()
        self.assertEqual(stiffness.shape, (21, 21))
        self.assertEqual(mass.shape, (21, 21))
        np.testing.assert_allclose((stiffness - stiffness.T).data, 0.0)
        np.testing.assert_allclose((mass - mass.T).data, 0.0)

        ones = np.ones(self.inlet.num_nodes)
        np.testing.assert_allclose(stiffness @ ones, 0.0, atol=1.0e-10)
        self.assertAlmostEqual(float(ones @ mass @ ones), 0.004)
        self.assertAlmostEqual(float(load.sum()), 0.004)

    def test_poisson_profile(self):
        _, _, load = self.inlet.laplacian_matrices()
        profile = self.inlet.poisson_profile()
        np.testing.assert_allclose(profile[self.inlet.rim_node_ids], 0.0)
        self.assertGreater(float(profile.max()), 0.0)
        self.assertAlmostEqual(float(load @ profile), 1.0)

    def test_laplacian_modes(self):
        _, mass, _ = self.inlet.laplacian_matrices()
        eigenvalues, modes = self.inlet.laplacian_modes(3)
        self.assertEqual(eigenvalues.shape, (3,))
        self.assertEqual(modes.shape, (21, 3))
        self.assertTrue(np.all(eigenvalues > 0.0))
        np.testing.assert_allclose(modes[self.inlet.rim_node_ids], 0.0)
        np.testing.assert_allclose(modes.T @ mass @ modes, np.eye(3), atol=1.0e-10)

    def test_missing_boundary(self):
        with self.assertRaises(RuntimeError):
            self.mesh.boundary("missing")


if __name__ == "__main__":
    unittest.main()
