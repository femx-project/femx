import unittest

import numpy as np

import femx


class EnsembleBasisTest(unittest.TestCase):
    def setUp(self):
        self.basis = femx.EnsembleBasis(
            mean=np.array([1.0, -2.0, 0.5]),
            perturbations=np.array(
                [
                    [2.0, 0.0],
                    [0.0, -1.0],
                    [0.5, 3.0],
                ]
            ),
        )

    def test_maps_coefficients_and_physical_vectors(self):
        self.assertEqual(self.basis.value_size, 3)
        self.assertEqual(self.basis.num_physical_parameters, 3)
        self.assertEqual(self.basis.num_coefficients, 2)
        np.testing.assert_allclose(
            self.basis.evaluate(np.array([0.25, -2.0])),
            [1.5, 0.0, -5.375],
        )
        np.testing.assert_allclose(
            self.basis.apply_transpose(np.array([1.0, 2.0, -1.0])),
            [1.5, -5.0],
        )

    def test_array_properties_are_copies(self):
        mean = self.basis.mean
        perturbations = self.basis.perturbations
        mean[:] = 0.0
        perturbations[:] = 0.0

        np.testing.assert_array_equal(self.basis.mean, [1.0, -2.0, 0.5])
        np.testing.assert_array_equal(
            self.basis.perturbations,
            [[2.0, 0.0], [0.0, -1.0], [0.5, 3.0]],
        )

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(RuntimeError):
            femx.EnsembleBasis(np.array([]), np.empty((0, 1)))
        with self.assertRaises(RuntimeError):
            femx.EnsembleBasis(np.ones(2), np.ones((3, 1)))
        with self.assertRaises(RuntimeError):
            femx.EnsembleBasis(np.array([np.nan]), np.ones((1, 1)))
        with self.assertRaises(RuntimeError):
            self.basis.evaluate(np.ones(3))
        with self.assertRaises(RuntimeError):
            self.basis.evaluate(np.array([0.0, np.inf]))
        with self.assertRaises(RuntimeError):
            self.basis.apply_transpose(np.ones(2))


if __name__ == "__main__":
    unittest.main()
