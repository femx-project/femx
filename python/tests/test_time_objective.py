import unittest

import numpy as np

import femx


class TimeRegularizationTest(unittest.TestCase):
    def setUp(self):
        self.trajectory = femx.TimeTrajectory(2, 3)
        self.reference = np.array([0.2, 0.4, 0.6, 0.8])
        self.objective = femx.TimeRegularization(
            num_steps=2,
            num_states=3,
            num_levels=2,
            block_size=2,
            difference_weight=3.0,
            value_weight=2.0,
            reference=self.reference,
        )

    def test_value_and_gradient(self):
        parameters = np.array([0.3, 0.1, 0.9, 1.0])
        increment = parameters - self.reference
        level_difference = increment[2:] - increment[:2]
        expected_value = (
            0.5 * 2.0 * increment @ increment
            + 0.5 * 3.0 * level_difference @ level_difference
        )
        expected_gradient = 2.0 * increment
        expected_gradient[:2] -= 3.0 * level_difference
        expected_gradient[2:] += 3.0 * level_difference

        self.assertAlmostEqual(
            self.objective.value(self.trajectory, parameters),
            expected_value,
            places=14,
        )
        np.testing.assert_allclose(
            self.objective.param_grad(
                self.trajectory, parameters
            ),
            expected_gradient,
            rtol=0.0,
            atol=1.0e-14,
        )
        np.testing.assert_array_equal(
            self.objective.state_grad(
                1, self.trajectory, parameters
            ),
            np.zeros(3),
        )

    def test_rejects_invalid_weights_and_reference(self):
        with self.assertRaisesRegex(RuntimeError, "invalid inputs"):
            femx.TimeRegularization(2, 3, 2, 2, np.inf)
        with self.assertRaisesRegex(RuntimeError, "inconsistent size"):
            femx.TimeRegularization(
                2, 3, 2, 2, 0.0, reference=np.zeros(3)
            )


class TimeBlockRegularizationTest(unittest.TestCase):
    def test_sparse_quadratic_value_and_gradient(self):
        objective = femx.TimeBlockRegularization(
            1,
            2,
            2,
            2,
            rows=np.array([0, 0, 1, 1]),
            cols=np.array([0, 1, 0, 1]),
            vals=np.array([2.0, -1.0, -1.0, 3.0]),
            weight=2.0,
            reference=np.array([1.0, 1.0, 0.0, 0.0]),
        )
        trajectory = femx.TimeTrajectory(1, 2)
        param = np.array([2.0, 0.0, 1.0, 2.0])

        self.assertAlmostEqual(objective.value(trajectory, param), 17.0)
        np.testing.assert_allclose(
            objective.param_grad(trajectory, param),
            [6.0, -8.0, 0.0, 10.0],
        )


class SumTimeObjectiveTest(unittest.TestCase):
    def test_adds_objective_terms_and_keeps_them_alive(self):
        trajectory = femx.TimeTrajectory(2, 3)
        first = femx.TimeRegularization(
            2, 3, 2, 1, 0.0, 2.0, np.array([0.0, 0.0])
        )
        second = femx.TimeRegularization(
            2, 3, 2, 1, 0.0, 3.0, np.array([1.0, 1.0])
        )
        objective = femx.SumTimeObjective(2, 3, 2)
        returned = objective.add(first)
        objective.add(second)
        del first, second

        parameters = np.array([0.25, 0.75])
        expected_value = (
            0.5 * 2.0 * np.sum(parameters**2)
            + 0.5 * 3.0 * np.sum((parameters - 1.0) ** 2)
        )
        expected_gradient = (
            2.0 * parameters + 3.0 * (parameters - 1.0)
        )

        self.assertIs(returned, objective)
        self.assertAlmostEqual(
            objective.value(trajectory, parameters), expected_value
        )
        np.testing.assert_allclose(
            objective.param_grad(trajectory, parameters),
            expected_gradient,
        )

    def test_rejects_incompatible_term(self):
        objective = femx.SumTimeObjective(2, 3, 2)
        incompatible = femx.TimeRegularization(2, 3, 3, 1, 0.0)

        with self.assertRaisesRegex(RuntimeError, "inconsistent dimensions"):
            objective.add(incompatible)


if __name__ == "__main__":
    unittest.main()
