import unittest
from pathlib import Path

import numpy as np

import femx


TINY_MESH_FILE = Path(__file__).parent / "data" / "2d_tiny_tube.msh"


class VelocityObservationOperatorTest(unittest.TestCase):
    def setUp(self):
        self.model = femx.NavierStokesModel(
            TINY_MESH_FILE,
            num_steps=2,
            dt=0.25,
            rho=1.0,
            mu=0.01,
        )

    def analytic_trajectory(self):
        trajectory = femx.TimeTrajectory(
            self.model.num_steps,
            self.model.num_states,
        )
        values = np.asarray(trajectory)
        coordinates = self.model.mesh.coordinates[:, :2]
        velocity_dofs = self.model.velocity_dofs.reshape(
            coordinates.shape[0], 2
        )
        for level in range(self.model.num_steps + 1):
            time = level * self.model.dt
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            values[level, velocity_dofs[:, 0]] = x + 2.0 * y + 3.0 * time
            values[level, velocity_dofs[:, 1]] = -x + y - time
        return trajectory

    def test_space_time_interpolation_and_layout(self):
        points = np.array([[0.25, 0.25], [0.75, 0.5]])
        times = np.array([0.0, 0.125, 0.5])
        operator = femx.VelocityObservationOperator(
            self.model,
            points=points,
            times=times,
            components=(1, 0),
        )

        prediction = operator.predict(self.analytic_trajectory())
        expected = np.empty(operator.shape)
        for level, time in enumerate(times):
            expected[level, :, 0] = (
                -points[:, 0] + points[:, 1] - time
            )
            expected[level, :, 1] = (
                points[:, 0] + 2.0 * points[:, 1] + 3.0 * time
            )

        self.assertEqual(operator.shape, (3, 2, 2))
        np.testing.assert_allclose(operator.reshape(prediction), expected)

    def test_transpose_satisfies_adjoint_identity(self):
        operator = femx.VelocityObservationOperator(
            self.model,
            points=((0.25, 0.25), (0.75, 0.5)),
            times=(0.0, 0.125, 0.5),
            components=(1, 0),
        )
        trajectory = self.analytic_trajectory()
        direction = np.linspace(-1.0, 1.0, operator.num_obs)

        transpose = operator.apply_transpose(direction)

        self.assertEqual(
            transpose.shape,
            (self.model.num_steps + 1, self.model.num_states),
        )
        self.assertAlmostEqual(
            float(direction @ operator.predict(trajectory)),
            float(np.sum(transpose * trajectory.values)),
            places=13,
        )

    def test_gaussian_misfit_uses_component_standard_deviation(self):
        operator = femx.VelocityObservationOperator(
            self.model,
            points=((0.25, 0.25), (0.75, 0.5)),
            times=(0.0, 0.5),
        )
        prediction = operator.predict(self.analytic_trajectory())
        observation = femx.GaussianObservation(
            operator,
            values=prediction,
            std=(0.5, 2.0),
        )

        self.assertEqual(observation.misfit(prediction), 0.0)
        shifted = prediction + observation.std
        self.assertAlmostEqual(
            observation.misfit(shifted),
            0.5 * operator.num_obs,
        )

    def test_cpp_objective_matches_gaussian_misfit_and_gradient(self):
        operator = femx.VelocityObservationOperator(
            self.model,
            points=((0.25, 0.25), (0.75, 0.5)),
            times=(0.0, 0.125, 0.5),
            components=(1, 0),
        )
        trajectory = self.analytic_trajectory()
        prediction = operator.predict(trajectory)
        deviation = np.tile((0.5, 2.0), 6)
        observed = prediction + np.linspace(
            -0.2,
            0.3,
            operator.num_obs,
        )
        observation = femx.GaussianObservation(
            operator,
            values=observed,
            std=deviation,
        )

        self.assertAlmostEqual(
            observation.value(trajectory),
            observation.misfit(prediction),
            places=13,
        )
        expected_gradient = operator.apply_transpose(
            (prediction - observed) / np.square(deviation)
        )
        np.testing.assert_allclose(
            observation.state_grad(trajectory),
            expected_gradient,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

        objective = observation.objective(num_param=2)
        self.assertIsInstance(objective, femx.TimeLeastSquaresObjective)
        self.assertEqual(objective.num_param, 2)
        self.assertAlmostEqual(
            objective.value(trajectory, np.array([0.1, 0.2])),
            observation.misfit(prediction),
            places=13,
        )
        np.testing.assert_array_equal(
            objective.param_grad(
                trajectory,
                np.array([0.1, 0.2]),
            ),
            np.zeros(2),
        )

    def test_time_obs_data_preserves_values_and_times(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        times = np.array([0.125, 0.5])

        data = femx.TimeObservationData(values, times=times)

        self.assertEqual(data.num_levels, 2)
        self.assertEqual(data.num_obs, 2)
        np.testing.assert_array_equal(data.values, values)
        np.testing.assert_array_equal(data.times, times)
        self.assertIsNone(data.time_levels)

    def test_rejects_point_outside_mesh(self):
        with self.assertRaisesRegex(ValueError, "outside the mesh"):
            femx.VelocityObservationOperator(
                self.model,
                points=((2.0, 0.5),),
                times=(0.25,),
            )


class ControlledObservationIntegrationTest(unittest.TestCase):
    def test_synthetic_ctr_generates_obs_vector(self):
        model = femx.NavierStokesModel(
            TINY_MESH_FILE,
            num_steps=3,
            dt=0.25,
            rho=1.0,
            mu=0.01,
        )
        problem = femx.NavierStokesProblem(model)
        problem.add_bc(femx.DirichletBC("wall", "velocity", (0.0, 0.0)))
        problem.add_bc(femx.DirichletBC("outlet", "pressure", 0.0))
        problem.add_ctr(
            femx.NormalVelocityControl(
                "inlet",
                normal=(1.0, 0.0),
                times=(0.0, 0.5),
            )
        )
        truth = problem.solve(np.array([0.2, 0.6]))
        operator = femx.VelocityObservationOperator(
            model,
            points=((0.25, 0.5), (0.75, 0.5)),
            times=(0.25, 0.5, 0.75),
            components=(0,),
        )

        prediction = operator.predict(truth)
        observation = femx.GaussianObservation(
            operator,
            values=prediction,
            std=0.01,
        )

        self.assertEqual(prediction.shape, (6,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertGreater(np.linalg.norm(prediction), 0.0)
        self.assertEqual(observation.misfit(prediction), 0.0)


if __name__ == "__main__":
    unittest.main()
