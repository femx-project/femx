import gc
import unittest

import numpy as np

import femx


class ScalarRecurrence(femx.TimeResidual):
    """Residual ``next - current - parameter = 0`` for binding tests."""

    def __init__(self, num_steps):
        super().__init__()
        self._num_steps = num_steps

    def dims(self):
        dims = femx.TimeDims()
        dims.num_steps = self._num_steps
        dims.num_states = 1
        dims.num_param = 1
        dims.num_res = 1
        dims.num_history_states = 1
        return dims

    def residual(self, context):
        return (
            context["next_state"]
            - context["history"][0]
            - context["parameters"]
        )

    def apply_jacobian(self, context, variable, direction):
        del context
        if variable.is_next_state:
            return direction
        if variable.is_history_state or variable.is_parameter:
            return -direction
        raise AssertionError("unexpected variable block")

    def apply_jacobian_transpose(self, context, variable, adjoint):
        return self.apply_jacobian(context, variable, adjoint)

    def assemble_jacobian(self, context, variable):
        del context
        if variable.is_next_state:
            return np.array([[1.0]])
        if variable.is_history_state or variable.is_parameter:
            return np.array([[-1.0]])
        return None


class TimeTrajectoryTest(unittest.TestCase):
    def test_contiguous_numpy_view(self):
        trajectory = femx.TimeTrajectory(2, 3)
        values = np.asarray(trajectory)

        self.assertEqual(trajectory.shape, (3, 3))
        self.assertEqual(values.shape, (3, 3))
        values[:] = np.arange(9, dtype=float).reshape(3, 3)

        np.testing.assert_array_equal(trajectory.values, values)
        np.testing.assert_array_equal(trajectory[1], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(trajectory[-1], [6.0, 7.0, 8.0])

        retained_view = trajectory.values
        del trajectory
        gc.collect()
        np.testing.assert_array_equal(retained_view[2], [6.0, 7.0, 8.0])


class TimeLinearIntegratorTest(unittest.TestCase):
    @staticmethod
    def make_integrator():
        problem = ScalarRecurrence(num_steps=3)
        matrix = femx.DenseMatrix()
        linear_solver = femx.DenseLinearSolver()
        integrator = femx.TimeLinearIntegrator(problem, matrix, linear_solver)
        integrator.set_initial_state(np.array([1.0]))
        return integrator

    def test_solves_python_residual(self):
        integrator = self.make_integrator()
        trajectory = integrator.solve(np.array([2.0]))

        np.testing.assert_allclose(
            trajectory.values[:, 0],
            [1.0, 3.0, 5.0, 7.0],
        )
        self.assertEqual(integrator.assembly_calls, 3)
        self.assertEqual(integrator.solve_calls, 3)

    def test_constructor_keeps_dependencies_alive(self):
        integrator = self.make_integrator()
        gc.collect()

        trajectory = integrator.solve(np.array([0.5]))
        np.testing.assert_allclose(
            trajectory.values[:, 0],
            [1.0, 1.5, 2.0, 2.5],
        )

    def test_rejects_wrong_param_size(self):
        integrator = self.make_integrator()
        with self.assertRaises(RuntimeError):
            integrator.solve(np.array([1.0, 2.0]))

    def test_reports_each_completed_forward_step_in_order(self):
        integrator = self.make_integrator()
        events = []

        trajectory = integrator.solve(
            np.array([2.0]),
            progress=lambda event: events.append(dict(event)),
        )

        np.testing.assert_allclose(
            trajectory.values[:, 0],
            [1.0, 3.0, 5.0, 7.0],
        )
        self.assertEqual([event["step"] for event in events], [1, 2, 3])
        self.assertTrue(
            all(
                event["type"] == "solve"
                and event["phase"] == "forward"
                and event["total"] == 3
                for event in events
            )
        )
        self.assertTrue(
            all(event["assembly_seconds"] >= 0.0 for event in events)
        )
        self.assertTrue(
            all(event["linear_solve_seconds"] >= 0.0 for event in events)
        )

    def test_callback_exception_clears_monitor_and_solver_is_reusable(self):
        integrator = self.make_integrator()
        steps = []

        def stop_after_second_step(event):
            steps.append(event["step"])
            if event["step"] == 2:
                raise RuntimeError("stop from progress callback")

        with self.assertRaisesRegex(
            RuntimeError, "stop from progress callback"
        ):
            integrator.solve(
                np.array([2.0]),
                progress=stop_after_second_step,
            )
        self.assertEqual(steps, [1, 2])

        trajectory = integrator.solve(np.array([0.5]))
        np.testing.assert_allclose(
            trajectory.values[:, 0],
            [1.0, 1.5, 2.0, 2.5],
        )

    def test_keyboard_interrupt_clears_monitor_and_solver_is_reusable(self):
        integrator = self.make_integrator()

        def interrupt(event):
            if event["step"] == 1:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            integrator.solve(np.array([2.0]), progress=interrupt)

        trajectory = integrator.solve(np.array([1.0]))
        np.testing.assert_allclose(
            trajectory.values[:, 0],
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_rejects_non_callable_progress(self):
        integrator = self.make_integrator()
        with self.assertRaisesRegex(TypeError, "progress must be callable"):
            integrator.solve(np.array([1.0]), progress=object())


if __name__ == "__main__":
    unittest.main()
