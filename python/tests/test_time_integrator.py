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
        dims.num_hist = 1
        return dims

    def residual(self, context):
        return (
            context["next_state"]
            - context["history"][0]
            - context["parameters"]
        )

    def apply_jac_transpose(self, context, variable, adjoint):
        del context
        if variable.is_history_state or variable.is_parameter:
            return -adjoint
        raise AssertionError("unexpected variable block")

    def assemble_next(self, context):
        del context
        return np.array([[1.0]])


class SetupScalarRecurrence(ScalarRecurrence):
    def setup(self, context, rhs):
        del context
        self.prepared_type = type(rhs)
        rhs[0] = 6.0


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


class TimeIntegratorTest(unittest.TestCase):
    @staticmethod
    def make_integrator():
        problem = ScalarRecurrence(num_steps=3)
        integrator = femx.TimeIntegrator(problem)
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

    def test_observes_completed_host_linear_systems(self):
        problem = ScalarRecurrence(num_steps=3)
        systems = []
        integrator = femx.TimeIntegrator(
            problem,
            linear_system_observer=lambda sample: systems.append(dict(sample)),
        )
        integrator.set_initial_state(np.array([1.0]))

        trajectory = integrator.solve(np.array([2.0]))

        np.testing.assert_allclose(
            trajectory.values[:, 0],
            [1.0, 3.0, 5.0, 7.0],
        )
        self.assertEqual(len(systems), 3)
        for sample in systems:
            self.assertEqual(sample["rows"], 1)
            self.assertEqual(sample["cols"], 1)
            np.testing.assert_array_equal(sample["row_ptr"], [0, 1])
            np.testing.assert_array_equal(sample["col_ind"], [0])
            np.testing.assert_allclose(sample["values"], [1.0])
            np.testing.assert_allclose(
                sample["values"] * sample["solution"],
                sample["rhs"],
            )

    def test_rejects_non_callable_linear_system_observer(self):
        with self.assertRaisesRegex(
            TypeError,
            "linear_system_observer must be callable",
        ):
            femx.TimeIntegrator(
                ScalarRecurrence(num_steps=1),
                linear_system_observer=object(),
            )

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

    def test_runs_without_trajectory_and_samples_selected_levels(self):
        integrator = self.make_integrator()
        samples = []
        events = []

        result = integrator.run(
            np.array([2.0]),
            sample_every=2,
            sample=lambda level, state: samples.append(
                (level, np.asarray(state).copy())
            ),
            progress=lambda event: events.append(dict(event)),
        )

        self.assertIsNone(result)
        self.assertEqual([level for level, _ in samples], [0, 2, 3])
        np.testing.assert_allclose(
            [state[0] for _, state in samples],
            [1.0, 5.0, 7.0],
        )
        self.assertEqual([event["step"] for event in events], [1, 2, 3])
        self.assertEqual(integrator.assembly_calls, 3)
        self.assertEqual(integrator.solve_calls, 3)

    def test_run_rejects_invalid_sampling_arguments(self):
        integrator = self.make_integrator()

        with self.assertRaisesRegex(ValueError, "sample_every"):
            integrator.run(np.array([2.0]), sample_every=0)
        with self.assertRaisesRegex(TypeError, "sample must be callable"):
            integrator.run(np.array([2.0]), sample=object())

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

    def test_setup_receives_mutable_rhs(self):
        problem = SetupScalarRecurrence(num_steps=1)
        integrator = femx.TimeIntegrator(problem)
        integrator.set_initial_state(np.array([1.0]))

        trajectory = integrator.solve(np.array([0.0]))

        self.assertIs(problem.prepared_type, np.ndarray)
        np.testing.assert_allclose(trajectory.values[:, 0], [1.0, 6.0])


if __name__ == "__main__":
    unittest.main()
