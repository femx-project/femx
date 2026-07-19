import unittest
from pathlib import Path

import numpy as np

import femx
import femx.navier_stokes as navier_stokes


ROOT = Path(__file__).resolve().parents[2]
MESH_FILE = ROOT / "data" / "meshes" / "2d_rectangle.msh"
TINY_MESH_FILE = Path(__file__).parent / "data" / "2d_tiny_tube.msh"


class NavierStokesProblemTest(unittest.TestCase):
    def setUp(self):
        self.model = femx.NavierStokesModel(
            MESH_FILE,
            num_steps=3,
            dt=0.25,
            rho=1.0,
            mu=0.01,
        )

    def test_constant_velocity_and_automatic_pressure_gauge(self):
        problem = femx.NavierStokesProblem(self.model)
        problem.add_bc(
            femx.DirichletBC(
                boundary="wall",
                field="velocity",
                value=(0.0, 0.0),
            )
        )
        problem.build()

        wall_dofs = self.model.velocity_boundary_dofs("wall")
        self.assertTrue(set(wall_dofs).issubset(set(problem.fixed_dofs)))
        self.assertEqual(problem.fixed_dofs.size, wall_dofs.size + 1)
        np.testing.assert_array_equal(problem.fixed_values, 0.0)
        np.testing.assert_array_equal(problem.initial_state, 0.0)

        dims = problem.residual.dims()
        self.assertEqual(dims.num_steps, self.model.num_steps)
        self.assertEqual(dims.num_states, self.model.num_states)
        self.assertEqual(dims.num_param, 0)

    def test_name_and_tag_select_the_same_boundary(self):
        by_name = femx.NavierStokesProblem(self.model)
        by_name.add_bc(femx.DirichletBC("inlet", "velocity", (1.0, 0.0)))
        by_name.build()

        by_tag = femx.NavierStokesProblem(self.model)
        by_tag.add_bc(femx.DirichletBC(4, "velocity", (1.0, 0.0)))
        by_tag.build()

        np.testing.assert_array_equal(by_name.fixed_dofs, by_tag.fixed_dofs)
        np.testing.assert_array_equal(by_name.fixed_values, by_tag.fixed_values)

    def test_callable_is_sampled_by_point_and_time(self):
        calls = []

        def inlet_value(point, time):
            calls.append((point.copy(), time))
            return point[1] + time, -time

        problem = femx.NavierStokesProblem(self.model)
        problem.add_bc(
            femx.DirichletBC(
                boundary="inlet",
                field="velocity",
                value=inlet_value,
            )
        )
        problem.build()

        inlet_dofs = self.model.velocity_boundary_dofs("inlet")
        columns = {dof: i for i, dof in enumerate(problem.fixed_dofs)}
        for dof in inlet_dofs[1::2]:
            np.testing.assert_allclose(
                problem.fixed_values[:, columns[int(dof)]],
                [-0.25, -0.5, -0.75],
            )
        self.assertEqual({time for _, time in calls}, {0.0, 0.25, 0.5, 0.75})
        self.assertTrue(all(point.shape == (2,) for point, _ in calls))

    def test_pressure_bc_disables_automatic_point_gauge(self):
        problem = femx.NavierStokesProblem(self.model)
        problem.add_bc(femx.DirichletBC("outlet", "pressure", 0.0))
        problem.build()

        outlet_nodes = self.model.mesh.boundary("outlet").mesh_node_ids
        self.assertEqual(problem.fixed_dofs.size, outlet_nodes.size)

    def test_conflicting_corner_values_are_rejected(self):
        problem = femx.NavierStokesProblem(self.model)
        problem.add_bc(femx.DirichletBC("wall", "velocity", (0.0, 0.0)))
        problem.add_bc(femx.DirichletBC("inlet", "velocity", (1.0, 0.0)))
        with self.assertRaisesRegex(ValueError, "conflicting Dirichlet"):
            problem.build()

    def test_rejects_invalid_velocity_shape(self):
        problem = femx.NavierStokesProblem(self.model)
        problem.add_bc(femx.DirichletBC("inlet", "velocity", 1.0))
        with self.assertRaisesRegex(ValueError, "2 values"):
            problem.build()

    def test_normal_ctr_runs_dense_forward_with_time_interpolation(self):
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
        problem.build()

        self.assertEqual(problem.param_shape, (2, 1))
        self.assertEqual(problem.num_param, 2)
        self.assertEqual(problem.ctr_state_dofs.size, 2)
        inlet = model.mesh.boundary("inlet")
        np.testing.assert_array_equal(
            problem.ctr_mesh_node_ids,
            inlet.mesh_node_ids[inlet.interior_node_ids],
        )
        self.assertTrue(
            set(problem.ctr_state_dofs).isdisjoint(problem.fixed_dofs)
        )

        trajectory = problem.solve(np.array([0.2, 0.6]))
        values = np.asarray(trajectory)
        normal_dof, tangential_dof = problem.ctr_state_dofs
        np.testing.assert_allclose(
            values[1:, normal_dof],
            [0.4, 0.6, 0.6],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(values[1:, tangential_dof], 0.0)

        solver = problem.create_solver()
        self.assertIsInstance(solver, femx.NavierStokesSolver)
        self.assertEqual(solver.backend, "dense")
        second = solver.solve(np.array([0.2, 0.6]))
        third = solver.solve(np.array([0.1, 0.3]))
        self.assertEqual(solver.num_solves, 2)
        self.assertEqual(solver.assembly_calls, model.num_steps)
        self.assertEqual(solver.linear_solve_calls, model.num_steps)
        np.testing.assert_allclose(second.values, trajectory.values)
        self.assertFalse(np.array_equal(third.values, second.values))

        problem.add_bc(femx.DirichletBC("inlet", "pressure", 0.0))
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            solver.solve(np.array([0.2, 0.6]))

    def test_fixed_initial_state_runs_without_adding_parameters(self):
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
        problem.build()

        state = problem.initial_state
        free = np.setdiff1d(
            model.velocity_dofs,
            np.union1d(problem.fixed_dofs, problem.ctr_state_dofs),
        )
        state[free[0]] = 0.5
        problem.set_initial_state(state)
        problem.build()

        self.assertEqual(problem.num_init_param, 0)
        self.assertEqual(problem.num_param, 2)
        np.testing.assert_array_equal(problem.initial_state, state)
        traj = problem.solve(np.array([0.2, 0.6]))
        np.testing.assert_array_equal(traj.values[0], state)

    def test_fixed_initial_state_preserves_boundary_values(self):
        problem = femx.NavierStokesProblem(self.model)
        problem.add_bc(femx.DirichletBC("inlet", "velocity", (1.0, 0.0)))
        problem.build()

        state = problem.initial_state
        state[problem.fixed_dofs[0]] = 2.0
        problem.set_initial_state(state)
        with self.assertRaisesRegex(ValueError, "fixed boundary"):
            problem.build()

    def test_affine_initial_state_ctr_runs_forward(self):
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
        problem.build()

        constrained = np.union1d(problem.fixed_dofs, problem.ctr_state_dofs)
        free_velocity = np.setdiff1d(model.velocity_dofs, constrained)
        self.assertGreater(free_velocity.size, 0)

        mean = problem.initial_state
        modes = np.zeros((model.num_states, 1))
        modes[free_velocity[0], 0] = 0.5
        problem.add_init_ctr(femx.InitialStateControl(modes, mean=mean))
        problem.build()

        self.assertEqual(problem.num_init_param, 1)
        self.assertEqual(problem.ctr_param_offset, 1)
        self.assertEqual(problem.num_ctr_param, 2)
        self.assertEqual(problem.num_param, 3)
        self.assertEqual(problem.param_shape, (2, 1))
        np.testing.assert_array_equal(problem.initial_state_modes, modes)
        np.testing.assert_array_equal(modes[problem.fixed_dofs], 0.0)
        np.testing.assert_array_equal(modes[problem.ctr_state_dofs], 0.0)
        pressure_dofs = np.setdiff1d(
            np.arange(model.num_states), model.velocity_dofs
        )
        np.testing.assert_array_equal(modes[pressure_dofs], 0.0)

        param = np.array([0.4, 0.2, 0.6])
        expected_initial = mean + modes[:, 0] * param[0]
        normal_dof, tangential_dof = problem.ctr_state_dofs
        expected_initial[normal_dof] = param[1]
        expected_initial[tangential_dof] = 0.0
        np.testing.assert_allclose(
            problem.initial_state_from_param(param),
            expected_initial,
        )

        solver = problem.create_solver()
        events = []

        def progress(event):
            events.append(
                (
                    dict(event),
                    solver.linear_solve_calls,
                    solver.num_solves,
                )
            )

        trajectory = solver.solve(param, progress=progress)
        values = np.asarray(trajectory)
        self.assertEqual(
            [event[0]["step"] for event in events],
            [1, 2, 3],
        )
        self.assertEqual(
            [event[1] for event in events],
            [1, 2, 3],
        )
        self.assertTrue(all(event[2] == 0 for event in events))
        np.testing.assert_allclose(values[0], expected_initial)
        np.testing.assert_allclose(
            values[1:, normal_dof],
            [0.4, 0.6, 0.6],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(values[1:, tangential_dof], 0.0)

        def interrupt(event):
            if event["step"] == 1:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            solver.solve(param, progress=interrupt)
        self.assertEqual(solver.num_solves, 1)

        changed = solver.solve(np.array([-0.2, 0.2, 0.6]))
        self.assertEqual(solver.num_solves, 2)
        self.assertNotEqual(
            trajectory[0][free_velocity[0]],
            changed[0][free_velocity[0]],
        )
        self.assertFalse(np.array_equal(trajectory[-1], changed[-1]))
        self.assertGreaterEqual(solver.assembly_seconds, 0.0)
        self.assertGreaterEqual(solver.solve_seconds, 0.0)
        solver.reset_timing()
        self.assertEqual(solver.assembly_seconds, 0.0)
        self.assertEqual(solver.solve_seconds, 0.0)

    def test_periodic_ctr_shares_first_and_last_time_level(self):
        model = femx.NavierStokesModel(
            TINY_MESH_FILE,
            num_steps=4,
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
                times=(0.0, 0.5, 1.0),
                periodic=True,
            )
        )
        problem.build()

        self.assertEqual(problem.param_shape, (2, 1))
        trajectory = problem.solve(np.array([0.2, 0.6]))
        normal_dof, tangential_dof = problem.ctr_state_dofs
        np.testing.assert_allclose(
            trajectory.values[1:, normal_dof],
            [0.4, 0.6, 0.4, 0.2],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            trajectory.values[1:, tangential_dof],
            0.0,
        )

        with self.assertRaisesRegex(ValueError, "explicit times"):
            femx.NormalVelocityControl(
                "inlet",
                normal=(1.0, 0.0),
                periodic=True,
            )

    def test_explicit_ctr_rejects_time_levels_unused_by_solver(self):
        for periodic in (False, True):
            with self.subTest(periodic=periodic):
                model = femx.NavierStokesModel(
                    TINY_MESH_FILE,
                    num_steps=1,
                    dt=1.0,
                    rho=1.0,
                    mu=0.01,
                )
                problem = femx.NavierStokesProblem(model)
                problem.add_bc(
                    femx.DirichletBC("wall", "velocity", (0.0, 0.0))
                )
                problem.add_bc(
                    femx.DirichletBC("outlet", "pressure", 0.0)
                )
                problem.add_ctr(
                    femx.NormalVelocityControl(
                        "inlet",
                        normal=(1.0, 0.0),
                        times=(0.0, 0.1, 1.0),
                        periodic=periodic,
                    )
                )

                with self.assertRaisesRegex(
                    ValueError, "not sampled by any solver step"
                ):
                    problem.build()

    def test_initial_modes_are_exactly_zero_on_constrained_dofs(self):
        base = femx.NavierStokesProblem(self.model)
        base.add_bc(femx.DirichletBC("wall", "velocity", (0.0, 0.0)))
        base.add_bc(femx.DirichletBC("outlet", "pressure", 0.0))
        base.add_ctr(
            femx.NormalVelocityControl(
                "inlet",
                normal=(1.0, 0.0),
                times=(0.0, 0.5),
            )
        )
        base.build()

        for dof, message in (
            (base.fixed_dofs[0], "fixed boundary dofs"),
            (base.ctr_state_dofs[0], "control boundary dofs"),
        ):
            with self.subTest(dof=int(dof)):
                modes = np.zeros((self.model.num_states, 1))
                modes[dof, 0] = 1.0e-12
                problem = femx.NavierStokesProblem(self.model)
                problem.add_bc(
                    femx.DirichletBC("wall", "velocity", (0.0, 0.0))
                )
                problem.add_bc(
                    femx.DirichletBC("outlet", "pressure", 0.0)
                )
                problem.add_ctr(
                    femx.NormalVelocityControl(
                        "inlet",
                        normal=(1.0, 0.0),
                        times=(0.0, 0.5),
                    )
                )
                problem.add_init_ctr(femx.InitialStateControl(modes))

                with self.assertRaisesRegex(ValueError, message):
                    problem.build()

    def test_solver_factory_rejects_unknown_backend(self):
        problem = femx.NavierStokesProblem(self.model)

        self.assertIn("dense", femx.solver_backends())
        with self.assertRaisesRegex(TypeError, "string"):
            problem.create_solver(None)
        with self.assertRaisesRegex(ValueError, "available: dense"):
            problem.create_solver("missing")

    def test_dense_solver_rejects_resolve_options(self):
        problem = femx.NavierStokesProblem(self.model)

        with self.assertRaisesRegex(
            ValueError, "not supported for backend 'dense'"
        ):
            problem.create_solver("dense", options={"rtol": 1.0e-10})

    @unittest.skipUnless(
        "resolve" in femx.solver_backends(),
        "femx was built without ReSolve",
    )
    def test_resolve_solver_accepts_validated_options(self):
        values = {
            "rtol": np.float64(1.0e-10),
            "max_its": np.int64(5000),
            "restart": 200,
        }
        options = navier_stokes._resolve_options(values)
        self.assertEqual(options.rtol, 1.0e-10)
        self.assertEqual(options.max_its, 5000)
        self.assertEqual(options.restart, 200)
        self.assertEqual(options.solve, "fgmres")
        self.assertEqual(options.precond, "ilu0")

        problem = femx.NavierStokesProblem(self.model)
        solver = problem.create_solver("resolve", options=values)
        self.assertEqual(solver.backend, "resolve")

    @unittest.skipUnless(
        "resolve" in femx.solver_backends(),
        "femx was built without ReSolve",
    )
    def test_resolve_solver_rejects_invalid_options(self):
        problem = femx.NavierStokesProblem(self.model)

        with self.assertRaisesRegex(TypeError, "options must be a mapping"):
            problem.create_solver("resolve", options=[("rtol", 1.0e-10)])
        with self.assertRaisesRegex(ValueError, "unknown ReSolve option"):
            problem.create_solver("resolve", options={"tolerance": 1.0e-10})

        invalid = (
            ({"rtol": True}, TypeError, "real number"),
            ({"rtol": 0.0}, ValueError, "finite and positive"),
            ({"rtol": np.inf}, ValueError, "finite and positive"),
            ({"max_its": 2.5}, TypeError, "integer"),
            ({"max_its": 0}, ValueError, "positive"),
            ({"restart": False}, TypeError, "integer"),
            ({"restart": -1}, ValueError, "positive"),
        )
        for values, error, message in invalid:
            with self.subTest(options=values):
                with self.assertRaisesRegex(error, message):
                    problem.create_solver("resolve", options=values)


if __name__ == "__main__":
    unittest.main()
