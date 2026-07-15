import unittest
from pathlib import Path

import numpy as np

import femx


TINY_MESH_FILE = Path(__file__).parent / "data" / "2d_tiny_tube.msh"

DENSE_PARAM = np.array([0.1, 0.2])
DENSE_PRED = np.array(
    [
        0.13657757076642157,
        0.12648074632172848,
        0.19079668098283917,
        0.18630583122682456,
        0.1952533492447734,
        0.1948341198666849,
    ]
)
DENSE_OBJ = 0.5532572993929018
DENSE_GRAD = np.array(
    [-0.3178728821910878, -2.083814260251649]
)
DENSE_LEVEL_SUMS = np.array(
    [0.0, 1.9035677198503913, 1.297850612585905, 0.7427631422339811]
)
DENSE_LEVEL_NORMS = np.array(
    [0.0, 0.6835514195517696, 0.4519850928518233, 0.34656623294316113]
)
DENSE_FINAL_STATE = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.19050669848954677,
        0.0,
        0.19916154124382301,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.030081071407519158,
        0.019020501652607182,
        0.0,
        0.034868108207795429,
        0.020023648172563167,
        0.0,
        0.030081071407519155,
        0.019020501652607179,
        0.0,
    ]
)


class DenseNavierStokesReducedFunctionalTest(unittest.TestCase):
    def setUp(self):
        self.model = femx.NavierStokesModel(
            TINY_MESH_FILE,
            num_steps=3,
            dt=0.25,
            rho=1.0,
            mu=0.01,
        )
        self.problem = femx.NavierStokesProblem(self.model)
        self.problem.add_bc(
            femx.DirichletBC("wall", "velocity", (0.0, 0.0))
        )
        self.problem.add_bc(
            femx.DirichletBC("outlet", "pressure", 0.0)
        )
        self.problem.add_ctr(
            femx.NormalVelocityControl(
                "inlet",
                normal=(1.0, 0.0),
                times=(0.0, 0.5),
            )
        )
        self.operator = femx.VelocityObservationOperator(
            self.model,
            points=((0.25, 0.5), (0.75, 0.5)),
            times=(0.25, 0.5, 0.75),
            components=(0,),
        )
        truth = self.problem.solve(np.array([0.3, 0.7]))
        self.observation = femx.GaussianObservation(
            self.operator,
            values=self.operator.predict(truth),
        )
        objective = self.observation.objective(
            num_param=self.problem.num_param
        )
        self.reduced = self.problem.create_reduced(
            objective
        )
        self.assertIsInstance(
            self.reduced,
            femx.NavierStokesReducedFunctional,
        )
        self.assertEqual(self.reduced.backend, "dense")

    def test_dense_reference_signature(self):
        traj = self.problem.solve(DENSE_PARAM)
        pred = self.operator.predict(traj)
        obj, grad = self.reduced.value_grad(DENSE_PARAM)

        np.testing.assert_allclose(
            pred,
            DENSE_PRED,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        self.assertAlmostEqual(obj, DENSE_OBJ, places=12)
        np.testing.assert_allclose(
            grad,
            DENSE_GRAD,
            rtol=1.0e-11,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            traj.values.sum(axis=1),
            DENSE_LEVEL_SUMS,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            np.linalg.norm(traj.values, axis=1),
            DENSE_LEVEL_NORMS,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            traj.values[-1],
            DENSE_FINAL_STATE,
            rtol=1.0e-12,
            atol=1.0e-13,
        )

    def test_adjoint_gradient_matches_central_difference(self):
        parameters = np.array([0.1, 0.2])

        value, gradient = self.reduced.value_grad(parameters)

        trajectory = self.problem.solve(parameters)
        expected_value = self.observation.misfit(
            self.operator.predict(trajectory)
        )
        self.assertAlmostEqual(value, expected_value, places=13)
        self.assertEqual(self.reduced.solve_calls, self.model.num_steps)
        self.assertGreater(
            self.reduced.assembly_calls,
            self.reduced.solve_calls,
        )

        step = 1.0e-6
        finite_difference = np.empty(parameters.size)
        for i in range(parameters.size):
            perturbation = np.zeros(parameters.size)
            perturbation[i] = step
            finite_difference[i] = (
                self.reduced.value(parameters + perturbation)
                - self.reduced.value(parameters - perturbation)
            ) / (2.0 * step)

        np.testing.assert_allclose(
            gradient,
            finite_difference,
            rtol=2.0e-6,
            atol=1.0e-9,
        )

    @unittest.skipUnless(
        "resolve" in femx.solver_backends(),
        "femx was built without ReSolve",
    )
    def test_resolve_matches_dense_forward_value_and_gradient(self):
        traj = self.problem.solve(DENSE_PARAM, backend="resolve")
        reduced = self.problem.create_reduced(
            self.observation.objective(num_param=self.problem.num_param),
            backend="resolve",
        )
        obj, grad = reduced.value_grad(DENSE_PARAM)

        np.testing.assert_allclose(
            traj.values,
            self.problem.solve(DENSE_PARAM).values,
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        self.assertAlmostEqual(obj, DENSE_OBJ, places=8)
        np.testing.assert_allclose(
            grad,
            DENSE_GRAD,
            rtol=2.0e-7,
            atol=2.0e-9,
        )

    @unittest.skipUnless(
        "petsc" in femx.solver_backends(),
        "femx was built without PETSc",
    )
    def test_petsc_matches_dense_forward_value_and_gradient(self):
        traj = self.problem.solve(DENSE_PARAM, backend="petsc")
        reduced = self.problem.create_reduced(
            self.observation.objective(num_param=self.problem.num_param),
            backend="petsc",
        )
        obj, grad = reduced.value_grad(DENSE_PARAM)

        np.testing.assert_allclose(
            traj.values,
            self.problem.solve(DENSE_PARAM).values,
            rtol=2.0e-8,
            atol=2.0e-10,
        )
        self.assertAlmostEqual(obj, DENSE_OBJ, places=8)
        np.testing.assert_allclose(
            grad,
            DENSE_GRAD,
            rtol=2.0e-7,
            atol=2.0e-9,
        )

    def test_rejects_stale_problem_configuration(self):
        self.problem.add_bc(femx.DirichletBC("inlet", "pressure", 0.0))

        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            self.reduced.value(np.array([0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
