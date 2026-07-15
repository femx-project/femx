import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

import femx


MESH_FILE = Path(__file__).parent / "data" / "2d_tiny_tube.msh"


class NavierStokesIoTest(unittest.TestCase):
    def setUp(self):
        self.model = femx.NavierStokesModel(
            MESH_FILE,
            num_steps=3,
            dt=0.25,
            rho=1.0,
            mu=0.01,
        )
        self.prob = femx.NavierStokesProblem(self.model)
        self.prob.add_bc(
            femx.DirichletBC("wall", "velocity", (0.0, 0.0))
        )
        self.prob.add_bc(
            femx.DirichletBC("outlet", "pressure", 0.0)
        )

    def test_writes_velocity_and_pressure_time_series(self):
        traj = self.prob.solve()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "flow.xdmf"
            self.model.write_xdmf(base, traj)

            h5_path = Path(directory) / "flow.h5"
            xdmf_path = Path(directory) / "flow.xdmf"
            self.assertTrue(h5_path.is_file())
            self.assertTrue(xdmf_path.is_file())

            with h5py.File(h5_path, "r") as h5:
                np.testing.assert_allclose(
                    h5["Mesh/Geometry"],
                    self.model.mesh.coordinates,
                )
                self.assertEqual(
                    list(h5["Data"]),
                    [f"Step{level:05d}" for level in range(4)],
                )

                velocity_dofs = self.model.velocity_dofs.reshape(
                    self.model.mesh.num_nodes,
                    self.model.mesh.dimension,
                )
                pressure_dofs = np.setdiff1d(
                    np.arange(self.model.num_states),
                    velocity_dofs,
                )
                for level in range(traj.num_time_levels):
                    group = h5[f"Data/Step{level:05d}"]
                    velocity = np.zeros(
                        (self.model.mesh.num_nodes, 3)
                    )
                    velocity[:, : self.model.mesh.dimension] = (
                        traj.values[level, velocity_dofs]
                    )
                    np.testing.assert_allclose(
                        group["velocity"],
                        velocity,
                    )
                    np.testing.assert_allclose(
                        group["pressure"],
                        traj.values[level, pressure_dofs],
                    )

            root = ET.parse(xdmf_path).getroot()
            times = [float(item.attrib["Value"]) for item in root.iter("Time")]
            np.testing.assert_allclose(times, [0.0, 0.25, 0.5, 0.75])
            self.assertEqual(
                {item.attrib["Name"] for item in root.iter("Attribute")},
                {"velocity", "pressure"},
            )

    def test_rejects_incompatible_trajectory(self):
        traj = femx.TimeTrajectory(1, 1)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.model.write_xdmf("flow", traj)


if __name__ == "__main__":
    unittest.main()
