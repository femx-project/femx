"""Observation operators and Gaussian observation data."""

import numpy as np

from . import _core
from .navier_stokes import NavierStokesModel


class VelocityObservationOperator:
    """Sample velocity components at physical points and times.

    Predictions are flattened in ``(time, point, component)`` order. The
    corresponding three-dimensional shape is available through ``shape``.
    """

    def __init__(self, model, points, times, components=None):
        if not isinstance(model, NavierStokesModel):
            raise TypeError("model must be a NavierStokesModel")

        points = np.asarray(points, dtype=float)
        dim = model.mesh.dimension
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] not in (dim, 3)
        ):
            raise ValueError(
                "points must have shape (num_points, mesh_dimension) or "
                "(num_points, 3)"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")

        times = np.asarray(times, dtype=float)
        end_time = model.num_steps * model.dt
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a nonempty one-dimensional array")
        if (
            not np.all(np.isfinite(times))
            or np.any(times < 0.0)
            or np.any(times > end_time)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "times must be finite, increasing, and inside the simulation"
            )

        if components is None:
            comps = tuple(range(dim))
        else:
            comps = tuple(components)
        if not comps:
            raise ValueError("components must not be empty")
        if any(
            isinstance(comp, (bool, np.bool_))
            or not isinstance(comp, (int, np.integer))
            for comp in comps
        ):
            raise TypeError("components must contain integer component ids")
        comps = tuple(int(comp) for comp in comps)
        if len(set(comps)) != len(comps):
            raise ValueError("components must not contain duplicates")
        if any(
            comp < 0 or comp >= dim
            for comp in comps
        ):
            raise ValueError("component is out of range")

        self._model = model
        self._points = np.ascontiguousarray(points)
        self._times = np.ascontiguousarray(times)
        self._components = np.asarray(comps, dtype=np.int32)
        try:
            self._impl = _core._VelocityPointSampler(
                model._impl,
                self._points,
                self._components,
            )
        except RuntimeError as error:
            raise ValueError(str(error)) from error

        solve_times = np.arange(model.num_steps + 1) * model.dt
        upper = np.searchsorted(solve_times, self._times, side="left")
        self._upper_levels = np.minimum(upper, model.num_steps)
        self._lower_levels = np.maximum(self._upper_levels - 1, 0)
        denominator = (
            solve_times[self._upper_levels]
            - solve_times[self._lower_levels]
        )
        self._upper_weights = np.zeros(self._times.size)
        varying = self._upper_levels != self._lower_levels
        self._upper_weights[varying] = (
            self._times[varying] - solve_times[self._lower_levels[varying]]
        ) / denominator[varying]

    @property
    def model(self):
        return self._model

    @property
    def points(self):
        return self._points.copy()

    @property
    def times(self):
        return self._times.copy()

    @property
    def components(self):
        return self._components.copy()

    @property
    def shape(self):
        return (
            self._times.size,
            self._points.shape[0],
            self._components.size,
        )

    @property
    def num_obs(self):
        return int(np.prod(self.shape))

    def sample_solve_levels(self, traj):
        """Sample every solve time, returning ``(level, point, component)``."""

        try:
            return np.asarray(self._impl.sample(traj))
        except RuntimeError as error:
            raise ValueError(str(error)) from error

    def predict(self, traj):
        """Return observations flattened in time-point-component order."""

        solve_values = self.sample_solve_levels(traj)
        pred = (
            (1.0 - self._upper_weights[:, None, None])
            * solve_values[self._lower_levels]
            + self._upper_weights[:, None, None]
            * solve_values[self._upper_levels]
        )
        return np.ascontiguousarray(pred).reshape(-1)

    def apply_transpose(self, values):
        """Apply ``H.T`` and return gradients at every solve time level."""

        dirs = np.asarray(values, dtype=float)
        if dirs.size != self.num_obs:
            raise ValueError(
                f"values must contain {self.num_obs} observations"
            )
        dirs = np.ascontiguousarray(dirs).reshape(
            self._times.size,
            -1,
        )
        if not np.all(np.isfinite(dirs)):
            raise ValueError("values must be finite")

        solve_dirs = np.zeros(
            (self._model.num_steps + 1, dirs.shape[1]),
            dtype=float,
        )
        np.add.at(
            solve_dirs,
            self._lower_levels,
            (1.0 - self._upper_weights[:, None]) * dirs,
        )
        np.add.at(
            solve_dirs,
            self._upper_levels,
            self._upper_weights[:, None] * dirs,
        )
        try:
            return np.asarray(self._impl.apply_transpose(solve_dirs))
        except RuntimeError as error:
            raise ValueError(str(error)) from error

    def reshape(self, values):
        """View a flat observation vector as ``(time, point, component)``."""

        array = np.asarray(values, dtype=float)
        if array.size != self.num_obs:
            raise ValueError(
                f"observation vector must contain {self.num_obs} values"
            )
        return array.reshape(self.shape)


class GaussianObservation:
    """Observed values with independent Gaussian standard deviations."""

    def __init__(self, op, values, std=1.0):
        if not isinstance(op, VelocityObservationOperator):
            raise TypeError("op must be a VelocityObservationOperator")

        obs = np.asarray(values, dtype=float)
        if obs.size != op.num_obs:
            raise ValueError(
                f"values must contain {op.num_obs} observations"
            )
        obs = np.ascontiguousarray(obs).reshape(-1)
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation values must be finite")

        std = np.asarray(std, dtype=float)
        if std.ndim == 1 and std.size == op.num_obs:
            std = std.reshape(op.shape)
        try:
            std = np.broadcast_to(std, op.shape)
        except ValueError as error:
            raise ValueError(
                "std must be scalar or broadcast to the "
                "observation shape"
            ) from error
        std = np.ascontiguousarray(std).reshape(-1)
        if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError("std must be finite and positive")

        self._op = op
        self._values = obs.copy()
        self._std = std.copy()

    @property
    def op(self):
        return self._op

    @property
    def values(self):
        return self._values.copy()

    @property
    def std(self):
        return self._std.copy()

    def predict(self, traj):
        return self._op.predict(traj)

    def residual(self, pred):
        predicted = np.asarray(pred, dtype=float).reshape(-1)
        if predicted.size != self._values.size:
            raise ValueError(
                f"prediction must contain {self._values.size} observations"
            )
        if not np.all(np.isfinite(predicted)):
            raise ValueError("prediction must be finite")
        return predicted - self._values

    def whitened_residual(self, pred):
        return self.residual(pred) / self._std

    def misfit(self, pred):
        residual = self.whitened_residual(pred)
        return 0.5 * float(residual @ residual)

    def objective(self, *, num_param=0, level_wts=1.0):
        """Build the C++ trajectory objective for these observations."""

        if (
            isinstance(num_param, (bool, np.bool_))
            or not isinstance(num_param, (int, np.integer))
            or num_param < 0
        ):
            raise ValueError("num_param must be a nonnegative integer")

        wts = np.asarray(level_wts, dtype=float)
        try:
            wts = np.broadcast_to(
                wts,
                (self._op.model.num_steps + 1,),
            )
        except ValueError as error:
            raise ValueError(
                "level_wts must be scalar or match the solve levels"
            ) from error
        wts = np.ascontiguousarray(wts)
        if not np.all(np.isfinite(wts)) or np.any(wts < 0.0):
            raise ValueError("level_wts must be finite and nonnegative")

        if num_param == 0:
            sampler = self._op._impl
        else:
            sampler = _core._VelocityPointSampler(
                self._op.model._impl,
                self._op._points,
                self._op._components,
                int(num_param),
            )
        values = self._values.reshape(self._op.shape[0], -1)
        obs_wts = (
            1.0 / np.square(self._std)
        ).reshape(values.shape)
        data = _core.TimeObservationData(
            values,
            times=self._op._times,
        )
        return _core.TimeLeastSquaresObjective(
            sampler,
            data,
            wts,
            obs_wts,
            self._op.model.dt,
        )

    def value(self, traj, param=None, *, level_wts=1.0):
        """Evaluate the C++ least-squares objective on a trajectory."""

        param = (
            np.empty(0, dtype=float)
            if param is None
            else np.ascontiguousarray(param, dtype=float).reshape(-1)
        )
        obj = self.objective(
            num_param=param.size,
            level_wts=level_wts,
        )
        return obj.value(traj, param)

    def state_grad(self, traj, param=None, *, level_wts=1.0):
        """Return the C++ objective gradient at every trajectory level."""

        param = (
            np.empty(0, dtype=float)
            if param is None
            else np.ascontiguousarray(param, dtype=float).reshape(-1)
        )
        obj = self.objective(
            num_param=param.size,
            level_wts=level_wts,
        )
        return np.stack(
            [
                obj.state_grad(
                    level,
                    traj,
                    param,
                )
                for level in range(self._op.model.num_steps + 1)
            ]
        )
