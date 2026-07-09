# Experimental Python Bridge

This directory contains an experimental, opt-in Python module for small femx
probes. It is not part of the stable v0.1.0 API and may change without notice.

The first module, `femx_experimental`, exposes a tiny mesh bridge:

- `read_mesh_info(path)`
- `boundary_center(path, tag_or_name)`
- `structured_quad_info(nx, ny, x_min=0, x_max=1, y_min=0, y_max=1)`

From the femx repository root, install it with:

```shell
pip install .
```

From the parent research checkout, the same command is:

```shell
pip install ./third-party/femx
```

For development, the CMake preset is still available:

```shell
pip install pybind11
cmake --preset python-experimental
cmake --build --preset python-experimental
```

With a conda environment that already has pybind11, run CMake from that
environment:

```shell
conda run -n pinn-gpu-env cmake --preset python-experimental
conda run -n pinn-gpu-env cmake --build --preset python-experimental
```

Try it with:

```shell
python3 python/experimental/examples/mesh_info.py data/meshes/2d_straighttube.msh
```
