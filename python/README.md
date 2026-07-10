# Experimental Python Bindings

Experimental Python bindings for small femx libraries.

Install from the femx repository root:

```shell
pip install .
```

Use the module as:

```python
import femx_experimental as femx
```

Currently exposed:

- `read_mesh_info(path)`
- `boundary_center(path, tag_or_name)`
- `structured_quad_info(nx, ny, x_min=0, x_max=1, y_min=0, y_max=1)`

Example:

```shell
python3 python/experimental/examples/mesh_info.py data/meshes/2d_straighttube.msh
```
