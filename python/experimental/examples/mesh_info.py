#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import pprint


def import_femx_experimental():
    try:
        import femx_experimental as fx
        return fx
    except ModuleNotFoundError:
        femx_root = Path(__file__).resolve().parents[3]
        build_module_dir = femx_root / "build" / "python-experimental" / "python" / "experimental"
        if build_module_dir.is_dir():
            sys.path.insert(0, str(build_module_dir))
            import femx_experimental as fx
            return fx
        raise


def main() -> None:
    fx = import_femx_experimental()
    femx_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh",
        nargs="?",
        default=str(femx_root / "data" / "meshes" / "2d_straighttube.msh"))
    parser.add_argument("--boundary")
    args = parser.parse_args()

    pprint.pp(fx.read_mesh_info(args.mesh))
    if args.boundary is not None:
        try:
            boundary = int(args.boundary)
        except ValueError:
            boundary = args.boundary
        print("boundary center:", fx.boundary_center(args.mesh, boundary))


if __name__ == "__main__":
    main()
