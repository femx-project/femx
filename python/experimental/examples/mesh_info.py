#!/usr/bin/env python3

import argparse
from pathlib import Path
import pprint
import femx_experimental as femx


def main() -> None:
    femx_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh",
        nargs="?",
        default=str(femx_root / "data" / "meshes" / "2d_straighttube.msh"))
    parser.add_argument("--boundary")
    args = parser.parse_args()

    pprint.pp(femx.read_mesh_info(args.mesh))
    if args.boundary is not None:
        try:
            boundary = int(args.boundary)
        except ValueError:
            boundary = args.boundary
        print("boundary center:", femx.boundary_center(args.mesh, boundary))


if __name__ == "__main__":
    main()
