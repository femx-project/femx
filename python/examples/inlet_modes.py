#!/usr/bin/env python3
"""Compute an arbitrary-section Poisson profile and Laplacian modes."""

import argparse

import femx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--boundary", default="inlet")
    parser.add_argument("--modes", type=int, default=3)
    args = parser.parse_args()

    mesh = femx.Mesh.read(args.mesh)
    surface = mesh.boundary(args.boundary)
    profile = surface.poisson_profile()
    eigenvalues, modes = surface.laplacian_modes(args.modes)

    print(f"boundary nodes: {surface.num_nodes}")
    print(f"boundary elements: {surface.num_elements}")
    print(f"rim nodes: {surface.rim_node_ids.size}")
    print(f"profile range: [{profile.min():.6e}, {profile.max():.6e}]")
    print(f"eigenvalues: {eigenvalues}")
    print(f"mode matrix shape: {modes.shape}")


if __name__ == "__main__":
    main()
