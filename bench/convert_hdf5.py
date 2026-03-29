#!/usr/bin/env python3
"""Convert ann-benchmarks HDF5 files to FVECS/IVECS format for Go consumption.

Usage:
    python3 bench/convert_hdf5.py bench/datasets/siftsmall-128-euclidean.hdf5

Creates:
    bench/datasets/siftsmall-128-euclidean/
        train.fvecs      -- float32 vectors (base set)
        test.fvecs       -- float32 vectors (query set)
        neighbors.ivecs  -- int32 ground truth indices
        meta.txt         -- dataset metadata
"""

import sys
import os
import struct
import numpy as np

try:
    import h5py
except ImportError:
    print("pip install h5py numpy", file=sys.stderr)
    sys.exit(1)

def write_fvecs(path, data):
    """Write float32 vectors in FVECS format: [dim:uint32][v0:f32]...[vn:f32] per vector."""
    data = data.astype(np.float32)
    n, d = data.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<I', d))
            f.write(data[i].tobytes())
    print(f"  Wrote {path}: {n} vectors, d={d}")

def write_ivecs(path, data):
    """Write int32 vectors in IVECS format: [dim:uint32][v0:i32]...[vn:i32] per vector."""
    data = data.astype(np.int32)
    n, d = data.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<I', d))
            f.write(data[i].tobytes())
    print(f"  Wrote {path}: {n} vectors, k={d}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.hdf5>", file=sys.stderr)
        sys.exit(1)

    hdf5_path = sys.argv[1]
    name = os.path.splitext(os.path.basename(hdf5_path))[0]
    out_dir = os.path.join(os.path.dirname(hdf5_path), name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting {hdf5_path} -> {out_dir}/")

    with h5py.File(hdf5_path, 'r') as f:
        train = f['train'][:]
        test = f['test'][:]
        neighbors = f['neighbors'][:]

        # Get distance metric from attributes
        distance = f.attrs.get('distance', 'unknown')
        if isinstance(distance, bytes):
            distance = distance.decode('utf-8')

        print(f"  Dataset: {name}")
        print(f"  Distance: {distance}")
        print(f"  Train: {train.shape}")
        print(f"  Test: {test.shape}")
        print(f"  Neighbors: {neighbors.shape}")

        write_fvecs(os.path.join(out_dir, 'train.fvecs'), train)
        write_fvecs(os.path.join(out_dir, 'test.fvecs'), test)
        write_ivecs(os.path.join(out_dir, 'neighbors.ivecs'), neighbors)

        # Write metadata
        with open(os.path.join(out_dir, 'meta.txt'), 'w') as mf:
            mf.write(f"name={name}\n")
            mf.write(f"distance={distance}\n")
            mf.write(f"dim={train.shape[1]}\n")
            mf.write(f"train_size={train.shape[0]}\n")
            mf.write(f"test_size={test.shape[0]}\n")
            mf.write(f"neighbors_k={neighbors.shape[1]}\n")

    print(f"  Done! Files in {out_dir}/")

if __name__ == '__main__':
    main()
