"""Energy-only all-pairs nonbon8 oracle using the compiled rotation/physics code."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from jax_scorer import NbFusedStepData, NbGlobalData, NbRunConfig, _as_ptr
from pdb_io import parse_reduced_pdb


class CompiledGridlessScoreOracle:
    def __init__(self, *, receptor_ensemble=None, receptor_atomtypes=None,
                 receptor_charges=None, receptor_ens_list=None, ligand_ensemble=None,
                 ligand_atomtypes=None, ligand_charges=None, ligand_pdb=None,
                 attract_par_npz, lig_pivot, epsilon=15.0, dof_type="rotvec"):
        if dof_type not in {"rotvec", "euler"}:
            raise ValueError("compiled gridless scoring requires rotvec or euler DOFs")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if receptor_ensemble is None:
            if receptor_ens_list is None:
                raise ValueError("Either receptor_ens_list or receptor_ensemble must be provided")
            listing = Path(receptor_ens_list)
            paths = [listing.parent / line.strip() for line in listing.read_text().splitlines() if line.strip()]
            if not paths:
                raise ValueError("Empty receptor ensemble list")
            records = [parse_reduced_pdb(str(path)) for path in paths]
            receptor_ensemble = np.asarray([record[0] for record in records])
            receptor_atomtypes = records[0][1]
            receptor_charges = np.asarray([record[2] for record in records])
            if any(not np.array_equal(record[1], receptor_atomtypes) for record in records):
                raise ValueError("receptor ensembles must have matching atom types")
        if ligand_ensemble is None:
            if ligand_pdb is None:
                raise ValueError("Either ligand_pdb or ligand_ensemble must be provided")
            ligand_ensemble, ligand_atomtypes, ligand_charges, _ = parse_reduced_pdb(ligand_pdb)
        rec = self._coordinates(receptor_ensemble, "receptor")
        lig = self._coordinates(ligand_ensemble, "ligand")
        rec_types = self._vector(receptor_atomtypes, rec.shape[1], "receptor atom types", np.int32)
        lig_types = self._vector(ligand_atomtypes, lig.shape[1], "ligand atom types", np.int32)
        rec_charge = np.zeros(rec.shape[:2]) if receptor_charges is None else np.asarray(receptor_charges, dtype=np.float64)
        if rec_charge.shape == (rec.shape[1],):
            rec_charge = np.broadcast_to(rec_charge, rec.shape[:2])
        if rec_charge.shape != rec.shape[:2] or not np.all(np.isfinite(rec_charge)):
            raise ValueError("receptor charges must match receptor atom/ensemble counts")
        lig_charge = self._vector(ligand_charges, lig.shape[1], "ligand charges")
        rec_mask, lig_mask = rec_types != 99, lig_types != 99
        rec, lig = rec[:, rec_mask], lig[:, lig_mask]
        rec_types, lig_types = rec_types[rec_mask], lig_types[lig_mask]
        if not rec.shape[1] or not lig.shape[1]:
            raise ValueError("receptor and ligand must contain non-dummy atoms")
        rec_alphabet, rec_type = np.unique(rec_types, return_inverse=True)
        lig_alphabet, lig_type = np.unique(lig_types, return_inverse=True)
        if np.any(rec_alphabet < 1) or np.any(lig_alphabet < 1):
            raise ValueError("ATTRACT atom types must be positive")
        with np.load(attract_par_npz) as parameters:
            rc = parameters["rc"][rec_alphabet - 1][:, lig_alphabet - 1].astype(np.float64)
            ac = parameters["ac"][rec_alphabet - 1][:, lig_alphabet - 1].astype(np.float64)
            ivor = parameters["ivor"][rec_alphabet - 1][:, lig_alphabet - 1].astype(np.int8)
        with np.errstate(invalid="ignore", divide="ignore"):
            emin = -27.0 * ac**4 / (256.0 * rc**3)
            rmin2 = 4.0 * rc / (3.0 * ac)
        felec = math.sqrt(332.053986 / epsilon)
        self._data = {
            "rec": np.ascontiguousarray(rec, dtype=np.float64),
            "lig": np.ascontiguousarray(lig, dtype=np.float64),
            "rec_type": np.ascontiguousarray(rec_type, dtype=np.int16),
            "lig_type": np.ascontiguousarray(lig_type, dtype=np.int16),
            "rec_charge": np.ascontiguousarray(rec_charge[:, rec_mask] * felec),
            "lig_charge": np.ascontiguousarray(lig_charge[lig_mask] * felec),
            "pivot": self._vector(lig_pivot, 3, "ligand pivot"),
            "rc": np.ascontiguousarray(rc), "ac": np.ascontiguousarray(ac),
            "emin": np.ascontiguousarray(emin), "rmin2": np.ascontiguousarray(rmin2),
            "ivor": np.ascontiguousarray(ivor),
        }
        pointer = lambda name, ctype=ctypes.c_double: _as_ptr(self._data[name].reshape(-1), ctype)
        self._global = NbGlobalData(
            nrec=rec.shape[1], nens=rec.shape[0], nb_concat_len=0, nb_concat=None,
            rec_coord=pointer("rec"), rec_type=pointer("rec_type", ctypes.c_int16),
            rec_charge=pointer("rec_charge"), nrec_types=len(rec_alphabet), nlig_types=len(lig_alphabet),
            rc=pointer("rc"), ac=pointer("ac"), emin=pointer("emin"), rmin2=pointer("rmin2"),
            ivor=pointer("ivor", ctypes.c_int8), plateaudissq=0.0,
        )
        self._cfg = NbRunConfig(num_threads=int(os.environ.get("OMP_NUM_THREADS", "0")), kernel_variant=1)
        kernel_dir = Path(__file__).resolve().parents[1] / "nb_kernel"
        library = kernel_dir / "nb_kernel_gridless_nonbon8.so"
        dependencies = [kernel_dir / name for name in (
            "Makefile", "gridless_nonbon8.cpp", "nb_kernel.h", "include/euler_rot.h", "include/rotvec_rot.h",
            "forcefields/nonbon8/lj.h", "forcefields/nonbon8/elec.h")]
        if not library.exists() or any(path.stat().st_mtime > library.stat().st_mtime for path in dependencies):
            subprocess.run(["make", "-C", str(kernel_dir), library.name], check=True)
        self._library = ctypes.CDLL(str(library))
        self._kernel = getattr(self._library, f"nb_kernel_gridless_{dof_type}_energy")
        self._kernel.argtypes = [ctypes.POINTER(NbFusedStepData), ctypes.POINTER(NbGlobalData),
                                ctypes.POINTER(NbRunConfig), ctypes.POINTER(ctypes.c_int32),
                                ctypes.c_int32, ctypes.POINTER(ctypes.c_double)]
        self._kernel.restype = ctypes.c_int
        self._total_kernel_calls = 0
        self._total_kernel_time = 0.0

    @staticmethod
    def _coordinates(value, name):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[2] != 3 or not len(array) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} coordinates must have finite shape (E,N,3)")
        return array

    @staticmethod
    def _vector(value, length, name, dtype=np.float64):
        array = np.zeros(length, dtype=dtype) if value is None else np.asarray(value, dtype=dtype)
        if array.shape != (length,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite vector of length {length}")
        return np.ascontiguousarray(array)

    def score_batch(self, ens, dofs, conformers=None):
        dofs = np.ascontiguousarray(dofs, dtype=np.float64)
        n = len(dofs)
        if dofs.shape != (n, 6) or not np.all(np.isfinite(dofs)):
            raise ValueError("DOFs must have finite shape (N,6)")
        ens = np.asarray(ens)
        conformers = np.zeros(n, dtype=np.int32) if conformers is None else np.asarray(conformers)
        if ens.shape != (n,) or not np.issubdtype(ens.dtype, np.integer) or np.any(ens < 1) or np.any(ens > self._global.nens):
            raise ValueError("receptor ensemble indices must be valid 1-based integers")
        if conformers.shape != (n,) or not np.issubdtype(conformers.dtype, np.integer) or np.any(conformers < 0) or np.any(conformers >= len(self._data["lig"])):
            raise ValueError("conformer indices must be valid 0-based integers")
        ens = np.ascontiguousarray(ens - 1, dtype=np.int16)
        conformers = np.ascontiguousarray(conformers, dtype=np.int32)
        energies = np.empty(n, dtype=np.float64)
        if n:
            step = NbFusedStepData(
                nposes=n, natoms=self._data["lig"].shape[1], dofs=_as_ptr(dofs, ctypes.c_double),
                ens=_as_ptr(ens, ctypes.c_int16), lig_coords=_as_ptr(self._data["lig"], ctypes.c_double),
                lig_pivot=_as_ptr(self._data["pivot"], ctypes.c_double),
                lig_type=_as_ptr(self._data["lig_type"], ctypes.c_int16),
                lig_charge=_as_ptr(self._data["lig_charge"], ctypes.c_double),
            )
            started = time.monotonic()
            status = self._kernel(ctypes.byref(step), ctypes.byref(self._global), ctypes.byref(self._cfg),
                                  _as_ptr(conformers, ctypes.c_int32), len(self._data["lig"]),
                                  _as_ptr(energies, ctypes.c_double))
            if status:
                raise RuntimeError(f"compiled gridless kernel rejected its inputs ({status})")
            self._total_kernel_calls += 1
            self._total_kernel_time += time.monotonic() - started
        return energies, np.empty((0, 6), dtype=np.float64)

    def print_stats(self):
        print(f"COMPILED gridless kernel stats: {self._total_kernel_calls} calls, {self._total_kernel_time:.3f}s total")
