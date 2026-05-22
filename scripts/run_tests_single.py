# #!/usr/bin/env python3
# """
# batch_eval_all_polys_rmsd.py

# Runs your PolyDiff sampler for each polyhedron in test_one_polyhedron.json,
# computes Hungarian-Kabsch RMSD, counts how many are below 1.0 Å,
# prints per-polyhedron RMSD and a final summary including per-shape match percentages.
# """

# import argparse
# import ast
# import json
# import os
# import re
# import subprocess
# import sys
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from collections import defaultdict


# # -------------------------
# # Kabsch + Hungarian
# # -------------------------
# # def kabsch_numpy(P: np.ndarray, Q: np.ndarray):
# #     assert P.shape == Q.shape
# #     centroid_P = P.mean(axis=0)
# #     centroid_Q = Q.mean(axis=0)
# #     P_c, Q_c = P - centroid_P, Q - centroid_Q
# #     H = P_c.T @ Q_c
# #     U, S, Vt = np.linalg.svd(H)
# #     if np.linalg.det(Vt.T @ U.T) < 0:
# #         Vt[-1, :] *= -1
# #     R = Vt.T @ U.T
# #     t = centroid_Q - centroid_P @ R.T
# #     rmsd = np.sqrt(np.mean(np.sum((P_c @ R.T - Q_c) ** 2, axis=1)))
# #     return R, t, rmsd


# # def kabsch_with_optimal_assignment(P: np.ndarray, Q: np.ndarray):
# #     assert P.shape == Q.shape
# #     P_c = P - P.mean(axis=0, keepdims=True)
# #     Q_c = Q - Q.mean(axis=0, keepdims=True)
# #     D = np.linalg.norm(P_c[:, None, :] - Q_c[None, :, :], axis=2)
# #     row_ind, col_ind = linear_sum_assignment(D)
# #     Q_matched = Q_c[col_ind]
# #     R, t, rmsd = kabsch_numpy(P_c, Q_matched)
# #     return R, t, rmsd, col_ind


# def kabsch_numpy(P: np.ndarray, Q: np.ndarray):
#     """
#     Computes the optimal rotation and translation to align two sets of points (P -> Q),
#     and their RMSD.

#     :param P: A Nx3 matrix of points
#     :param Q: A Nx3 matrix of points
#     :return: A tuple containing the optimal rotation matrix, the optimal
#              translation vector, and the RMSD.
#     """
#     assert P.shape == Q.shape, "Matrix dimensions must match"

#     # Compute centroids
#     centroid_P = np.mean(P, axis=0)
#     centroid_Q = np.mean(Q, axis=0)

#     # Optimal translation
#     t = centroid_Q - centroid_P

#     # Center the points
#     p = P - centroid_P
#     q = Q - centroid_Q

#     # Compute the covariance matrix
#     H = np.dot(p.T, q)

#     # SVD
#     U, S, Vt = np.linalg.svd(H)

#     # Validate right-handed coordinate system
#     if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
#         Vt[-1, :] *= -1.0

#     # Optimal rotation
#     R = np.dot(Vt.T, U.T)

#     # RMSD
#     rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

#     return R, t, rmsd


# def kabsch_with_optimal_assignment(P: np.ndarray, Q: np.ndarray):
#     assert P.shape == Q.shape
#     # Save centroids so we can return a correct translation later
#     centroid_P = P.mean(axis=0)
#     centroid_Q = Q.mean(axis=0)
#     P_c = P - centroid_P
#     Q_c = Q - centroid_Q

#     # Hungarian on centered coordinates
#     D = np.linalg.norm(P_c[:, None, :] - Q_c[None, :, :], axis=2)
#     row_ind, col_ind = linear_sum_assignment(D)
#     Q_matched = Q_c[col_ind]

#     # kabsch_numpy expects inputs already centered -> no internal centering
#     R, _, rmsd = kabsch_numpy(P_c, Q_matched)

#     # translation in original coordinates:
#     t = centroid_Q - centroid_P @ R.T
#     return R, t, rmsd, col_ind


# # -------------------------
# # Parse tensor output
# # -------------------------
# def parse_predicted_points(output_text: str) -> np.ndarray:
#     tensor_re = re.compile(
#         r"tensor\(\s*(\[[\s\S]*?\])\s*(?:, device=.*)?\)", re.MULTILINE
#     )
#     matches = list(tensor_re.finditer(output_text))
#     block = matches[-1].group(1) if matches else None
#     if not block:
#         raise ValueError("No tensor found in output.")

#     try:
#         arr = ast.literal_eval(block)
#         a = np.array(arr, dtype=float)
#         if a.ndim != 2 or a.shape[1] != 3:
#             raise ValueError(f"Unexpected array shape {a.shape}")
#         return a
#     except Exception as e:
#         raise ValueError(f"Failed to parse predicted points: {e}")


# # -------------------------
# # Helpers
# # -------------------------
# def normalize_base_label(label: str) -> str:
#     return re.sub(r"_[0-9]+$", "", label)


# def extract_gt_points(poly_dict: dict) -> np.ndarray:
#     c = np.asarray(poly_dict["center"]["coords"], float)
#     n = [np.asarray(x["coords"], float) for x in poly_dict["neighbors"]]
#     return np.vstack([c] + n)


# # -------------------------
# # Main
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--json_path", required=True, help="path to test_one_polyhedron.json"
#     )
#     parser.add_argument("--model_path", required=True, help="path to model directory")
#     parser.add_argument(
#         "--sample_py", default="scripts/sample_poly.py", help="path to sample script"
#     )
#     parser.add_argument(
#         "--python_bin", default=sys.executable, help="python binary to use"
#     )
#     parser.add_argument(
#         "--num_process",
#         type=int,
#         default=None,
#         help="If provided, only process this many polyhedra",
#     )
#     parser.add_argument(
#         "--extra_args", default="", help="extra args for sample_poly.py"
#     )
#     args = parser.parse_args()

#     with open(args.json_path, "r") as f:
#         data = json.load(f)

#     items = []
#     for entry in data:
#         for poly in entry.get("polyhedra", []):
#             label = poly.get("label")
#             if label:
#                 base = normalize_base_label(label)
#                 items.append((base, poly))

#     print(f"Found {len(items)} polyhedra in {args.json_path}")

#     total_rmsd = 0.0
#     matched_count = 0
#     processed = 0
#     failed = 0

#     shape_stats = defaultdict(lambda: {"total": 0, "matched": 0})

#     for idx, (formula, poly) in enumerate(items, start=1):
#         motif = poly.get("motif_type", "Unknown")
#         print(f"\n[{idx}/{len(items)}] {formula} — {motif}")

#         cmd = [
#             args.python_bin,
#             args.sample_py,
#             "--model_path",
#             args.model_path,
#             "--formula",
#             formula,
#             "--num_evals",
#             "5",
#         ]
#         if args.extra_args:
#             cmd += args.extra_args.split()

#         try:
#             proc = subprocess.run(
#                 cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 check=False,
#             )
#             out = proc.stdout
#         except Exception as e:
#             print(f"  subprocess failed: {e}")
#             failed += 1
#             continue

#         try:
#             pred = parse_predicted_points(out)
#         except Exception as e:
#             print(f"  parse failed: {e}")
#             failed += 1
#             continue

#         gt = extract_gt_points(poly)

#         if pred.shape[0] != gt.shape[0]:
#             print(
#                 f"  point count mismatch: pred={pred.shape[0]} gt={gt.shape[0]} -> skipped"
#             )
#             failed += 1
#             continue

#         # pred -= pred.mean(axis=0, keepdims=True)
#         # gt -= gt.mean(axis=0, keepdims=True)

#         try:
#             _, _, rmsd, _ = kabsch_with_optimal_assignment(pred, gt)
#         except Exception as e:
#             print(f"  alignment failed: {e}")
#             failed += 1
#             continue

#         processed += 1
#         total_rmsd += rmsd
#         is_match = rmsd < 1.0
#         if is_match:
#             matched_count += 1

#         shape_stats[motif]["total"] += 1
#         if is_match:
#             shape_stats[motif]["matched"] += 1

#         print(f"  RMSD = {rmsd:.4f} Å  -> {'MATCH' if is_match else 'NO MATCH'}")

#         if args.num_process and processed == args.num_process:
#             print(f"\nReached --num_process limit of {args.num_process}. Stopping.")
#             break

#     avg_rmsd = total_rmsd / processed if processed > 0 else float("nan")

#     print("\n========== SUMMARY ==========")
#     print(f"Processed polyhedra: {processed}")
#     print(f"Failed runs: {failed}")
#     print(f"Matched (<1.0 Å): {matched_count}")
#     print(f"Average RMSD: {avg_rmsd:.4f} Å")
#     print("=============================\n")

#     print("Shape-wise match summary:")
#     for shape, stats in sorted(shape_stats.items()):
#         total = stats["total"]
#         matched = stats["matched"]
#         pct = 100 * matched / total if total > 0 else 0.0
#         print(f"  {shape}: {matched} / {total} ({pct:.2f}%)")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
batch_eval_all_polys_rmsd.py

Runs your PolyDiff sampler for each polyhedron in test_one_polyhedron.json,
computes Hungarian-Kabsch RMSD, counts how many are below 1.0 Å,
prints per-polyhedron RMSD and a final summary including per-shape match percentages.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def kabsch_numpy(P: np.ndarray, Q: np.ndarray):
    assert P.shape == Q.shape, "Matrix dimensions must match"
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    t = centroid_Q - centroid_P
    p = P - centroid_P
    q = Q - centroid_Q
    H = np.dot(p.T, q)
    U, S, Vt = np.linalg.svd(H)
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0
    R = np.dot(Vt.T, U.T)
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])
    return R, t, rmsd


def kabsch_with_optimal_assignment(P: np.ndarray, Q: np.ndarray):
    assert P.shape == Q.shape
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q
    D = np.linalg.norm(P_c[:, None, :] - Q_c[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(D)
    Q_matched = Q_c[col_ind]
    R, _, rmsd = kabsch_numpy(P_c, Q_matched)
    t = centroid_Q - centroid_P @ R.T
    return R, t, rmsd, col_ind


### changed — parse multiple samples separated by ===SAMPLE N=== markers
def parse_all_samples(output_text: str) -> list:
    """Parse output with multiple ===SAMPLE N=== blocks. Returns list of np arrays."""
    # Split by sample markers
    parts = re.split(r"===SAMPLE \d+===", output_text)
    # First part is before any marker (model loading prints etc), skip it
    sample_blocks = parts[1:]  # everything after first marker

    tensor_re = re.compile(
        r"tensor\(\s*(\[[\s\S]*?\])\s*(?:, device=.*)?\)", re.MULTILINE
    )

    samples = []
    for block in sample_blocks:
        matches = list(tensor_re.finditer(block))
        if not matches:
            continue
        # Take the last tensor in this block
        raw = matches[-1].group(1)
        try:
            arr = ast.literal_eval(raw)
            a = np.array(arr, dtype=float)
            if a.ndim == 2 and a.shape[1] == 3:
                samples.append(a)
        except Exception:
            continue

    # Fallback: if no markers found, try old single-tensor parsing
    if not samples:
        matches = list(tensor_re.finditer(output_text))
        if matches:
            raw = matches[-1].group(1)
            try:
                arr = ast.literal_eval(raw)
                a = np.array(arr, dtype=float)
                if a.ndim == 2 and a.shape[1] == 3:
                    samples.append(a)
            except Exception:
                pass

    return samples


def normalize_base_label(label: str) -> str:
    return re.sub(r"_[0-9]+$", "", label)


def extract_gt_points(poly_dict: dict) -> np.ndarray:
    c = np.asarray(poly_dict["center"]["coords"], float)
    n = [np.asarray(x["coords"], float) for x in poly_dict["neighbors"]]
    return np.vstack([c] + n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path", required=True, help="path to test_one_polyhedron.json"
    )
    parser.add_argument("--model_path", required=True, help="path to model directory")
    parser.add_argument(
        "--sample_py", default="scripts/sample_poly.py", help="path to sample script"
    )
    parser.add_argument(
        "--python_bin", default=sys.executable, help="python binary to use"
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=None,
        help="If provided, only process this many polyhedra",
    )
    ### changed — add num_evals argument
    parser.add_argument(
        "--num_evals",
        type=int,
        default=5,
        help="Number of samples per polyhedron (takes best RMSD)",
    )
    parser.add_argument(
        "--extra_args", default="", help="extra args for sample_poly.py"
    )
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    items = []
    for entry in data:
        for poly in entry.get("polyhedra", []):
            label = poly.get("label")
            if label:
                base = normalize_base_label(label)
                items.append((base, poly))

    print(f"Found {len(items)} polyhedra in {args.json_path}")
    print(f"Generating {args.num_evals} sample(s) per polyhedron, taking best RMSD\n")

    all_rmsds = []
    matched_count = 0
    processed = 0
    failed = 0

    shape_stats = defaultdict(lambda: {"total": 0, "matched": 0, "rmsds": []})

    for idx, (formula, poly) in enumerate(items, start=1):
        motif = poly.get("motif_type", "Unknown")
        print(f"\n[{idx}/{len(items)}] {formula} — {motif}")

        ### changed — pass num_evals to sample_poly.py
        cmd = [
            args.python_bin,
            args.sample_py,
            "--model_path",
            args.model_path,
            "--formula",
            formula,
            "--num_evals",
            str(args.num_evals),
        ]
        if args.extra_args:
            cmd += args.extra_args.split()

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            out = proc.stdout
        except Exception as e:
            print(f"  subprocess failed: {e}")
            failed += 1
            continue

        ### changed — parse multiple samples
        pred_samples = parse_all_samples(out)
        if not pred_samples:
            print(f"  parse failed: no valid tensors found")
            failed += 1
            continue

        gt = extract_gt_points(poly)

        ### changed — compute RMSD for each sample, take best
        sample_rmsds = []
        for si, pred in enumerate(pred_samples):
            if pred.shape[0] != gt.shape[0]:
                continue
            try:
                _, _, rmsd, _ = kabsch_with_optimal_assignment(pred, gt)
                sample_rmsds.append(rmsd)
            except Exception:
                continue

        if not sample_rmsds:
            print(
                f"  all {len(pred_samples)} samples failed (shape mismatch or alignment error)"
            )
            failed += 1
            continue

        best_rmsd = min(sample_rmsds)
        processed += 1
        all_rmsds.append(best_rmsd)
        is_match = best_rmsd < 1.0
        if is_match:
            matched_count += 1

        shape_stats[motif]["total"] += 1
        shape_stats[motif]["rmsds"].append(best_rmsd)
        if is_match:
            shape_stats[motif]["matched"] += 1

        ### changed — print best of N and scale ranges for debugging
        print(
            f"  best RMSD = {best_rmsd:.4f} Å (best of {len(sample_rmsds)}/{len(pred_samples)}) "
            f"-> {'MATCH' if is_match else 'NO MATCH'}"
        )
        # Scale check on first valid sample
        first_valid = pred_samples[0]
        print(
            f"  pred range: [{first_valid.min():.2f}, {first_valid.max():.2f}]  "
            f"gt range: [{gt.min():.2f}, {gt.max():.2f}]"
        )

        if args.num_process and processed == args.num_process:
            print(f"\nReached --num_process limit of {args.num_process}. Stopping.")
            break

    ### changed — added median RMSD and per-shape median
    avg_rmsd = np.mean(all_rmsds) if all_rmsds else float("nan")
    median_rmsd = np.median(all_rmsds) if all_rmsds else float("nan")

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Processed polyhedra : {processed}")
    print(f"Failed runs         : {failed}")
    print(
        f"Matched (<1.0 Å)    : {matched_count} / {processed} ({100*matched_count/max(processed,1):.1f}%)"
    )
    print(f"Mean RMSD           : {avg_rmsd:.4f} Å")
    print(f"Median RMSD         : {median_rmsd:.4f} Å")
    print("=" * 40)

    print("\nShape-wise summary:")
    for shape, stats in sorted(shape_stats.items(), key=lambda x: -x[1]["total"]):
        total = stats["total"]
        matched = stats["matched"]
        pct = 100 * matched / total if total > 0 else 0.0
        med = np.median(stats["rmsds"]) if stats["rmsds"] else float("nan")
        print(f"  {shape:30s}: {matched:3d}/{total:3d} ({pct:5.1f}%)  median={med:.4f}")


if __name__ == "__main__":
    main()
