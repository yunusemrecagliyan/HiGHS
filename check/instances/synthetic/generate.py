"""Synthetic feed-formulation-style MIPs matching YemYap model sizes.

Structure mirrors the real models: per-formula blocks (variable upper
bound rows + sparse nutrient rows + a total-balance row) plus a few
coupling rows spanning blocks. Feasibility is planted: pick x*, set row
bounds around activities. Objective: minimize random costs.
"""
import numpy as np


def build(n_blocks, vars_per_block, int_frac, n_bound, n_nutrient,
          nut_degree, n_combo, n_coupling, coupling_degree, seed):
    rng = np.random.default_rng(seed)
    V = vars_per_block
    n_col = n_blocks * V
    is_int = np.zeros(n_col, dtype=bool)
    for b in range(n_blocks):
        idx = np.arange(b * V, (b + 1) * V)
        n_int = int(round(V * int_frac))
        is_int[rng.choice(idx, size=n_int, replace=False)] = True
    U = np.round(rng.uniform(50, 400, size=n_col), 3)
    xi = rng.integers(0, (U / 4).astype(int) + 1).astype(float)
    xc = np.round(rng.uniform(0, U / 4), 3)
    xstar = np.where(is_int, xi, xc)
    cost = np.round(rng.uniform(1, 20, size=n_col), 3)

    rows = []  # (name, sense, rhs, [(col, coef)])
    # 1) ingredient bound rows (degree 1-2, like min/max rows)
    for b in range(n_blocks):
        idx = np.arange(b * V, (b + 1) * V)
        for t in range(n_bound):
            j = int(rng.choice(idx))
            if rng.random() < 0.7:
                rows.append((f"bd{b}_{t}", "L", U[j] * rng.uniform(0.8, 1.5),
                             [(j, 1.0)]))
            else:
                j2 = int(rng.choice(idx))
                rows.append((f"bd{b}_{t}", "L", U[j] * rng.uniform(0.8, 1.5),
                             [(j, 1.0), (j2, rng.uniform(0.1, 0.5))]))
    # 2) nutrient rows per block (sparse)
    for b in range(n_blocks):
        idx = np.arange(b * V, (b + 1) * V)
        for t in range(n_nutrient):
            k = max(2, int(rng.normal(nut_degree, nut_degree / 3)))
            cols = rng.choice(idx, size=min(k, V), replace=False)
            coefs = np.round(rng.uniform(0.01, 3.0, size=len(cols)), 3)
            act = float(np.dot(coefs, xstar[cols]))
            r = rng.random()
            if r < 0.45:
                rows.append((f"n{b}_{t}", "L", act * rng.uniform(1.02, 1.25),
                             list(zip(cols.tolist(), coefs.tolist()))))
            elif r < 0.9:
                rows.append((f"n{b}_{t}", "G", act * rng.uniform(0.75, 0.98),
                             list(zip(cols.tolist(), coefs.tolist()))))
            else:
                rows.append((f"n{b}_{t}", "E", act,
                             list(zip(cols.tolist(), coefs.tolist()))))
    # 3) per-block total balance (dense within block)
    for b in range(n_blocks):
        idx = np.arange(b * V, (b + 1) * V)
        act = float(xstar[idx].sum())
        rows.append((f"tot{b}", "E", act,
                     list(zip(idx.tolist(), [1.0] * V))))
    # 3b) combination rows (degree 4)
    for b in range(n_blocks):
        idx = np.arange(b * V, (b + 1) * V)
        for t in range(n_combo):
            cols = rng.choice(idx, size=4, replace=False)
            coefs = np.round(rng.uniform(0.5, 1.5, size=4), 3)
            act = float(np.dot(coefs, xstar[cols]))
            rows.append((f"cb{b}_{t}", "L", act * rng.uniform(1.05, 1.3),
                         list(zip(cols.tolist(), coefs.tolist()))))
    # 4) coupling rows across blocks
    for a in range(n_coupling):
        cols = rng.choice(n_col, size=coupling_degree, replace=False)
        coefs = np.round(rng.uniform(0.5, 2.0, size=len(cols)), 3)
        act = float(np.dot(coefs, xstar[cols]))
        if rng.random() < 0.6:
            rows.append((f"cpl{a}", "L", act * rng.uniform(1.03, 1.2),
                         list(zip(cols.tolist(), coefs.tolist()))))
        else:
            rows.append((f"cpl{a}", "G", act * rng.uniform(0.8, 0.97),
                         list(zip(cols.tolist(), coefs.tolist()))))
    return n_col, is_int, U, cost, rows


def write_mps(path, n_col, is_int, U, cost, rows):
    by_col = [[] for _ in range(n_col)]
    for ri, (name, sense, rhs, entries) in enumerate(rows):
        for (c, v) in entries:
            by_col[c].append((name, v))
    with open(path, "w") as f:
        f.write("NAME          SYNTH\nROWS\n N  Obj\n")
        for name, sense, rhs, _ in rows:
            f.write(f" {sense}  {name}\n")
        f.write("COLUMNS\n")
        for j in range(n_col):
            if is_int[j]:
                f.write(f"    MK{j}  'MARKER'  'INTORG'\n")
            f.write(f"    x{j}  Obj  {cost[j]:.6f}\n")
            for (name, v) in by_col[j]:
                f.write(f"    x{j}  {name}  {v:.6f}\n")
            if is_int[j]:
                f.write(f"    MK{j}  'MARKER'  'INTEND'\n")
        f.write("RHS\n")
        for name, sense, rhs, _ in rows:
            f.write(f"    RHS  {name}  {rhs:.6f}\n")
        f.write("BOUNDS\n")
        for j in range(n_col):
            f.write(f" UP BND  x{j}  {U[j]:.6f}\n")
        f.write("ENDATA\n")
    nnz = sum(len(e) for _, _, _, e in rows)
    print(f"{path}: rows={len(rows)} cols={n_col} nz={nnz} int={int(is_int.sum())}")


if __name__ == "__main__":
    import os
    os.makedirs("/tmp/synth", exist_ok=True)
    cfgs = [
        # (name, blocks, V, int_frac, bound/blk, nutr/blk, nut_deg, combo/blk, cpl, cpl_deg, seed)
        ("synth_adana", 30, 580, 0.156, 25, 45, 12, 6, 2, 40, 7),
        ("synth_salihli", 45, 790, 0.143, 25, 68, 9, 7, 3, 45, 11),
        ("synth_onlyadana", 45, 790, 0.143, 25, 68, 9, 7, 1, 41, 21),
        ("synth_problem", 45, 790, 0.143, 25, 68, 9, 7, 2, 43, 31),
    ]
    for name, *args in cfgs:
        n_col, is_int, U, cost, rows = build(*args)
        write_mps(f"/tmp/synth/{name}.mps", n_col, is_int, U, cost, rows)
