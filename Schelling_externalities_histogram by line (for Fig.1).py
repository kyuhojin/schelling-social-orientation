# schelling_seg_distribution_fixed.py
import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count, set_start_method

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Optional KDE smoothing. If SciPy is unavailable, we use a smoothed-hist fallback.
try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================
# Config
# =========================
@dataclass
class Config:
    grid_size: int = 32
    vacancy_ratio: float = 0.20
    threshold: float = 0.50
    neighborhood: str = "moore"     # "moore" or "vonneumann"
    wrap: bool = True
    reps: int = 10000
    cores: int = 16
    max_steps: int = 10_000
    initial_mix: Tuple[float, float] = (0.5, 0.5)  # +1/-1 among occupied agents
    seed: int = 20250930
    out_root: str = "out"

    def outdir(self) -> str:
        tag = f"GS{self.grid_size}_VR{self.vacancy_ratio:.2f}_thr{self.threshold:.2f}_{self.neighborhood}_seed_{self.seed}_reps{self.reps}"
        return os.path.join(self.out_root, f"{tag}_seg_distributions_fixed")


# =========================
# Neighborhood helpers
# =========================
def neighborhood_offsets(kind: str) -> List[Tuple[int, int]]:
    k = kind.lower()
    if k in ("moore", "moore8", "moore-8"):
        return [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
    if k in ("vonneumann", "von-neumann", "vn", "neumann"):
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    raise ValueError("neighborhood must be 'moore' or 'vonneumann'")


def neighbors_of(r: int, c: int, n: int, offs: List[Tuple[int, int]], wrap: bool) -> List[Tuple[int, int]]:
    res = []
    for dr, dc in offs:
        rr, cc = r + dr, c + dc
        if wrap:
            res.append((rr % n, cc % n))
        else:
            if 0 <= rr < n and 0 <= cc < n:
                res.append((rr, cc))
    return res


# =========================
# Grid & satisfaction
# =========================
def make_initial_grid(n: int, vacancy_ratio: float, initial_mix: Tuple[float, float], rng: random.Random) -> np.ndarray:
    total = n * n
    num_vac = int(round(vacancy_ratio * total))
    num_agents = total - num_vac
    num_pos = int(round(initial_mix[0] * num_agents))
    num_neg = num_agents - num_pos
    arr = np.array([+1] * num_pos + [-1] * num_neg + [0] * num_vac, dtype=np.int8)
    rng.shuffle(arr)
    return arr.reshape((n, n))


def is_satisfied(grid: np.ndarray, r: int, c: int, threshold: float, offs, wrap: bool) -> bool:
    n = grid.shape[0]
    s = grid[r, c]
    if s == 0:
        return True
    neigh = neighbors_of(r, c, n, offs, wrap)
    likes, tot = 0, 0
    for rr, cc in neigh:
        val = grid[rr, cc]
        if val != 0:
            tot += 1
            if val == s:
                likes += 1
    if tot == 0:
        return True
    return (likes / tot) >= threshold


def satisfied_mask(grid: np.ndarray, threshold: float, offs, wrap: bool) -> np.ndarray:
    n = grid.shape[0]
    sat = np.ones((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            if grid[r, c] != 0:
                sat[r, c] = is_satisfied(grid, r, c, threshold, offs, wrap)
    return sat


# =========================
# Candidate evaluation utilities
# =========================
def candidate_sites_that_satisfy_agent(grid: np.ndarray, agent_sign: int, threshold: float,
                                       offs, wrap: bool) -> List[Tuple[int, int]]:
    n = grid.shape[0]
    cands = []
    for r in range(n):
        for c in range(n):
            if grid[r, c] == 0:
                neigh = neighbors_of(r, c, n, offs, wrap)
                likes, tot = 0, 0
                for rr, cc in neigh:
                    val = grid[rr, cc]
                    if val != 0:
                        tot += 1
                        if val == agent_sign:
                            likes += 1
                if tot == 0 or (likes / tot) >= threshold:
                    cands.append((r, c))
    return cands


def count_satisfied_neighbors_if_move_in(grid: np.ndarray, r: int, c: int, agent_sign: int,
                                         threshold: float, offs, wrap: bool) -> Tuple[int, int]:
    n = grid.shape[0]
    neigh = neighbors_of(r, c, n, offs, wrap)
    sat_before = 0
    neighbor_positions = []
    for rr, cc in neigh:
        if grid[rr, cc] != 0:
            neighbor_positions.append((rr, cc))
            if is_satisfied(grid, rr, cc, threshold, offs, wrap):
                sat_before += 1
    sat_after = 0
    for rr, cc in neighbor_positions:
        s = grid[rr, cc]
        local = neighbors_of(rr, cc, n, offs, wrap)
        likes, tot = 0, 0
        for r2, c2 in local:
            val = agent_sign if (r2 == r and c2 == c) else grid[r2, c2]
            if val != 0:
                tot += 1
                if val == s:
                    likes += 1
        if tot == 0 or (likes / tot) >= threshold:
            sat_after += 1
    return sat_before, sat_after


def would_any_currently_satisfied_neighbor_become_unsatisfied(grid: np.ndarray, r: int, c: int,
                                                              agent_sign: int, threshold: float,
                                                              offs, wrap: bool) -> bool:
    n = grid.shape[0]
    neigh = neighbors_of(r, c, n, offs, wrap)
    sat_now = []
    for rr, cc in neigh:
        if grid[rr, cc] != 0 and is_satisfied(grid, rr, cc, threshold, offs, wrap):
            sat_now.append((rr, cc))
    for rr, cc in sat_now:
        s = grid[rr, cc]
        local = neighbors_of(rr, cc, n, offs, wrap)
        likes, tot = 0, 0
        for r2, c2 in local:
            val = agent_sign if (r2 == r and c2 == c) else grid[r2, c2]
            if val != 0:
                tot += 1
                if val == s:
                    likes += 1
        if not (tot == 0 or (likes / tot) >= threshold):
            return True
    return False


def num_unsatisfied_neighbors_after_move(grid: np.ndarray, r: int, c: int, agent_sign: int,
                                         threshold: float, offs, wrap: bool) -> int:
    n = grid.shape[0]
    neigh = neighbors_of(r, c, n, offs, wrap)
    cnt = 0
    for rr, cc in neigh:
        if grid[rr, cc] != 0:
            s = grid[rr, cc]
            local = neighbors_of(rr, cc, n, offs, wrap)
            likes, tot = 0, 0
            for r2, c2 in local:
                val = agent_sign if (r2 == r and c2 == c) else grid[r2, c2]
                if val != 0:
                    tot += 1
                    if val == s:
                        likes += 1
            if not (tot == 0 or (likes / tot) >= threshold):
                cnt += 1
    return cnt


# =========================
# Metrics
# =========================
def segregation_stat(grid: np.ndarray, offs, wrap: bool) -> float:
    """
    S = (1 / (2 * N)) * sum_{i,j} A_ij S_i S_j, undirected edges counted twice.
    S_i ∈ {+1,-1,0}; vacancies contribute 0.
    """
    n = grid.shape[0]
    num_agents = int(np.count_nonzero(grid))
    if num_agents == 0:
        return 0.0
    total = 0.0
    for r in range(n):
        for c in range(n):
            Si = grid[r, c]
            if Si == 0:
                continue
            for rr, cc in neighbors_of(r, c, n, offs, wrap):
                Sj = grid[rr, cc]
                if Sj != 0:
                    total += (Si * Sj)
    return float(total) / (2.0 * num_agents)


# =========================
# Move rules (Schelling, PEF, PEO, NEA)
# =========================
def move_rule_schelling(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    s = grid[r, c]
    cands = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def move_rule_pep(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    # Satisfied self AND (neighbors' satisfied count after >= before). Pick randomly.
    s = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    cands = []
    for rr, cc in raw:
        before, after = count_satisfied_neighbors_if_move_in(grid, rr, cc, s, threshold, offs, wrap)
        if after >= before:
            cands.append((rr, cc))
    if not cands: return None
    return cands[rng.randrange(len(cands))]


def move_rule_peo(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    # Among PEF-feasible sites, maximize neighbors' satisfaction (or minimize unsatisfied neighbors).
    s = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    filtered = []
    for rr, cc in raw:
        before, after = count_satisfied_neighbors_if_move_in(grid, rr, cc, s, threshold, offs, wrap)
        if after >= before:
            unsat_after = num_unsatisfied_neighbors_after_move(grid, rr, cc, s, threshold, offs, wrap)
            filtered.append(((rr, cc), unsat_after))
    if not filtered: return None
    min_unsat = min(v for (_, v) in filtered)
    top = [pos for (pos, v) in filtered if v == min_unsat]
    return top[rng.randrange(len(top))]


def move_rule_nea(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    # Satisfied self AND no currently satisfied neighbor becomes unsatisfied. Pick randomly.
    s = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    cands = []
    for rr, cc in raw:
        if not would_any_currently_satisfied_neighbor_become_unsatisfied(grid, rr, cc, s, threshold, offs, wrap):
            cands.append((rr, cc))
    if not cands: return None
    return cands[rng.randrange(len(cands))]


MOVE_RULES = {
    "SCHELLING": move_rule_schelling,
    "PEF": move_rule_pep,
    "PEO": move_rule_peo,
    "NEA": move_rule_nea,
}


# =========================
# Simulation core
# =========================
def step_once(grid: np.ndarray, threshold: float, offs, wrap: bool,
              strategy: str, rng: random.Random) -> Tuple[np.ndarray, bool, int]:
    n = grid.shape[0]
    sat = satisfied_mask(grid, threshold, offs, wrap)
    unsat_positions = [(r, c) for r in range(n) for c in range(n)
                       if grid[r, c] != 0 and not sat[r, c]]
    rng.shuffle(unsat_positions)

    any_move, moves = False, 0
    move_fn = MOVE_RULES[strategy]
    for r, c in unsat_positions:
        dest = move_fn(grid, r, c, threshold, offs, wrap, rng)
        if dest is not None:
            rr, cc = dest
            if grid[rr, cc] == 0:
                agent = grid[r, c]
                grid[r, c] = 0
                grid[rr, cc] = agent
                any_move = True
                moves += 1
    return grid, any_move, moves


def run_one(cfg: Config, strategy: str, rep: int) -> Dict:
    n = cfg.grid_size
    offs = neighborhood_offsets(cfg.neighborhood)

    # deterministic seed per (strategy, rep)
    seed_base = (cfg.seed + 9000 * {"SCHELLING": 1, "PEF": 2, "PEO": 3, "NEA": 4}[strategy] + rep)
    rng = random.Random(seed_base)

    grid = make_initial_grid(n, cfg.vacancy_ratio, cfg.initial_mix, rng)

    total_moves, steps = 0, 0
    for steps in range(1, cfg.max_steps + 1):
        grid, any_move, moves = step_once(grid, cfg.threshold, offs, cfg.wrap, strategy, rng)
        total_moves += moves
        if not any_move:
            break

    seg = segregation_stat(grid, offs, cfg.wrap)

    return {
        "strategy": strategy,
        "rep": rep,
        "final_segregation": seg,
        "steps_to_equilibrium": steps,
        "total_moves": total_moves
    }


def _worker(job):
    cfg_dict, strategy, rep = job
    cfg = Config(**cfg_dict)
    return run_one(cfg, strategy, rep)


# =========================
# Plotting — fixed x-ranges + dot-connected lines
# =========================
def _compute_xrange_from_data(df: pd.DataFrame) -> tuple[float, float]:
    xmin = float(df["final_segregation"].min())
    xmax = float(df["final_segregation"].max())
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = -1.0, 1.0  # degenerate fallback
    pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    return xmin - pad, xmax + pad


def _smooth_hist_line(values: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Fallback smoothing if SciPy is unavailable: smoothed histogram density
       over the SAME x-range as the line plot."""
    if len(values) <= 1:
        return np.zeros_like(xs)
    hist_bins = max(20, int(np.sqrt(len(values))) * 2)
    hist, bin_edges = np.histogram(values, bins=hist_bins, range=(xs[0], xs[-1]), density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    k = max(3, len(hist) // 30)
    pad_left = np.repeat(hist[0], k)
    pad_right = np.repeat(hist[-1], k)
    padded = np.concatenate([pad_left, hist, pad_right])
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.interp(xs, centers, smoothed)


def plot_overlaid_hist(df: pd.DataFrame, out_dir: str, bins: int = 30):
    xmin, xmax = _compute_xrange_from_data(df)
    plt.figure(figsize=(7.2, 4.6), dpi=160)
    for strat, sub in df.groupby("strategy", sort=False):
        plt.hist(sub["final_segregation"], bins=bins, range=(xmin, xmax),
                 density=True, alpha=0.35, label=strat)
    plt.xlabel("Final segregation (S)")
    plt.ylabel("Probability density")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segregation_distribution_overlaid.png"))
    plt.savefig(os.path.join(out_dir, "segregation_distribution_overlaid.pdf"))
    plt.close()


def plot_grid_hists(df: pd.DataFrame, out_dir: str, bins: int = 30):
    xmin, xmax = _compute_xrange_from_data(df)
    rules = ["SCHELLING", "PEF", "PEO", "NEA"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=160)
    axes = axes.ravel()
    for ax, rule in zip(axes, rules):
        sub = df[df["strategy"] == rule]
        ax.hist(sub["final_segregation"], bins=bins, range=(xmin, xmax), density=True, alpha=0.6)
        ax.set_title(rule)
        ax.set_xlabel("S")
        ax.set_ylabel("Density")
        ax.set_xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segregation_distribution_grid.png"))
    plt.savefig(os.path.join(out_dir, "segregation_distribution_grid.pdf"))
    plt.close()


def plot_line_pdf(df: pd.DataFrame, out_dir: str):
    """Smoothed PDF (KDE if SciPy available, else smoothed histogram),
    drawn as connected lines with dot markers over the data’s actual range."""
    xmin, xmax = _compute_xrange_from_data(df)
    #xs = np.linspace(xmin, xmax, 400)
    xs = np.linspace(xmin, xmax, 2000)
    plt.figure(figsize=(7.2, 4.6), dpi=160)
    for strat, sub in df.groupby("strategy", sort=False):
        data = sub["final_segregation"].values
        if _HAS_SCIPY and len(data) > 1 and np.std(data) > 1e-12:
            kde = gaussian_kde(data, bw_method=0.5)  # 0.5 makes it smoother (default ~ n^(-1/5)) Scott’s rule by default
            ys = kde(xs)
        else:
            ys = _smooth_hist_line(data, xs)
        #plt.plot(xs, ys, marker="o", markevery=max(1, len(xs)//20), linewidth=2, label=strat)
        plt.plot(xs, ys, marker="o", markevery=max(1, len(xs)//50), linewidth=2, label=strat)
    plt.xlabel("Final segregation (S)")
    plt.ylabel("Probability density")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segregation_distribution_lines.png"))
    plt.savefig(os.path.join(out_dir, "segregation_distribution_lines.pdf"))
    plt.close()


def plot_ecdf(df: pd.DataFrame, out_dir: str):
    xmin, xmax = _compute_xrange_from_data(df)
    plt.figure(figsize=(7.2, 4.6), dpi=160)
    for strat, sub in df.groupby("strategy", sort=False):
        x = np.sort(sub["final_segregation"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=strat)
    plt.xlabel("Final segregation (S)")
    plt.ylabel("ECDF")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segregation_distribution_ecdf.png"))
    plt.savefig(os.path.join(out_dir, "segregation_distribution_ecdf.pdf"))
    plt.close()


def plot_line_ecdf(df: pd.DataFrame, out_dir: str):
    xmin, xmax = _compute_xrange_from_data(df)
    plt.figure(figsize=(7.2, 4.6), dpi=160)
    for strat, sub in df.groupby("strategy", sort=False):
        x = np.sort(sub["final_segregation"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        #plt.plot(x, y, marker="o", markevery=max(1, len(x)//20), linewidth=2, label=strat)
        plt.plot(x, y, marker="o", markevery=max(1, len(x)//50), linewidth=2, label=strat)
    plt.xlabel("Final segregation (S)")
    plt.ylabel("ECDF")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segregation_distribution_ecdf_lines.png"))
    plt.savefig(os.path.join(out_dir, "segregation_distribution_ecdf_lines.pdf"))
    plt.close()


# =========================
# Main
# =========================
def main():
    cfg = Config(
        grid_size=32,
        vacancy_ratio=0.20,
        threshold=0.50,
        neighborhood="moore",   # or "vonneumann"
        wrap=True,
        reps=10000,
        cores=min(32, cpu_count()),
        max_steps=10_000,
        initial_mix=(0.5, 0.5),
        seed=20251003,
        out_root="out"
    )

    outdir = cfg.outdir()
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    strategies = ["SCHELLING", "PEF", "PEO", "NEA"]

    # Build jobs
    jobs = []
    cfg_dict = asdict(cfg)
    for strat in strategies:
        for rep in range(cfg.reps):
            jobs.append((cfg_dict, strat, rep))

    # Run in parallel with progress bar
    set_start_method("spawn", force=True)
    results = []
    with Pool(processes=min(cfg.cores, cpu_count())) as pool:
        for r in tqdm(pool.imap_unordered(_worker, jobs),
                      total=len(jobs), desc="Running simulations", unit="run"):
            results.append(r)

    # Save raw per-run segregation
    df = pd.DataFrame(results).sort_values(["strategy", "rep"]).reset_index(drop=True)
    df.to_csv(os.path.join(outdir, "results_segregation_raw.csv"), index=False)

    # Quick table of mean ± SE
    def se(x: pd.Series) -> float:
        n = len(x)
        if n <= 1: return 0.0
        return float(x.std(ddof=1) / math.sqrt(n))

    summary = df.groupby("strategy").agg(
        mean_S=("final_segregation", "mean"),
        se_S=("final_segregation", se),
        mean_steps=("steps_to_equilibrium", "mean"),
        se_steps=("steps_to_equilibrium", se),
    ).reset_index()
    summary.to_csv(os.path.join(outdir, "results_segregation_summary.csv"), index=False)
    print("\n=== Final segregation by rule (mean ± SE) ===")
    print(summary.to_string(index=False))
    print("S range:", df["final_segregation"].min(), "to", df["final_segregation"].max())

    # Figures — bars AND lines (with correct x-range)
    plot_overlaid_hist(df, figdir, bins=30)
    plot_grid_hists(df, figdir, bins=30)
    plot_ecdf(df, figdir)                # step ECDF (bar-style)
    plot_line_pdf(df, figdir)            # KDE / smoothed line with dots
    plot_line_ecdf(df, figdir)           # ECDF as dot-connected lines

    print(f"\nSaved raw results to : {os.path.join(outdir, 'results_segregation_raw.csv')}")
    print(f"Saved summary to     : {os.path.join(outdir, 'results_segregation_summary.csv')}")
    print(f"Saved figures in     : {figdir}")
    if not _HAS_SCIPY:
        print("Note: SciPy not found; used smoothed histograms for line PDFs. Install scipy for KDE: pip install scipy")


if __name__ == "__main__":
    main()
