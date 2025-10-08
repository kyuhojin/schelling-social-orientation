# schelling_ext_representative_figs.py
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
from matplotlib.colors import ListedColormap


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
    initial_mix: Tuple[float, float] = (0.5, 0.5)  # +1 / -1 among occupied
    seed: int = 20250930
    out_root: str = "out"

    def outdir(self) -> str:
        tag = f"GS{self.grid_size}_VR{self.vacancy_ratio:.2f}_thr{self.threshold:.2f}_{self.neighborhood}_seed_{self.seed}_GRID_rep{self.reps}"
        return os.path.join(self.out_root, tag)


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
# Candidate evaluation
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


def fraction_satisfied(grid: np.ndarray, threshold: float, offs, wrap: bool) -> float:
    n = grid.shape[0]
    sat, tot = 0, 0
    for r in range(n):
        for c in range(n):
            if grid[r, c] != 0:
                tot += 1
                if is_satisfied(grid, r, c, threshold, offs, wrap):
                    sat += 1
    return 0.0 if tot == 0 else (sat / tot)


# =========================
# Move rules (four strategies)
# =========================
def move_rule_schelling(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    s = grid[r, c]
    cands = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def move_rule_pep(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    s = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    cands = []
    for rr, cc in raw:
        before, after = count_satisfied_neighbors_if_move_in(grid, rr, cc, s, threshold, offs, wrap)
        if after >= before:
            cands.append((rr, cc))
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def move_rule_pem(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    s = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    filtered = []
    for rr, cc in raw:
        before, after = count_satisfied_neighbors_if_move_in(grid, rr, cc, s, threshold, offs, wrap)
        if after >= before:
            unsat_after = num_unsatisfied_neighbors_after_move(grid, rr, cc, s, threshold, offs, wrap)
            filtered.append(((rr, cc), unsat_after))
    if not filtered:
        return None
    min_unsat = min(v for (_, v) in filtered)
    top = [pos for (pos, v) in filtered if v == min_unsat]
    return top[rng.randrange(len(top))]


def move_rule_nep(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    s = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, s, threshold, offs, wrap)
    cands = []
    for rr, cc in raw:
        if not would_any_currently_satisfied_neighbor_become_unsatisfied(grid, rr, cc, s, threshold, offs, wrap):
            cands.append((rr, cc))
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


MOVE_RULES = {
    "SCHELLING": move_rule_schelling,
    "PEF": move_rule_pep,
    "PEO": move_rule_pem,
    "NEA": move_rule_nep,
}


# =========================
# Simulation
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
    seed_base = (cfg.seed + 5000 * {"SCHELLING": 1, "PEF": 2, "PEO": 3, "NEA": 4}[strategy] + rep)
    rng = random.Random(seed_base)

    init_grid = make_initial_grid(n, cfg.vacancy_ratio, cfg.initial_mix, rng).copy()
    grid = init_grid.copy()

    total_moves, steps = 0, 0
    for steps in range(1, cfg.max_steps + 1):
        grid, any_move, moves = step_once(grid, cfg.threshold, offs, cfg.wrap, strategy, rng)
        total_moves += moves
        if not any_move:
            break

    frac_sat = fraction_satisfied(grid, cfg.threshold, offs, cfg.wrap)
    seg = segregation_stat(grid, offs, cfg.wrap)
    num_agents = int(np.count_nonzero(grid))
    mpa = (total_moves / num_agents) if num_agents > 0 else 0.0

    return {
        "strategy": strategy,
        "rep": rep,
        "final_fraction_satisfied": frac_sat,
        "final_segregation": seg,
        "steps_to_equilibrium": steps,
        "total_moves": total_moves,
        "moves_per_agent": mpa,
        "init_grid": init_grid,    # keep arrays (small for 32x32)
        "final_grid": grid
    }


def _worker(job):
    cfg_dict, strategy, rep = job
    cfg = Config(**cfg_dict)
    return run_one(cfg, strategy, rep)


# =========================
# Representative selection
# =========================
def select_representative_runs(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    For each strategy, select the rep whose (metrics) vector is closest to that
    strategy's mean (Euclidean distance in standardized metric space).
    Returns a dataframe with one row per strategy (including rep and metrics).
    """
    reps = []
    for strat, g in df.groupby("strategy"):
        means = g[metrics].mean()
        stds = g[metrics].std(ddof=1).replace(0, 1.0)  # avoid div-by-zero
        z = (g[metrics] - means) / stds
        dist = (z**2).sum(axis=1) ** 0.5
        idx = dist.idxmin()
        row = g.loc[idx].copy()
        row["distance_to_mean"] = dist.loc[idx]
        reps.append(row)
    return pd.DataFrame(reps).reset_index(drop=True)


# =========================
# Plotting
# =========================
def plot_grid(ax, grid: np.ndarray, title: str):
    # Map: -1 -> 0 (e.g., dark color), 0 -> 1 (white), +1 -> 2 (light color); customize as needed.
    # Here: [-1, 0, +1] -> [0,1,2] with a clear, colorblind-friendly palette.
    mapping = { -1: 0, 0: 1, +1: 2 }
    disp = np.vectorize(mapping.get)(grid)
    cmap = ListedColormap(["#FF2802B3", "#f2f2f2", "#003464"])  # blue (−1), light gray (vacancy), green (+1)
    ax.imshow(disp, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def save_strategy_figures(reps_df: pd.DataFrame, outdir: str):
    # Individual side-by-side figures per strategy
    ind_dir = os.path.join(outdir, "figures_by_strategy")
    os.makedirs(ind_dir, exist_ok=True)

    for _, row in reps_df.iterrows():
        strat = row["strategy"]
        init_grid = row["init_grid"]
        final_grid = row["final_grid"]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=160)
        plot_grid(axes[0], init_grid, f"{strat}: Initial grid")
        plot_grid(axes[1], final_grid, f"{strat}: Final grid")
        plt.tight_layout()
        fig.savefig(os.path.join(ind_dir, f"{strat}_initial_final.png"))
        fig.savefig(os.path.join(ind_dir, f"{strat}_initial_final.pdf"))
        plt.close(fig)

    # Combined 4x2 panel
    order = ["SCHELLING", "NEA", "PEF", "PEO"]
    reps_df = reps_df.set_index("strategy").loc[order].reset_index()

    fig, axes = plt.subplots(len(order), 2, figsize=(8, 8), dpi=170)
    for i, (_, row) in enumerate(reps_df.iterrows()):
        plot_grid(axes[i, 0], row["init_grid"], f"{row['strategy']}: Initial")
        plot_grid(axes[i, 1], row["final_grid"], f"{row['strategy']}: Final")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison_initial_final_all.png"))
    fig.savefig(os.path.join(outdir, "comparison_initial_final_all.pdf"))
    plt.close(fig)


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
        seed=20251002,
        out_root="out"
    )

    outdir = cfg.outdir()
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    strategies = ["SCHELLING", "PEF", "PEO", "NEA"]

    # Build jobs
    jobs = []
    cfg_dict = asdict(cfg)
    for strat in strategies:
        for rep in range(cfg.reps):
            jobs.append((cfg_dict, strat, rep))

    # Run
    set_start_method("spawn", force=True)
    results = []
    with Pool(processes=min(cfg.cores, cpu_count())) as pool:
        for r in tqdm(pool.imap_unordered(_worker, jobs),
                      total=len(jobs), desc="Running simulations", unit="run"):
            results.append(r)

    # Collect into DataFrame (keep grids for representative selection)
    df = pd.DataFrame(results)

    # Save per-run metrics (without bulky grids) for record
    slim = df.drop(columns=["init_grid", "final_grid"]).copy()
    slim.to_csv(os.path.join(outdir, "results_raw.csv"), index=False)

    # Select representative run per strategy
    # Use three core metrics to define "average": satisfaction, segregation, steps.
    metrics = ["final_fraction_satisfied", "final_segregation", "steps_to_equilibrium"]
    reps_df = select_representative_runs(df, metrics)

    # Save record of chosen reps
    reps_info = reps_df.drop(columns=["init_grid", "final_grid"]).copy()
    reps_info.to_csv(os.path.join(outdir, "representative_runs.csv"), index=False)

    # Save figures (per strategy + combined)
    save_strategy_figures(reps_df, outdir)

    print("\n=== Representative runs (closest to strategy means) ===")
    print(reps_info.to_string(index=False))
    print(f"\nSaved per-run metrics to       : {os.path.join(outdir, 'results_raw.csv')}")
    print(f"Saved representative run table : {os.path.join(outdir, 'representative_runs.csv')}")
    print(f"Saved figures in               : {outdir}")


if __name__ == "__main__":
    main()
