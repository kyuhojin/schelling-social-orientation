# schelling_ext_representative_5runs_fixed.py
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
    neighborhood: str = "moore"   # "moore" or "vonneumann"
    wrap: bool = True
    reps: int = 10000
    cores: int = 16
    max_steps: int = 10_000
    initial_mix: Tuple[float, float] = (0.5, 0.5)  # +1 / -1 among occupied agents
    seed: int = 20250930
    out_root: str = "out"

    def outdir(self) -> str:
        tag = f"GS{self.grid_size}_{self.neighborhood}_Final Grids_5_reps_{self.reps}"
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
    # Satisfied self AND (neighbors' satisfied count after >= before). Pick randomly.
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


def move_rule_peo(grid, r, c, threshold, offs, wrap, rng) -> Optional[Tuple[int, int]]:
    # Among PEF-feasible sites, choose the one minimizing unsatisfied neighbors after the move (tie → random).
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
    # Satisfied self AND no currently satisfied neighbor becomes unsatisfied. Pick randomly.
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
    "PEO": move_rule_peo,
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

    return {
        "strategy": strategy,
        "rep": rep,
        "final_fraction_satisfied": frac_sat,
        "final_segregation": seg,
        "steps_to_equilibrium": steps,
        "total_moves": total_moves,
        "init_grid": init_grid,
        "final_grid": grid
    }


def _worker(job):
    cfg_dict, strategy, rep = job
    cfg = Config(**cfg_dict)
    return run_one(cfg, strategy, rep)


# =========================
# Representative selection (top-k closest to mean)
# =========================
def select_k_representatives(df: pd.DataFrame, metrics: List[str], k: int = 5) -> pd.DataFrame:
    out = []
    for strat, g in df.groupby("strategy"):
        means = g[metrics].mean()
        stds = g[metrics].std(ddof=1).replace(0, 1.0)  # avoid div-by-zero
        z = (g[metrics] - means) / stds
        dist = (z**2).sum(axis=1) ** 0.5
        gg = g.assign(distance_to_mean=dist).sort_values("distance_to_mean").head(k)
        out.append(gg)
    return pd.concat(out, ignore_index=True)


# =========================
# Plotting
# =========================
def plot_strategy_five(rows: List[Dict], strategy: str, figdir: str):
    rows = sorted(rows, key=lambda r: r["distance_to_mean"])
    k = len(rows)
    fig, axes = plt.subplots(k, 2, figsize=(8, 2.6 * k), dpi=170)
    if k == 1:
        axes = np.array([axes])
    cmap = ListedColormap(["#ff6600", "#f2f2f2", "#00249b"])  # -1 blue, 0 gray, +1 green

    for i, row in enumerate(rows):
        for j, (title, grid) in enumerate([("Initial", row["init_grid"]), ("Final", row["final_grid"])]):
            disp = np.where(grid == -1, 0, np.where(grid == 0, 1, 2))
            ax = axes[i, j]
            ax.imshow(disp, cmap=cmap, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_title(f"{strategy}: {title} — rep {row['rep']}")
            else:
                ax.set_title(f"{strategy}")
    plt.tight_layout()
    path = os.path.join(figdir, f"{strategy}_initial_final_top5.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()


def plot_combined_all_strategies(rep_df: pd.DataFrame, figdir: str):
    # Combined panel: for each strategy, 5 rows × 2 cols stacked block-wise
    order = ["SCHELLING", "PEF", "PEO", "NEA"]
    blocks = []
    for strat in order:
        sub = rep_df[rep_df["strategy"] == strat].sort_values("distance_to_mean")
        blocks.append((strat, sub))
    total_rows = sum(len(sub) for _, sub in blocks)
    fig, axes = plt.subplots(total_rows, 2, figsize=(8, 2.6 * total_rows), dpi=170)
    cmap = ListedColormap(["#ff6600", "#f2f2f2", "#00249b"])
    row_idx = 0
    for strat, sub in blocks:
        for _, row in sub.iterrows():
            for j, (title, grid) in enumerate([("Initial", row["init_grid"]), ("Final", row["final_grid"])]):
                disp = np.where(grid == -1, 0, np.where(grid == 0, 1, 2))
                ax = axes[row_idx, j]
                ax.imshow(disp, cmap=cmap, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{strat}")
            row_idx += 1
    plt.tight_layout()
    path = os.path.join(figdir, "ALL_STRATEGIES_initial_final_top5.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()


# =========================
# Main
# =========================
def main():
    cfg = Config(
        grid_size=32,
        vacancy_ratio=0.20,
        threshold=0.50,
        neighborhood="moore",  # or "vonneumann"
        wrap=True,
        reps=10000,
        cores=min(16, cpu_count()),
        max_steps=10_000,
        initial_mix=(0.5, 0.5),
        seed=20251004,
        out_root="out",
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
    results: List[Dict] = []
    with Pool(processes=min(cfg.cores, cpu_count())) as pool:
        for r in tqdm(pool.imap_unordered(_worker, jobs),
                      total=len(jobs), desc="Running simulations", unit="run"):
            results.append(r)

    # DataFrame of all runs
    df = pd.DataFrame(results)

    # Save per-run metrics (without bulky grids)
    slim = df.drop(columns=["init_grid", "final_grid"]).copy()
    slim_path = os.path.join(outdir, "results_raw.csv")
    slim.to_csv(slim_path, index=False)

    # Select top-5 closest to mean per strategy (in z-space over key metrics)
    metrics = ["final_fraction_satisfied", "final_segregation", "steps_to_equilibrium"]
    reps_df = select_k_representatives(df, metrics, k=5)

    # Save info about selected reps
    reps_info = reps_df.drop(columns=["init_grid", "final_grid"]).copy()
    reps_info_path = os.path.join(outdir, "representative_top5.csv")
    reps_info.to_csv(reps_info_path, index=False)

    # Plot per-strategy figures (5 rows × 2 cols)
    for strat, g in reps_df.groupby("strategy"):
        plot_strategy_five([row._asdict() if hasattr(row, "_asdict") else row.to_dict()
                            for _, row in g.iterrows()], strat, figdir)

    # Optional combined panel across all strategies
    plot_combined_all_strategies(reps_df, figdir)

    print("\n=== Top-5 representative runs per strategy (closest to mean) ===")
    print(reps_info.sort_values(["strategy", "distance_to_mean"]).to_string(index=False))
    print(f"\nSaved per-run metrics to     : {slim_path}")
    print(f"Saved representative table to: {reps_info_path}")
    print(f"Figures saved in             : {figdir}")
    print(f"Output directory             : {outdir}")


if __name__ == "__main__":
    main()
