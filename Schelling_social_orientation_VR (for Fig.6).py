# schelling_ext_sweep.py
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


# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    grid_size: int = 32
    threshold: float = 0.5              # satisfaction threshold
    neighborhood: str = "moore"         # "moore" or "vonneumann"
    wrap: bool = True                   # torus boundary
    reps: int = 100                     # simulations per strategy per vacancy ratio
    cores: int = 16                     # up to 16
    max_steps: int = 10_000             # safety cap
    initial_mix: Tuple[float, float] = (0.5, 0.5)  # share of +1 and -1 among occupied
    seed: int = 20250930                # master seed
    out_root: str = "out"               # root for outputs

    # naming for sweep folder
    def sweep_outdir(self) -> str:
        name = f"GS{self.grid_size}_thr{self.threshold:.2f}_{self.neighborhood}_seed_{self.seed}_sweep_VR_rep{self.reps}"
        return os.path.join(self.out_root, name)


# -------------------------------
# Neighborhood helpers
# -------------------------------
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


# -------------------------------
# Grid & satisfaction
# -------------------------------
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


# -------------------------------
# Candidate evaluation utilities
# -------------------------------
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


# -------------------------------
# Metrics
# -------------------------------
def segregation_stat(grid: np.ndarray, offs, wrap: bool) -> float:
    """
    S = (1 / (2 * N)) * sum_{i,j} A_ij S_i S_j
    where S_i ∈ {+1,-1,0} and A_ij = 1 if neighbors else 0 (no self loops).
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
    return float(total) / (2.0 * num_agents)  # undirected edges counted twice


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


# -------------------------------
# Move rules (four strategies)
# -------------------------------
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


# -------------------------------
# Simulation core
# -------------------------------
def step_once(grid: np.ndarray, threshold: float, offs, wrap: bool,
              agent_types: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, bool, int]:
    n = grid.shape[0]
    sat = satisfied_mask(grid, threshold, offs, wrap)
    unsat_positions = [(r, c) for r in range(n) for c in range(n)
                       if grid[r, c] != 0 and not sat[r, c]]
    rng.shuffle(unsat_positions)

    any_move, moves = False, 0
    for r, c in unsat_positions:
        t = agent_types[r, c]  # string: strategy name
        dest = MOVE_RULES[t](grid, r, c, threshold, offs, wrap, rng)
        if dest is not None:
            rr, cc = dest
            if grid[rr, cc] == 0:  # still vacant
                agent = grid[r, c]
                grid[r, c] = 0
                grid[rr, cc] = agent
                agent_types[rr, cc] = agent_types[r, c]
                agent_types[r, c] = ""
                any_move = True
                moves += 1
    return grid, any_move, moves


def strat_code(s: str) -> int:
    return {"SCHELLING": 1, "PEF": 2, "PEO": 3, "NEA": 4}[s]


def run_one(cfg_dict: Dict, strategy: str, vacancy_ratio: float, rep: int) -> Dict:
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__annotations__})
    n = cfg.grid_size
    offs = neighborhood_offsets(cfg.neighborhood)

    # deterministic per (strategy, vacancy, rep)
    seed_base = (cfg.seed
                 + 10_000 * int(round(vacancy_ratio * 100))
                 + 1_000 * strat_code(strategy)
                 + rep)
    rng = random.Random(seed_base)

    grid = make_initial_grid(n, vacancy_ratio, cfg.initial_mix, rng)

    # everyone uses the same strategy in a single run
    agent_types = np.empty((n, n), dtype=object)
    agent_types[:] = ""
    for r in range(n):
        for c in range(n):
            if grid[r, c] != 0:
                agent_types[r, c] = strategy

    total_moves = 0
    steps = 0
    for steps in range(1, cfg.max_steps + 1):
        grid, any_move, moves = step_once(grid, cfg.threshold, offs, cfg.wrap, agent_types, rng)
        total_moves += moves
        if not any_move:
            break

    frac_sat = fraction_satisfied(grid, cfg.threshold, offs, cfg.wrap)
    seg = segregation_stat(grid, offs, cfg.wrap)
    num_agents = int(np.count_nonzero(grid))
    moves_per_agent = (total_moves / num_agents) if num_agents > 0 else 0.0

    return {
        "strategy": strategy,
        "vacancy_ratio": float(vacancy_ratio),
        "rep": rep,
        "final_fraction_satisfied": frac_sat,
        "final_segregation": seg,
        "steps_to_equilibrium": steps,
        "total_moves": total_moves,
        "moves_per_agent": moves_per_agent,
        "grid_size": cfg.grid_size,
        "threshold": cfg.threshold,
        "neighborhood": cfg.neighborhood,
    }


def _worker(job):
    cfg_dict, strategy, vacancy_ratio, rep = job
    return run_one(cfg_dict, strategy, vacancy_ratio, rep)


# -------------------------------
# Plotting
# -------------------------------
def _ci95(series: pd.Series) -> float:
    n = len(series)
    if n <= 1:
        return 0.0
    se = float(series.std(ddof=1)) / math.sqrt(n)
    return 1.96 * se


def plot_with_ci(df_agg: pd.DataFrame, ycol: str, ylabel: str, out_base: str):
    """
    df_agg has columns: strategy, vacancy_ratio, mean_<metric>, ci95_<metric>
    """
    plt.figure(figsize=(7, 4.5), dpi=160)
    for strat, sub in df_agg.groupby("strategy", sort=False):
        x = sub["vacancy_ratio"].values
        y = sub[f"mean_{ycol}"].values
        ci = sub[f"ci95_{ycol}"].values
        plt.plot(x, y, label=strat, linewidth=2)
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)
    plt.xlabel("Vacancy ratio")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_base + ".png")
    plt.savefig(out_base + ".pdf")
    plt.close()


# -------------------------------
# Main
# -------------------------------
def main():
    cfg = Config(
        grid_size=32,
        threshold=0.50,
        neighborhood="vonneumann",   # or "vonneumann"
        wrap=True,
        reps=10000,
        cores=min(32, cpu_count()),
        max_steps=10_000,
        initial_mix=(0.5, 0.5),
        seed=20251003,
        out_root="out"
    )

    # Sweep vacancy ratios
    vacancy_ratios = np.round(np.arange(0.05, 0.31, 0.025), 2)

    outdir = cfg.sweep_outdir()
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    strategies = ["SCHELLING", "PEF", "PEO", "NEA"]

    # Build jobs
    jobs = []
    cfg_dict = asdict(cfg)
    for vac in vacancy_ratios:
        for strat in strategies:
            for rep in range(cfg.reps):
                jobs.append((cfg_dict, strat, float(vac), rep))

    # Run in parallel with progress bar
    set_start_method("spawn", force=True)
    results = []
    with Pool(processes=min(cfg.cores, cpu_count())) as pool:
        for r in tqdm(pool.imap_unordered(_worker, jobs),
                      total=len(jobs), desc="Running simulations", unit="run"):
            results.append(r)

    # Save raw
    df = pd.DataFrame(results).sort_values(["strategy", "vacancy_ratio", "rep"]).reset_index(drop=True)
    raw_path = os.path.join(outdir, "results_raw.csv")
    df.to_csv(raw_path, index=False)

    # Aggregate by (strategy, vacancy_ratio)
    def agg_ci(g: pd.DataFrame, col: str) -> Tuple[float, float]:
        mean = g[col].mean()
        ci = _ci95(g[col])
        return mean, ci

    rows = []
    for (strat, vac), g in df.groupby(["strategy", "vacancy_ratio"]):
        ms, cs = agg_ci(g, "final_fraction_satisfied")
        mseg, cseg = agg_ci(g, "final_segregation")
        msteps, csteps = agg_ci(g, "steps_to_equilibrium")
        mmoves, cmoves = agg_ci(g, "moves_per_agent")
        rows.append({
            "strategy": strat,
            "vacancy_ratio": vac,
            "mean_fraction_satisfied": ms, "ci95_fraction_satisfied": cs,
            "mean_segregation": mseg, "ci95_segregation": cseg,
            "mean_steps": msteps, "ci95_steps": csteps,
            "mean_moves_per_agent": mmoves, "ci95_moves_per_agent": cmoves,
            "reps": len(g)
        })
    agg = pd.DataFrame(rows).sort_values(["strategy", "vacancy_ratio"]).reset_index(drop=True)
    agg_path = os.path.join(outdir, "results_summary.csv")
    agg.to_csv(agg_path, index=False)

    # One plot per metric (PNG + PDF), LaTeX-ready
    plot_with_ci(agg, "fraction_satisfied", "Fraction satisfied", os.path.join(figdir, "fraction_satisfied_vs_vacancy"))
    plot_with_ci(agg, "segregation", "Segregation (S)", os.path.join(figdir, "segregation_vs_vacancy"))
    plot_with_ci(agg, "steps", "Steps to equilibrium", os.path.join(figdir, "steps_to_equilibrium_vs_vacancy"))
    plot_with_ci(agg, "moves_per_agent", "Moves per agent", os.path.join(figdir, "moves_per_agent_vs_vacancy"))

    # Console summary preview
    print("\n=== Summary head (strategy × vacancy) ===")
    print(agg.head(16).to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to    : {agg_path}")
    print(f"Figures saved in    : {figdir}")


if __name__ == "__main__":
    main()
