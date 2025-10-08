# schelling_ext_mix_sweep.py
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
    vacancy_ratio: float = 0.20         # fixed VR per request
    threshold: float = 0.50
    neighborhood: str = "moore"         # "moore" or "vonneumann"
    wrap: bool = True                    # torus
    reps: int = 10000
    cores: int = 16
    max_steps: int = 10_000
    initial_mix: Tuple[float, float] = (0.5, 0.5)  # +1 / -1 among occupied agents
    seed: int = 20250930
    out_root: str = "out"
    baseline_type: str = "SCHELLING"    # remaining share uses this type

    def outdir(self) -> str:
        name = f"GS{self.grid_size}_VR{self.vacancy_ratio:.2f}_thr{self.threshold:.2f}_{self.neighborhood}_mix0.05_{self.baseline_type}_rep{self.reps}"
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
    where S_i ∈ {+1,-1,0}, vacancies contribute 0; undirected edges counted twice.
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


# -------------------------------
# Move rules
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
def assign_agent_types_for_mix(grid: np.ndarray, focal: str, frac_focal: float,
                               baseline: str, rng: random.Random) -> np.ndarray:
    """
    Randomly assign 'frac_focal' of occupied agents to 'focal' type, remainder to 'baseline'.
    """
    n = grid.shape[0]
    occ = [(r, c) for r in range(n) for c in range(n) if grid[r, c] != 0]
    k = int(round(frac_focal * len(occ)))
    rng.shuffle(occ)
    agent_types = np.empty((n, n), dtype=object)
    agent_types[:] = ""
    focal_set = set(occ[:k])
    for r, c in occ:
        agent_types[r, c] = focal if (r, c) in focal_set else baseline
    return agent_types


def step_once(grid: np.ndarray, threshold: float, offs, wrap: bool,
              agent_types: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, bool, int]:
    n = grid.shape[0]
    sat = satisfied_mask(grid, threshold, offs, wrap)
    unsat_positions = [(r, c) for r in range(n) for c in range(n)
                       if grid[r, c] != 0 and not sat[r, c]]
    rng.shuffle(unsat_positions)

    any_move, moves = False, 0
    for r, c in unsat_positions:
        t = agent_types[r, c]  # "SCHELLING", "PEF", "PEO", or "NEA"
        dest = MOVE_RULES[t](grid, r, c, threshold, offs, wrap, rng)
        if dest is not None:
            rr, cc = dest
            if grid[rr, cc] == 0:
                agent = grid[r, c]
                grid[r, c] = 0
                grid[rr, cc] = agent
                agent_types[rr, cc] = agent_types[r, c]
                agent_types[r, c] = ""
                any_move = True
                moves += 1
    return grid, any_move, moves


def run_one_mix(cfg_dict: Dict, focal_strategy: str, frac_focal: float, rep: int) -> Dict:
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__annotations__})
    n = cfg.grid_size
    offs = neighborhood_offsets(cfg.neighborhood)

    # deterministic seed per (focal, frac, rep)
    seed_base = (cfg.seed
                 + 1000 * ({"PEF": 1, "PEO": 2, "NEA": 3}[focal_strategy])
                 + 100 * int(round(frac_focal * 10))
                 + rep)
    rng = random.Random(seed_base)

    # initial grid
    grid = make_initial_grid(n, cfg.vacancy_ratio, cfg.initial_mix, rng)

    # mixed agent types: frac_focal of 'focal_strategy', remainder = baseline
    agent_types = assign_agent_types_for_mix(grid, focal_strategy, frac_focal, cfg.baseline_type, rng)

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
    mpa = (total_moves / num_agents) if num_agents > 0 else 0.0

    return {
        "focal_strategy": focal_strategy,
        "fraction_focal": round(frac_focal, 2),
        "rep": rep,
        "final_fraction_satisfied": frac_sat,
        "final_segregation": seg,
        "steps_to_equilibrium": steps,
        "total_moves": total_moves,
        "moves_per_agent": mpa,
        "grid_size": cfg.grid_size,
        "vacancy_ratio": cfg.vacancy_ratio,
        "threshold": cfg.threshold,
        "neighborhood": cfg.neighborhood,
        "baseline_type": cfg.baseline_type,
    }


def _worker(job):
    cfg_dict, focal_strategy, frac_focal, rep = job
    return run_one_mix(cfg_dict, focal_strategy, frac_focal, rep)


# -------------------------------
# Plotting
# -------------------------------
def _ci95(series: pd.Series) -> float:
    n = len(series)
    if n <= 1:
        return 0.0
    se = float(series.std(ddof=1)) / math.sqrt(n)
    return 1.96 * se


def plot_with_ci_by_focal(agg: pd.DataFrame, ycol: str, ylabel: str, out_base: str):
    """
    agg columns include:
      focal_strategy, fraction_focal, mean_<ycol>, ci95_<ycol>
    Draw one combined figure (lines for each focal strategy).
    """
    plt.figure(figsize=(7.2, 4.6), dpi=160)
    for focal, sub in agg.groupby("focal_strategy", sort=False):
        x = sub["fraction_focal"].values
        y = sub[f"mean_{ycol}"].values
        ci = sub[f"ci95_{ycol}"].values
        plt.plot(x, y, label=focal, linewidth=2)
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)
    plt.xlabel("Fraction of focal agent type")
    plt.ylabel(ylabel)
    plt.legend(title="Focal strategy", frameon=False)
    plt.tight_layout()
    plt.savefig(out_base + ".png")
    plt.savefig(out_base + ".pdf")
    plt.close()


def plot_per_focal(agg: pd.DataFrame, ycol: str, ylabel: str, figdir: str, file_stub: str):
    """
    One figure per focal strategy.
    """
    for focal, sub in agg.groupby("focal_strategy", sort=False):
        out_base = os.path.join(figdir, f"{file_stub}_{focal}")
        plt.figure(figsize=(6.6, 4.2), dpi=160)
        x = sub["fraction_focal"].values
        y = sub[f"mean_{ycol}"].values
        ci = sub[f"ci95_{ycol}"].values
        plt.plot(x, y, linewidth=2)
        plt.fill_between(x, y - ci, y + ci, alpha=0.25)
        plt.xlabel("Fraction of focal agent type")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs. fraction ({focal} vs {sub['baseline_type'].iloc[0]})")
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
        vacancy_ratio=0.20,
        threshold=0.50,
        neighborhood="moore",   # or "vonneumann"
        wrap=True,
        reps=10000,
        cores=min(32, cpu_count()),
        max_steps=10_000,
        initial_mix=(0.5, 0.5),
        seed=20251004,
        out_root="out",
        baseline_type="SCHELLING"
    )

    outdir = cfg.outdir()
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    focal_strategies = ["PEF", "PEO", "NEA"]
    fractions = [round(x * 0.1, 2) for x in range(0, 11)]  # 0.0 ... 1.0

    # Build jobs: 3 focal strategies × 11 fractions × reps
    jobs = []
    cfg_dict = asdict(cfg)
    for focal in focal_strategies:
        for frac in fractions:
            for rep in range(cfg.reps):
                jobs.append((cfg_dict, focal, float(frac), rep))

    # Run in parallel with progress bar
    set_start_method("spawn", force=True)
    results = []
    with Pool(processes=min(cfg.cores, cpu_count())) as pool:
        for r in tqdm(pool.imap_unordered(_worker, jobs),
                      total=len(jobs), desc="Running simulations", unit="run"):
            results.append(r)

    # Save raw
    df = pd.DataFrame(results).sort_values(["focal_strategy", "fraction_focal", "rep"]).reset_index(drop=True)
    raw_path = os.path.join(outdir, "results_raw.csv")
    df.to_csv(raw_path, index=False)

    # Aggregate by (focal_strategy, fraction_focal)
    def agg_ci(g: pd.DataFrame, col: str) -> Tuple[float, float]:
        mean = g[col].mean()
        ci = _ci95(g[col])
        return mean, ci

    rows = []
    for (focal, frac), g in df.groupby(["focal_strategy", "fraction_focal"]):
        ms, cs = agg_ci(g, "final_fraction_satisfied")
        mseg, cseg = agg_ci(g, "final_segregation")
        msteps, csteps = agg_ci(g, "steps_to_equilibrium")
        mmoves, cmoves = agg_ci(g, "moves_per_agent")
        rows.append({
            "focal_strategy": focal,
            "fraction_focal": frac,
            "mean_fraction_satisfied": ms, "ci95_fraction_satisfied": cs,
            "mean_segregation": mseg, "ci95_segregation": cseg,
            "mean_steps": msteps, "ci95_steps": csteps,
            "mean_moves_per_agent": mmoves, "ci95_moves_per_agent": cmoves,
            "reps": len(g),
            "baseline_type": g["baseline_type"].iloc[0]
        })
    agg = pd.DataFrame(rows).sort_values(["focal_strategy", "fraction_focal"]).reset_index(drop=True)
    agg_path = os.path.join(outdir, "results_summary.csv")
    agg.to_csv(agg_path, index=False)

    # Combined figures (lines: focal strategies)
    plot_with_ci_by_focal(agg, "fraction_satisfied", "Fraction satisfied",
                          os.path.join(figdir, "fraction_satisfied_vs_fraction"))
    plot_with_ci_by_focal(agg, "segregation", "Segregation (S)",
                          os.path.join(figdir, "segregation_vs_fraction"))
    plot_with_ci_by_focal(agg, "steps", "Steps to equilibrium",
                          os.path.join(figdir, "steps_to_equilibrium_vs_fraction"))
    plot_with_ci_by_focal(agg, "moves_per_agent", "Moves per agent",
                          os.path.join(figdir, "moves_per_agent_vs_fraction"))

    # Per-focal figures
    plot_per_focal(agg, "fraction_satisfied", "Fraction satisfied", figdir, "fraction_satisfied_vs_fraction")
    plot_per_focal(agg, "segregation", "Segregation (S)", figdir, "segregation_vs_fraction")
    plot_per_focal(agg, "steps", "Steps to equilibrium", figdir, "steps_to_equilibrium_vs_fraction")
    plot_per_focal(agg, "moves_per_agent", "Moves per agent", figdir, "moves_per_agent_vs_fraction")

    # Preview
    print("\n=== Summary head (focal × fraction) ===")
    print(agg.head(15).to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to    : {agg_path}")
    print(f"Figures saved in    : {figdir}")
    print(f"Output directory    : {outdir}")


if __name__ == "__main__":
    main()
