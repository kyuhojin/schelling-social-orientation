# schelling_ext_pep_pem_nep.py
import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count, set_start_method
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    grid_size: int = 32
    vacancy_ratio: float = 0.20
    threshold: float = 0.5              # satisfaction threshold (fraction of like neighbors)
    neighborhood: str = "moore"         # "moore" or "vonneumann"
    wrap: bool = True                   # torus boundary (recommended)
    reps: int = 10000                   # simulations per strategy
    cores: int = 32                     # up to 16
    max_steps: int = 10_000             # safety cap
    initial_mix: Tuple[float, float] = (0.5, 0.5)  # share of +1 and -1 agents among occupied cells
    seed: int = 20250930                # master seed for reproducibility
    out_root: str = "out"               # root directory for outputs

    def outdir(self) -> str:
        name = f"GS{self.grid_size}_vac{self.vacancy_ratio:.2f}_thr{self.threshold:.2f}_{self.neighborhood}_rep{self.reps}"
        return os.path.join(self.out_root, name)


# -------------------------------
# Utilities
# -------------------------------
def rng_choice(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def neighborhood_offsets(kind: str) -> List[Tuple[int, int]]:
    if kind.lower() in ("moore", "moore8", "moore-8"):
        offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
    elif kind.lower() in ("vonneumann", "von-neumann", "vn", "neumann"):
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("neighborhood must be 'moore' or 'vonneumann'")
    return offs


def neighbors_of(r: int, c: int, n: int, offs: List[Tuple[int, int]], wrap: bool) -> List[Tuple[int, int]]:
    res = []
    for dr, dc in offs:
        rr, cc = r + dr, c + dc
        if wrap:
            rr %= n
            cc %= n
            res.append((rr, cc))
        else:
            if 0 <= rr < n and 0 <= cc < n:
                res.append((rr, cc))
    return res


def make_initial_grid(cfg: Config, rng: random.Random) -> np.ndarray:
    """Grid values: +1, -1 for agents; 0 for vacancy."""
    n = cfg.grid_size
    total_cells = n * n
    num_vacant = int(round(cfg.vacancy_ratio * total_cells))
    num_agents = total_cells - num_vacant

    num_pos = int(round(cfg.initial_mix[0] * num_agents))
    num_neg = num_agents - num_pos

    arr = np.array([+1] * num_pos + [-1] * num_neg + [0] * num_vacant, dtype=np.int8)
    rng.shuffle(arr)
    return arr.reshape((n, n))


def is_satisfied(grid: np.ndarray, r: int, c: int, cfg: Config, offs: List[Tuple[int, int]]) -> bool:
    """Agent is satisfied if fraction of like (occupied) neighbors >= threshold.
       If no occupied neighbors, treat as satisfied (standard choice)."""
    n = cfg.grid_size
    s = grid[r, c]
    if s == 0:
        return True
    neigh = neighbors_of(r, c, n, offs, cfg.wrap)
    likes = 0
    tot = 0
    for rr, cc in neigh:
        val = grid[rr, cc]
        if val != 0:
            tot += 1
            if val == s:
                likes += 1
    if tot == 0:
        return True
    return (likes / tot) >= cfg.threshold


def satisfied_mask(grid: np.ndarray, cfg: Config, offs: List[Tuple[int, int]]) -> np.ndarray:
    n = cfg.grid_size
    sat = np.ones((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            if grid[r, c] != 0:
                sat[r, c] = is_satisfied(grid, r, c, cfg, offs)
    return sat


def candidate_sites_that_satisfy_agent(grid: np.ndarray, agent_sign: int, cfg: Config,
                                       offs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return vacant sites where the agent would be satisfied if placed there."""
    n = cfg.grid_size
    cands = []
    # Temporarily place agent to test satisfaction (local, no need to consider origin removal)
    for r in range(n):
        for c in range(n):
            if grid[r, c] == 0:
                # compute satisfaction fraction for this hypothetical placement
                neigh = neighbors_of(r, c, n, offs, cfg.wrap)
                likes = 0
                tot = 0
                for rr, cc in neigh:
                    val = grid[rr, cc]
                    if val != 0:
                        tot += 1
                        if val == agent_sign:
                            likes += 1
                if tot == 0 or (likes / tot) >= cfg.threshold:
                    cands.append((r, c))
    return cands


def count_satisfied_neighbors_if_move_in(grid: np.ndarray, r: int, c: int, agent_sign: int,
                                         cfg: Config, offs: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    For the neighborhood around (r,c), return:
      (num_satisfied_before, num_satisfied_after)
    considering only currently occupied neighbors around (r,c).
    """
    n = cfg.grid_size
    neigh = neighbors_of(r, c, n, offs, cfg.wrap)

    # current satisfaction of neighbors (before we move in)
    sat_before = 0
    neighbor_positions = []
    for rr, cc in neigh:
        if grid[rr, cc] != 0:
            neighbor_positions.append((rr, cc))
            if is_satisfied(grid, rr, cc, cfg, offs):
                sat_before += 1

    # after we move in: only neighbors in this neighborhood can change due to the added neighbor.
    # We simulate their satisfaction by temporarily treating (r,c)=agent_sign.
    sat_after = 0
    for rr, cc in neighbor_positions:
        # check satisfaction of neighbor if (r,c) becomes agent_sign
        s = grid[rr, cc]
        local_neigh = neighbors_of(rr, cc, n, offs, cfg.wrap)
        likes = 0
        tot = 0
        for r2, c2 in local_neigh:
            if (r2 == r) and (c2 == c):
                val = agent_sign
            else:
                val = grid[r2, c2]
            if val != 0:
                tot += 1
                if val == s:
                    likes += 1
        if tot == 0 or (likes / tot) >= cfg.threshold:
            sat_after += 1

    return sat_before, sat_after


def would_any_currently_satisfied_neighbor_become_unsatisfied(grid: np.ndarray, r: int, c: int,
                                                              agent_sign: int, cfg: Config,
                                                              offs: List[Tuple[int, int]]) -> bool:
    """
    NEA constraint: among neighbors of (r,c) that are currently satisfied,
    none should become unsatisfied after we move in at (r,c).
    """
    n = cfg.grid_size
    neigh = neighbors_of(r, c, n, offs, cfg.wrap)

    satisfied_now: List[Tuple[int, int]] = []
    for rr, cc in neigh:
        if grid[rr, cc] != 0 and is_satisfied(grid, rr, cc, cfg, offs):
            satisfied_now.append((rr, cc))

    # Evaluate each of these with (r,c) occupied by agent_sign
    for rr, cc in satisfied_now:
        s = grid[rr, cc]
        local_neigh = neighbors_of(rr, cc, n, offs, cfg.wrap)
        likes = 0
        tot = 0
        for r2, c2 in local_neigh:
            if (r2 == r) and (c2 == c):
                val = agent_sign
            else:
                val = grid[r2, c2]
            if val != 0:
                tot += 1
                if val == s:
                    likes += 1
        if not (tot == 0 or (likes / tot) >= cfg.threshold):
            return True  # some currently satisfied neighbor would become unsatisfied
    return False


def num_unsatisfied_neighbors_after_move(grid: np.ndarray, r: int, c: int, agent_sign: int,
                                         cfg: Config, offs: List[Tuple[int, int]]) -> int:
    """Count how many occupied neighbors around (r,c) would be unsatisfied after we move in."""
    n = cfg.grid_size
    neigh = neighbors_of(r, c, n, offs, cfg.wrap)
    cnt = 0
    for rr, cc in neigh:
        if grid[rr, cc] != 0:
            s = grid[rr, cc]
            local_neigh = neighbors_of(rr, cc, n, offs, cfg.wrap)
            likes = 0
            tot = 0
            for r2, c2 in local_neigh:
                if (r2 == r) and (c2 == c):
                    val = agent_sign
                else:
                    val = grid[r2, c2]
                if val != 0:
                    tot += 1
                    if val == s:
                        likes += 1
            if not (tot == 0 or (likes / tot) >= cfg.threshold):
                cnt += 1
    return cnt


def segregation_stat(grid: np.ndarray, cfg: Config, offs: List[Tuple[int, int]]) -> float:
    """
    S = (1 / (2 * num_agents)) * sum_{i,j} A_ij S_i S_j,
    where A_ij = 1 if i and j are neighbors, else 0 (no self loops).
    S_i in {+1, -1, 0}, vacancies contribute 0.
    """
    n = cfg.grid_size
    s = 0.0
    num_agents = np.count_nonzero(grid)
    if num_agents == 0:
        return 0.0

    for r in range(n):
        for c in range(n):
            Si = grid[r, c]
            if Si == 0:
                continue
            for rr, cc in neighbors_of(r, c, n, offs, cfg.wrap):
                Sj = grid[rr, cc]
                if Sj != 0:
                    s += (Si * Sj)
    # Each undirected edge counted twice; the 1/(2 * num_agents) matches your definition.
    return float(s) / (2.0 * num_agents)


def fraction_satisfied(grid: np.ndarray, cfg: Config, offs: List[Tuple[int, int]]) -> float:
    n = cfg.grid_size
    sat = 0
    tot = 0
    for r in range(n):
        for c in range(n):
            if grid[r, c] != 0:
                tot += 1
                if is_satisfied(grid, r, c, cfg, offs):
                    sat += 1
    return 0.0 if tot == 0 else (sat / tot)


# -------------------------------
# Agent move rules
# -------------------------------
def move_rule_schelling(grid: np.ndarray, r: int, c: int, cfg: Config, offs: List[Tuple[int, int]],
                        rng: random.Random) -> Optional[Tuple[int, int]]:
    """Panel A: random among all satisfying vacant sites (original Schelling)."""
    s = grid[r, c]
    cands = candidate_sites_that_satisfy_agent(grid, s, cfg, offs)
    if not cands:
        return None
    return rng_choice(rng, cands)


def move_rule_pep(grid: np.ndarray, r: int, c: int, cfg: Config, offs: List[Tuple[int, int]],
                  rng: random.Random) -> Optional[Tuple[int, int]]:
    """
    Panel B (PEF): among sites where agent is satisfied and
    (num_satisfied_neighbors_after >= num_satisfied_neighbors_before),
    choose one at random.
    """
    sgn = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, sgn, cfg, offs)
    cands = []
    for rr, cc in raw:
        before, after = count_satisfied_neighbors_if_move_in(grid, rr, cc, sgn, cfg, offs)
        if after >= before:
            cands.append((rr, cc))
    if not cands:
        return None
    return rng_choice(rng, cands)


def move_rule_pem(grid: np.ndarray, r: int, c: int, cfg: Config, offs: List[Tuple[int, int]],
                  rng: random.Random) -> Optional[Tuple[int, int]]:
    """
    Panel C (PEO): among sites where agent is satisfied and
    (num_satisfied_neighbors_after >= num_satisfied_neighbors_before),
    choose the site that minimizes the number of unsatisfied neighbors after the move.
    Break ties randomly.
    """
    sgn = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, sgn, cfg, offs)
    filtered = []
    for rr, cc in raw:
        before, after = count_satisfied_neighbors_if_move_in(grid, rr, cc, sgn, cfg, offs)
        if after >= before:
            unsat_after = num_unsatisfied_neighbors_after_move(grid, rr, cc, sgn, cfg, offs)
            filtered.append(((rr, cc), unsat_after))
    if not filtered:
        return None
    # minimize unsatisfied neighbors after move
    min_unsat = min(v for (_, v) in filtered)
    top = [pos for (pos, v) in filtered if v == min_unsat]
    return rng_choice(rng, top)


def move_rule_nep(grid: np.ndarray, r: int, c: int, cfg: Config, offs: List[Tuple[int, int]],
                  rng: random.Random) -> Optional[Tuple[int, int]]:
    """
    Panel D (NEA): among sites that satisfy the agent AND do not dissatisfy any
    currently satisfied neighbor in that neighborhood, choose one at random.
    (This is stricter than PEF: it forbids making any currently satisfied neighbor unhappy.)
    """
    sgn = grid[r, c]
    raw = candidate_sites_that_satisfy_agent(grid, sgn, cfg, offs)
    cands = []
    for rr, cc in raw:
        if not would_any_currently_satisfied_neighbor_become_unsatisfied(grid, rr, cc, sgn, cfg, offs):
            cands.append((rr, cc))
    if not cands:
        return None
    return rng_choice(rng, cands)


MOVE_RULES = {
    "SCHELLING": move_rule_schelling,
    "PEF": move_rule_pep,
    "PEO": move_rule_pem,
    "NEA": move_rule_nep,
}


# -------------------------------
# Simulation
# -------------------------------
def step_once(grid: np.ndarray, cfg: Config, offs: List[Tuple[int, int]],
              agent_types: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, bool, int]:
    """
    One asynchronous sweep:
      - Randomize order of all currently unsatisfied agents
      - Each moves according to its type's rule (if a feasible site found)
    Returns: (grid, any_move_made, moves_count)
    """
    n = cfg.grid_size
    sat = satisfied_mask(grid, cfg, offs)
    unsatisfied_positions = [(r, c) for r in range(n) for c in range(n)
                             if grid[r, c] != 0 and not sat[r, c]]
    rng.shuffle(unsatisfied_positions)
    any_move = False
    moves = 0

    for r, c in unsatisfied_positions:
        t = agent_types[r, c]  # string tag
        move_fn = MOVE_RULES[t]
        dest = move_fn(grid, r, c, cfg, offs, rng)
        if dest is not None:
            rr, cc = dest
            if grid[rr, cc] == 0:  # still vacant
                agent = grid[r, c]
                grid[r, c] = 0
                grid[rr, cc] = agent
                agent_types[rr, cc] = agent_types[r, c]
                agent_types[r, c] = ""  # vacancy carries no type
                any_move = True
                moves += 1
    return grid, any_move, moves


def run_one(cfg: Config, strategy: str, rep: int, seed_offset: int) -> Dict:
    """
    Run a single simulation where ALL agents have the same type = strategy.
    Returns summary statistics.
    """
    rng = random.Random(cfg.seed + 1000 * seed_offset + rep)
    np_rng = np.random.default_rng(cfg.seed + 2000 * seed_offset + rep)

    grid = make_initial_grid(cfg, rng)
    n = cfg.grid_size
    offs = neighborhood_offsets(cfg.neighborhood)

    # Assign type to all occupied cells
    agent_types = np.empty((n, n), dtype=object)
    agent_types[:, :] = ""
    for r in range(n):
        for c in range(n):
            if grid[r, c] != 0:
                agent_types[r, c] = strategy

    total_moves = 0
    steps = 0
    for steps in range(1, cfg.max_steps + 1):
        grid, any_move, moves = step_once(grid, cfg, offs, agent_types, rng)
        total_moves += moves
        if not any_move:
            break

    frac_sat = fraction_satisfied(grid, cfg, offs)
    seg = segregation_stat(grid, cfg, offs)
    num_agents = int(np.count_nonzero(grid))
    moves_per_agent = (total_moves / num_agents) if num_agents > 0 else 0.0

    return {
        "strategy": strategy,
        "rep": rep,
        "final_fraction_satisfied": frac_sat,
        "final_segregation": seg,
        "steps_to_equilibrium": steps,
        "total_moves": total_moves,
        "moves_per_agent": moves_per_agent,
        "grid_size": cfg.grid_size,
        "vacancy_ratio": cfg.vacancy_ratio,
        "threshold": cfg.threshold,
        "neighborhood": cfg.neighborhood,
    }


def _worker(job):
    cfg_dict, strategy, rep, seed_offset = job
    cfg = Config(**cfg_dict)
    return run_one(cfg, strategy, rep, seed_offset)


def main():
    cfg = Config(
        grid_size=32,
        vacancy_ratio=0.20,
        threshold=0.50,
        neighborhood="moore",   # change to "vonneumann" if desired
        wrap=True,
        reps=10000,
        cores=min(32, cpu_count()),
        max_steps=10_000,
        initial_mix=(0.5, 0.5),
        seed=20250930,
        out_root="out"
    )

    outdir = cfg.outdir()
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Build jobs: 4 strategies × reps
    strategies = ["SCHELLING", "PEF", "PEO", "NEA"]
    jobs = []
    for strat in strategies:
        for rep in range(cfg.reps):
            jobs.append((asdict(cfg), strat, rep, hash(strat) % (10**6)))

    # Run in parallel with progress bar
    set_start_method("spawn", force=True)
    results = []
    with Pool(processes=min(cfg.cores, cpu_count())) as pool:
        for r in tqdm(pool.imap_unordered(_worker, jobs),
                      total=len(jobs), desc="Running simulations", unit="run"):
            results.append(r)

    # Save raw results
    df = pd.DataFrame(results)
    raw_path = os.path.join(outdir, "results_raw.csv")
    df.to_csv(raw_path, index=False)

    # Aggregate means & standard errors by strategy
    def se(x: pd.Series) -> float:
        n = len(x)
        if n <= 1:
            return 0.0
        return float(x.std(ddof=1) / math.sqrt(n))

    agg = df.groupby("strategy").agg(
        mean_fraction_satisfied=("final_fraction_satisfied", "mean"),
        se_fraction_satisfied=("final_fraction_satisfied", se),
        mean_segregation=("final_segregation", "mean"),
        se_segregation=("final_segregation", se),
        mean_steps=("steps_to_equilibrium", "mean"),
        se_steps=("steps_to_equilibrium", se),
        mean_total_moves=("total_moves", "mean"),
        se_total_moves=("total_moves", se),
        mean_moves_per_agent=("moves_per_agent", "mean"),
        se_moves_per_agent=("moves_per_agent", se),
        reps=("rep", "count"),
    ).reset_index()

    agg_path = os.path.join(outdir, "results_summary.csv")
    agg.to_csv(agg_path, index=False)

    # Pretty print
    print("\n=== Summary by Strategy ===")
    print(agg.to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to    : {agg_path}")
    print(f"Output directory    : {outdir}")


if __name__ == "__main__":
    main()
