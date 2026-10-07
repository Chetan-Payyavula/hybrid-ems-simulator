"""
run_dp_benchmark.py -- how close are ECMS and the rule-based controller to the global optimum?

    python run_dp_benchmark.py                  # writes into ./results
    python run_dp_benchmark.py --out my_dir --n-soc 201 --n-u 81

DP is NON-CAUSAL (it knows the whole cycle in advance), so it is a bound on what ANY controller could achieve
under this model, not a controller you could actually ship.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hybrid_sim import (ECMS, IceOnly, Params, RuleBased, simulate, solve_dp, standard_cycles, tune_ecms_s0,
                        tune_rule_soc_target, shadow_price, priced_l100)

FIXED_S0 = 2.8   # one equivalence factor for every cycle (what a deployable ECMS would have to use)
COL = {"Hybrid: rule-based": "#7fb3d9", "Hybrid: rule-based (tuned)": "#2a7fbf", "Hybrid: ECMS (tuned)": "#d9531e",
       "Hybrid: ECMS (fixed s0)": "#f2a07b", "Hybrid: DP (global optimum)": "#2ca02c", "ICE-only baseline": "#888888"}
HYBRIDS = ["Hybrid: rule-based", "Hybrid: rule-based (tuned)", "Hybrid: ECMS (tuned)", "Hybrid: ECMS (fixed s0)"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-soc", type=int, default=151)
    ap.add_argument("--n-u", type=int, default=61)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = Params()
    cycles = standard_cycles()

    rows, runs_by_cycle = [], {}
    for name, cyc in cycles.items():
        s0 = tune_ecms_s0(cyc, p)
        tgt = tune_rule_soc_target(cyc, p)
        sol = solve_dp(cyc, p, n_soc=args.n_soc, n_u=args.n_u)
        runs = {
            "ICE-only baseline": simulate(cyc, IceOnly(), p),
            "Hybrid: rule-based": simulate(cyc, RuleBased(), p),
            "Hybrid: rule-based (tuned)": simulate(cyc, RuleBased(soc_target=tgt), p),
            "Hybrid: ECMS (tuned)": simulate(cyc, ECMS(s0=s0), p),
            "Hybrid: ECMS (fixed s0)": simulate(cyc, ECMS(s0=FIXED_S0), p),
            "Hybrid: DP (global optimum)": simulate(cyc, sol.strategy(), p),
        }
        runs_by_cycle[name] = runs
        sh = shadow_price(sol)
        ice = runs["ICE-only baseline"].summary["l_per_100km_raw"]
        priced = {sname: (r.summary["l_per_100km_raw"] if sname == "ICE-only baseline"
                          else priced_l100(r.summary, p, sh)) for sname, r in runs.items()}
        dp = priced["Hybrid: DP (global optimum)"]
        for sname, r in runs.items():
            x = priced[sname]
            hyb = sname != "ICE-only baseline"
            rows.append({
                "cycle": name, "strategy": sname,
                "l_per_100km_dp_priced": round(x, 4),
                "l_per_100km_raw": round(r.summary["l_per_100km_raw"], 4),
                "pct_above_dp": round(100 * (x - dp) / dp, 2),
                "share_of_achievable_saving_captured_pct": round(100 * (ice - x) / (ice - dp), 1),
                "soc_end": round(r.summary["soc_end"], 3) if hyb else "",
                "regen_share_of_braking": round(r.summary["regen_share_of_braking"], 3) if hyb else "",
                "engine_on_fraction": round(r.summary["engine_on_fraction"], 3),
                "engine_starts": r.summary["engine_starts"],
                "mean_engine_eff_when_on": round(r.summary["mean_engine_eff_when_on"], 3),
                "dp_shadow_price": round(sh, 3) if sname.startswith("Hybrid: DP") else "",
                "ecms_s0": round(s0 if "tuned" in sname else FIXED_S0, 3) if sname.startswith("Hybrid: ECMS") else "",
                "rule_soc_target": round(tgt, 3) if sname == "Hybrid: rule-based (tuned)" else "",
            })
    write_csv(os.path.join(args.out, "dp_benchmark.csv"), rows)

    # ------------------------------------------------------------------ convergence on urban
    conv = []
    for ns, nu in ((41, 21), (61, 31), (101, 41), (151, 61), (201, 81), (301, 121)):
        sol = solve_dp(cycles["urban"], p, n_soc=ns, n_u=nu)
        r = simulate(cycles["urban"], sol.strategy(), p).summary
        conv.append({"n_soc": ns, "n_u": nu, "dp_estimate_l": round(sol.fuel_estimate_l(), 5),
                     "replayed_fuel_l": round(r["fuel_l"], 5),
                     "l_per_100km_soc_corrected": round(r["l_per_100km_soc_corrected"], 4),
                     "soc_end": round(r["soc_end"], 4)})
    write_csv(os.path.join(args.out, "dp_grid_convergence_urban.csv"), conv)

    # ------------------------------------------------------------------ figures
    names = list(cycles)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    w = 0.2
    for j, sname in enumerate(HYBRIDS):
        gap = [next(r["pct_above_dp"] for r in rows if r["cycle"] == c and r["strategy"] == sname) for c in names]
        cap = [next(r["share_of_achievable_saving_captured_pct"] for r in rows if r["cycle"] == c and r["strategy"] == sname)
               for c in names]
        x = np.arange(len(names)) + (j - 1.5) * w
        b1 = ax[0].bar(x, gap, w, color=COL[sname], label=sname.replace("Hybrid: ", ""))
        b2 = ax[1].bar(x, cap, w, color=COL[sname], label=sname.replace("Hybrid: ", ""))
        for b, v in zip(b1, gap):
            ax[0].text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.2, f"{v:.1f}", ha="center", fontsize=7)
        for b, v in zip(b2, cap):
            ax[1].text(b.get_x() + b.get_width() / 2, max(v, 0) + 1.5, f"{v:.0f}", ha="center", fontsize=7)
    ax[0].set_title("Fuel above the global optimum [%]  (0 = optimal)")
    ax[1].set_title("Share of the achievable hybrid saving captured [%]")
    ax[1].axhline(100, color="#2ca02c", ls="--", lw=1)
    ax[1].set_ylim(-10, 118)
    ax[0].axhline(0, color="#2ca02c", lw=1)
    for a in ax:
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names)
        a.grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig5_dp_gap.png"), dpi=150)
    plt.close(fig)

    runs = runs_by_cycle["mixed"]
    cyc = cycles["mixed"]
    t = cyc.t[:-1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    ax[0].plot(t, runs["Hybrid: ECMS (tuned)"].arrays["v"] * 3.6, color="k", lw=1)
    ax[0].set_ylabel("Speed [km/h]")
    for s in ("Hybrid: rule-based", "Hybrid: ECMS (tuned)", "Hybrid: DP (global optimum)"):
        ax[1].plot(t, runs[s].arrays["soc"], color=COL[s], label=s, lw=1.4)
    ax[1].set_ylabel("Battery SOC")
    ax[1].legend(loc="upper right", fontsize=8)
    for s, ls in (("Hybrid: ECMS (tuned)", "-"), ("Hybrid: DP (global optimum)", "-")):
        ax[2].plot(t, runs[s].arrays["p_eng"] / 1e3, color=COL[s], lw=1, ls=ls, label=f"engine, {s.split(': ')[1]}")
    ax[2].set_ylabel("Engine power [kW]")
    ax[2].set_xlabel("Time [s]")
    ax[2].legend(loc="upper right", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].set_title("'mixed' cycle: what the global optimum does differently")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig6_dp_vs_ecms_mixed.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ console report
    print(f"\nDP grid: {args.n_soc} SOC x {args.n_u} controls\n")
    print("=== FUEL USE [L/100 km, leftover battery priced at DP marginal value] and gap to the global optimum ===")
    cols = ["Hybrid: rule-based", "Hybrid: rule-based (tuned)", "Hybrid: ECMS (tuned)", "Hybrid: ECMS (fixed s0)"]
    print(f"{'cycle':11s}{'ICE':>7s}{'DP':>7s} |" + "".join(f"{c.replace('Hybrid: ',''):>27s}" for c in cols))
    print(f"{'':25s} |" + "".join(f"{'L/100  gap   captures  socEnd':>27s}" for _ in cols))
    for c in names:
        g = lambda sname, k: next(r[k] for r in rows if r["cycle"] == c and r["strategy"] == sname)
        line = f"{c:11s}{g('ICE-only baseline','l_per_100km_dp_priced'):7.2f}{g('Hybrid: DP (global optimum)','l_per_100km_dp_priced'):7.2f} |"
        for sname in cols:
            line += (f"{g(sname,'l_per_100km_dp_priced'):6.2f}{g(sname,'pct_above_dp'):+6.1f}%"
                     f"{g(sname,'share_of_achievable_saving_captured_pct'):6.0f}%{g(sname,'soc_end'):7.2f}  ")
        print(line)
    print("\nDP implied value of stored energy vs tuned ECMS s0 (J fuel per J battery):")
    for c in names:
        g = lambda sname, k: next(r[k] for r in rows if r["cycle"] == c and r["strategy"] == sname)
        print(f"  {c:11s} DP {g('Hybrid: DP (global optimum)','dp_shadow_price'):.2f}   ECMS s0 {g('Hybrid: ECMS (tuned)','ecms_s0'):.2f}"
              f"   tuned rule-based SOC target {g('Hybrid: rule-based (tuned)','rule_soc_target'):.2f}")
    print("\n=== GRID CONVERGENCE (urban) ===")
    print(f"{'n_soc':>6s}{'n_u':>6s}{'DP estimate L':>15s}{'replayed L':>12s}{'L/100 corr':>12s}{'SOC end':>9s}")
    for r in conv:
        print(f"{r['n_soc']:6d}{r['n_u']:6d}{r['dp_estimate_l']:15.5f}{r['replayed_fuel_l']:12.5f}"
              f"{r['l_per_100km_soc_corrected']:12.4f}{r['soc_end']:9.4f}")
    print(f"\nFiles written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
