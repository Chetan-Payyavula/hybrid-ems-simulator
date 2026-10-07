"""
run_demo.py -- runs the full study and writes results + figures.

    python run_demo.py                 # writes into ./results
    python run_demo.py --out my_dir
    python run_demo.py --cycle-csv wltp.csv   # optional: add an official cycle (time_s,speed_kmh at 1 s steps)
"""
import argparse
import csv
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hybrid_sim import (ECMS, IceOnly, Params, RuleBased, load_cycle_csv, max_speed_for_range, simulate,
                        standard_cycles, steady_speed_sweep, top_speed_report, tune_ecms_s0)

C = {"ICE-only baseline": "#888888", "Hybrid: rule-based": "#2a7fbf", "Hybrid: ECMS": "#d9531e"}


def run_strategies(cycle, p):
    s0 = tune_ecms_s0(cycle, p)
    runs = {
        "ICE-only baseline": simulate(cycle, IceOnly(), p),
        "Hybrid: rule-based": simulate(cycle, RuleBased(), p),
        "Hybrid: ECMS": simulate(cycle, ECMS(s0=s0), p),
    }
    return runs, s0


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sensitivity(base: Params, cycles: dict):
    """Hybrid (ECMS) fuel saving vs the matching ICE-only baseline under changed assumptions.
    Evaluated on a gentle cycle ('mixed') and a hard-driving one ('aggressive')."""
    variants = {
        "baseline assumptions": base,
        "MGU-H off": replace(base, mguh=replace(base.mguh, enabled=False)),
        "MGU-H backpressure penalty 0": replace(base, mguh=replace(base.mguh, backpressure_penalty=0.0)),
        "MGU-H backpressure penalty 3": replace(base, mguh=replace(base.mguh, backpressure_penalty=3.0)),
        "MGU-H generous (F1-like, phi 0.25)": replace(base, mguh=replace(base.mguh, phi_max=0.25, backpressure_penalty=0.5)),
        "battery 0.75 kWh": replace(base, elec=replace(base.elec, cap_wh=750.0)),
        "battery 3.0 kWh": replace(base, elec=replace(base.elec, cap_wh=3000.0)),
        "e-machine 20 kW": replace(base, elec=replace(base.elec, motor_max_w=20e3)),
        "e-machine 60 kW": replace(base, elec=replace(base.elec, motor_max_w=60e3)),
        "Cd 0.28": replace(base, vehicle=replace(base.vehicle, cd=0.28)),
        "Cd 0.32": replace(base, vehicle=replace(base.vehicle, cd=0.32)),
        "regen limited to 50 % of braking": replace(base, elec=replace(base.elec, regen_frac_max=0.50)),
    }
    rows = []
    for name, p in variants.items():
        row = {"variant": name}
        for cname in ("mixed", "aggressive"):
            cyc = cycles[cname]
            ice = simulate(cyc, IceOnly(), p).summary["l_per_100km_soc_corrected"]
            ecms = simulate(cyc, ECMS(s0=tune_ecms_s0(cyc, p)), p).summary
            row[f"{cname}_ice_l_per_100km"] = round(ice, 3)
            row[f"{cname}_ecms_l_per_100km"] = round(ecms["l_per_100km_soc_corrected"], 3)
            row[f"{cname}_saving_pct"] = round(100 * (ice - ecms["l_per_100km_soc_corrected"]) / ice, 2)
            row[f"{cname}_mguh_kwh"] = round(ecms["mguh_kwh"], 4)
        t = top_speed_report(p)
        row["hybrid_burst_top_speed_kmh"] = round(t["hybrid_burst_top_speed_kmh"], 1)
        row["burst_duration_s_initial_soc"] = round(t["burst_duration_s_from_initial_soc"], 1)
        rows.append(row)
    return rows


def fig_summary(table, cycles, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(C)
    w = 0.26
    for j, s in enumerate(names):
        vals = [table[(c, s)]["l_per_100km_soc_corrected"] for c in cycles]
        bars = ax.bar(np.arange(len(cycles)) + (j - 1) * w, vals, w, label=s, color=C[s])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
    ax.axhline(9.4, color="k", ls="--", lw=1)
    ax.text(2.0, 9.65, "report target 9.4 L/100 km", ha="center", fontsize=9)
    ax.set_xticks(range(len(cycles)))
    ax.set_xticklabels(cycles)
    ax.set_ylabel("Fuel use [L/100 km], SOC-corrected")
    ax.set_title("Fuel use by drive cycle and energy-management strategy (synthetic cycles)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_timeseries(runs, cycle, out):
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = cycle.t[:-1]
    ax[0].plot(t, runs["Hybrid: ECMS"].arrays["v"] * 3.6, color="k", lw=1)
    ax[0].set_ylabel("Speed [km/h]")
    for s in ("Hybrid: rule-based", "Hybrid: ECMS"):
        ax[1].plot(t, runs[s].arrays["soc"], color=C[s], label=s)
    ax[1].set_ylabel("Battery SOC")
    ax[1].legend(loc="upper right")
    A = runs["Hybrid: ECMS"].arrays
    ax[2].plot(t, A["p_eng"] / 1e3, color="#d9531e", lw=1, label="Engine")
    ax[2].plot(t, A["p_m"] / 1e3, color="#2a7fbf", lw=1, label="MGU-K (+assist / -regen or charge)")
    ax[2].plot(t, A["p_h"] / 1e3, color="#2ca02c", lw=1, label="MGU-H electric")
    ax[2].set_ylabel("Power [kW]")
    ax[2].legend(loc="upper right", fontsize=8)
    for s, r in runs.items():
        ax[3].plot(t, np.cumsum(r.arrays["fuel_w"]) * cycle.dt / r.p.engine.lhv_j_per_kg / r.p.engine.density_kg_per_l,
                   color=C[s], label=s)
    ax[3].set_ylabel("Cumulative fuel [L]")
    ax[3].set_xlabel("Time [s]")
    ax[3].legend(loc="upper left")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].set_title(f"'{cycle.name}' cycle: ECMS vs rule-based operation")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_range(p, out):
    rows = steady_speed_sweep(p, range(40, 281, 5))
    v500 = max_speed_for_range(p, 500.0)
    t = top_speed_report(p)
    ok = [r for r in rows if r["engine_alone_feasible"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r["speed_kmh"] for r in ok], [r["range_km"] for r in ok], color="#d9531e", lw=2)
    ax.axhline(500, color="k", ls="--", lw=1)
    ax.axvline(v500, color="k", ls=":", lw=1)
    ax.text(v500 + 3, 1000, f"500 km range holds\nup to ~{v500:.0f} km/h", fontsize=9)
    ax.axvspan(t["engine_only_top_speed_kmh"], t["hybrid_burst_top_speed_kmh"], color="#2a7fbf", alpha=0.2)
    ax.text(t["engine_only_top_speed_kmh"] - 2, 700,
            f"battery-assist burst zone\n{t['engine_only_top_speed_kmh']:.0f}-{t['hybrid_burst_top_speed_kmh']:.0f} km/h\n"
            f"(~{t['burst_duration_s_from_initial_soc']:.0f}-{t['burst_duration_s_from_full_soc']:.0f} s)",
            fontsize=8, ha="right")
    ax.set_xlabel("Steady speed [km/h]")
    ax.set_ylabel("Range on 47 L [km]")
    ax.set_ylim(0, 1300)
    ax.set_title("Range vs steady speed (engine-only, flat road, no battery credit)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_sensitivity(rows, out):
    fig, ax = plt.subplots(figsize=(10, 7))
    names = [r["variant"] for r in rows][::-1]
    y = np.arange(len(names))
    m = [r["mixed_saving_pct"] for r in rows][::-1]
    a = [r["aggressive_saving_pct"] for r in rows][::-1]
    ax.barh(y + 0.2, m, 0.38, color="#2a7fbf", label="'mixed' cycle (gentle)")
    ax.barh(y - 0.2, a, 0.38, color="#d9531e", label="'aggressive' cycle")
    for i, (vm, va) in enumerate(zip(m, a)):
        ax.text(vm + 0.3, i + 0.2, f"{vm:.1f}", va="center", fontsize=7)
        ax.text(va + 0.3, i - 0.2, f"{va:.1f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("ECMS fuel saving vs ICE-only baseline [%]")
    ax.set_title("How much do the assumptions matter?")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--cycle-csv", default=None, help="official cycle CSV with columns time_s,speed_kmh")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    p = Params()
    cycles = standard_cycles()
    if args.cycle_csv:
        c = load_cycle_csv(args.cycle_csv, name=os.path.splitext(os.path.basename(args.cycle_csv))[0])
        cycles[c.name] = c

    table, rows, s0s = {}, [], {}
    for name, cyc in cycles.items():
        runs, s0 = run_strategies(cyc, p)
        s0s[name] = s0
        for s, r in runs.items():
            table[(name, s)] = r.summary
            rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.summary.items()})
        if name == "mixed":
            fig_timeseries(runs, cyc, os.path.join(args.out, "fig2_timeseries_mixed.png"))
    write_csv(os.path.join(args.out, "results_by_cycle.csv"), rows)

    names = list(cycles)
    fig_summary(table, names, os.path.join(args.out, "fig1_fuel_by_cycle.png"))
    fig_range(p, os.path.join(args.out, "fig3_range_vs_speed.png"))

    sens = sensitivity(p, cycles)
    write_csv(os.path.join(args.out, "sensitivity_mixed.csv"), sens)
    fig_sensitivity(sens, os.path.join(args.out, "fig4_sensitivity.png"))
    write_csv(os.path.join(args.out, "steady_speed_range.csv"),
              [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
               for r in steady_speed_sweep(p, range(40, 281, 10))])

    # ------------------------------------------------------------------ console report
    t = top_speed_report(p)
    print("\n=== TOP SPEED / BURST ===")
    for k, v in t.items():
        print(f"  {k:42s} {v:8.1f}")
    print(f"  max steady speed with 500 km range        {max_speed_for_range(p):8.1f} km/h")
    print("\n=== FUEL USE [L/100 km, SOC-corrected] ===")
    print(f"{'cycle':11s}{'avg km/h':>9s}{'ICE-only':>10s}{'rule':>8s}{'ECMS':>8s}{'ECMS saving':>13s}{'range@ECMS km':>15s}")
    for name in names:
        a, b, c = (table[(name, s)] for s in C)
        sv = 100 * (a["l_per_100km_soc_corrected"] - c["l_per_100km_soc_corrected"]) / a["l_per_100km_soc_corrected"]
        print(f"{name:11s}{a['avg_speed_kmh']:9.1f}{a['l_per_100km_soc_corrected']:10.2f}{b['l_per_100km_soc_corrected']:8.2f}"
              f"{c['l_per_100km_soc_corrected']:8.2f}{sv:12.1f}%{c['range_km_on_tank']:15.0f}")
    print("\n=== SENSITIVITY: ECMS fuel saving vs ICE-only [%] ===")
    print(f"  {'variant':38s}{'mixed':>7s}{'aggr.':>7s}{'MGU-H kWh (mixed)':>19s}{'burst top speed':>17s}")
    for r in sens:
        print(f"  {r['variant']:38s}{r['mixed_saving_pct']:7.1f}{r['aggressive_saving_pct']:7.1f}"
              f"{r['mixed_mguh_kwh']:19.3f}{r['hybrid_burst_top_speed_kmh']:9.0f} km/h {r['burst_duration_s_initial_soc']:4.0f} s")
    print(f"\nFiles written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
