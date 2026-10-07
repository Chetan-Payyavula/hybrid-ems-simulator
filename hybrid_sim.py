"""
hybrid_sim.py  --  Quasi-static energy-management simulator for the petrol turbo-hybrid.

Architecture modelled (P2 parallel hybrid):
    engine --+--> transmission --> wheels
             |
    MGU-K (e-machine, assist + regen) <--> battery
    MGU-H (turbo-shaft generator) --------> battery

The report lists a separate "electric motor" and "MGU-K". Here they are LUMPED into one
e-machine (called MGU-K), which is the standard way to model a P2 hybrid.

Method: backward-facing. A speed-vs-time cycle is prescribed; wheel power is computed from
road-load physics; an energy-management strategy decides how the engine and MGU-K share it.

EVERY numeric parameter below is labelled REPORT (taken from the project report), or
ASSUMPTION (chosen by me, not validated against measured data). Change them freely.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

G = 9.81


# ----------------------------------------------------------------------------- components
@dataclass(frozen=True)
class Vehicle:
    mass_kg: float = 1550.0           # ASSUMPTION (hybrid car, incl. driver)
    hybrid_hw_mass_kg: float = 90.0   # ASSUMPTION battery + e-machine + MGU-H; removed for the ICE-only baseline
    cd: float = 0.30                  # REPORT
    frontal_area_m2: float = 2.0      # REPORT
    air_density: float = 1.225        # REPORT
    crr0: float = 0.009               # ASSUMPTION rolling resistance coefficient at low speed
    crr_v2: float = 8e-7              # ASSUMPTION speed-dependent rolling term [s^2/m^2]
    rot_mass_factor: float = 1.04     # ASSUMPTION rotating inertia
    trans_eff: float = 0.93           # ASSUMPTION transmission efficiency

    def mass(self, hybrid: bool = True) -> float:
        return self.mass_kg if hybrid else self.mass_kg - self.hybrid_hw_mass_kg

    def road_load_w(self, v: float, hybrid: bool = True) -> float:
        """Steady-speed power needed at the wheels [W] on flat road."""
        drag = 0.5 * self.air_density * self.cd * self.frontal_area_m2 * v**2
        rr = (self.crr0 + self.crr_v2 * v**2) * self.mass(hybrid) * G
        return (drag + rr) * v


@dataclass(frozen=True)
class Engine:
    p_max_w: float = 160e3            # REPORT range 200-230 hp (150-170 kW); mid-range ~215 hp
    idle_fuel_gps: float = 0.25       # ASSUMPTION idle fuel flow [g/s]
    willans_a: float = 2.35           # ASSUMPTION marginal fuel/power ratio (1/a = 42.6 % marginal efficiency)
    willans_b_per_kw: float = 0.0028  # ASSUMPTION quadratic high-load / enrichment loss [1/kW]
    lhv_j_per_kg: float = 43.4e6      # petrol lower heating value
    density_kg_per_l: float = 0.745   # REPORT
    start_fuel_g: float = 0.4         # ASSUMPTION fuel burned per engine restart

    def idle_fuel_w(self) -> float:
        return self.idle_fuel_gps * 1e-3 * self.lhv_j_per_kg

    def fuel_power_w(self, p_w):
        """Willans-line fuel power for engine shaft power p_w (engine ON). Works on arrays."""
        p_kw = np.asarray(p_w, dtype=float) / 1e3
        return self.idle_fuel_w() + 1e3 * (self.willans_a * p_kw + self.willans_b_per_kw * p_kw**2)

    def efficiency(self, p_w):
        p_w = np.asarray(p_w, dtype=float)
        return np.where(p_w > 0, p_w / self.fuel_power_w(p_w), 0.0)


@dataclass(frozen=True)
class MguH:
    enabled: bool = True
    exhaust_frac: float = 0.30        # ASSUMPTION share of fuel energy leaving as exhaust enthalpy
    phi_max: float = 0.08             # ASSUMPTION share of exhaust power recoverable at full load (scales with load)
    gen_eff: float = 0.92             # ASSUMPTION generator + inverter efficiency
    load_threshold: float = 0.25      # ASSUMPTION engine load fraction below which exhaust flow is too low
    backpressure_penalty: float = 1.0 # ASSUMPTION extra fuel power per unit MGU-H electric power (turbine backpressure)

    def elec_w(self, engine: Engine, p_eng_w, fuel_base_w):
        """Electric power MGU-H can generate at this engine operating point (arrays OK)."""
        p_eng_w = np.asarray(p_eng_w, dtype=float)
        if not self.enabled:
            return np.zeros_like(p_eng_w)
        x = np.clip(p_eng_w / engine.p_max_w, 0.0, 1.0)
        out = self.gen_eff * self.exhaust_frac * np.asarray(fuel_base_w) * self.phi_max * x
        return np.where(x >= self.load_threshold, out, 0.0)


@dataclass(frozen=True)
class Electrical:
    motor_max_w: float = 40e3         # REPORT 40-60 hp assist -> 40 kW (~54 hp)
    em_eff: float = 0.92              # ASSUMPTION e-machine + inverter, each direction
    cap_wh: float = 1500.0            # ASSUMPTION ~1.5 kWh pack (F1 ES is ~1.1 kWh usable)
    v_oc: float = 350.0               # ASSUMPTION open-circuit voltage [V]
    r_int: float = 0.09               # ASSUMPTION pack internal resistance [ohm]
    soc_min: float = 0.25             # ASSUMPTION
    soc_max: float = 0.85             # ASSUMPTION
    soc_init: float = 0.55            # ASSUMPTION
    p_batt_max_w: float = 50e3        # ASSUMPTION terminal power limit (both directions)
    aux_w: float = 300.0              # ASSUMPTION 12 V accessory load drawn from the battery
    regen_frac_max: float = 0.85      # ASSUMPTION max share of wheel braking power sent through the e-machine
    regen_min_speed: float = 1.5      # ASSUMPTION [m/s] no regen below this speed

    def chem_power_w(self, p_term):
        """Chemical (internal) battery power for terminal power p_term (+ = discharge). Arrays OK."""
        p_term = np.asarray(p_term, dtype=float)
        disc = np.maximum(self.v_oc**2 - 4.0 * self.r_int * p_term, 0.0)
        current = (self.v_oc - np.sqrt(disc)) / (2.0 * self.r_int)
        return self.v_oc * current

    def mech_to_elec(self, p_m):
        p_m = np.asarray(p_m, dtype=float)
        return np.where(p_m > 0, p_m / self.em_eff, p_m * self.em_eff)

    def motor_bounds(self, soc: float) -> tuple[float, float]:
        """Feasible mechanical e-machine power range [lo, hi] (hi = assist, lo = generate), ignoring MGU-H."""
        hi = min(self.motor_max_w, (self.p_batt_max_w - self.aux_w) * self.em_eff)
        lo = -min(self.motor_max_w, (self.p_batt_max_w + self.aux_w) / self.em_eff)
        if soc <= self.soc_min:
            hi = 0.0
        if soc >= self.soc_max:
            lo = 0.0
        return lo, hi


@dataclass(frozen=True)
class Params:
    vehicle: Vehicle = field(default_factory=Vehicle)
    engine: Engine = field(default_factory=Engine)
    mguh: MguH = field(default_factory=MguH)
    elec: Electrical = field(default_factory=Electrical)
    alt_eff: float = 0.65             # ASSUMPTION alternator efficiency (ICE-only baseline powers its accessories)
    soc_eq_eff: float = 0.37          # fuel->battery exchange rate for SOC-corrected fuel (1/2.7). Set from the DP
                                      # shadow price (2.5-2.8 J fuel per J stored); v1 used 0.30, which over-credited
                                      # runs that finish with extra charge
    tank_l: float = 47.0              # REPORT (35 kg petrol)


# ----------------------------------------------------------------------------- drive cycles
@dataclass(frozen=True)
class Cycle:
    name: str
    t: np.ndarray        # [s]
    v: np.ndarray        # [m/s]

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def build_cycle(name: str, segments: list[tuple], dt: float = 1.0) -> Cycle:
    """segments: ('idle', s) | ('cruise', s) | ('to', v_kmh, accel_ms2)  (accel is magnitude; sign inferred)."""
    v = [0.0]
    for seg in segments:
        kind = seg[0]
        if kind in ("idle", "cruise"):
            n = int(round(seg[1] / dt))
            v.extend([v[-1]] * n)
        elif kind == "to":
            target, acc = seg[1] / 3.6, abs(seg[2])
            while abs(v[-1] - target) > 1e-9:
                step = acc * dt
                v.append(min(v[-1] + step, target) if target > v[-1] else max(v[-1] - step, target))
        else:
            raise ValueError(kind)
    v = np.array(v)
    return Cycle(name, np.arange(len(v)) * dt, v)


def standard_cycles() -> dict[str, Cycle]:
    """SYNTHETIC cycles (NOT the official WLTP/FTP). Use load_cycle_csv() for official traces."""
    urban = []
    for _ in range(14):
        urban += [("to", 50, 1.3), ("cruise", 12), ("to", 0, 1.6), ("idle", 10)]
    suburban = []
    for _ in range(6):
        suburban += [("to", 70, 1.0), ("cruise", 35), ("to", 45, 0.8), ("cruise", 15), ("to", 0, 1.3), ("idle", 8)]
    highway = [("to", 120, 0.9), ("cruise", 240), ("to", 90, 0.5), ("cruise", 90), ("to", 130, 0.5),
               ("cruise", 300), ("to", 100, 0.5), ("cruise", 120), ("to", 0, 1.1), ("idle", 5)]
    mixed = urban[: 4 * 7] + suburban[: 6 * 6] + highway[:8] + [("to", 0, 1.1), ("idle", 5)] + urban[: 2 * 4]
    aggressive = []
    for _ in range(5):
        aggressive += [("to", 110, 2.2), ("cruise", 8), ("to", 40, 2.8), ("to", 130, 2.0), ("cruise", 6),
                       ("to", 0, 3.0), ("idle", 6)]
    return {
        "urban": build_cycle("urban", urban),
        "suburban": build_cycle("suburban", suburban),
        "highway": build_cycle("highway", highway),
        "mixed": build_cycle("mixed", mixed),
        "aggressive": build_cycle("aggressive", aggressive),
    }


def load_cycle_csv(path: str, name: str | None = None) -> Cycle:
    """Load an official cycle: CSV with header 'time_s,speed_kmh' at uniform 1 s steps."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return Cycle(name or path, np.asarray(data["time_s"], float), np.asarray(data["speed_kmh"], float) / 3.6)


# ----------------------------------------------------------------------------- strategies
class Strategy:
    name = "base"
    hybrid = True

    def reset(self, p: Params) -> None:
        pass

    def begin_step(self, i: int) -> None:
        """Called once per timestep before any decision (lets time-dependent strategies, e.g. DP, track time)."""

    def request(self, p: Params, soc: float, p_dem: float, v: float, dt: float, engine_on: bool) -> float:
        """Return requested MGU-K mechanical power [W]; + assist, - generate."""
        raise NotImplementedError


class IceOnly(Strategy):
    name = "ICE-only baseline"
    hybrid = False


@dataclass
class RuleBased(Strategy):
    name: str = "Hybrid: rule-based"
    hybrid: bool = True
    soc_target: float = 0.55
    ev_power_max_w: float = 12e3
    ev_speed_max: float = 50 / 3.6
    ev_soc_min: float = 0.40
    k_assist_w_per_soc: float = 250e3
    k_charge_w_per_soc: float = 250e3

    def request(self, p, soc, p_dem, v, dt, engine_on):
        e = p.elec
        lo, hi = e.motor_bounds(soc)
        # electric-only launch / creep
        if p_dem <= self.ev_power_max_w and v <= self.ev_speed_max and soc > self.ev_soc_min:
            return min(p_dem, hi)
        # forced assist when engine alone cannot meet demand
        req = max(p_dem - p.engine.p_max_w, 0.0)
        # SOC-driven discretionary assist / charging
        dsoc = soc - self.soc_target
        if dsoc > 0:
            req = max(req, min(self.k_assist_w_per_soc * dsoc, 0.5 * p_dem))
        else:
            headroom = max(p.engine.p_max_w - p_dem, 0.0)
            req = min(req, 0.0) - min(self.k_charge_w_per_soc * (-dsoc), headroom)
        return float(np.clip(req, lo, hi))


@dataclass
class ECMS(Strategy):
    """Equivalent Consumption Minimisation Strategy with SOC-penalty equivalence factor."""
    name: str = "Hybrid: ECMS"
    hybrid: bool = True
    s0: float = 2.6                   # equivalence factor (fuel-power per unit battery chemical power)
    soc_ref: float = 0.55
    n_grid: int = 81
    _prev_on: bool = False

    def reset(self, p):
        self._prev_on = False

    def s_eff(self, p, soc):
        half = 0.5 * (p.elec.soc_max - p.elec.soc_min)
        x = np.clip((soc - self.soc_ref) / half, -1.0, 1.0)
        return self.s0 * (1.0 - x**3)

    def request(self, p, soc, p_dem, v, dt, engine_on):
        e, eng = p.elec, p.engine
        lo, hi = e.motor_bounds(soc)
        lo_eff = max(lo, p_dem - eng.p_max_w)
        hi_eff = min(hi, p_dem)
        if lo_eff > hi_eff:                       # engine + motor cannot meet demand -> full assist
            return hi
        pm = np.linspace(lo_eff, hi_eff, self.n_grid)
        p_eng = p_dem - pm
        on = p_eng > 1.0
        fuel_base = np.where(on, eng.fuel_power_w(np.maximum(p_eng, 0.0)), 0.0)
        p_h = np.where(on, p.mguh.elec_w(eng, p_eng, fuel_base), 0.0)
        fuel = fuel_base + p.mguh.backpressure_penalty * p_h
        fuel = fuel + np.where(on & (not engine_on), eng.start_fuel_g * 1e-3 * eng.lhv_j_per_kg / dt, 0.0)
        p_term = e.mech_to_elec(pm) + e.aux_w - p_h
        chem = e.chem_power_w(np.clip(p_term, -e.p_batt_max_w, e.p_batt_max_w))
        cost = fuel + self.s_eff(p, soc) * chem
        return float(pm[int(np.argmin(cost))])


# ----------------------------------------------------------------------------- simulation
@dataclass
class Result:
    cycle: Cycle
    strategy: str
    p: Params
    arrays: dict
    summary: dict


def simulate(cycle: Cycle, strategy: Strategy, p: Params | None = None) -> Result:
    p = p or Params()
    veh, eng, e, mh = p.vehicle, p.engine, p.elec, p.mguh
    hybrid = strategy.hybrid
    m = veh.mass(hybrid)
    n = len(cycle.v) - 1
    dt = cycle.dt
    strategy.reset(p)

    keys = ["v", "p_wheel", "p_dem", "p_eng", "p_m", "p_h", "p_term", "p_chem", "soc", "fuel_w",
            "regen_wheel", "fric_wheel", "shortfall", "engine_on"]
    A = {k: np.zeros(n) for k in keys}
    soc = e.soc_init
    engine_on_prev = False
    starts = 0
    start_fuel_j = 0.0

    for i in range(n):
        strategy.begin_step(i)
        v0, v1 = cycle.v[i], cycle.v[i + 1]
        vm = 0.5 * (v0 + v1)
        a = (v1 - v0) / dt
        force = (veh.rot_mass_factor * m * a
                 + 0.5 * veh.air_density * veh.cd * veh.frontal_area_m2 * vm**2
                 + (veh.crr0 + veh.crr_v2 * vm**2) * m * G)
        p_wheel = force * vm
        p_m = p_eng = p_h = p_term = p_chem = fuel_w = regen_wheel = fric_wheel = shortfall = 0.0
        p_dem = 0.0
        on = False

        if p_wheel >= 0:                                       # ---- propulsion / coast
            p_dem = p_wheel / veh.trans_eff
            if not hybrid:                                     # ICE-only baseline
                aux_mech = e.aux_w / p.alt_eff
                if p_dem > 0 or vm < 0.5:
                    p_eng = min(p_dem + aux_mech, eng.p_max_w)
                    shortfall = max(p_dem + aux_mech - eng.p_max_w, 0.0)
                    fuel_w = float(eng.fuel_power_w(p_eng))
                    on = True
            else:
                req = strategy.request(p, soc, p_dem, vm, dt, engine_on_prev)
                lo, hi = e.motor_bounds(soc)
                p_m = float(np.clip(req, lo, hi))
                p_eng = p_dem - p_m
                if p_eng > eng.p_max_w:                        # engine saturated -> push assist to the limit
                    p_m = float(np.clip(p_dem - eng.p_max_w, lo, hi))
                    p_eng = p_dem - p_m
                    if p_eng > eng.p_max_w:
                        shortfall = p_eng - eng.p_max_w
                        p_eng = eng.p_max_w
                if p_eng < 0:                                  # never motor harder than demand
                    p_m, p_eng = p_dem, 0.0
                on = p_eng > 1.0
                if on:
                    fuel_base = float(eng.fuel_power_w(p_eng))
                    p_h_avail = float(mh.elec_w(eng, p_eng, fuel_base))
                    p_elec_m = float(e.mech_to_elec(p_m))
                    p_h = p_h_avail
                    if soc >= e.soc_max:                       # battery full: only offset the load
                        p_h = min(p_h, max(p_elec_m + e.aux_w, 0.0))
                    p_h = min(p_h, max(p_elec_m + e.aux_w + e.p_batt_max_w, 0.0))   # charge-power cap
                    fuel_w = fuel_base + mh.backpressure_penalty * p_h
        else:                                                   # ---- braking
            p_brake = -p_wheel
            regen_mech = 0.0
            if hybrid and vm >= e.regen_min_speed:
                want = min(e.motor_max_w, e.regen_frac_max * p_brake * veh.trans_eff)
                lo, _ = e.motor_bounds(soc)
                p_m = float(max(-want, lo))
                regen_mech = -p_m
            regen_wheel = regen_mech / veh.trans_eff
            fric_wheel = p_brake - regen_wheel
            if not hybrid and vm < 0.5:
                fuel_w, on, p_eng = float(eng.fuel_power_w(0.0)), True, 0.0

        if hybrid:
            p_term = float(e.mech_to_elec(p_m)) + e.aux_w - p_h
            p_chem = float(e.chem_power_w(p_term))
            soc = soc - p_chem * dt / (3600.0 * e.cap_wh)
        if on and not engine_on_prev and hybrid:
            starts += 1
            start_fuel_j += eng.start_fuel_g * 1e-3 * eng.lhv_j_per_kg
        engine_on_prev = on

        for k, val in zip(keys, [vm, p_wheel, p_dem, p_eng, p_m, p_h, p_term, p_chem, soc, fuel_w,
                                 regen_wheel, fric_wheel, shortfall, float(on)]):
            A[k][i] = val

    return Result(cycle, strategy.name, p, A, summarize(cycle, strategy, p, A, starts, start_fuel_j))


def summarize(cycle, strategy, p, A, starts, start_fuel_j) -> dict:
    eng, e = p.engine, p.elec
    dt = cycle.dt
    dist_km = float(np.sum(A["v"]) * dt / 1000.0)
    fuel_kg = float((np.sum(A["fuel_w"]) * dt + start_fuel_j) / eng.lhv_j_per_kg)
    soc0 = e.soc_init if strategy.hybrid else 0.0
    soc_end = float(A["soc"][-1]) if strategy.hybrid else 0.0
    d_batt_j = (soc0 - soc_end) * e.cap_wh * 3600.0                       # + = battery drained
    fuel_corr_kg = fuel_kg + d_batt_j / (eng.lhv_j_per_kg * p.soc_eq_eff)
    to_l = lambda kg: kg / eng.density_kg_per_l
    l100 = to_l(fuel_kg) / dist_km * 100.0 if dist_km > 0 else float("nan")
    l100c = to_l(fuel_corr_kg) / dist_km * 100.0 if dist_km > 0 else float("nan")
    on = A["engine_on"] > 0.5
    eng_shaft_j = float(np.sum(A["p_eng"][on]) * dt)
    eng_fuel_j = float(np.sum(A["fuel_w"][on]) * dt) if on.any() else 0.0
    brake_wheel_j = float(np.sum(A["regen_wheel"] + A["fric_wheel"]) * dt)
    regen_j = float(np.sum(A["regen_wheel"]) * dt)
    return {
        "cycle": cycle.name, "strategy": strategy.name,
        "distance_km": dist_km, "duration_s": float(cycle.t[-1]),
        "avg_speed_kmh": float(np.mean(A["v"]) * 3.6),
        "fuel_kg": fuel_kg, "fuel_l": to_l(fuel_kg),
        "l_per_100km_raw": l100, "l_per_100km_soc_corrected": l100c,
        "range_km_on_tank": p.tank_l / l100c * 100.0 if l100c == l100c and l100c > 0 else float("nan"),
        "soc_start": soc0 if strategy.hybrid else float("nan"),
        "soc_end": soc_end if strategy.hybrid else float("nan"),
        "soc_min_seen": float(np.min(A["soc"])) if strategy.hybrid else float("nan"),
        "soc_max_seen": float(np.max(A["soc"])) if strategy.hybrid else float("nan"),
        "regen_wheel_kwh": regen_j / 3.6e6,
        "braking_wheel_kwh": brake_wheel_j / 3.6e6,
        "regen_share_of_braking": regen_j / brake_wheel_j if brake_wheel_j > 0 else float("nan"),
        "mguh_kwh": float(np.sum(A["p_h"]) * dt / 3.6e6),
        "engine_on_fraction": float(np.mean(A["engine_on"])),
        "engine_starts": starts,
        "mean_engine_eff_when_on": eng_shaft_j / eng_fuel_j if eng_fuel_j > 0 else float("nan"),
        "shortfall_kwh": float(np.sum(A["shortfall"]) * dt / 3.6e6),
        "max_shortfall_kw": float(np.max(A["shortfall"]) / 1e3),
    }


# ----------------------------------------------------------------------------- analysis helpers
def max_speed_kmh(p: Params, crank_power_w: float, hybrid: bool = True) -> float:
    """Highest steady speed where wheel road-load equals crank power * trans_eff (bisection)."""
    lo, hi = 1.0, 150.0
    avail = crank_power_w * p.vehicle.trans_eff
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if p.vehicle.road_load_w(mid, hybrid) > avail:
            hi = mid
        else:
            lo = mid
    return lo * 3.6


def top_speed_report(p: Params) -> dict:
    """Engine-only vs engine+MGU-K top speed, and how long the battery can sustain the assist."""
    e, eng = p.elec, p.engine
    v_engine_only = max_speed_kmh(p, eng.p_max_w, hybrid=True)
    v_hybrid = max_speed_kmh(p, eng.p_max_w + e.motor_max_w, hybrid=True)
    v_ms = v_hybrid / 3.6
    fuel_base = float(eng.fuel_power_w(eng.p_max_w))
    p_h = float(p.mguh.elec_w(eng, eng.p_max_w, fuel_base))
    net_draw_w = e.motor_max_w / e.em_eff + e.aux_w - p_h
    usable_wh_full = (e.soc_max - e.soc_min) * e.cap_wh
    usable_wh_init = (e.soc_init - e.soc_min) * e.cap_wh
    return {
        "engine_only_top_speed_kmh": v_engine_only,
        "hybrid_burst_top_speed_kmh": v_hybrid,
        "net_battery_draw_at_full_assist_kw": net_draw_w / 1e3,
        "burst_duration_s_from_full_soc": usable_wh_full * 3600.0 / net_draw_w if net_draw_w > 0 else float("inf"),
        "burst_duration_s_from_initial_soc": usable_wh_init * 3600.0 / net_draw_w if net_draw_w > 0 else float("inf"),
        "power_needed_280kmh_at_crank_kw": p.vehicle.road_load_w(280 / 3.6) / p.vehicle.trans_eff / 1e3,
        "power_available_kw": (eng.p_max_w + e.motor_max_w) / 1e3,
    }


def tune_ecms_s0(cycle: Cycle, p: Params, lo: float = 1.2, hi: float = 5.0, iters: int = 14,
                 tol: float = 0.01) -> float:
    """Bisect s0 so the cycle ends near the starting SOC (charge-sustaining). Offline calibration."""
    best_s, best_err = None, 9.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = simulate(cycle, ECMS(s0=mid), p)
        err = r.summary["soc_end"] - p.elec.soc_init
        if abs(err) < best_err:
            best_s, best_err = mid, abs(err)
        if abs(err) < tol:
            break
        if err > 0:        # ended too full -> use the battery more -> lower s
            hi = mid
        else:
            lo = mid
    return best_s


def steady_speed_sweep(p: Params, speeds_kmh=range(40, 281, 10)) -> list[dict]:
    """Engine-only steady-speed fuel use and range on a full tank (ignores battery and MGU-H: conservative)."""
    eng = p.engine
    rows = []
    for kmh in speeds_kmh:
        v = kmh / 3.6
        p_crank = p.vehicle.road_load_w(v, hybrid=True) / p.vehicle.trans_eff
        if p_crank > eng.p_max_w:
            rows.append({"speed_kmh": kmh, "crank_kw": p_crank / 1e3, "l_per_100km": float("nan"),
                         "range_km": float("nan"), "engine_alone_feasible": False})
            continue
        fuel_lps = float(eng.fuel_power_w(p_crank)) / eng.lhv_j_per_kg / eng.density_kg_per_l
        l100 = fuel_lps / (v / 1000.0) * 100.0
        rows.append({"speed_kmh": kmh, "crank_kw": p_crank / 1e3, "l_per_100km": l100,
                     "range_km": p.tank_l / l100 * 100.0, "engine_alone_feasible": True})
    return rows


def max_speed_for_range(p: Params, range_km: float = 500.0) -> float:
    """Highest steady speed [km/h] at which the full tank still reaches range_km (engine-only, bisection)."""
    lo, hi = 20.0, 280.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = steady_speed_sweep(p, [mid])[0]
        ok = r["engine_alone_feasible"] and r["range_km"] >= range_km
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


# ----------------------------------------------------------------------------- dynamic programming benchmark
_BIG = 1e13   # finite stand-in for "infeasible" so interpolation stays well-defined


@dataclass
class DPSolution:
    cycle: Cycle
    p: Params
    soc_grid: np.ndarray
    policy: np.ndarray        # (T, 2, n_soc) optimal MGU-K power [W] for (time, engine_was_on, SOC)
    value0: np.ndarray        # (2, n_soc) optimal cost-to-go at t = 0 [J of fuel energy]

    def fuel_estimate_l(self) -> float:
        """DP's own estimate of optimal fuel [L], starting engine-off at the initial SOC (grid-interpolated)."""
        j = float(np.interp(self.p.elec.soc_init, self.soc_grid, self.value0[0]))
        return j / self.p.engine.lhv_j_per_kg / self.p.engine.density_kg_per_l

    def strategy(self) -> "DPPolicy":
        return DPPolicy(self)


class DPPolicy(Strategy):
    """Replays the DP-optimal policy inside the normal simulator (nearest SOC grid point)."""
    name = "Hybrid: DP (global optimum)"
    hybrid = True

    def __init__(self, sol: DPSolution):
        self.sol = sol
        self._t = 0

    def reset(self, p):
        self._t = 0

    def begin_step(self, i):
        self._t = i

    def request(self, p, soc, p_dem, v, dt, engine_on):
        g = self.sol.soc_grid
        k = int(np.clip(np.rint((soc - g[0]) / (g[1] - g[0])), 0, len(g) - 1))
        return float(self.sol.policy[self._t, int(engine_on), k])


def solve_dp(cycle: Cycle, p: Params | None = None, n_soc: int = 151, n_u: int = 61,
             terminal_price: float = 8.0, shortfall_price: float = 20.0) -> DPSolution:
    """
    Backward dynamic programming over (time, SOC, engine-was-on) for a KNOWN drive cycle.

    Minimises total fuel energy (incl. engine-start fuel and MGU-H backpressure penalty). The cost-to-go at the end
    penalises finishing below the initial SOC at `terminal_price` J of fuel per J of battery energy (above any real
    exchange rate, so the optimum is charge-sustaining); finishing above gives no credit.

    This is a NON-CAUSAL benchmark (it knows the future cycle). It uses exactly the same component models as
    simulate(), so its gap to ECMS / rule-based is a fair measure of what a perfect controller could still gain.
    """
    p = p or Params()
    veh, eng, e, mh = p.vehicle, p.engine, p.elec, p.mguh
    dt = cycle.dt
    n = len(cycle.v) - 1
    m = veh.mass(True)
    v = cycle.v
    vm = 0.5 * (v[:-1] + v[1:])
    acc = np.diff(v) / dt
    p_wheel = (veh.rot_mass_factor * m * acc + 0.5 * veh.air_density * veh.cd * veh.frontal_area_m2 * vm**2
               + (veh.crr0 + veh.crr_v2 * vm**2) * m * G) * vm

    grid = np.linspace(e.soc_min, e.soc_max, n_soc)
    cap_j = e.cap_wh * 3600.0
    hi_s = np.full(n_soc, min(e.motor_max_w, (e.p_batt_max_w - e.aux_w) * e.em_eff))
    lo_s = np.full(n_soc, -min(e.motor_max_w, (e.p_batt_max_w + e.aux_w) / e.em_eff))
    hi_s[grid <= e.soc_min] = 0.0
    lo_s[grid >= e.soc_max] = 0.0
    S = grid[:, None]
    start_pen = eng.start_fuel_g * 1e-3 * eng.lhv_j_per_kg
    tol = 1e-9

    V = np.tile(terminal_price * np.maximum(e.soc_init - grid, 0.0) * cap_j, (2, 1))
    policy = np.zeros((n, 2, n_soc), dtype=np.float32)
    ar = np.arange(n_soc)

    for t in range(n - 1, -1, -1):
        Vn = V
        Vnew = np.empty_like(V)
        if p_wheel[t] >= 0:                                           # propulsion / coast
            p_dem = p_wheel[t] / veh.trans_eff
            extras = [p_dem, p_dem - eng.p_max_w]
            u = np.concatenate([np.linspace(-e.motor_max_w, e.motor_max_w, n_u),
                                [x for x in extras if -e.motor_max_w <= x <= e.motor_max_w]])
            u = np.unique(u)
            u = u[u <= p_dem + 1e-9]                                  # never motor harder than the demand
            U = u[None, :]
            p_eng_raw = p_dem - U
            shortfall = np.maximum(p_eng_raw - eng.p_max_w, 0.0)
            p_eng = np.minimum(p_eng_raw, eng.p_max_w)
            on = p_eng > 1.0
            fuel_base = np.where(on, eng.fuel_power_w(np.maximum(p_eng, 0.0)), 0.0)
            p_h = np.where(on, mh.elec_w(eng, p_eng, fuel_base), 0.0)
            p_elec_m = e.mech_to_elec(U)
            p_h = np.broadcast_to(p_h, (n_soc, u.size))
            p_h = np.where(S >= e.soc_max, np.minimum(p_h, np.maximum(p_elec_m + e.aux_w, 0.0)), p_h)
            p_h = np.minimum(p_h, np.maximum(p_elec_m + e.aux_w + e.p_batt_max_w, 0.0))
            p_h = np.where(on, p_h, 0.0)
            fuel_w = fuel_base + mh.backpressure_penalty * p_h
            chem = e.chem_power_w(p_elec_m + e.aux_w - p_h)
            soc_next = S - chem * dt / (3600.0 * e.cap_wh)
            feasible = ((U >= lo_s[:, None] - tol) & (U <= hi_s[:, None] + tol)
                        & (soc_next >= e.soc_min - tol) & (soc_next <= e.soc_max + tol))
            stage = fuel_w * dt + shortfall_price * shortfall * dt
            v_next = np.where(on, np.interp(soc_next, grid, Vn[1]), np.interp(soc_next, grid, Vn[0]))
            for prev in (0, 1):
                c = stage + v_next + (np.where(on, start_pen, 0.0) if prev == 0 else 0.0)
                c = np.where(feasible, c, _BIG)
                j = np.argmin(c, axis=1)
                Vnew[prev] = c[ar, j]
                policy[t, prev] = u[j]
        else:                                                           # braking: fixed max-regen policy, engine off
            if vm[t] >= e.regen_min_speed:
                want = min(e.motor_max_w, e.regen_frac_max * (-p_wheel[t]) * veh.trans_eff)
                p_m = np.maximum(-want, lo_s)
            else:
                p_m = np.zeros(n_soc)
            chem = e.chem_power_w(e.mech_to_elec(p_m) + e.aux_w)
            soc_next = grid - chem * dt / (3600.0 * e.cap_wh)
            ok = (soc_next >= e.soc_min - tol) & (soc_next <= e.soc_max + tol)
            val = np.where(ok, np.interp(soc_next, grid, Vn[0]), _BIG)
            Vnew[0] = Vnew[1] = val
            policy[t, :, :] = p_m[None, :]
        V = Vnew
    return DPSolution(cycle, p, grid, policy, V)


def tune_rule_soc_target(cycle: Cycle, p: Params, lo: float = 0.30, hi: float = 0.80, iters: int = 14,
                         tol: float = 0.005) -> float:
    """Bisect the rule-based controller's SOC target so the cycle ends near the starting SOC (charge-sustaining)."""
    best_t, best_err = None, 9.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = simulate(cycle, RuleBased(soc_target=mid), p)
        err = r.summary["soc_end"] - p.elec.soc_init
        if abs(err) < best_err:
            best_t, best_err = mid, abs(err)
        if abs(err) < tol:
            break
        if err > 0:        # ended too full -> aim lower (assist more, charge less)
            hi = mid
        else:
            lo = mid
    return best_t


def shadow_price(sol: DPSolution) -> float:
    """DP's implied value of stored energy [J fuel per J battery], from the cost-to-go slope near the start SOC."""
    p = sol.p
    g, v = sol.soc_grid, sol.value0[0]
    i = int(np.searchsorted(g, p.elec.soc_init))
    lo, hi = max(i - 10, 0), min(i + 10, len(g))
    return -float(np.polyfit(g[lo:hi], v[lo:hi], 1)[0]) / (p.elec.cap_wh * 3600.0)


def priced_l100(summary: dict, p: Params, shadow: float) -> float:
    """L/100 km with leftover battery energy priced at the DP marginal value (fair when final SOC differs).
    First-order: the price is a local slope, so treat runs ending far (> ~0.1) from the start SOC as approximate."""
    d_batt_j = (p.elec.soc_init - summary["soc_end"]) * p.elec.cap_wh * 3600.0
    kg = summary["fuel_kg"] + d_batt_j * shadow / p.engine.lhv_j_per_kg
    return kg / p.engine.density_kg_per_l / summary["distance_km"] * 100.0
