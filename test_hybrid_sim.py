import unittest

import numpy as np

from hybrid_sim import (ECMS, G, IceOnly, Params, RuleBased, build_cycle, max_speed_for_range,
                        max_speed_kmh, simulate, solve_dp, standard_cycles, steady_speed_sweep,
                        priced_l100, shadow_price, top_speed_report, tune_ecms_s0, tune_rule_soc_target)

P = Params()
CYCLES = standard_cycles()


class TestVehicleModel(unittest.TestCase):
    def test_wheel_energy_matches_kinetic_plus_road_load(self):
        """Sum of wheel energy must equal kinetic-energy change + drag + rolling work (exact for midpoint scheme)."""
        cyc = CYCLES["mixed"]
        r = simulate(cyc, IceOnly(), P)
        veh, m = P.vehicle, P.vehicle.mass(False)
        v = cyc.v
        e_wheel = np.sum(r.arrays["p_wheel"]) * cyc.dt
        vm = 0.5 * (v[:-1] + v[1:])
        d_ke = 0.5 * veh.rot_mass_factor * m * (v[-1] ** 2 - v[0] ** 2)
        road = np.sum((0.5 * veh.air_density * veh.cd * veh.frontal_area_m2 * vm**2
                       + (veh.crr0 + veh.crr_v2 * vm**2) * m * G) * vm) * cyc.dt
        self.assertAlmostEqual(e_wheel, d_ke + road, delta=1e-6 * abs(e_wheel))


class TestPowerBalance(unittest.TestCase):
    def _check(self, strat, cyc):
        r = simulate(cyc, strat, P)
        A = r.arrays
        prop = A["p_wheel"] >= 0
        # shaft balance: engine + MGU-K = demand (unless engine saturated -> shortfall)
        resid = A["p_eng"][prop] + A["p_m"][prop] + A["shortfall"][prop] - A["p_dem"][prop]
        self.assertLess(np.max(np.abs(resid)), 1e-6)
        # braking balance: regen + friction = braking power at wheels
        brk = ~prop
        resid_b = A["regen_wheel"][brk] + A["fric_wheel"][brk] + A["p_wheel"][brk]
        self.assertLess(np.max(np.abs(resid_b)), 1e-6)
        return r

    def test_rule_based_balances(self):
        for c in CYCLES.values():
            self._check(RuleBased(), c)

    def test_ecms_balances(self):
        for c in CYCLES.values():
            self._check(ECMS(), c)


class TestConstraints(unittest.TestCase):
    def test_limits_respected(self):
        e, eng = P.elec, P.engine
        for strat in (RuleBased(), ECMS()):
            for c in CYCLES.values():
                A = simulate(c, strat, P).arrays
                self.assertLessEqual(np.max(A["p_eng"]), eng.p_max_w + 1e-6)
                self.assertGreaterEqual(np.min(A["p_eng"]), -1e-6)
                self.assertLessEqual(np.max(np.abs(A["p_m"])), e.motor_max_w + 1e-6)
                self.assertLessEqual(np.max(np.abs(A["p_term"])), e.p_batt_max_w + 1e-6)
                # SOC may drift slightly past the window because of the constant accessory load
                self.assertGreater(np.min(A["soc"]), e.soc_min - 0.03)
                self.assertLess(np.max(A["soc"]), e.soc_max + 0.01)

    def test_no_regen_when_battery_full(self):
        """Battery stays exactly full (no assist, no accessory load, no MGU-H) -> regen must be refused."""
        from dataclasses import replace
        from hybrid_sim import Strategy

        class NoAssist(Strategy):
            name, hybrid = "no assist", True

            def request(self, p, soc, p_dem, v, dt, engine_on):
                return 0.0

        e_full = replace(P.elec, soc_init=P.elec.soc_max, aux_w=0.0)
        p_full = replace(P, elec=e_full, mguh=replace(P.mguh, enabled=False))
        cyc = build_cycle("brake", [("to", 100, 1.0), ("to", 0, 2.0)])
        r = simulate(cyc, NoAssist(), p_full)
        self.assertLess(r.summary["regen_wheel_kwh"], 1e-9)
        self.assertGreater(r.summary["braking_wheel_kwh"], 0.1)       # there was real braking to recover
        # control: with room in the battery, the same cycle does regen
        p_room = replace(p_full, elec=replace(e_full, soc_init=0.55))
        self.assertGreater(simulate(cyc, NoAssist(), p_room).summary["regen_wheel_kwh"], 0.05)


class TestEngineModel(unittest.TestCase):
    def test_efficiency_in_plausible_range(self):
        x = np.linspace(5e3, P.engine.p_max_w, 200)
        eta = P.engine.efficiency(x)
        self.assertGreater(eta.max(), 0.33)
        self.assertLess(eta.max(), 0.42)              # no road petrol engine beats ~40 %
        self.assertTrue(np.all(eta < 0.45))

    def test_mean_engine_efficiency_sane(self):
        for strat in (IceOnly(), RuleBased(), ECMS()):
            s = simulate(CYCLES["mixed"], strat, P).summary
            self.assertGreater(s["mean_engine_eff_when_on"], 0.10)
            self.assertLess(s["mean_engine_eff_when_on"], 0.40)


class TestTopSpeed(unittest.TestCase):
    def test_hybrid_faster_than_engine_only(self):
        t = top_speed_report(P)
        self.assertGreater(t["hybrid_burst_top_speed_kmh"], t["engine_only_top_speed_kmh"] + 5)
        self.assertGreater(t["hybrid_burst_top_speed_kmh"], 255)
        self.assertLess(t["hybrid_burst_top_speed_kmh"], 300)

    def test_max_speed_consistent_with_road_load(self):
        v = max_speed_kmh(P, 200e3) / 3.6
        self.assertAlmostEqual(P.vehicle.road_load_w(v), 200e3 * P.vehicle.trans_eff, delta=50.0)


class TestHybridBenefit(unittest.TestCase):
    def test_hybrid_beats_baseline_in_urban(self):
        base = simulate(CYCLES["urban"], IceOnly(), P).summary["l_per_100km_soc_corrected"]
        for strat in (RuleBased(), ECMS()):
            hyb = simulate(CYCLES["urban"], strat, P).summary["l_per_100km_soc_corrected"]
            self.assertLess(hyb, base)

    def test_soc_correction_is_charge_neutral_consistent(self):
        s = simulate(CYCLES["mixed"], ECMS(), P).summary
        self.assertLess(abs(s["l_per_100km_soc_corrected"] - s["l_per_100km_raw"]) / s["l_per_100km_raw"], 0.25)


class TestRange(unittest.TestCase):
    def test_steady_sweep_matches_cycle_model_at_120(self):
        row = steady_speed_sweep(P, [120])[0]
        base = simulate(CYCLES["highway"], IceOnly(), P).summary["l_per_100km_raw"]
        self.assertLess(abs(row["l_per_100km"] - base) / base, 0.12)     # same physics, different driving

    def test_fuel_use_rises_with_speed_and_range_falls(self):
        rows = [r for r in steady_speed_sweep(P, range(100, 261, 10)) if r["engine_alone_feasible"]]
        l100 = [r["l_per_100km"] for r in rows]
        self.assertTrue(all(b > a for a, b in zip(l100, l100[1:])))

    def test_range_speed_limit_consistent(self):
        v = max_speed_for_range(P, 500.0)
        self.assertAlmostEqual(steady_speed_sweep(P, [v])[0]["range_km"], 500.0, delta=1.0)
        self.assertGreater(v, 120)
        self.assertLess(v, 260)


class TestRegression(unittest.TestCase):
    """Pins headline numbers so later changes cannot silently alter existing results."""

    def test_pinned_values(self):
        c = CYCLES["urban"]
        self.assertAlmostEqual(simulate(c, IceOnly(), P).summary["l_per_100km_raw"], 9.04, delta=0.01)
        self.assertAlmostEqual(simulate(c, RuleBased(), P).summary["l_per_100km_soc_corrected"], 4.23, delta=0.01)   # was 4.19 with soc_eq_eff=0.30 (v1)
        self.assertAlmostEqual(simulate(CYCLES["mixed"], ECMS(s0=2.92), P).summary["l_per_100km_soc_corrected"], 4.98, delta=0.02)
        self.assertAlmostEqual(top_speed_report(P)["hybrid_burst_top_speed_kmh"], 278.3, delta=0.1)


class TestDP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sol = {n: solve_dp(CYCLES[n], P, n_soc=101, n_u=41) for n in ("urban", "suburban", "mixed")}

    def test_forward_sim_consistent_with_dp_value(self):
        """DP's own optimal-cost estimate should match replaying its policy in the simulator."""
        for n, sol in self.sol.items():
            r = simulate(CYCLES[n], sol.strategy(), P).summary
            self.assertLess(abs(sol.fuel_estimate_l() - r["fuel_l"]) / r["fuel_l"], 0.06, n)

    def test_dp_is_not_worse_than_ecms_or_rule_based(self):
        for n, sol in self.sol.items():
            dp = simulate(CYCLES[n], sol.strategy(), P).summary["l_per_100km_soc_corrected"]
            ecms = simulate(CYCLES[n], ECMS(s0=tune_ecms_s0(CYCLES[n], P)), P).summary["l_per_100km_soc_corrected"]
            rule = simulate(CYCLES[n], RuleBased(), P).summary["l_per_100km_soc_corrected"]
            self.assertLessEqual(dp, ecms * 1.01, f"{n}: DP {dp:.3f} vs ECMS {ecms:.3f}")
            self.assertLessEqual(dp, rule * 1.01, f"{n}: DP {dp:.3f} vs rule {rule:.3f}")

    def test_dp_policy_satisfies_balances_and_limits(self):
        e, eng = P.elec, P.engine
        for n, sol in self.sol.items():
            A = simulate(CYCLES[n], sol.strategy(), P).arrays
            prop = A["p_wheel"] >= 0
            self.assertLess(np.max(np.abs(A["p_eng"][prop] + A["p_m"][prop] + A["shortfall"][prop] - A["p_dem"][prop])), 1e-6)
            self.assertLess(np.max(np.abs(A["regen_wheel"][~prop] + A["fric_wheel"][~prop] + A["p_wheel"][~prop])), 1e-6)
            self.assertLessEqual(np.max(A["p_eng"]), eng.p_max_w + 1e-6)
            self.assertLessEqual(np.max(np.abs(A["p_m"])), e.motor_max_w + 1e-6)
            self.assertLessEqual(np.max(np.abs(A["p_term"])), e.p_batt_max_w + 1e-6)
            self.assertGreater(np.min(A["soc"]), e.soc_min - 0.02)
            self.assertLess(np.max(A["soc"]), e.soc_max + 0.01)
            self.assertEqual(float(np.sum(A["shortfall"])), 0.0)

    def test_dp_is_charge_sustaining(self):
        for n, sol in self.sol.items():
            end = simulate(CYCLES[n], sol.strategy(), P).summary["soc_end"]
            self.assertGreater(end, P.elec.soc_init - 0.03, n)
            self.assertLess(end, P.elec.soc_init + 0.08, n)

    def test_value_strictly_decreases_with_more_initial_charge(self):
        """More initial charge can never cost more fuel: cost-to-go must be non-increasing in SOC (no tolerance)."""
        for n, sol in self.sol.items():
            v = sol.value0[0]
            self.assertTrue(np.all(v < 1e12), n)
            self.assertLessEqual(float(np.max(np.diff(v))), 0.0, n)

    def test_dp_shadow_price_agrees_with_tuned_ecms(self):
        """DP's implied value of stored energy (J fuel per J battery) should match the independently tuned ECMS s0."""
        for n, sol in self.sol.items():
            shadow = shadow_price(sol)
            s0 = tune_ecms_s0(CYCLES[n], P)
            self.assertGreater(shadow, 2.0, n)
            self.assertLess(shadow, 3.5, n)
            self.assertLess(abs(shadow - s0) / s0, 0.15, f"{n}: DP {shadow:.2f} vs ECMS {s0:.2f}")

    def test_nothing_beats_dp_when_priced_fairly(self):
        """With leftover battery priced at DP's marginal value, no controller may beat the optimum (0.5 % grid noise)."""
        for n, sol in self.sol.items():
            sh = shadow_price(sol)
            dp = priced_l100(simulate(CYCLES[n], sol.strategy(), P).summary, P, sh)
            others = [RuleBased(), RuleBased(soc_target=tune_rule_soc_target(CYCLES[n], P)),
                      ECMS(s0=tune_ecms_s0(CYCLES[n], P)), ECMS(s0=2.8), ECMS(s0=2.4), ECMS(s0=3.2)]
            for strat in others:
                x = priced_l100(simulate(CYCLES[n], strat, P).summary, P, sh)
                self.assertGreaterEqual(x, dp * 0.995, f"{n}: {strat.name} {x:.3f} beat DP {dp:.3f}")

    def test_rule_based_suburban_gap_is_a_parameter_artifact(self):
        """The default rule-based controller's big suburban gap comes from its 50 km/h EV ceiling, not from rule-based
        control as such: widening the EV window must recover most of it (documented finding in the README)."""
        sol = self.sol["suburban"]
        sh = shadow_price(sol)
        dp = priced_l100(simulate(CYCLES["suburban"], sol.strategy(), P).summary, P, sh)
        default = priced_l100(simulate(CYCLES["suburban"], RuleBased(), P).summary, P, sh)
        wide = priced_l100(simulate(CYCLES["suburban"], RuleBased(ev_speed_max=70 / 3.6, ev_power_max_w=25e3), P).summary, P, sh)
        self.assertGreater((default - dp) / dp, 0.15)
        self.assertLess((wide - dp) / dp, 0.06)

    def test_grid_convergence(self):
        c = CYCLES["urban"]
        vals = [simulate(c, solve_dp(c, P, n_soc=ns, n_u=nu).strategy(), P).summary["l_per_100km_soc_corrected"]
                for ns, nu in ((61, 31), (151, 61))]
        self.assertLess(abs(vals[0] - vals[1]) / vals[1], 0.02)


class TestRuleTuning(unittest.TestCase):
    def test_tuned_rule_based_is_charge_sustaining(self):
        for n in ("urban", "mixed", "aggressive"):
            tgt = tune_rule_soc_target(CYCLES[n], P)
            end = simulate(CYCLES[n], RuleBased(soc_target=tgt), P).summary["soc_end"]
            self.assertLess(abs(end - P.elec.soc_init), 0.02, f"{n}: target {tgt:.3f} ended {end:.3f}")

    def test_soc_exchange_rate_matches_dp_marginal_value(self):
        """The default SOC-correction exchange rate must agree with what DP says stored energy is worth."""
        shadow = shadow_price(solve_dp(CYCLES["mixed"], P, n_soc=101, n_u=41))
        self.assertLess(abs(1.0 / P.soc_eq_eff - shadow) / shadow, 0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)
