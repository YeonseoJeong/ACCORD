from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import math

import numpy as np
import yaml


def dbm_to_w(dbm: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** ((np.asarray(dbm) - 30.0) / 10.0)


def w_to_dbm(w: float | np.ndarray) -> float | np.ndarray:
    w = np.maximum(np.asarray(w), 1e-30)
    return 10.0 * np.log10(w) + 30.0


@dataclass
class StepMetrics:  # 환경 performance metrics
    normalized_energy: float    # Et_bar
    energy_w: float             # Et
    service_degradation: float  # Vt
    outage_fraction: float      # Ot
    handover_rate: float        # total Ht
    mlb_handover_rate: float    # Ht excluding forced HO from ES switch-off
    forced_es_handover_rate: float
    switching_count: int
    jain_load_fairness: float
    active_load_fairness: float
    load_balance_loss: float


class ORANLoadBalancingEnv:
    """
    Open RAN load-balancing environment.

    Agent controls
    --------------
    ES agent:
        Binary cell activation vector s in {0,1}^B, accepted only at t % K == 0. 
        Dwell-ineligible cells retain their current state.

    MLB agent:
        Per-cell CIO vector theta in {-6,...,+6} dB, updated every slot.

    step(action)
    ------------
    action = {
        "es":  np.ndarray(shape=(B,), dtype=int) or None,
        "mlb": np.ndarray(shape=(B,), values in CIO grid)
    }

    The environment returns a global state matching Eq. (20):
        loads, previous activation, previous CIO, per-cell demand history, dwell timers.

    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.seed = int(self.cfg.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        n = self.cfg["network"]
        ts = self.cfg["traffic_service"]
        c = self.cfg["control"]
        p = self.cfg["power"]
        impl = self.cfg["implementation"]

        self.B = int(n["num_cells"])
        if self.B != 7:
            raise ValueError("Current geometry helper implements the paper's 7-cell cluster.")

        self.isd = float(n["inter_site_distance_m"])
        self.fc = float(n["carrier_frequency_hz"])
        self.bandwidth = float(n["bandwidth_hz"])
        self.N_prb = int(n["num_prbs"])
        self.W_prb = float(n["prb_bandwidth_hz"])
        self.tx_power_total_w = float(dbm_to_w(float(n["cell_tx_power_dbm"])))
        self.tx_power_prb_w = self.tx_power_total_w / self.N_prb
        self.pathloss_exp = float(n["pathloss_exponent"])
        self.shadow_std_db = float(n["shadowing_std_db"])
        self.noise_psd_dbm_hz = float(n["noise_psd_dbm_per_hz"])
        self.noise_figure_db = float(n["noise_figure_db"])
        self.rsrp_min_dbm = float(n["rsrp_min_dbm"])
        self.U = int(n["num_ues"])
        self.mobility_std = float(n["mobility_std_m_per_slot"])

        self.nominal_demand = float(ts["nominal_demand_bps"])
        self.xi_min = float(ts["profile_floor"])

        self.slot_duration_s = float(n["slot_duration_s"])
        self.episode_length_slots = int(self.cfg["accord_training"]["rollout_slots"])
        self.T = self.episode_length_slots  # backward-compatible alias used by runners
        self.diurnal_cycle_slots = int(ts["diurnal_cycle_slots"])
        self.start_time_resolution_minutes = int(ts.get("start_time_resolution_minutes", 1)) # 분 단위로 학습 시작 가능

        if self.diurnal_cycle_slots <= 0:
            raise ValueError("diurnal_cycle_slots must be positive.")

        activity_cfg = ts["active_ue_schedule"]
        self.busy_start_hour = float(activity_cfg["busy_start_hour"])
        self.busy_end_hour = float(activity_cfg["busy_end_hour"])
        self.deep_night_start_hour = float(activity_cfg["deep_night_start_hour"])
        self.deep_night_end_hour = float(activity_cfg["deep_night_end_hour"])
        self.busy_active_ratio = float(activity_cfg["busy_active_ratio"])
        self.transition_active_ratio = float(activity_cfg["transition_active_ratio"])
        self.deep_night_active_ratio = float(activity_cfg["deep_night_active_ratio"])

        peak_cfg = ts["peak_time"]
        self.peak_time_distribution = str(peak_cfg["distribution"]).lower()
        self.peak_time_low_hour = float(peak_cfg["low"])
        self.peak_time_high_hour = float(peak_cfg["high"])
        self.peak_time_low = self.peak_time_low_hour * 3600.0 / self.slot_duration_s
        self.peak_time_high = self.peak_time_high_hour * 3600.0 / self.slot_duration_s   # t_peak: [25200, 82800)

        if not (0.0 <= self.peak_time_low_hour < self.peak_time_high_hour <= 24.0):
            raise ValueError("peak_time low_hour/high_hour must satisfy 0 <= low < high <= 24.")

        self.beta = float(ts["service_fraction_beta"])
        self.Vmax = float(ts["service_level_vmax"])
        self.wh = float(ts["handover_weight"])

        self.cio_min = float(c["cio_min_db"])
        self.cio_max = float(c["cio_max_db"])
        self.cio_step = float(c["cio_step_db"])
        self.cio_values = np.arange(
            self.cio_min, self.cio_max + 0.5 * self.cio_step, self.cio_step
        )
        self.K = int(c["es_epoch_slots"])
        self.Tdwell = int(c["dwell_time_slots"])
        self.W = int(c["demand_window_slots"])

        self.P_fix = float(p["active_fixed_power_w"])
        self.delta_power = float(p["load_slope"])
        self.P_sleep = float(p["sleep_power_w"])
        self.zeta = float(p["transition_cost_j"])

        self.d0 = float(impl["reference_distance_m"])
        self.initial_all_cells_on = bool(impl["initial_all_cells_on"])
        self.initial_cio_db = float(impl["initial_cio_db"])
        self.mobility_boundary = str(impl["mobility_boundary"])
        self.shadowing_mode = str(impl.get("shadowing_mode", "episode_static"))

        self.cell_xy = self._make_7cell_centers(self.isd)
        self.hex_radius = self.isd / math.sqrt(3.0)

        # Thermal noise on one PRB.
        noise_dbm = (
            self.noise_psd_dbm_hz
            + 10.0 * math.log10(self.W_prb)
            + self.noise_figure_db
        )
        self.noise_w = float(dbm_to_w(noise_dbm))

        # FSPL(d0) implementation assumption.
        c0 = 299_792_458.0
        wavelength = c0 / self.fc
        self.pl0_db = 20.0 * math.log10(4.0 * math.pi * self.d0 / wavelength) # reference path loss

        # Max reference in Eq. (12), all active and fully loaded.
        self.energy_norm_den = self.B * (
            self.P_fix + self.delta_power * self.tx_power_total_w
        )

        self.reset()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _make_7cell_centers(isd: float) -> np.ndarray:
        """Central site plus six neighbors at 60-degree spacing."""
        centers = [(0.0, 0.0)]
        for k in range(6):
            a = math.radians(60.0 * k)
            centers.append((isd * math.cos(a), isd * math.sin(a)))
        return np.asarray(centers, dtype=np.float64)

    def _sample_point_in_hex(self, center: np.ndarray) -> np.ndarray: # 속한 hexagon에서 random 위치 생성
        """
        Uniform rejection sampling inside a regular pointy-top hexagon with
        circumradius self.hex_radius.
        """
        R = self.hex_radius
        while True:
            x = self.rng.uniform(-math.sqrt(3) * R / 2.0, math.sqrt(3) * R / 2.0)
            y = self.rng.uniform(-R, R)
            # pointy-top regular hexagon inequalities
            if abs(y) <= R - abs(x) / math.sqrt(3.0):
                return center + np.array([x, y], dtype=np.float64)

    def _inside_hex(self, point: np.ndarray, center: np.ndarray) -> bool: # 7개 hexagon 중 하나에 속하는지 확인
        x, y = point - center
        R = self.hex_radius
        return (
            abs(x) <= math.sqrt(3) * R / 2.0 + 1e-12
            and abs(y) <= R - abs(x) / math.sqrt(3.0) + 1e-12
        )

    def _inside_cluster(self, point: np.ndarray) -> bool: # 7개 cluster 내부에 속하는지 확인
        return any(self._inside_hex(point, c) for c in self.cell_xy)

    def _sample_ue_positions(self) -> np.ndarray: # ue가 속할 hexagon 선택
        home = self.rng.integers(0, self.B, size=self.U)
        return np.stack([self._sample_point_in_hex(self.cell_xy[b]) for b in home])

    # ------------------------------------------------------------------
    # Channel, traffic, association, resource model
    # ------------------------------------------------------------------
    def _large_scale_gain(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
            step 1: distance: (U,B) distance matrix, minimum distance d0
            step 2: pl_db: (U,B) path loss in dB + shadowing in dB
            step 3: gain_linear: (U,B) linear channel gain, based on per-PRB transmit power
            step 4: rsrp_dbm: (U,B) RSRP in dBm, used for association and outage detection
        """
        d = np.linalg.norm(
            self.ue_xy[:, None, :] - self.cell_xy[None, :, :], axis=-1
        )
        d = np.maximum(d, self.d0) # (U, B) distance matrix, minimum distance d0

        pl_db = self.pl0_db + 10.0 * self.pathloss_exp * np.log10(d / self.d0)
        if self.shadowing_mode == "episode_static": # episode 동안 shadowing 고정
            shadow_db = self.shadow_db
        elif self.shadowing_mode == "iid_per_slot": # slot마다 iid로 shadowing 생성
            shadow_db = self.rng.normal(0.0, self.shadow_std_db, size=(self.U, self.B))
        else:
            raise NotImplementedError(f"Unknown shadowing mode: {self.shadowing_mode}")

        # Received power in dBm = Tx(dBm per PRB) - PL + shadowing.
        tx_prb_dbm = float(w_to_dbm(self.tx_power_prb_w))
        rsrp_dbm = tx_prb_dbm - pl_db + shadow_db
        gain_linear = 10.0 ** ((-pl_db + shadow_db) / 10.0)
        return gain_linear, rsrp_dbm

    def _clock_slot(self, t: int) -> int:
        """Physical slot-of-day corresponding to episode-relative slot t."""
        return int((self.start_clock_slot + t) % self.diurnal_cycle_slots)

    def _hour_of_day(self, t: int) -> float:
        seconds = self._clock_slot(t) * self.slot_duration_s
        return (seconds / 3600.0) % 24.0

    def _hour_index(self, t: int) -> int:
        return int(math.floor(self._hour_of_day(t))) % 24

    def _target_active_ratio(self, t: int) -> float: # 현재 시간 기준 몇 %의 UE가 active인지 결정
        """Time-of-day UE activity schedule: 100%, 70%, or 30%."""
        hour = self._hour_of_day(t)
        if self.busy_start_hour <= hour < self.busy_end_hour:
            return self.busy_active_ratio
        if self.deep_night_start_hour <= hour < self.deep_night_end_hour:
            return self.deep_night_active_ratio
        return self.transition_active_ratio

    def _active_ue_mask(self, t: int) -> np.ndarray:
        # ue_acitivity_mask: reset()에서 랜덤하게 정해진 ue 순위
        ratio = self._target_active_ratio(t)
        num_active = int(round(ratio * self.U))
        return self.ue_activity_rank < num_active # 활성화 비율 안으로 active 

    def _cell_traffic_factor(self, t: int) -> np.ndarray:
        """
        Continuous 24-hour cell-specific cosine traffic profile.
        T=86400 기준 cosine profile로 변화하는데 매 슬롯마다 traffic profile 함수를 호출
        """
        clock_slot = self._clock_slot(t)
        phase = 2.0 * math.pi * (clock_slot - self.t_peak) / self.diurnal_cycle_slots
        return self.xi_min + (1.0 - self.xi_min) * (1.0 + np.cos(phase)) / 2.0

    def _traffic_demand(self, t: int, active_ue_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Compute the traffic demand for each UE at time t.
            du,t = au(t) * dbar * xi_homecell(t)
        """
        if active_ue_mask is None:
            active_ue_mask = self._active_ue_mask(t)
        xi_cell = self._cell_traffic_factor(t)
        demand = self.nominal_demand * xi_cell[self.home_cell]
        return np.where(active_ue_mask, demand, 0.0)

    def _associate(
        self, rsrp_dbm: np.ndarray, cio_db: np.ndarray,
        active_ue_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]: 
        '''
            Associate each UE with the best serving cell based on RSRP and CIO.
            serving[u] = 0, 1, ..., B-1: UE u is served by cell serving[u]
            serving[u] = -1: UE u is in outage 
            모든 ue에 대해 결정X, active_ue에 대해서만 ua 결정, inactive ue(serving=-1, outage=False)
        '''
        if active_ue_mask is None:
            active_ue_mask = np.ones(self.U, dtype=bool)
        active_ue_mask = np.asarray(active_ue_mask, dtype=bool)

        active_cells = np.flatnonzero(self.s == 1)
        active_users = np.flatnonzero(active_ue_mask) # active_ue만 ho 계산
        serving = np.full(self.U, -1, dtype=np.int64)
        outage = np.zeros(self.U, dtype=bool)

        # Inactive UEs are neither served nor counted as outage.
        if len(active_users) == 0:
            return serving, outage
        if len(active_cells) == 0:
            outage[active_users] = True
            return serving, outage

        active_rsrp = rsrp_dbm[np.ix_(active_users, active_cells)]
        strongest_active = np.max(active_rsrp, axis=1)
        user_outage = strongest_active < self.rsrp_min_dbm

        score = active_rsrp + cio_db[active_cells][None, :]
        best_local = np.argmax(score, axis=1)
        served_users = active_users[~user_outage]
        serving[served_users] = active_cells[best_local[~user_outage]]
        outage[active_users[user_outage]] = True
        return serving, outage

    def _sinr_and_rates(
        self,
        gain: np.ndarray,
        serving: np.ndarray,
        prev_load: np.ndarray,
    ) -> np.ndarray:
        """
            Compute SINR and rates for each UE.
            Eq. (8)-(9), using previous-slot load in interference activity factor.
            Returns per-PRB rates [bit/s] for all UEs; outage UEs get 0.
            ru^prb(t) = W_prb * log2(1 + SINR_u(t))
        """
        rates = np.zeros(self.U, dtype=np.float64)
        active = np.flatnonzero(self.s == 1)
        activity = np.minimum(prev_load, 1.0)

        for u in range(self.U):
            b = serving[u]
            if b < 0:
                continue # outage rate = 0
            signal = self.tx_power_prb_w * gain[u, b]
            interf = 0.0
            for bp in active:
                if bp == b:
                    continue
                interf += activity[bp] * self.tx_power_prb_w * gain[u, bp]
            sinr = signal / (interf + self.noise_w)
            rates[u] = self.W_prb * math.log2(1.0 + sinr)
        return rates

    def _load_and_delivery(
        self,
        serving: np.ndarray,
        per_prb_rate: np.ndarray,
        demand: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Compute the required PRBs and delivery for each UE.
            req_prb Nu_prb(t) = du(t) / ru^prb(t)
            Load rho_b(t) = sum_{u in cell b} Nu_prb(t) / N_prb
            Delivered rate R_u(t) = du(t) / max(rho_b(t), 1)
        """
        req_prbs = np.zeros(self.U, dtype=np.float64)
        served = serving >= 0
        req_prbs[served] = demand[served] / np.maximum(per_prb_rate[served], 1e-12)

        load = np.zeros(self.B, dtype=np.float64)
        for b in range(self.B):
            load[b] = np.sum(req_prbs[serving == b]) / self.N_prb

        delivered = np.zeros(self.U, dtype=np.float64)
        for u in np.flatnonzero(served):
            b = serving[u]
            delivered[u] = demand[u] / max(load[b], 1.0)

        return load, delivered, req_prbs

    # ------------------------------------------------------------------
    # Metrics and state
    # ------------------------------------------------------------------
    def _compute_handover_rates(
        self,
        serving: np.ndarray,
        outage: np.ndarray,
        prev_s: np.ndarray,
        active_ue_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, float, float]:
        """Return total, MLB-attributed, and forced-ES handover rates.

        A handover is excluded from the MLB penalty only when the UE's previous
        serving cell was switched OFF by the ES action in the current slot.
        Handover to a newly switched-ON cell is not forced and therefore remains
        part of the MLB handover cost.
        """
        if active_ue_mask is None:
            active_ue_mask = np.ones(self.U, dtype=bool)
        active_ue_mask = np.asarray(active_ue_mask, dtype=bool)
        active_users = np.flatnonzero(active_ue_mask)
        if len(active_users) == 0:
            self.last_serving[:] = -1
            return 0.0, 0.0, 0.0

        # If a UE becomes inactive, terminate the previous session. When it
        # becomes active again, the new association is not counted as HO.
        self.last_serving[~active_ue_mask] = -1

        total_handovers = 0
        forced_es_handovers = 0

        for u in active_users:
            if outage[u] or serving[u] < 0:
                continue

            old_bs = int(self.last_serving[u])
            is_handover = old_bs >= 0 and int(serving[u]) != old_bs

            if is_handover:
                total_handovers += 1
                forced_by_es = bool(prev_s[old_bs] == 1 and self.s[old_bs] == 0)
                if forced_by_es:
                    forced_es_handovers += 1

            self.last_serving[u] = serving[u]

        denom = float(len(active_users))
        total_rate = total_handovers / denom
        forced_es_rate = forced_es_handovers / denom
        mlb_rate = (total_handovers - forced_es_handovers) / denom
        return total_rate, mlb_rate, forced_es_rate

    def _compute_energy(self, prev_s: np.ndarray, load: np.ndarray) -> Tuple[float, float, int]:
        switching_count = int(np.sum(np.abs(self.s - prev_s)))
        active_term = self.s * (
            self.P_fix + self.delta_power * np.minimum(load, 1.0) * self.tx_power_total_w
        )
        sleep_term = (1 - self.s) * self.P_sleep

        # Eq. (12) is written as power/consumption per slot; zeta is J.
        # With T0=1 s in Table II, adding zeta per transition is numerically
        # equivalent in J/slot. For a different T0, divide by slot duration.
        T0 = float(self.cfg["network"]["slot_duration_s"])
        transition_power_equiv = self.zeta * switching_count / T0

        E = float(np.sum(active_term + sleep_term) + transition_power_equiv)
        E_bar = E / self.energy_norm_den
        return E, E_bar, switching_count

    @staticmethod
    def _jain(x: np.ndarray) -> float: # BS load balancing 계산 - 고르게 분배 JFI = 1, 하나에 몰리면 JFI 낮아짐
        x = np.asarray(x, dtype=np.float64)
        denom = len(x) * np.sum(x * x)
        if denom <= 1e-15: # all BS load = 0, JFI = 1.0
            return 1.0
        return float(np.sum(x) ** 2 / denom)

    def _update_demand_history(self, serving: np.ndarray, demand: np.ndarray) -> None:
        d_cell = np.zeros(self.B, dtype=np.float64)
        for b in range(self.B):
            d_cell[b] = float(np.sum(demand[serving == b]))
        self.demand_history[:, 1:] = self.demand_history[:, :-1]
        self.demand_history[:, 0] = d_cell

    def _get_state(self) -> Dict[str, np.ndarray]:
        return {
            "load": self.load.astype(np.float32).copy(),
            "activation": self.s.astype(np.float32).copy(),
            "cio_db": self.theta.astype(np.float32).copy(),
            "demand_history_bps": self.demand_history.astype(np.float32).copy(),
            "dwell_timer": self.dwell_timer.astype(np.float32).copy(),
        }

    def flat_state(self) -> np.ndarray:
        s = self._get_state()
        return np.concatenate(
            [
                s["load"].ravel(), # 7 
                s["activation"].ravel(), # 7 
                s["cio_db"].ravel(), # 7
                s["demand_history_bps"].ravel(), # 70
                s["dwell_timer"].ravel(), # 7
            ] # total 98 vector
        )

    @property
    def state_dim(self) -> int:
        # B loads + B activations + B CIOs + B*W demand history + B timers
        return self.B * (4 + self.W)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _apply_es_action(self, requested: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
            Apply ES action at an ES epoch using the dwell mask of Eq. (29).
            Returns (previous activation, eligibility mask).
        """
        prev_s = self.s.copy()
        eligible = self.dwell_timer >= self.Tdwell - 1

        if requested is None:
            return prev_s, eligible

        requested = np.asarray(requested, dtype=np.int64)
        if requested.shape != (self.B,):
            raise ValueError(f"ES action must have shape ({self.B},)")
        if not np.all(np.isin(requested, [0, 1])):
            raise ValueError("ES action must be binary.")

        self.s[eligible] = requested[eligible]

        if np.sum(self.s) == 0: # 모든 cell이 off되면 가장 load가 높은 cell을 켬
            b = int(np.argmax(self.load))
            self.s[b] = 1

        return prev_s, eligible

    def _apply_mlb_action(self, cio: np.ndarray) -> None:
        cio = np.asarray(cio, dtype=np.float64)
        if cio.shape != (self.B,):
            raise ValueError(f"MLB action must have shape ({self.B},)")

        idx = np.rint((cio - self.cio_min) / self.cio_step)
        quantized = self.cio_min + idx * self.cio_step
        if np.any(np.abs(cio - quantized) > 1e-8):
            raise ValueError("Every CIO value must lie on the configured quantization grid.")
        if np.any((cio < self.cio_min) | (cio > self.cio_max)):
            raise ValueError("CIO action outside configured range.")
        self.theta = cio.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        start_clock_slot: Optional[int] = None,
        episode_length_slots: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]: # episode 초기화
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        self.t = 0
        self.episode_length_slots = int(
            self.cfg["accord_training"]["rollout_slots"]
            if episode_length_slots is None
            else episode_length_slots
        )
        self.T = self.episode_length_slots

        if start_clock_slot is None:
            slots_per_minute = int(round(60.0 / self.slot_duration_s))      # 60 slots/min
            resolution_slots = self.start_time_resolution_minutes * slots_per_minute
            choices = self.diurnal_cycle_slots // resolution_slots
            self.start_clock_slot = int(self.rng.integers(0, choices) * resolution_slots)
        else:
            self.start_clock_slot = int(start_clock_slot) % self.diurnal_cycle_slots

        self.ue_xy = self._sample_ue_positions()
        self.initial_ue_xy = self.ue_xy.copy()
        self.home_cell = np.argmin(
            np.linalg.norm(
                self.initial_ue_xy[:, None, :] - self.cell_xy[None, :, :],
                axis=-1,
            ),
            axis=1,
        )

        if self.peak_time_distribution == "uniform":
            self.t_peak = self.rng.uniform(
                self.peak_time_low, self.peak_time_high, size=self.B
            ) # 각 BS 마다 peak time 다르게 해서 spatial imbalance 유지됨
        else:
            raise NotImplementedError(
                f"Unknown peak-time distribution: {self.peak_time_distribution}"
            )

        # 100명에게 랜덤 순위 생성 30명 ⊂ 70명 ⊂ 100명 
        order = self.rng.permutation(self.U)
        self.ue_activity_rank = np.empty(self.U, dtype=np.int64)
        self.ue_activity_rank[order] = np.arange(self.U, dtype=np.int64)

        self.s = (
            np.ones(self.B, dtype=np.int64)
            if self.initial_all_cells_on
            else np.zeros(self.B, dtype=np.int64)
        ) 
        self.theta = np.full(self.B, self.initial_cio_db, dtype=np.float64)

        # Initialize cells as immediately eligible. The paper gives the dwell
        # rule but does not prescribe the episode-start timer value.
        self.dwell_timer = np.full(self.B, self.Tdwell, dtype=np.int64)

        self.last_serving = np.full(self.U, -1, dtype=np.int64)
        self.shadow_db = self.rng.normal(0.0, self.shadow_std_db, size=(self.U, self.B))
        self.load = np.zeros(self.B, dtype=np.float64)
        self.prev_load = np.zeros(self.B, dtype=np.float64)
        self.demand_history = np.zeros((self.B, self.W), dtype=np.float64)

        # Warm-start state by evaluating the initial configuration once
        # without advancing mobility/time or counting handovers.
        gain, rsrp = self._large_scale_gain()
        active_ue_mask = self._active_ue_mask(0)
        demand = self._traffic_demand(0, active_ue_mask)
        serving, outage = self._associate(rsrp, self.theta, active_ue_mask)
        per_prb_rate = self._sinr_and_rates(gain, serving, self.prev_load)
        self.load, _, _ = self._load_and_delivery(serving, per_prb_rate, demand)
        self.prev_load = self.load.copy()
        self._update_demand_history(serving, demand)

        info = {
            "t_peak": self.t_peak.copy(),
            "cell_positions_m": self.cell_xy.copy(),
            "cio_values_db": self.cio_values.copy(),
            "start_clock_slot": self.start_clock_slot,
            "start_hour_of_day": self._hour_of_day(0),
            "episode_length_slots": self.episode_length_slots,
            "active_ue_ratio": float(np.mean(active_ue_mask)),
            "target_active_ue_ratio": self._target_active_ratio(0),
            "cell_traffic_factor": self._cell_traffic_factor(0).copy(),
        }
        return self._get_state(), info

    def _move_ues(self) -> None:
        proposed = self.ue_xy + self.rng.normal(
            0.0, self.mobility_std, size=self.ue_xy.shape
        )
        if self.mobility_boundary == "reject":
            for u in range(self.U):
                if self._inside_cluster(proposed[u]):
                    self.ue_xy[u] = proposed[u]
        else:
            raise NotImplementedError(
                f"Unknown mobility boundary mode: {self.mobility_boundary}"
            )

    def step(
        self, action: Dict[str, Optional[np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict[str, Any]]:
        '''
            ES action -> MLB action -> channel 계산
            -> active ue 결정 -> cosine traffic 계산
            -> active ue association -> sinr/per-prb rate
            -> load/delivery -> handover rate, service degradation, energy, reward
            -> dwell timer update -> ue mobility
        '''
        if self.t >= self.episode_length_slots:
            raise RuntimeError("Episode is done. Call reset().")

        prev_s = self.s.copy()
        eligible = np.zeros(self.B, dtype=bool)

        if self.t % self.K == 0:
            prev_s, eligible = self._apply_es_action(action.get("es"))
        elif action.get("es") is not None:
            raise ValueError("ES action may only be supplied when t % K == 0.")

        if action.get("mlb") is None:
            raise ValueError("MLB action is required every slot.")
        self._apply_mlb_action(action["mlb"])

        gain, rsrp = self._large_scale_gain()
        active_ue_mask = self._active_ue_mask(self.t)
        demand = self._traffic_demand(self.t, active_ue_mask)
        serving, outage = self._associate(rsrp, self.theta, active_ue_mask)
        per_prb_rate = self._sinr_and_rates(gain, serving, self.prev_load)
        load, delivered, req_prbs = self._load_and_delivery(
            serving, per_prb_rate, demand
        )

        handover_rate, mlb_handover_rate, forced_es_handover_rate = (
            self._compute_handover_rates(
                serving, outage, prev_s, active_ue_mask
            )
        )

        num_active = int(np.sum(active_ue_mask)) # service degradation -> active ue 기준
        if num_active > 0:
            below_min = (
                active_ue_mask
                & (~outage)
                & (serving >= 0)
                & (delivered < self.beta * demand)
            )
            active_outage = outage & active_ue_mask
            V = float((np.sum(active_outage) + np.sum(below_min)) / num_active)
            outage_fraction = float(np.sum(active_outage) / num_active)
        else:
            V = 0.0
            outage_fraction = 0.0

        E, E_bar, switching_count = self._compute_energy(prev_s, load)

        active_load = load[self.s == 1]

        if len(active_load) == 0:
            raise RuntimeError("No active BS: all-off state should have been prevented.")
        
        mean_active_load = float(np.mean(active_load))
        lb_loss = float(np.mean((active_load - mean_active_load) ** 2))

        active_load_fairness = self._jain(active_load)
        reward_mlb = -lb_loss - self.wh * mlb_handover_rate
        # Per-slot ES contribution; training can average this over K slots
        # exactly as Eq. (25) does.
        reward_es_slot = -E_bar
        constraint_cost = V

        switched = self.s != prev_s
        self.dwell_timer[switched] = 0
        self.dwell_timer[~switched] = np.minimum(
            self.dwell_timer[~switched] + 1, self.Tdwell
        )

        self.load = load
        self.prev_load = load.copy()
        self._update_demand_history(serving, demand)

        metrics = StepMetrics(
            normalized_energy=E_bar,
            energy_w=E,
            service_degradation=V,
            outage_fraction=outage_fraction,
            handover_rate=handover_rate,
            mlb_handover_rate=mlb_handover_rate,
            forced_es_handover_rate=forced_es_handover_rate,
            switching_count=switching_count,
            jain_load_fairness=self._jain(load),
            active_load_fairness=active_load_fairness,
            load_balance_loss=lb_loss,
        )

        info = {
            "metrics": metrics.__dict__.copy(),
            "serving_cell": serving.copy(),
            "outage": outage.copy(),
            "active_ue_mask": active_ue_mask.copy(),
            "num_active_ues": num_active,
            "active_ue_ratio": float(num_active / self.U),
            "target_active_ue_ratio": float(self._target_active_ratio(self.t)),
            "hour_of_day": float(self._hour_of_day(self.t)),
            "hour_index": int(self._hour_index(self.t)),
            "clock_slot": int(self._clock_slot(self.t)),
            "traffic_profile_continuous": True,
            "cell_traffic_factor": self._cell_traffic_factor(self.t).copy(),
            "delivered_rate_bps": delivered.copy(),
            "demand_bps": demand.copy(),
            "per_prb_rate_bps": per_prb_rate.copy(),
            "required_prbs": req_prbs.copy(),
            "rsrp_dbm": rsrp.copy(),
            "es_eligible_mask": eligible.copy(),
            "es_epoch_start": (self.t % self.K == 0),
            "constraint_violation": V - self.Vmax,
        }

        rewards = {
            "es_slot": float(reward_es_slot),
            "mlb": float(reward_mlb),
            "constraint_cost": float(constraint_cost),
        }

        self.t += 1
        terminated = self.t >= self.episode_length_slots
        truncated = False

        if not terminated:
            self._move_ues()

        return self._get_state(), rewards, terminated, truncated, info


def load_env(config_path: str | Path | None = None) -> ORANLoadBalancingEnv:
    if config_path is None:
        config_path = Path(__file__).with_name("config.yaml")
    return ORANLoadBalancingEnv(config_path)
