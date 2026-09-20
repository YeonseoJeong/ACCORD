import numpy as np
from collections import deque

class AllOnESController:
    def __init__(self, num_cells: int, epoch_slots: int):
        self.B = int(num_cells)
        self.K = int(epoch_slots) # 7개의 BS, 60 슬롯마다 action

    def reset(self):
        pass

    def observe(self, obs: dict) -> None:
        pass

    def act(self, obs: dict, t: int) -> np.ndarray | None:
        if t % self.K != 0:
            return None
        return np.ones(self.B, dtype=np.int64)  # Keep all cells ON every K slots: [1, 1, 1, ..., 1] 

class RuleBasedESController:
    '''
        A rule-based controller for the Energy Saving (ES) component.
        OFF: 최근 K slots 동안 load가 off_threshold보다 낮으면 OFF
        ON: 인접 BS 부하가 on_threshold보다 높으면 ON
        Dwell time: 최소 dwell_slots 동안은 상태를 유지
    '''
    def __init__(self, cell_xy: np.ndarray, 
                 isd: float, 
                 epoch_slots: int, 
                 dwell_slots: int, 
                 off_threshold: float = 0.2, 
                 on_threshold: float = 0.8, 
                ):
        xy = np.asarray(cell_xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("cell_xy must be a 2D array with shape (num_cells, 2)")
        
        self.B = cell_xy.shape[0]
        self.K = int(epoch_slots)
        self.Tdwell = int(dwell_slots)
        self.off_threshold = float(off_threshold)
        self.on_threshold = float(on_threshold)

        if self.K < 1 or self.Tdwell < 1:
            raise ValueError("epoch_slots and dwell_slots must be positive integers")
        if not 0 <= self.off_threshold < self.on_threshold:
            raise ValueError("Expected 0 <= off_threshold < on_threshold")
        
        distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1) # (B, B) pairwise distance matrix
        self.neighbors = (distance > 1e-9) & np.isclose(distance, float(isd), rtol=1e-5, atol=1e-6) # boolean matrix indicating neighboring BSs

        self.reset()

    def reset(self) -> None:
        self._load_history = deque(maxlen=self.K)
        self._active_history = deque(maxlen=self.K)

    def observe(self, obs: dict) -> None: # 과거 네트워크 상태 기억
        load = np.asarray(obs["load"], dtype=np.float64)
        active = np.asarray(obs["activation"], dtype=np.int64)

        if load.shape != (self.B,) or active.shape != (self.B,):
            raise ValueError("ES observation arrays must have shape (B,)")

        self._load_history.append(load.copy())
        self._active_history.append(active.copy())

    def act(self, obs: dict, t: int) -> np.ndarray | None:
        if t % self.K != 0:
            return None

        active = np.asarray(obs["activation"], dtype=np.int64)
        load = np.asarray(obs["load"], dtype=np.float64)
        dwell = np.asarray(obs["dwell_timer"], dtype=np.float64)
        if any(a.shape != (self.B,) for a in (active, load, dwell)):
            raise ValueError("ES observation arrays must have shape (B,)")
        if not np.all(np.isin(active, (0, 1))):
            raise ValueError("Activation values must be 0 or 1")

        requested = active.copy()
        eligible = dwell >= self.Tdwell - 1

        for b in range(self.B): # BS ON 조건
            if active[b] == 0 and eligible[b] and np.any(load[self.neighbors[b]] > self.on_threshold):
                requested[b] = 1

        if len(self._load_history) == self.K: # BS OFF 조건
            low_throughout = np.all(
                np.stack(self._load_history, axis=0) < self.off_threshold,
                axis=0,
            )
            active_throughout = np.all(
                np.stack(self._active_history, axis=0) == 1,
                axis=0,
            )
            to_switch_off = (active == 1) & eligible & low_throughout & active_throughout
            requested[to_switch_off] = 0

        if not np.any(requested) and np.any(active): # 모든 BS OFF 방지
            keep = int(np.argmax(np.where(active == 1, load, -np.inf)))
            requested[keep] = 1

        return requested