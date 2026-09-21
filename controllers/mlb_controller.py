import numpy as np

class FixedCIOController:
    def __init__(self, num_cells: int):
        self.B = int(num_cells)

    def reset(self) -> None:
        pass

    def act(self, obs:dict, t: int) -> np.ndarray:
        return np.zeros(self.B, dtype=np.float64)  # Keep CIO=0 dB for all cells

class RuleBasedMLBController:
    '''
        A rule-based controller for the Mobility Load Balancing (MLB) component.
        If the load difference between a BS and its neighbor exceeds load_threshold,
        the controller adjusts the CIO of the BS to balance the load.
    '''
    def __init__(self, 
                 cell_xy: np.ndarray, 
                 isd: float, 
                 cio_min: float, 
                 cio_max: float, 
                 cio_step: float, 
                 load_threshold: float = 0.2, 
                 ):
        xy = np.asarray(cell_xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("cell_xy must be a 2D array with shape (num_cells, 2)")
        
        self.B = cell_xy.shape[0]
        self.cio_min = float(cio_min)
        self.cio_max = float(cio_max)
        self.cio_step = float(cio_step)
        self.load_threshold = float(load_threshold)
        if self.cio_step <= 0 or self.cio_min > self.cio_max:
            raise ValueError("Expected cio_step > 0 and cio_min <= cio_max")

        distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1) # (B, B) pairwise distance matrix
        matrix = (distance > 1e-9) & np.isclose(distance, float(isd), rtol=1e-5, atol=1e-6) # neighboring BSs Db,j ≈ ISD

        self.neighbors = [np.flatnonzero(matrix[b]) for b in range(self.B)]
        self.max_idx = int(np.rint((self.cio_max - self.cio_min) / self.cio_step))
        if not np.isclose(self.cio_min + self.max_idx * self.cio_step, self.cio_max):
            raise ValueError("cio_max must be reachable from cio_min by integer multiples of cio_step")

    def reset(self) -> None:
        pass

    def act(self, obs: dict, t: int) -> np.ndarray:
        """
        Compute the next CIO values based on the current load and CIO values.
        """
        load = np.asarray(obs["load"], dtype=np.float64)    # [0.65, 0.25, 0.80, 0.10, 0.35, 0.45, 0.20]
        cio = np.asarray(obs["cio_db"], dtype=np.float64)   # [0, 0, 0, 0, 0, 0, 0]

        if load.shape != (self.B,) or cio.shape != (self.B,):
            raise ValueError("MLB observation arrays must have shape (B,)")

        index = np.rint((cio - self.cio_min) / self.cio_step).astype(np.int64) # [-6, 6] -> [0, 12]
        if np.any(index < 0) or np.any(index > self.max_idx) or not np.allclose(self.cio_min + index * self.cio_step, cio, atol=1e-8):
            raise ValueError("CIO values must be within [cio_min, cio_max] and aligned with cio_step")

        next_index = index.copy()

        for b in range(self.B):
            nbrs = self.neighbors[b]
            if len(nbrs) == 0:
                continue  # No neighbors to consider

            diffs = load[b] - load[nbrs]
            j = int(np.argmax(np.abs(diffs)))  # Index of the neighbor with the largest load difference
            delta = float(diffs[j])

            if delta > self.load_threshold:
                next_index[b] = max(0, index[b] - 1)
            elif delta < -self.load_threshold:
                next_index[b] = min(self.max_idx, index[b] + 1)

        return self.cio_min + next_index.astype(np.float64) * self.cio_step # [0, 12] -> [-6, 6]