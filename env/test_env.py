import numpy as np

from env import load_env


def main():
    env = load_env()
    obs, info = env.reset(seed=0)

    print("state_dim =", env.state_dim)
    print("flat_state shape =", env.flat_state().shape)
    print("CIO values =", info["cio_values_db"])

    cumulative_v = 0.0
    cumulative_h = 0.0
    cumulative_e = 0.0

    for t in range(env.T):
        # Simple smoke-test controller:
        # keep all cells ON and CIO=0.
        es_action = np.ones(env.B, dtype=np.int64) if t % env.K == 0 else None # 60 slots 마다 on
        mlb_action = np.zeros(env.B, dtype=np.float64) # cio = 0 dB

        obs, rewards, terminated, truncated, info = env.step(
            {"es": es_action, "mlb": mlb_action}
        )

        m = info["metrics"]
        cumulative_v += m["service_degradation"]
        cumulative_h += m["handover_rate"]
        cumulative_e += m["normalized_energy"]

        if terminated or truncated:
            break

    n = t + 1
    print(f"slots             : {n}")
    print(f"mean V            : {cumulative_v / n:.6f}")
    print(f"mean HO           : {cumulative_h / n:.6f}")
    print(f"mean normalized E : {cumulative_e / n:.6f}")


if __name__ == "__main__":
    main()
