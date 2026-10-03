"""Train ACCORD with one environment first.

Run from repository root:
    python -m simulations.train_accord --seed 0 --iterations 원하는 만큼

"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from ca_mappo.buffer import ESRolloutBuffer, MLBRolloutBuffer
from ca_mappo.trainer import CAMAPPOTrainer
from ca_mappo.utils import flatten_normalized_observation, set_seed
from env.env import load_env


def make_rl_state(obs: dict, env) -> np.ndarray:
    return flatten_normalized_observation(
        obs,
        cio_abs_max=max(abs(env.cio_min), abs(env.cio_max)),
        dwell_time=env.Tdwell,
        nominal_demand_bps=env.nominal_demand,
        num_ues=env.U,
    )


def train(
    *,
    seed: int = 0,
    config_path: str | Path | None = None,
    iterations: int | None = None,
    checkpoint_dir: str | Path = "results/checkpoints",
) -> None:
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "env" / "config.yaml"

    env = load_env(config_path)
    cfg = env.cfg
    train_cfg = cfg["accord_training"]

    rollout_slots = int(train_cfg["rollout_slots"])
    if rollout_slots % env.K != 0:
        raise ValueError(
            f"rollout_slots={rollout_slots} must be divisible by K={env.K}."
        )

    num_iterations = int(
        train_cfg["training_iterations"] if iterations is None else iterations
    )

    set_seed(seed)
    trainer = CAMAPPOTrainer(
        state_dim=env.state_dim,
        num_cells=env.B,
        cio_values=env.cio_values,
        config=cfg,
        epoch_slots=env.K,
        vmax=env.Vmax,
    )

    print(f"Device: {trainer.device}")
    print(f"State dim: {env.state_dim}, B={env.B}, K={env.K}, Tdwell={env.Tdwell}")

    checkpoint_dir = Path(checkpoint_dir)

    for iteration in range(1, num_iterations + 1):
        # Different environment seed each rollout, reproducibly derived from base seed.
        obs, reset_info = env.reset(
            seed=seed * 100000 + iteration,
            episode_length_slots=rollout_slots,
        )
        start_hour = float(reset_info["start_hour_of_day"])

        mlb_buffer = MLBRolloutBuffer()
        es_buffer = ESRolloutBuffer()

        current_es_action: np.ndarray | None = None
        es_epoch_state: np.ndarray | None = None
        es_epoch_action: np.ndarray | None = None
        es_epoch_log_prob: float | None = None
        es_epoch_mask: np.ndarray | None = None
        es_epoch_value: float | None = None
        es_reward_acc = 0.0

        metric_sums = {
            "normalized_energy": 0.0,
            "service_degradation": 0.0,
            "handover_rate": 0.0,
            "switching_count": 0.0,
        }
        es_entropy_sum = 0.0
        mlb_entropy_sum = 0.0
        es_decisions = 0

        es_epoch_rewards= []
        mlb_rewards = []
        v_values = []
        wh_h_values = []

        for t in range(rollout_slots):
            state = make_rl_state(obs, env)

            # ----------------------------------------------------------
            # ES: decide only at epoch start.
            # Eligibility must be computed from the pre-action observation.
            # ----------------------------------------------------------
            if t % env.K == 0:
                eligible_mask = obs["dwell_timer"] >= (env.Tdwell - 1)
                (
                    current_es_action,
                    es_log_prob,
                    es_entropy,
                    es_value,
                ) = trainer.select_es_action(
                    state,
                    obs["activation"],
                    eligible_mask,
                    deterministic=False,
                )

                es_epoch_state = state.copy()
                es_epoch_action = current_es_action.copy()
                es_epoch_log_prob = es_log_prob
                es_epoch_mask = eligible_mask.copy()
                es_epoch_value = es_value
                es_reward_acc = 0.0
                es_entropy_sum += es_entropy
                es_decisions += 1

            # ----------------------------------------------------------
            # MLB: decide every slot.
            # ----------------------------------------------------------
            (
                mlb_action_idx,
                mlb_cio_db,
                mlb_log_prob,
                mlb_entropy,
                mlb_value,
                cost_value,
            ) = trainer.select_mlb_action(state, deterministic=False)
            mlb_entropy_sum += mlb_entropy

            next_obs, rewards, terminated, truncated, info = env.step(
                {
                    "es": current_es_action if t % env.K == 0 else None,
                    "mlb": mlb_cio_db,
                }
            )
            done = bool(terminated or truncated)

            mlb_rewards.append(rewards["mlb"])
            V_t = float(info["metrics"]["service_degradation"])
            H_t = float(info["metrics"]["handover_rate"])
            v_values.append(V_t)
            wh_h_values.append(env.wh * H_t)

            mlb_buffer.add(
                state=state,
                action=mlb_action_idx,
                old_log_prob=mlb_log_prob,
                reward=rewards["mlb"],
                cost=rewards["constraint_cost"],
                value=mlb_value,
                cost_value=cost_value,
                done=done,
            )

            # Eq. (19): epoch ES reward = average of slot energy rewards.
            es_reward_acc += rewards["es_slot"]

            for key in metric_sums:
                metric_sums[key] += float(info["metrics"][key])

            if (t + 1) % env.K == 0:
                assert es_epoch_state is not None
                assert es_epoch_action is not None
                assert es_epoch_log_prob is not None
                assert es_epoch_mask is not None
                assert es_epoch_value is not None

                es_epoch_reward = es_reward_acc / env.K
                es_epoch_rewards.append(es_epoch_reward)

                es_buffer.add(
                    state=es_epoch_state,
                    action=es_epoch_action,
                    old_log_prob=es_epoch_log_prob,
                    eligible_mask=es_epoch_mask,
                    reward=es_epoch_reward,
                    value=es_epoch_value,
                    done=done,
                )

            obs = next_obs

            if done:
                break

        # Current rollout ends exactly at episode termination, therefore the
        # bootstrap values are zero. If later using non-terminal rollouts,
        # evaluate the three critics on the final state here instead.
        stats = trainer.update(
            mlb_buffer,
            es_buffer,
            next_mlb_value=0.0,
            next_cost_value=0.0,
            next_es_value=0.0,
        )

        n_steps = len(mlb_buffer)

        def stat_string(values):
            x = np.asarray(values, dtype=np.float64)
            return (
                f"{x.mean():+.4f}±{x.std():.4f} "
                f"[{x.min():+.4f},{x.max():+.4f}]"
            )
        
        print(
            f"[Iter {iteration:04d}] "
            f"E={metric_sums['normalized_energy']/n_steps:.4f} | "
            f"V={metric_sums['service_degradation']/n_steps:.4f} | "
            f"HO={metric_sums['handover_rate']/n_steps:.4f} | "
            f"SW={metric_sums['switching_count']:.0f} | "
            f"lambda={stats['lagrange_lambda']:.4f} | "
            f"Cema={stats['cost_ema']:.4f} | "
            f"Lpi_E={stats['es_actor_loss']:.4f} | "
            f"Lpi_M={stats['mlb_actor_loss']:.4f} | "
            f"LV_E={stats['es_critic_loss']:.4f} | "
            f"LV_M={stats['mlb_critic_loss']:.4f} | "
            f"LV_C={stats['cost_critic_loss']:.4f} | "
            f"H_E={es_entropy_sum/max(es_decisions,1):.3f} | "
            f"H_M={mlb_entropy_sum/max(n_steps,1):.3f}"
        )
        print(
            f"           "
            f"StartHour={start_hour:.2f} | "
            f"R_ESepoch={stat_string(es_epoch_rewards)} | "
            f"V={stat_string(v_values)} | "
            f"wH*H={stat_string(wh_h_values)} | "
            f"R_MLB={stat_string(mlb_rewards)}"
        )
        print(
            f"           "
            f"Adv_ES={stats['es_adv_norm_abs_mean']:.3f} "
            f"[{stats['es_adv_norm_min']:+.3f},{stats['es_adv_norm_max']:+.3f}] | "
            f"Adv_C={stats['cost_adv_epoch_norm_abs_mean']:.3f} "
            f"[{stats['cost_adv_epoch_norm_min']:+.3f},{stats['cost_adv_epoch_norm_max']:+.3f}] | "
            f"lambda*Adv_C={stats['lambda_cost_adv_abs_mean']:.3f} "
            f"[{stats['lambda_cost_adv_min']:+.3f},{stats['lambda_cost_adv_max']:+.3f}] | "
            f"Adv_pen={stats['penalized_es_adv_abs_mean']:.3f} "
            f"[{stats['penalized_es_adv_min']:+.3f},{stats['penalized_es_adv_max']:+.3f}] | "
            f"Dom={100.0 * stats['constraint_dominance_frac']:.1f}% | "
            f"Flip={100.0 * stats['es_adv_sign_flip_frac']:.1f}%"
        )

        if iteration % 10 == 0 or iteration == num_iterations:
            trainer.save_checkpoint(
                checkpoint_dir / f"accord_seed{seed}_iter{iteration}.pt",
                iteration=iteration,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()

    train(
        seed=args.seed,
        config_path=args.config,
        iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
    )
