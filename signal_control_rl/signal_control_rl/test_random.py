from pathlib import Path

import numpy as np

from env.signal_env import SignalEnv, traci


def main():
    cfg = Path(__file__).resolve().parent / "sumo" / "go2_crossing.sumocfg"
    env = SignalEnv(cfg=str(cfg))
    rewards = []

    try:
        obs, info = env.reset()
        env.action_space.seed(42)
        assert env.observation_space.contains(obs), "初始观测不符合 observation_space"

        for step in range(1, 101):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step:03d} action={action} "
                f"obs.shape={obs.shape} obs.dtype={obs.dtype} reward={reward:.1f}",
                flush=True,
            )

            assert env.observation_space.contains(obs), f"第 {step} 步观测不符合空间"
            assert np.isfinite(reward) and reward <= 0, f"第 {step} 步奖励异常"
            assert not (terminated or truncated), f"第 {step} 步回合意外结束"
            rewards.append(reward)

        print(
            f"PASS: 100 steps; reward_min={min(rewards):.1f}; "
            f"reward_max={max(rewards):.1f}; "
            f"negative_steps={sum(r < 0 for r in rewards)}; "
            f"zero_steps={sum(r == 0 for r in rewards)}"
        )
    finally:
        env.close()
        if traci.isLoaded():
            traci.close()


if __name__ == "__main__":
    main()
