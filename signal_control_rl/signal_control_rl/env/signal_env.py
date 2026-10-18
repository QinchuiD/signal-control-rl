import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import sys
sys.path.insert(0, "/usr/share/sumo/tools")
import libsumo as traci

TLS_ID = "B1"          # 与基准脚本 signal_strategy_eval.py 一致
FROM_LANE = "B0B1_0"   # 过街进向车道（同基准脚本）
TO_LANE = "B1B2_0"
REQUEST_DURATION = 15.0
class SignalEnv(gym.Env):
    def __init__(self,cfg="sumo/go2_crossing.sumocfg",max_steps=1800):
        self.cfg = cfg
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.steps = 0
        self._phase_remaining = 0   # 决策周期计数: 每 10 个 0.1s 步决策一次
        self.idx=None
        self._episode_done=True
        self.req_i=0

    def dog_signal_index(self):
        ctrl = traci.trafficlight.getControlledLinks(TLS_ID)
        for idx, group in enumerate(ctrl):
            for link in group:
                if link[0] == FROM_LANE and link[1] == TO_LANE:
                    return idx
        return 0

    def dog_char(self,idx):
        state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
        return state[idx].lower() if 0 <= idx < len(state) else "?"
    
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except Exception:
            pass
        traci.start(["sumo", "-c", self.cfg, "--step-length", "0.1"])
        self.idx = self.dog_signal_index()
        state=traci.trafficlight.getRedYellowGreenState(TLS_ID)
        self._green_last = self.dog_char(self.idx) == "g"

        self._completed=0
        self._failed=0
        self.waits=[]
        self.steps = 0
        self._phase_remaining = 0
        self._pending = []
        interval=self.max_steps/36
        self.req_i=0
        self._episode_done=False

        self._request_times=sorted(round(max(1.0,i*interval+random.uniform(-5,5)),1)for i in range(36))

        return self._get_obs(),{}

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("去调用reset()")
        # 顺序铁律: 先执行动作 → 再推进仿真(10步) → 最后算奖励
        if action == 1:
            traci.trafficlight.setPhase(TLS_ID,
                (traci.trafficlight.getPhase(TLS_ID) + 1) % 4)
        traci.trafficlight.setPhaseDuration(TLS_ID, 1.1)
        self._update_vru()
        for _ in range(10):          # 1 个决策周期 = 10 × 0.1s
            traci.simulationStep()
            self._update_vru()

        self.steps += 1
        reward = -self._total_queue()
        done = self.steps >= self.max_steps
        self._episode_done=done
        if done:
            t = traci.simulation.getTime()
            for rt, onset in self._pending:
                if onset is not None and t - onset >= REQUEST_DURATION-1e-9:
                    self._completed += 1
                else:
                    self._failed += 1
            self._pending.clear()
        return self._get_obs(), float(reward), done, False, {}

    def _update_vru(self):
        t=traci.simulation.getTime()

        if self.req_i < len(self._request_times) and t>=self._request_times[self.req_i]-1e-9:
            self.req_i+=1
            if self.dog_char(self.idx)=="g":
                self.waits.append(0.0)
                self._pending.append((t,t))
            else:
                self._pending.append((t,None))
        state=traci.trafficlight.getRedYellowGreenState(TLS_ID)
        is_green = self.dog_char(self.idx) == "g"
        if is_green and not self._green_last:
            for i,(rt,onset) in enumerate(self._pending):
                if onset is None:
                    self._pending[i]=(rt,t)
                    self.waits.append(round(t-rt,2))
        elif not is_green and self._green_last:
            for rt,onset in self._pending:
                if onset is not None and t-onset>=REQUEST_DURATION:
                    self._completed+=1
                else:
                    self._failed+=1
            self._pending=[]

        self._green_last=is_green

        pass

    def _total_queue(self):
    # 获取 B1 控制的进口车道，并去重。
        lanes = set(traci.trafficlight.getControlledLanes(TLS_ID))

        total = 0
        for lane_id in lanes:
            # 获取这条车道上当前的停车车辆数。
            total += traci.lane.getLastStepHaltingNumber(lane_id)

        return float(total)
    def _get_obs(self):
        obs = np.zeros(16, dtype=np.float32) 
        incoming_lanes = [
        "B2B1_0",
        "C1B1_0",
        "B0B1_0",
        "A1B1_0",
    ]
        for i,lane_id in enumerate(incoming_lanes):
            obs[i]=traci.lane.getLastStepHaltingNumber(lane_id)
        for i in range(4,8):
            obs[i]=0
        phase=traci.trafficlight.getPhase(TLS_ID)
        obs[4+phase]=1.0
        obs[8]=traci.trafficlight.getSpentDuration(TLS_ID)
        obs[9] = float(bool(self._pending))
        t=traci.simulation.getTime()
        obs[10]=sum(max(0.0,t-rt) for rt,onset in self._pending if onset is None)
        for i in range(11,16):
            obs[i]=0
        return obs
    





