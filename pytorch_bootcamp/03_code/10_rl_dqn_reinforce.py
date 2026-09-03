"""RL：DQN + REINFORCE + A2C（對應 02_topics/08_RL.md）

★ 內建自製 CartPole 環境，不需要安裝 gymnasium。
   有裝 gymnasium 時會自動改用官方環境（--gym）。

實測參考（RTX 4060 + 自製環境）：
    REINFORCE 約 150 集就突破 195，500 集穩定在 300~420
    DQN 收斂較慢，600 集約 90，建議跑 1500 集以上

用法：
    python 10_rl_dqn_reinforce.py --algo dqn --episodes 1500
    python 10_rl_dqn_reinforce.py --algo reinforce --episodes 600
    python 10_rl_dqn_reinforce.py --algo a2c --episodes 600
    python 10_rl_dqn_reinforce.py --algo dqn --no-target      # ★ 觀察發散
    python 10_rl_dqn_reinforce.py --algo dqn --no-done-mask   # ★ 觀察 Q 值爆炸
    python 10_rl_dqn_reinforce.py --algo reinforce --no-baseline
    python 10_rl_dqn_reinforce.py --algo dqn --seeds 5        # ★ 多 seed 才有意義

===========================================================================
RL 的名詞對照（跟監督式學習完全不同的一套詞彙）
---------------------------------------------------------------------------
  state  s     現在的狀況（CartPole 是 [車位置, 車速, 桿角度, 角速度]）
  action a     可以做的動作（0=推左, 1=推右）
  reward r     做完動作拿到的分數（CartPole 每撐一步得 1 分）
  episode      一局（從 reset 到桿子倒下或撐滿 500 步）
  return       一局的總分（這才是要看的指標，不是 loss！）
  done         這一局結束了沒
  gamma        折扣因子，未來的獎勵打幾折（0.99 = 幾乎同等重視）
  epsilon      探索率，有多少機率隨機亂動而不是照策略走
===========================================================================
"""

import argparse
import math
import random
import sys
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.append(str(Path(__file__).resolve().parent))
from common import set_seed, get_device, save_json

OUT = Path(__file__).resolve().parent / "outputs"


# ============================================================
# 自製 CartPole（跟 Gym 的物理與獎勵一致）
# ============================================================
class CartPole:
    obs_dim, n_actions = 4, 2

    def __init__(self, seed=0, max_steps=500):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.g, self.mc, self.mp = 9.8, 1.0, 0.1
        self.mt = self.mc + self.mp
        self.l, self.fmag, self.tau = 0.5, 10.0, 0.02
        self.theta_thr, self.x_thr = 12 * math.pi / 180, 2.4

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        self.t = 0
        return self.state.astype(np.float32)

    def step(self, action):
        x, x_dot, th, th_dot = self.state
        force = self.fmag if action == 1 else -self.fmag
        cos, sin = math.cos(th), math.sin(th)
        temp = (force + self.mp * self.l * th_dot ** 2 * sin) / self.mt
        th_acc = ((self.g * sin - cos * temp)
                  / (self.l * (4.0 / 3.0 - self.mp * cos ** 2 / self.mt)))
        x_acc = temp - self.mp * self.l * th_acc * cos / self.mt

        x += self.tau * x_dot;      x_dot += self.tau * x_acc
        th += self.tau * th_dot;    th_dot += self.tau * th_acc
        self.state = np.array([x, x_dot, th, th_dot])
        self.t += 1

        terminated = bool(abs(x) > self.x_thr or abs(th) > self.theta_thr)
        truncated = self.t >= self.max_steps
        return (self.state.astype(np.float32), 1.0,
                terminated, truncated)


def make_env(seed, use_gym=False):
    if use_gym:
        try:
            import gymnasium as gym
            env = gym.make("CartPole-v1")
            print("[env] 使用 gymnasium CartPole-v1")
            return env, 4, 2, True
        except Exception as e:
            print(f"[env] gymnasium 不可用（{type(e).__name__}），改用自製環境")
    return CartPole(seed), CartPole.obs_dim, CartPole.n_actions, False


class EnvWrapper:
    """統一自製環境與 gymnasium 的介面。"""
    def __init__(self, env, is_gym, seed):
        self.env, self.is_gym, self.seed = env, is_gym, seed
    def reset(self):
        if self.is_gym:
            s, _ = self.env.reset(seed=self.seed); self.seed += 1
            return np.asarray(s, dtype=np.float32)
        return self.env.reset()
    def step(self, a):
        if self.is_gym:
            s, r, term, trunc, _ = self.env.step(a)
            return np.asarray(s, dtype=np.float32), float(r), bool(term), bool(trunc)
        return self.env.step(a)


# ============================================================
def mlp(inp, out, hidden=128):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


Transition = namedtuple("Transition", "s a r s2 done")


class ReplayBuffer:
    """經驗回放池：把走過的每一步存起來，訓練時隨機抽樣。

    解決什麼問題：連續的狀態高度相關（這一格跟下一格幾乎一樣），
    直接拿來訓練會違反「樣本獨立」的假設，導致發散。
    隨機抽樣打散了相關性，而且同一筆經驗可以重複使用（樣本效率高）。

    語法說明：deque(maxlen=N) 是「雙端佇列」，滿了之後從頭自動擠掉舊的，
              不用自己寫刪除邏輯。
    """
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)     # 50_000 的底線只是給人看的分隔
    def push(self, *a):
        self.buf.append(Transition(*a))
    def sample(self, bs, device):
        # 這一行做了三件事，由內往外讀：
        #  ① random.sample(self.buf, bs)  隨機抽 bs 筆 Transition
        #  ② zip(*...)                    「轉置」：把 bs 個 (s,a,r,s2,done)
        #                                  變成 (所有s), (所有a), (所有r)...
        #  ③ Transition(*...)             再包回具名的容器，之後能寫 b.s / b.a
        b = Transition(*zip(*random.sample(self.buf, bs)))
        t = lambda x, dt: torch.tensor(np.array(x), dtype=dt, device=device)
        return (t(b.s, torch.float32), t(b.a, torch.int64), t(b.r, torch.float32),
                t(b.s2, torch.float32), t(b.done, torch.float32))
    def __len__(self):
        return len(self.buf)


# ============================================================
# DQN
# ============================================================
def run_dqn(args, seed):
    set_seed(seed)
    device = get_device(verbose=False)
    env_raw, obs_dim, n_act, is_gym = make_env(seed, args.gym)
    env = EnvWrapper(env_raw, is_gym, seed)

    q_net = mlp(obs_dim, n_act).to(device)
    target_net = mlp(obs_dim, n_act).to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer)

    eps, returns, q_means, step_count = 1.0, [], [], 0
    for ep in range(1, args.episodes + 1):
        s, ep_ret, done = env.reset(), 0.0, False
        while not done:
            # epsilon-greedy：以 eps 的機率隨機亂動（探索），
            # 否則照目前學到的最好動作走（利用）。eps 隨訓練逐漸變小。
            if random.random() < eps:
                a = random.randrange(n_act)
            else:
                with torch.no_grad():
                    a = q_net(torch.as_tensor(s, device=device)[None]).argmax(1).item()
            s2, r, term, trunc = env.step(a)
            done = term or trunc
            # ★ 只有「真正終止」才算 done；時間到（truncated）不算，否則會低估價值
            # ★ 存的是 term（真的失敗）而不是 done（含時間到）。
            #   時間到只是我們喊卡，環境本身沒結束，不該把未來價值歸零。
            buf.push(s, a, r, s2, float(term))
            s, ep_ret, step_count = s2, ep_ret + r, step_count + 1

            if len(buf) >= args.warmup:
                sb, ab, rb, s2b, db = buf.sample(args.batch_size, device)
                # q_net(sb) 回傳每個動作的 Q 值 (B, n_actions)，
                # 但我們只要「當初實際採取的那個動作」的 Q 值。
                #   ab[:, None] 把 (B,) 變成 (B,1)（gather 要求 index 維度數相同）
                #   gather(1, ...) 沿第 1 維（動作維）各挑一個
                #   squeeze(1) 把 (B,1) 壓回 (B,)
                q_sa = q_net(sb).gather(1, ab[:, None]).squeeze(1)

                if args.no_target_net:
                    # ★ 對照組：直接用 q_net 當 target -> 追著自己的尾巴跑
                    q_next = q_net(s2b).max(1).values.detach()
                else:
                    with torch.no_grad():                     # ★ target 不能有梯度
                        if args.double:
                            a2 = q_net(s2b).argmax(1, keepdim=True)
                            q_next = target_net(s2b).gather(1, a2).squeeze(1)
                        else:
                            q_next = target_net(s2b).max(1).values
                # ★ (1 - done)：這一局結束的那一步，後面沒有未來了，
                #   所以未來獎勵那一項要歸零。忘了乘會讓 Q 值無限膨脹。
                mask = 1.0 if args.no_done_mask else (1 - db)
                # Bellman 目標：Q(s,a) 應該等於「立即獎勵 + 折扣後的未來最好價值」
                target = rb + args.gamma * q_next * mask

                # smooth_l1（Huber）：誤差小的時候像 MSE，大的時候像 MAE。
                # RL 的 target 本來就很吵，用 MSE 會被離群值帶偏。
                loss = F.smooth_l1_loss(q_sa, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                q_means.append(q_sa.mean().item())

            # ★ 定期把 q_net 的權重複製給 target_net。
            #   target_net 提供「暫時固定的目標」，避免追著自己的尾巴跑而發散。
            if step_count % args.target_update == 0 and not args.no_target_net:
                target_net.load_state_dict(q_net.state_dict())
            eps = max(args.eps_min, eps * args.eps_decay)

        returns.append(ep_ret)
        if ep % max(1, args.episodes // 10) == 0:
            avg = np.mean(returns[-50:])
            qm = np.mean(q_means[-500:]) if q_means else 0.0
            flag = "  <-- ★ Q 值爆炸" if qm > 200 else ""
            print(f"  ep {ep:4d} | 最近50集平均 return {avg:7.1f} | "
                  f"eps {eps:.3f} | Q均值 {qm:8.2f}{flag}")
    return returns


# ============================================================
# REINFORCE / A2C
# ============================================================
def run_pg(args, seed):
    set_seed(seed)
    device = get_device(verbose=False)
    env_raw, obs_dim, n_act, is_gym = make_env(seed, args.gym)
    env = EnvWrapper(env_raw, is_gym, seed)

    actor = mlp(obs_dim, n_act).to(device)          # ★ 輸出 logits，不加 softmax
    critic = mlp(obs_dim, 1).to(device) if args.algo == "a2c" else None
    params = list(actor.parameters()) + (list(critic.parameters()) if critic else [])
    opt = torch.optim.Adam(params, lr=args.lr)

    returns_hist = []
    for ep in range(1, args.episodes + 1):
        s, done = env.reset(), False
        log_probs, rewards, values, entropies = [], [], [], []
        while not done:
            st = torch.as_tensor(s, device=device)[None]
            # Categorical(logits=...) 直接吃未正規化的分數，內部自己做 softmax，
            # 比先算機率再傳進去更數值穩定。s[None] 是加上 batch 維度。
            dist = Categorical(logits=actor(st))
            a = dist.sample()
            # ★ log_prob 必須留在計算圖上！存成 .item() 就完全學不到東西
            # ★★ log_prob 必須「留在計算圖上」！
            #   寫成 .item() 或 .detach() 的話梯度就斷了，模型完全不學，
            #   而且「不會報錯」—— 這是 REINFORCE 最惡名昭彰的 bug。
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            if critic is not None:
                values.append(critic(st).squeeze())
            s, r, term, trunc = env.step(a.item())
            rewards.append(r)
            done = term or trunc

        # 折扣回報
        G, rets = 0.0, []
        for r in reversed(rewards):
            G = r + args.gamma * G
            rets.append(G)
        rets = torch.tensor(list(reversed(rets)), dtype=torch.float32, device=device)

        if critic is not None:
            V = torch.stack(values)
            adv = (rets - V).detach()                       # ★ advantage 要 detach
            critic_loss = F.mse_loss(V, rets)
        else:
            if args.baseline:
                # ★ baseline：減掉平均、除以標準差。
                #   數學上不改變梯度的期望值，但大幅降低變異數 -> 訓練穩定很多。
                #   直覺：「比平均好的動作才加強」，而不是「所有拿到正分的都加強」。
                adv = (rets - rets.mean()) / (rets.std() + 1e-8)
            else:
                adv = rets
            critic_loss = torch.zeros((), device=device)

        lp = torch.stack(log_probs).squeeze()
        # ★★ 這個負號很重要：策略梯度的目標是「最大化」期望回報，
        #   但 PyTorch 的優化器只會「最小化」loss，所以取負號。
        #   loss 不一定要是「誤差」，它可以是任何你想最小化的量。
        actor_loss = -(lp * adv).sum()
        ent = torch.stack(entropies).mean()
        loss = actor_loss + 0.5 * critic_loss - args.ent_coef * ent

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

        returns_hist.append(sum(rewards))
        if ep % max(1, args.episodes // 10) == 0:
            print(f"  ep {ep:4d} | 最近50集平均 return "
                  f"{np.mean(returns_hist[-50:]):7.1f} | policy 熵 {ent.item():.3f}")
    return returns_hist


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "reinforce", "a2c"], default="dqn")   # 選演算法
    ap.add_argument("--episodes", type=int, default=600,
                    help="reinforce 約 500 就夠；dqn 建議 1500+")
    # 這不是隨機種子本身，是「要跑幾個不同的種子」，
    # main() 主體會用 for i in range(args.seeds) 迴圈跑好幾次取平均。
    ap.add_argument("--seeds", type=int, default=1, help="★ 做研究至少 5")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=50000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--target-update", type=int, default=500)
    ap.add_argument("--eps-decay", type=float, default=0.9995)
    ap.add_argument("--eps-min", type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--double", action="store_true", help="Double DQN")
    ap.add_argument("--no-target", dest="no_target_net", action="store_true",
                    help="★ 對照組：不用 target network")
    ap.add_argument("--no-done-mask", action="store_true",
                    help="★ 對照組：忘記乘 (1-done)")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false",
                    help="★ 對照組：REINFORCE 不用 baseline")
    ap.add_argument("--gym", action="store_true", help="用 gymnasium 而非自製環境")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_runs = []
    for i in range(args.seeds):
        seed = args.seed + i
        print(f"\n--- seed {seed} | algo={args.algo} ---")
        rets = run_dqn(args, seed) if args.algo == "dqn" else run_pg(args, seed)
        all_runs.append(rets)

    # 不同 seed 的長度可能不同，先截到最短的才能疊成一個矩陣
    arr = np.array([r[:min(len(x) for x in all_runs)] for r in all_runs])
    final = arr[:, -100:].mean(axis=1)
    print()
    print("=" * 66)
    print(f"最後 100 集平均 return：{final.mean():.1f} +/- {final.std():.1f}  "
          f"(n={args.seeds} seeds)")
    if args.seeds == 1:
        print("★ 只跑一個 seed 的 RL 結果沒有參考價值，做研究請用 --seeds 5")
    if final.mean() >= 195:
        print("★ CartPole 已解決（>= 195）")
    print("=" * 66)

    save_json(OUT / f"rl_{args.algo}" / "returns.json",
              {"runs": arr.tolist(), "final_mean": float(final.mean()),
               "final_std": float(final.std())})

    print("""
建議做的對照實驗：
  --no-target       target network 拿掉 -> 訓練發散
  --no-done-mask    忘記 (1-done)      -> Q 值無限膨脹
  --no-baseline     REINFORCE 沒 baseline -> return 曲線劇烈震盪
  --double          Double DQN -> Q 值高估減輕
  ★ 每個都用 --seeds 5 跑，比較 mean +/- std
""")


# 只有直接執行這個檔案才會呼叫 main()（完整解釋見 00_common.py 與 01_tensor_playground.py 開頭）
if __name__ == "__main__":
    main()