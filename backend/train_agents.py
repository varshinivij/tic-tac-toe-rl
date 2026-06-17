"""
Train Q-learning (easy) and DQN (hard) agents, then save artifacts.
Run from the backend/ directory:  python train_agents.py
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(__file__))

from rl_agents.q_learning import Board, TicTacToeGame
from rl_agents.dql import BasicNN, TicTacToeDQ
import torch
import torch.nn as nn

GAMMA = 0.9
QL_EPISODES  = 30_000
DQN_EPISODES = 30_000

# ===========================================================
# 1. Q-learning (easy mode)
# ===========================================================
print(f"Training Q-learning for {QL_EPISODES:,} episodes...", flush=True)

for episode in range(QL_EPISODES):
    eps = max(0.05, 1.0 - episode / (QL_EPISODES * 0.7))
    g = TicTacToeGame(Board(), epsilon=eps)
    while not g.board.game_over():
        g.store_user_move()
        if not g.board.game_over():
            g.make_move()
    g.update_Q(g.board.winner())
    if episode % 10_000 == 0:
        print(f"  ep {episode:>6,}  table={len(TicTacToeGame.all_game_history):,}", flush=True)

qt_path = os.path.join(os.path.dirname(__file__), "rl_agents", "q_table.json")
nested: dict = {}
for (state, action), value in TicTacToeGame.all_game_history.items():
    nested.setdefault(state, {})[str(action)] = value
with open(qt_path, "w") as f:
    json.dump(nested, f)
print(f"Q-table saved → {qt_path}  ({len(TicTacToeGame.all_game_history):,} pairs)\n", flush=True)

# ===========================================================
# 2. DQN (hard mode)  —  DQN=player 1, random opponent=-1
# ===========================================================
print(f"Training DQN for {DQN_EPISODES:,} episodes...", flush=True)

model     = BasicNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
wins = draws = losses = 0

for episode in range(DQN_EPISODES):
    epsilon = max(0.05, 1.0 - episode / (DQN_EPISODES * 0.6))
    board = [0] * 9

    while True:
        # ---- DQN turn ----
        moves = [i for i in range(9) if board[i] == 0]
        if not moves:
            draws += 1; break

        b_before = Board(board[:])
        if random.random() < epsilon:
            action = random.choice(moves)
        else:
            with torch.no_grad():
                action = int(torch.argmax(model(b_before)).item())

        board[action] = 1
        b_after_dqn = Board(board[:])

        if b_after_dqn.game_over():
            w = b_after_dqn.winner()
            r = 1.0 if w == 1 else 0.0
            loss = criterion(model(b_before)[0, action], torch.tensor(r))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if w == 1: wins += 1
            else: draws += 1
            break

        # ---- Opponent turn (random) ----
        opp_moves = [i for i in range(9) if board[i] == 0]
        board[random.choice(opp_moves)] = -1
        b_after_opp = Board(board[:])

        if b_after_opp.game_over():
            w = b_after_opp.winner()
            r = -1.0 if w == -1 else 0.0
            loss = criterion(model(b_before)[0, action], torch.tensor(r))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if w == -1: losses += 1
            else: draws += 1
            break

        # ---- Bellman TD update (mid-game) ----
        with torch.no_grad():
            nq = model(Board(board[:]))
        t = torch.tensor(GAMMA * nq.max().item())
        loss = criterion(model(b_before)[0, action], t)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    if episode % 10_000 == 0 and episode > 0:
        total = wins + draws + losses
        pct_w = 100 * wins / total if total else 0
        print(f"  ep {episode:>6,}  W={wins} D={draws} L={losses}  win%={pct_w:.1f}", flush=True)
        wins = draws = losses = 0

weights_path = os.path.join(os.path.dirname(__file__), "rl_agents", "weights_dql.pt")
torch.save(model.state_dict(), weights_path)
total = wins + draws + losses
print(f"  final  W={wins} D={draws} L={losses} / {total}", flush=True)
print(f"DQN weights saved → {weights_path}\n", flush=True)
print("Training complete.", flush=True)
