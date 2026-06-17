"""
Train Q-learning (easy) and DQN (hard) agents, then save artifacts.
Run from the backend/ directory:  python train_agents.py

DQN uses self-play: the network plays both sides with a perspective flip
so it must learn both attack AND defence.
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(__file__))

from rl_agents.q_learning import Board, TicTacToeGame
from rl_agents.dql import BasicNN, TicTacToeDQ
import torch
import torch.nn as nn

GAMMA = 0.9

# ===========================================================
# 1. Q-learning (easy mode)
# ===========================================================
QL_EPISODES = 30_000
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
# 2. DQN – self-play with perspective flip
#
# Both players use the SAME network. Before each move the board
# is multiplied by the current player's sign so the network
# always sees "I am +1, opponent is -1". This forces it to
# learn both attack (win fast) and defence (block threats).
# Final game outcome is propagated back to every move as a
# Monte-Carlo reward (+1 win / -1 loss / 0 draw).
# ===========================================================
DQN_EPISODES = 100_000
print(f"Training DQN (self-play) for {DQN_EPISODES:,} episodes...", flush=True)

model     = BasicNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
wins = draws = losses = 0

for episode in range(DQN_EPISODES):
    epsilon = max(0.05, 1.0 - episode / (DQN_EPISODES * 0.65))
    board   = [0] * 9   # absolute: +1 = first mover, -1 = second mover
    current = 1         # whose turn it is

    # (board_in_current_perspective, action, absolute_player)
    move_log: list = []

    while True:
        moves = [i for i in range(9) if board[i] == 0]
        if not moves:
            # Board full, no winner → draw
            for (bp, a, _) in move_log:
                loss = criterion(model(bp)[0, a], torch.tensor(0.0))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            draws += 1
            break

        # Flip board so current player always appears as +1
        persp  = [v * current for v in board]
        b_persp = Board(persp)

        if random.random() < epsilon:
            action = random.choice(moves)
        else:
            with torch.no_grad():
                action = int(torch.argmax(model(b_persp)).item())

        move_log.append((b_persp, action, current))
        board[action] = current

        # Check result on absolute board
        b_abs = Board(board[:])
        if b_abs.game_over():
            abs_winner = b_abs.winner()   # +1, -1, or 0

            for (bp, a, player) in move_log:
                if abs_winner == 0:
                    r = 0.0
                else:
                    r = 1.0 if player == abs_winner else -1.0
                loss = criterion(model(bp)[0, a], torch.tensor(r))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if   abs_winner ==  1: wins   += 1
            elif abs_winner == -1: losses += 1
            else:                  draws  += 1
            break

        current = -current

    if episode % 20_000 == 0 and episode > 0:
        total  = wins + draws + losses
        pct_d  = 100 * draws / total if total else 0
        print(f"  ep {episode:>7,}  W={wins} D={draws} L={losses}  draw%={pct_d:.1f}", flush=True)
        wins = draws = losses = 0

weights_path = os.path.join(os.path.dirname(__file__), "rl_agents", "weights_dql.pt")
torch.save(model.state_dict(), weights_path)
total = wins + draws + losses
pct_d = 100 * draws / total if total else 0
print(f"  final  W={wins} D={draws} L={losses}  draw%={pct_d:.1f}", flush=True)
print(f"DQN weights saved → {weights_path}\n", flush=True)
print("Training complete.", flush=True)
