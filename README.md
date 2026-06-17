# Tic Tac Toe RL

A full-stack Tic Tac Toe game where the AI is powered by reinforcement learning.

**Tech Stack:** React · FastAPI · PyTorch

## Features

- **Two difficulty modes:**
  - **Easy** — Tabular Q-Learning: a classic table-based RL algorithm that learns state→action Q-values over thousands of self-play games
  - **Hard** — Deep Q-Network (DQN): a neural network (9 → 12 → 9) trained with the Bellman equation and ε-greedy exploration, achieving ~92% win rate vs a random opponent

- Play as **X** or **O** — the AI always takes the other mark

- Win/draw detection with animated highlights, board glow, and confetti on victory

## How it works

```
Frontend (React)  ──POST /new, /move──▶  Backend (FastAPI)
                                               │
                              ┌────────────────┴────────────────┐
                              │ Easy: TicTacToeGame              │
                              │  Q-table (15k state-action pairs)│
                              │                                  │
                              │ Hard: BasicNN (PyTorch)          │
                              │  weights trained via DQN         │
                              └──────────────────────────────────┘
```

### RL Algorithms

**Q-Learning (Easy)**
- State: 9-cell board encoded as a string
- Action: cell index 0–8
- Update rule: `Q(s,a) ← Q(s,a) + α · (r − Q(s,a))`
- Greedy inference: `argmax_a Q(s, a)`

**Deep Q-Network (Hard)**
- Network: `Linear(9→12, ReLU) → Linear(12→9)`
- Illegal moves masked to −∞ before argmax
- Bellman target: `r + γ · max_a Q(s', a)`
- Loss: MSE between predicted and target Q-value
- Trained with ε-greedy decay (ε: 1.0 → 0.05 over 30k episodes)

## Running locally

```bash
# 1. Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

# 2. Frontend (separate terminal)
cd frontend
npm install
npm start
```

App runs at `http://localhost:3000`, API at `http://localhost:8000`.

To retrain the agents:
```bash
cd backend
python train_agents.py
```

<img width="1225" height="828" alt="image" src="https://github.com/user-attachments/assets/a4b1eb58-c684-4d52-98de-50d782b53275" />
