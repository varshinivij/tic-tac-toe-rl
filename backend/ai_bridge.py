import os
import json
import ast
import importlib.util
import sys
import types
from typing import List, Optional

# -----------------------------
# Artifact paths (with overrides)
# -----------------------------
QTABLE_PATH = os.getenv(
    "QTABLE_PATH",
    os.path.join(os.path.dirname(__file__), "rl_agents", "q_table.json")
)
WEIGHTS_PATH = os.getenv(
    "DQL_WEIGHTS",
    os.path.join(os.path.dirname(__file__), "rl_agents", "weights_dql.pt")
)

# -----------------------------
# Sanitary module loader (prevents running training code on import)
# -----------------------------
_MOD_CACHE = {}

def _find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except Exception:
        return None

def _sanitize_source(src: str):
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(node)
    new_tree = ast.Module(body=keep, type_ignores=[])
    return compile(new_tree, filename="<sanitized>", mode="exec")

def _load_sanitized(module_name: str) -> types.ModuleType:
    if module_name in _MOD_CACHE:
        return _MOD_CACHE[module_name]

    spec = _find_spec(module_name)
    if not spec or not spec.origin:
        raise RuntimeError(f"Cannot find module: {module_name}")

    with open(spec.origin, "r", encoding="utf-8") as f:
        src = f.read()
    code = _sanitize_source(src)

    mod = types.ModuleType(module_name)
    mod.__file__ = spec.origin

    if module_name.endswith(".q_learning"):
        sys.modules["q_learning"] = mod
    if module_name.endswith(".dql") and "q_learning" not in sys.modules:
        base = module_name.rsplit(".", 1)[0]
        _load_sanitized(f"{base}.q_learning")

    sys.modules[module_name] = mod
    exec(code, mod.__dict__)
    _MOD_CACHE[module_name] = mod
    return mod

# -----------------------------
# Board helpers
# -----------------------------
def _other(mark: str) -> str:
    return "O" if mark.upper() == "X" else "X"

def _to_numeric(board: List[str], ai_mark: str) -> List[int]:
    ai = ai_mark.upper()
    opp = _other(ai)
    out: List[int] = []
    for v in board:
        if v == ai:
            out.append(1)
        elif v == opp:
            out.append(-1)
        else:
            out.append(0)
    return out

def _legal_actions(board: List[str]) -> List[int]:
    return [i for i in range(9) if board[i] == ""]

# -----------------------------
# Q-table + weights loaders
# -----------------------------
_QTABLE_CACHE = None

def _load_qtable_once():
    global _QTABLE_CACHE
    if _QTABLE_CACHE is not None:
        return _QTABLE_CACHE
    if os.path.exists(QTABLE_PATH):
        with open(QTABLE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Nested format saved by train_agents.py: {state: {action_str: value}}
        if data and isinstance(next(iter(data.values()), None), dict):
            qt: dict = {}
            for state, actions in data.items():
                for action_str, value in actions.items():
                    qt[(state, int(action_str))] = value
            _QTABLE_CACHE = qt
        else:
            _QTABLE_CACHE = {}
    else:
        _QTABLE_CACHE = None
    return _QTABLE_CACHE

def _maybe_load_weights_into(model) -> bool:
    try:
        import torch
    except Exception:
        return False  
    if not os.path.exists(WEIGHTS_PATH):
        return False
    try:
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return True
    except Exception:
        return False

# -----------------------------
# Easy (tabular Q-learning)
# -----------------------------
def _easy_move(board_str: List[str], ai_mark: str) -> int:
    ql = _load_sanitized("rl_agents.q_learning")
    Board = getattr(ql, "Board", None)
    Game = getattr(ql, "TicTacToeGame", None)
    if Board is None or Game is None:
        raise RuntimeError("rl_agents.q_learning must define Board and TicTacToeGame")

    numeric = _to_numeric(board_str, ai_mark)
    b = Board()
    b.board = numeric[:]
    player_num = 1 if ai_mark.upper() == "X" else -1

    # Inject loaded Q-table into the class-level all_game_history so select_move uses it
    qt = _load_qtable_once()
    if qt is not None:
        Game.all_game_history = qt

    g = Game(b, epsilon=0.0, player=player_num)

    if hasattr(g, "select_move") and callable(getattr(g, "select_move")):
        idx = g.select_move(epsilon=0.0)
        if idx is None:
            import random as _rand
            return _rand.choice(_legal_actions(board_str))
        if not isinstance(idx, int):
            raise RuntimeError(f"Q-learning returned non-int: {idx}")
        if b.board[idx] != 0:
            raise RuntimeError(f"Q-learning chose occupied cell: {idx}")
        return idx

    before = b.board[:]
    if hasattr(g, "make_move") and callable(getattr(g, "make_move")):
        g.make_move()
    else:
        raise RuntimeError("No usable move method found in TicTacToeGame")

    for i in range(9):
        if before[i] == 0 and b.board[i] == player_num:
            return i

    raise RuntimeError("Q-learning did not change the board")

# -----------------------------
# Hard (DQL with weights)
# -----------------------------
def _hard_move(board_str: List[str], ai_mark: str) -> int:
    ql = _load_sanitized("rl_agents.q_learning")
    dql = _load_sanitized("rl_agents.dql")

    Board = getattr(ql, "Board", None)
    BasicNN = getattr(dql, "BasicNN", None)
    DQ = getattr(dql, "TicTacToeDQ", None)
    if Board is None or BasicNN is None or DQ is None:
        raise RuntimeError("rl_agents.dql must define BasicNN and TicTacToeDQ; q_learning must define Board")

    numeric = _to_numeric(board_str, ai_mark)
    b = Board()
    b.board = numeric[:]
    player_num = 1 if ai_mark.upper() == "X" else -1

    model = BasicNN()
    _maybe_load_weights_into(model)  # loads weights_dql.pt if present

    game = DQ(model, b, player=player_num)
    if not hasattr(game, "select_move") or not callable(getattr(game, "select_move")):
        raise RuntimeError("TicTacToeDQ must provide select_move(epsilon=...)")

    pos = game.select_move(epsilon=0.0)
    if not isinstance(pos, int):
        raise RuntimeError(f"DQL returned non-int: {pos}")
    if pos < 0 or pos > 8:
        raise RuntimeError(f"DQL returned out-of-range index: {pos}")
    if board_str[pos] != "":
        raise RuntimeError(f"DQL chose occupied cell: {pos}")

    return pos

# -----------------------------
# Public API used by FastAPI
# -----------------------------
def agent_move(board: List[str], ai_mark: str, difficulty: str) -> int:
    if len(board) != 9:
        raise RuntimeError("Board must have 9 cells")
    if ai_mark.upper() not in ("X", "O"):
        raise RuntimeError("ai_mark must be 'X' or 'O'")

    mode = (difficulty or "hard").lower()
    if mode == "easy":
        pos = _easy_move(board, ai_mark)
    elif mode == "hard":
        pos = _hard_move(board, ai_mark)
    else:
        raise RuntimeError("difficulty must be 'easy' or 'hard'")

    legal = _legal_actions(board)
    if pos not in legal:
        raise RuntimeError(f"Agent returned illegal move: {pos}")
    return pos
