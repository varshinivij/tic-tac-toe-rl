import React, { useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function apiNewGame(playerMark, difficulty) {
  const res = await fetch(`${API}/new`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_mark: playerMark, difficulty })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiMove(gameId, position) {
  const res = await fetch(`${API}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game_id: gameId, position })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6]
];

function winLine(board) {
  for (const [a, b, c] of LINES) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) return [a, b, c];
  }
  return null;
}

function useConfetti() {
  const running = useRef(false);
  return () => {
    if (running.current) return;
    running.current = true;
    const canvas = document.createElement("canvas");
    canvas.className = "confetti";
    document.body.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);
    const pieces = Array.from({ length: 100 }, () => ({
      x: Math.random() * canvas.width,
      y: -20 - Math.random() * canvas.height * 0.4,
      s: 4 + Math.random() * 6,
      vy: 2 + Math.random() * 3,
      vx: -2 + Math.random() * 4,
      r: Math.random() * Math.PI,
      vr: -0.1 + Math.random() * 0.2
    }));
    let t = 0;
    (function loop() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const p of pieces) {
        p.x += p.vx; p.y += p.vy; p.r += p.vr;
        if (p.y > canvas.height + 20) { p.y = -20; p.x = Math.random() * canvas.width; }
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r);
        ctx.fillStyle = ["#ff6b6b","#ffd93d","#6bcbef","#b8f2e6","#c77dff"][(p.x | 0) % 5];
        ctx.fillRect(-p.s / 2, -p.s / 2, p.s, p.s);
        ctx.restore();
      }
      t++;
      if (t < 160) window.requestAnimationFrame(loop);
      else {
        canvas.remove();
        window.removeEventListener("resize", resize);
        running.current = false;
      }
    })();
  };
}

export default function App() {
  const [board, setBoard] = useState(Array(9).fill(""));
  const [status, setStatus] = useState("in_progress"); // "in_progress" | "X" | "O" | "draw"
  const [gameId, setGameId] = useState(null);
  const [player, setPlayer] = useState("X");
  const [ai, setAi] = useState("O");
  const [started, setStarted] = useState(false);
  const [difficulty, setDifficulty] = useState("hard"); // "easy" | "hard"

  const resultOpen = started && status !== "in_progress";
  const wins = useMemo(() => winLine(board), [board]);
  const confetti = useConfetti();

  useEffect(() => {
    if (started && status === player) confetti();
  }, [status, player, started, confetti]);

  async function startGame() {
    const g = await apiNewGame(player, difficulty);
    setGameId(g.game_id);
    setBoard(g.board);
    setStatus(g.status);
    setPlayer(g.player);
    setAi(g.ai);
    setStarted(true);
  }

  async function clickCell(i) {
    if (!started || status !== "in_progress" || board[i] !== "") return;
    const r = await apiMove(gameId, i);
    setBoard(r.board);
    setStatus(r.status);
  }

  function resetAll() {
    setStarted(false);
    setBoard(Array(9).fill(""));
    setStatus("in_progress");
    setGameId(null);
    setAi(player === "X" ? "O" : "X");
  }

  return (
    <div className={`wrap ${resultOpen ? "modal-open" : ""}`}>
      <h1 className="title">Tic Tac Toe <span className="pill">RL</span></h1>

      <div className="controls">
        <div className="seg">
          <button
            className={`seg-btn ${player === "X" ? "active" : ""}`}
            onClick={() => { if (!started) { setPlayer("X"); setAi("O"); } }}
            disabled={started}
          >
            Play as X
          </button>
          <button
            className={`seg-btn ${player === "O" ? "active" : ""}`}
            onClick={() => { if (!started) { setPlayer("O"); setAi("X"); } }}
            disabled={started}
          >
            Play as O
          </button>
        </div>

        <div className="seg">
          <button
            className={`seg-btn ${difficulty === "easy" ? "active" : ""}`}
            onClick={() => { if (!started) setDifficulty("easy"); }}
            disabled={started}
          >
            Easy
          </button>
          <button
            className={`seg-btn ${difficulty === "hard" ? "active" : ""}`}
            onClick={() => { if (!started) setDifficulty("hard"); }}
            disabled={started}
          >
            Hard
          </button>
        </div>

        {!started ? (
          <button className="btn primary" onClick={startGame}>Start Game</button>
        ) : (
          <button className="btn small" onClick={resetAll}>Reset</button>
        )}
      </div>

      <div
        className={`board
          ${status === player ? "win-human" : ""}
          ${status === ai ? "win-ai" : ""}
          ${resultOpen ? "no-bg glow-first" : ""}`}
      >
        <div className={`grid ${status === ai ? "shake" : ""} ${resultOpen ? "finished" : ""}`}>
          {board.map((cell, i) => {
            const highlight = wins?.includes(i);
            return (
              <button
                key={i}
                className={`cell ${cell || "empty"} ${highlight ? "win" : ""}`}
                onClick={() => clickCell(i)}
                disabled={!started || !!cell || status !== "in_progress"}
              >
                {cell}
              </button>
            );
          })}
        </div>
      </div>

      <div className="result">
        {started && status === "in_progress" && (player ? `You are ${player}` : "Your turn")}
        {started && resultOpen && (
          status === "draw" ? "It's a draw." :
          status === player ? "You win! 🎉" : "AI wins! 🤖"
        )}
      </div>

      {resultOpen && (
        <div className="overlay">
          <div className={`modal ${status === player ? "good" : status === ai ? "bad" : ""}`}>
            <div className="modal-title single">
              {status === "draw"
                ? "It's a draw."
                : status === player
                ? "You won! 🎉"
                : "AI wins — try again! 🤖"}
            </div>
            <div className="modal-actions">
              <button className="btn" onClick={resetAll}>Change Mark</button>
              <button className="btn ghost" onClick={resetAll}>Quit</button>
              <button className="btn primary" onClick={startGame}>Play Again</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
