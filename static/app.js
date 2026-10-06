/**
 * Connect 4 AlphaGo Studio - Core Interactive Application
 * Real-time MCTS evaluation, candidate move expected values,
 * PV ghost previews, blunder detection, and game tree scrubbing.
 */

// Application State
let gameState = {
  board: Array(6).fill(null).map(() => Array(7).fill(0)),
  to_play: 1,
  game_over: false,
  winner: 0,
  win_coords: null,
  history: [],
  player_red: 'human',
  player_yellow: 'alphazero',
  simulations: 128,
  analysis: null,
  is_thinking: false,
  sandbox_active: false,
};

let ghostPieces = []; // [{row, col, piece, step}]
let chartInstance = null;
let aiVsAiInterval = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initBoardGrid();
  initWinRateChart();
  bindEventListeners();
  fetchState();
});

// --- 1. Board Grid Initialization & Rendering ---

function initBoardGrid() {
  const grid = document.getElementById('boardGrid');
  grid.innerHTML = '';

  // 6 rows (0 is bottom in Connect 4, but visually row 5 is top row in CSS grid)
  for (let r = 5; r >= 0; r--) {
    for (let c = 0; c < 7; c++) {
      const slot = document.createElement('div');
      slot.className = 'board-slot';
      slot.dataset.row = r;
      slot.dataset.col = c;
      slot.addEventListener('click', () => handleSlotClick(c));
      grid.appendChild(slot);
    }
  }
}

function renderBoard(options = {}) {
  const slots = document.querySelectorAll('.board-slot');
  const newlyDropped = options.newlyDropped || [];

  slots.forEach(slot => {
    const r = parseInt(slot.dataset.row);
    const c = parseInt(slot.dataset.col);
    slot.innerHTML = '';

    const piece = gameState.board[r][c];
    if (piece === 1 || piece === 2) {
      const token = document.createElement('div');
      token.className = `disc-token ${piece === 1 ? 'red' : 'yellow'}`;

      // Apply tactile drop physics animation if newly dropped
      if (newlyDropped.some(d => d.row === r && d.col === c)) {
        token.classList.add('dropped');
      }

      // Check if winning piece
      if (gameState.win_coords && gameState.win_coords.some(([wr, wc]) => wr === r && wc === c)) {
        token.classList.add('winning');
      }
      slot.appendChild(token);
    } else {
      // Check if ghost piece should be rendered
      const ghost = ghostPieces.find(g => g.row === r && g.col === c);
      if (ghost) {
        const ghostToken = document.createElement('div');
        ghostToken.className = `disc-token ghost ${ghost.piece === 1 ? 'red' : 'yellow'}`;
        ghostToken.innerText = `+${ghost.step}`;
        slot.appendChild(ghostToken);
      }
    }
  });
}

// --- 2. Live Win Rate Meter & Chart.js ---

function initWinRateChart() {
  const ctx = document.getElementById('winRateChart').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['0'],
      datasets: [{
        label: 'Red Win Rate %',
        data: [50.0],
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.15)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: '#ef4444',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Red: ${ctx.parsed.y.toFixed(1)}% | Yellow: ${(100 - ctx.parsed.y).toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 10 } },
          title: { display: true, text: 'Move Ply', color: '#64748b', font: { size: 10 } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 10 }, stepSize: 25 },
        }
      },
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const index = elements[0].index;
          jumpToPly(index);
        }
      }
    }
  });
}

function updateWinRateMeter(redRate, yellowRate) {
  const redText = document.getElementById('winRateRedText');
  const yellowText = document.getElementById('winRateYellowText');
  const meterRed = document.getElementById('evalMeterRed');
  const meterYellow = document.getElementById('evalMeterYellow');
  const divider = document.getElementById('evalDivider');

  redText.innerText = `${redRate.toFixed(1)}%`;
  yellowText.innerText = `${yellowRate.toFixed(1)}%`;
  meterRed.style.width = `${redRate}%`;
  meterYellow.style.width = `${yellowRate}%`;
  divider.style.left = `${redRate}%`;
}

function updateChart() {
  if (!chartInstance) return;

  const labels = ['0'];
  const data = [50.0];

  gameState.history.forEach((m, idx) => {
    labels.push(`${idx + 1}`);
    data.push(m.win_rate_red);
  });

  chartInstance.data.labels = labels;
  chartInstance.data.datasets[0].data = data;
  chartInstance.update();
}

// --- 3. Candidate HUD & Expected Values Overlay ---

function renderCandidateHud() {
  const hud = document.getElementById('candidateHud');
  hud.innerHTML = '';

  const candidates = gameState.analysis?.candidates || [];
  const tactical = gameState.analysis?.tactical;

  for (let c = 0; c < 7; c++) {
    const card = document.createElement('div');
    card.className = 'candidate-hud-card';
    card.dataset.col = c;

    const cand = candidates.find(item => item.col === c);
    const isWin = tactical?.immediate_wins?.includes(c);
    const isThreat = tactical?.opponent_threats?.includes(c);
    const isForbidden = tactical?.forbidden_moves?.includes(c);

    if (cand) {
      if (cand.is_best) card.classList.add('best');

      let badgeHtml = '';
      if (isWin) badgeHtml = '<span class="hud-tag threat">WIN</span>';
      else if (isThreat) badgeHtml = '<span class="hud-tag threat">BLOCK</span>';
      else if (isForbidden) badgeHtml = '<span class="hud-tag forbidden">WARN</span>';
      else if (cand.is_best) badgeHtml = '<span class="hud-tag best">#1 BEST</span>';

      card.innerHTML = `
        <div class="hud-win-rate">${cand.win_rate.toFixed(0)}%</div>
        <div class="hud-meta">N: ${cand.visits}</div>
        <div class="hud-meta">P: ${(cand.prior * 100).toFixed(0)}%</div>
        ${badgeHtml}
      `;

      // Hover to preview this candidate's Principal Variation
      card.addEventListener('mouseenter', () => previewCandidatePv(cand));
      card.addEventListener('mouseleave', () => clearGhostPieces());
      card.addEventListener('click', () => handleSlotClick(c));
    } else {
      card.style.opacity = '0.3';
      card.innerHTML = `<div class="hud-meta">FULL</div>`;
    }

    hud.appendChild(card);
  }
}

function renderCandidateTable() {
  const tbody = document.getElementById('candidateTableBody');
  tbody.innerHTML = '';

  const candidates = gameState.analysis?.candidates || [];
  document.getElementById('candPlyTag').innerText = `Ply ${gameState.history.length}`;

  candidates.forEach(c => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:700; color:var(--accent);">Col ${c.col}</td>
      <td style="font-weight:700;">${c.win_rate.toFixed(1)}%</td>
      <td>${c.visits}</td>
      <td>${(c.visit_share * 100).toFixed(1)}%</td>
      <td>${(c.prior * 100).toFixed(1)}%</td>
      <td>${c.q_value > 0 ? '+' : ''}${c.q_value.toFixed(2)}</td>
      <td>${c.is_best ? '<span class="hud-tag best">BEST</span>' : '-'}</td>
    `;
    tr.addEventListener('mouseenter', () => previewCandidatePv(c));
    tr.addEventListener('mouseleave', () => clearGhostPieces());
    tbody.appendChild(tr);
  });
}

// --- 4. Principal Variation (PV) & Ghost Piece Forecast ---

function previewCandidatePv(cand) {
  if (!cand || !cand.pv || cand.pv.length === 0) return;

  ghostPieces = [];
  const tempBoard = gameState.board.map(row => [...row]);
  let currentPiece = gameState.to_play;

  cand.pv.forEach((col, idx) => {
    // Find next open row in tempBoard
    let row = -1;
    for (let r = 0; r < 6; r++) {
      if (tempBoard[r][col] === 0) {
        row = r;
        break;
      }
    }

    if (row !== -1) {
      tempBoard[row][col] = currentPiece;
      ghostPieces.push({
        row: row,
        col: col,
        piece: currentPiece,
        step: idx + 1,
      });
      currentPiece = 3 - currentPiece; // alternate
    }
  });

  renderBoard();
}

function clearGhostPieces() {
  ghostPieces = [];
  renderBoard();
}

function renderPvBanner() {
  const pvSeq = document.getElementById('pvSequence');
  const depthTag = document.getElementById('pvDepthTag');
  const pv = gameState.analysis?.principal_variation || [];

  depthTag.innerText = `Depth ${pv.length}`;
  if (pv.length === 0) {
    pvSeq.innerHTML = '<span class="text-muted">No tactical line predicted.</span>';
    return;
  }

  let html = '';
  let piece = gameState.to_play;
  pv.forEach((col, idx) => {
    const pColor = piece === 1 ? 'red' : 'yellow';
    html += `
      <div class="pv-pill">
        <span class="disc-sm ${pColor}"></span>
        <span>+${idx + 1}: Col ${col}</span>
      </div>
    `;
    piece = 3 - piece;
  });
  pvSeq.innerHTML = html;
}

// --- 5. Match Review & Move Quality Scrubber ---

function renderReviewList() {
  const container = document.getElementById('reviewList');
  document.getElementById('reviewCountTag').innerText = `${gameState.history.length} Moves`;

  if (gameState.history.length === 0) {
    container.innerHTML = '<div class="empty-state">No moves played yet. Start dropping pieces!</div>';
    return;
  }

  container.innerHTML = '';
  gameState.history.forEach((m, idx) => {
    const item = document.createElement('div');
    item.className = 'review-item';
    const pieceName = m.piece === 1 ? 'Red' : 'Yellow';
    const qualityBadge = m.quality_label ? 
      `<span class="review-badge" style="background-color: ${m.quality_color};">${m.quality_label}</span>` : '';

    item.innerHTML = `
      <div>
        <span style="font-weight:700; margin-right:8px;">#${m.ply}</span>
        <span class="disc-sm ${m.piece === 1 ? 'red' : 'yellow'}" style="display:inline-block; vertical-align:middle; margin-right:6px;"></span>
        <span>${pieceName} played Col ${m.col}</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        ${m.delta_win_rate > 0 ? `<span style="font-size:0.75rem; color:var(--text-dim); font-family:monospace;">-Δ${m.delta_win_rate}%</span>` : ''}
        ${qualityBadge}
      </div>
    `;

    item.addEventListener('click', () => jumpToPly(idx + 1));
    container.appendChild(item);
  });
}

// --- 6. API Interactions ---

async function fetchState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    console.error('Failed to fetch state:', err);
  }
}

async function handleSlotClick(col) {
  if (gameState.is_thinking || gameState.game_over) return;

  // 1. Instant client-side validation
  let openRow = -1;
  for (let r = 0; r < 6; r++) {
    if (gameState.board[r][col] === 0) {
      openRow = r;
      break;
    }
  }
  if (openRow === -1) return; // Column is full, reject click immediately

  // 2. OPTIMISTIC INSTANT LOCAL UPDATE (0ms tactile reaction!)
  const humanPiece = gameState.to_play;
  gameState.board[openRow][col] = humanPiece;
  const nextPiece = 3 - humanPiece;
  gameState.to_play = nextPiece;

  // Animate human disc dropping into place immediately
  renderBoard({ newlyDropped: [{ row: openRow, col: col }] });

  // If opponent is AI, immediately update status to calculating
  const isOpponentAi = (nextPiece === 1 ? gameState.player_red : gameState.player_yellow) !== 'human' 
                       && !gameState.sandbox_active;
  if (isOpponentAi) {
    setThinking(true);
  }

  // 3. Send move to server with auto_ai flag
  try {
    const res = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ col: col, auto_ai: isOpponentAi }),
    });
    if (!res.ok) {
      // Revert optimistic move on failure
      gameState.board[openRow][col] = 0;
      gameState.to_play = humanPiece;
      renderBoard();
      setThinking(false);
      return;
    }
    const data = await res.json();
    
    // If AI replied in the same roundtrip, animate its disc dropping
    if (data.last_ai_move) {
      updateUI(data, { newlyDropped: [data.last_ai_move] });
    } else {
      updateUI(data);
    }
  } catch (err) {
    console.error('Move error:', err);
    gameState.board[openRow][col] = 0;
    gameState.to_play = humanPiece;
    renderBoard();
  } finally {
    setThinking(false);
  }
}

async function triggerAiMove() {
  if (gameState.is_thinking || gameState.game_over) return;
  setThinking(true);
  try {
    const res = await fetch('/api/ai_move', { method: 'POST' });
    const data = await res.json();
    if (data.last_ai_move) {
      updateUI(data, { newlyDropped: [data.last_ai_move] });
    } else {
      updateUI(data);
    }
  } catch (err) {
    console.error('AI Move error:', err);
  } finally {
    setThinking(false);
  }
}

async function runDeepAnalyze() {
  setThinking(true);
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine_type: 'alphazero',
        simulations: 512,
        custom_board: gameState.board,
        to_play: gameState.to_play,
      }),
    });
    const analysis = await res.json();
    gameState.analysis = analysis;
    updateUI(gameState);
  } catch (err) {
    console.error('Deep analyze error:', err);
  } finally {
    setThinking(false);
  }
}

async function jumpToPly(ply) {
  try {
    const res = await fetch('/api/jump', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ply: ply }),
    });
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    console.error('Jump error:', err);
  }
}

async function undoMove() {
  try {
    const res = await fetch('/api/undo', { method: 'POST' });
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    console.error('Undo error:', err);
  }
}

async function resetGame() {
  const pRed = document.getElementById('playerRedSelect').value;
  const pYellow = document.getElementById('playerYellowSelect').value;
  const sims = parseInt(document.getElementById('simsSlider').value);

  try {
    const res = await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        player_red: pRed,
        player_yellow: pYellow,
        simulations: sims,
      }),
    });
    const data = await res.json();
    updateUI(data);
    checkAndTriggerAi();
  } catch (err) {
    console.error('Reset error:', err);
  }
}

async function toggleSandbox() {
  const newActive = !gameState.sandbox_active;
  try {
    const res = await fetch('/api/sandbox', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active: newActive }),
    });
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    console.error('Sandbox toggle error:', err);
  }
}

function checkAndTriggerAi() {
  if (gameState.game_over || gameState.sandbox_active) return;
  const currentTurnPlayer = gameState.to_play === 1 ? gameState.player_red : gameState.player_yellow;
  if (currentTurnPlayer !== 'human') {
    setTimeout(triggerAiMove, 300);
  }
}

function toggleAiVsAi() {
  const btn = document.getElementById('aiVsAiBtnText');
  if (aiVsAiInterval) {
    clearInterval(aiVsAiInterval);
    aiVsAiInterval = null;
    btn.innerText = 'AI vs AI Battle';
  } else {
    btn.innerText = '⏹ Stop AI Battle';
    aiVsAiInterval = setInterval(() => {
      if (gameState.game_over) {
        clearInterval(aiVsAiInterval);
        aiVsAiInterval = null;
        btn.innerText = 'AI vs AI Battle';
        return;
      }
      triggerAiMove();
    }, 600);
  }
}

// --- 7. UI Update Aggregator ---

function updateUI(data, options = {}) {
  gameState = data;

  // 1. Board & Pieces
  renderBoard(options);

  // 2. Win Rate Bar
  const redRate = data.analysis?.win_rate_red ?? 50.0;
  const yellowRate = data.analysis?.win_rate_yellow ?? 50.0;
  updateWinRateMeter(redRate, yellowRate);

  if (data.analysis?.eval_source) {
    document.getElementById('evalSourceBadge').innerText = data.analysis.eval_source;
  }

  // 3. Status text
  const statusBadge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
  if (data.game_over) {
    statusBadge.className = 'status-badge ready';
    statusText.innerText = data.winner === 1 ? 'Red Won!' : (data.winner === 2 ? 'Yellow Won!' : 'Match Drawn');
  } else if (data.sandbox_active) {
    statusBadge.className = 'status-badge thinking';
    statusText.innerText = 'Sandbox Mode (What-If)';
  } else {
    statusBadge.className = 'status-badge ready';
    statusText.innerText = data.to_play === 1 ? 'Turn: Red (P1)' : 'Turn: Yellow (P2)';
  }

  // 4. Candidate Overlays & Tables
  renderCandidateHud();
  renderCandidateTable();

  // 5. Principal Variation
  renderPvBanner();

  // 6. Timeline Chart & Review List
  updateChart();
  renderReviewList();

  // 7. Sandbox button label
  document.getElementById('sandboxBtnText').innerText = data.sandbox_active ? 'Sandbox: ON' : 'Sandbox: OFF';
}

function setThinking(thinking) {
  gameState.is_thinking = thinking;
  const statusBadge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
  if (thinking) {
    statusBadge.className = 'status-badge thinking';
    statusText.innerText = 'AI Calculating MCTS...';
  }
}

// --- 8. Event Bindings ---

function bindEventListeners() {
  document.getElementById('newGameBtn').addEventListener('click', resetGame);
  document.getElementById('undoBtn').addEventListener('click', undoMove);
  document.getElementById('aiMoveBtn').addEventListener('click', triggerAiMove);
  document.getElementById('deepThinkBtn').addEventListener('click', runDeepAnalyze);
  document.getElementById('sandboxToggleBtn').addEventListener('click', toggleSandbox);
  document.getElementById('aiVsAiBtn').addEventListener('click', toggleAiVsAi);

  const slider = document.getElementById('simsSlider');
  const simsVal = document.getElementById('simsValue');
  slider.addEventListener('input', (e) => {
    simsVal.innerText = e.target.value;
    gameState.simulations = parseInt(e.target.value);
  });
}
