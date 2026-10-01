/**
 * app.js — VERTO frontend logic
 *
 * Communicates with the FastAPI backend at /api/*
 * Stores query history in localStorage.
 */

const BASE_URL = (window.VERTO_API_BASE || localStorage.getItem('verto_api_base') || '').replace(/\/+$/, '');

const API = {
  health:    `${BASE_URL}/api/health`,
  databases: `${BASE_URL}/api/databases`,
  query:     `${BASE_URL}/api/query`,
};

// Declared before `state`, which calls loadHistory() while it is being built.
const HISTORY_KEY = 'verto_history';
const MAX_HISTORY = 50;
const MAX_ROWS    = 200;


// ── State ─────────────────────────────────────────────────────────────────

const state = {
  schemaLinking: true,
  errorRecovery: true,
  history: loadHistory(),
  sessionQueries: 0,
  sessionSuccess:  0,
  sessionTimes:   [],
  dbCount: 0,
  lastSql: '',
};

// ── DOM refs ──────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const ui = {
  statusDot:         $('status-dot'),
  statusText:        $('status-text'),
  runtimeInfo:       $('runtime-info'),
  dbSelect:          $('db-select'),
  toggleSchema:      $('toggle-schema'),
  toggleRecovery:    $('toggle-recovery'),
  questionInput:     $('question-input'),
  shortcutMod:       $('shortcut-mod'),
  submitBtn:         $('submit-btn'),
  btnLabel:          $('btn-label'),
  resultPanel:       $('result-panel'),
  resultPlaceholder: $('result-placeholder'),
  resultContent:     $('result-content'),
  resultMeta:        $('result-meta'),
  resultAlert:       $('result-alert'),
  sqlSection:        $('sql-section'),
  sqlOutput:         $('sql-output'),
  copySqlBtn:        $('copy-sql-btn'),
  copyLabel:         $('copy-label'),
  rowsSection:       $('rows-section'),
  resultBody:        $('result-body'),
  historyList:       $('history-list'),
  statQueries:       $('stat-queries'),
  statSuccess:       $('stat-success'),
  statAvgTime:       $('stat-avg-time'),
  statDbs:           $('stat-dbs'),
};

// ── Boot ──────────────────────────────────────────────────────────────────

async function boot() {
  // Show the shortcut the way this platform writes it; the handler accepts either key.
  if (/mac|iphone|ipad/i.test(navigator.userAgentData?.platform || navigator.platform || '')) {
    ui.shortcutMod.textContent = '⌘';
  }
  renderHistory();
  updateStats();
  attachListeners();
  await checkHealth();
  await loadDatabases();
}

// ── Health check ──────────────────────────────────────────────────────────

async function checkHealth() {
  setStatus('', 'Connecting…');
  try {
    const res = await fetch(API.health);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const model = data.checkpoint_loaded ? 'T5-large + LoRA' : 'T5-large, no checkpoint';
    ui.runtimeInfo.textContent = `${model} · ${data.device}`;
    setStatus('online', 'Ready');
  } catch (err) {
    setStatus('error', 'Server offline');
    console.error('Health check failed:', err);
  }
}

// ── Database loader ───────────────────────────────────────────────────────

async function loadDatabases() {
  try {
    const res  = await fetch(API.databases);
    const data = await res.json();
    const dbs  = data.databases || [];

    ui.dbSelect.innerHTML = '<option value="" disabled selected>Choose a database</option>';
    dbs.forEach(db => {
      const opt = document.createElement('option');
      opt.value       = db;
      opt.textContent = db;
      ui.dbSelect.appendChild(opt);
    });

    // Restore last-used DB from history
    const lastDb = state.history[0]?.db_id;
    if (lastDb && dbs.includes(lastDb)) ui.dbSelect.value = lastDb;

    state.dbCount = dbs.length;
    updateStats();
    updateSubmitState();
  } catch (err) {
    ui.dbSelect.innerHTML = '<option value="" disabled selected>Couldn’t load databases</option>';
    console.error('loadDatabases:', err);
  }
}

// ── Listeners ─────────────────────────────────────────────────────────────

function attachListeners() {
  // Pipeline checkboxes
  ui.toggleSchema.addEventListener('change', () => {
    state.schemaLinking = ui.toggleSchema.checked;
  });
  ui.toggleRecovery.addEventListener('change', () => {
    state.errorRecovery = ui.toggleRecovery.checked;
  });

  // Auto-resize textarea
  ui.questionInput.addEventListener('input', () => {
    ui.questionInput.style.height = 'auto';
    ui.questionInput.style.height = Math.min(ui.questionInput.scrollHeight + 2, 200) + 'px';
    updateSubmitState();
  });

  // Ctrl+Enter to submit
  ui.questionInput.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!ui.submitBtn.disabled) runQuery();
    }
  });

  ui.dbSelect.addEventListener('change', updateSubmitState);
  ui.submitBtn.addEventListener('click', runQuery);
  ui.copySqlBtn.addEventListener('click', copySQL);
}

function updateSubmitState() {
  const ready = ui.dbSelect.value && ui.questionInput.value.trim().length > 0;
  ui.submitBtn.disabled = !ready;
}

// ── Query execution ───────────────────────────────────────────────────────

async function runQuery() {
  const question = ui.questionInput.value.trim();
  const db_id    = ui.dbSelect.value;
  if (!question || !db_id) return;

  setLoading(true);

  const t0 = performance.now();

  try {
    const res = await fetch(API.query, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        db_id,
        use_schema_linking: state.schemaLinking,
        use_error_recovery:  state.errorRecovery,
      }),
    });

    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResult(data, db_id, elapsed);

    // Update session stats
    state.sessionQueries++;
    if (!data.error) state.sessionSuccess++;
    state.sessionTimes.push(parseFloat(elapsed));
    updateStats();

    // Save to history
    addToHistory({ question, db_id, sql: data.sql, error: data.error, attempts: data.attempts, elapsed });
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
}

// ── Result rendering ──────────────────────────────────────────────────────

function renderResult(data, db_id, elapsed) {
  const failed   = Boolean(data.error);
  const attempts = data.attempts || 0;
  const rows     = Array.isArray(data.result) ? data.result : [];

  const facts = [db_id];
  if (!failed) facts.push(`${rows.length.toLocaleString()} ${rows.length === 1 ? 'row' : 'rows'}`);
  facts.push(`${elapsed} s`);

  ui.resultMeta.innerHTML = `
    <span class="tag ${failed ? 'failed' : 'ok'}">${outcomeLabel(failed, attempts)}</span>
    <span class="summary-text">${escHtml(facts.join(' · '))}</span>
    <span class="summary-score">Attempts ${attemptMarks(attempts, failed)}<strong>${attempts}</strong></span>`;

  ui.resultAlert.textContent = failed ? data.error : '';
  ui.resultAlert.classList.toggle('hidden', !failed);

  // SQL
  state.lastSql = data.sql || '';
  ui.sqlOutput.innerHTML = data.sql ? highlightSQL(data.sql) : '<span class="sql-none">No SQL was generated</span>';
  ui.copySqlBtn.disabled = !data.sql;
  ui.sqlSection.classList.remove('hidden');

  // Rows
  ui.rowsSection.classList.toggle('hidden', failed);
  if (!failed) renderTable(rows, data.columns);

  showResult();
}

function outcomeLabel(failed, attempts) {
  if (failed) return 'Failed';
  if (attempts <= 1) return 'Ran';
  return attempts === 2 ? 'Ran after a retry' : `Ran after ${attempts - 1} retries`;
}

// One square per generation attempt: red for a try that failed, green for the one that ran.
function attemptMarks(attempts, failed) {
  if (!attempts) return '';
  const marks = Array.from({ length: attempts }, (_, i) =>
    `<span class="mark ${failed || i < attempts - 1 ? 'fail' : 'ok'}"></span>`).join('');
  const label = failed
    ? `${attempts} failed ${attempts === 1 ? 'attempt' : 'attempts'}`
    : attempts === 1 ? 'Ran on the first attempt' : `Ran on attempt ${attempts}`;
  return `<span class="marks" role="img" aria-label="${label}">${marks}</span>`;
}

function renderTable(rows, columns) {
  if (rows.length === 0) {
    ui.resultBody.innerHTML = '<p class="empty">The query ran and returned no rows</p>';
    return;
  }

  const shown   = rows.slice(0, MAX_ROWS);
  const cells   = row => (Array.isArray(row) ? row : [row]);
  const width   = Math.max(...shown.map(row => cells(row).length));
  const numeric = Array.from({ length: width }, (_, c) => {
    const values = shown.map(row => cells(row)[c]).filter(v => v !== null && v !== undefined);
    return values.length > 0 && values.every(v => typeof v === 'number');
  });

  const label = c => escHtml((columns && columns[c]) || `Column ${c + 1}`);
  const head  = numeric.map((isNum, c) => `<th scope="col"${isNum ? ' class="num"' : ''}>${label(c)}</th>`).join('');
  const body = shown.map(row => `<tr>${numeric.map((isNum, c) => {
    const v    = cells(row)[c];
    const text = v === null || v === undefined ? '<span class="null">NULL</span>' : escHtml(String(v));
    return `<td${isNum ? ' class="num"' : ''}>${text}</td>`;
  }).join('')}</tr>`).join('');

  const note = rows.length > MAX_ROWS
    ? `<p class="help table-note">Showing the first ${MAX_ROWS} of ${rows.length.toLocaleString()} rows</p>`
    : '';

  ui.resultBody.innerHTML = `
    <div class="table-wrap rows-wrap">
      <table class="table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
    </div>${note}`;
}

function renderError(message) {
  ui.resultMeta.innerHTML = '<span class="tag failed">Request failed</span>';
  ui.resultAlert.textContent = message;
  ui.resultAlert.classList.remove('hidden');
  state.lastSql = '';
  ui.sqlSection.classList.add('hidden');
  ui.rowsSection.classList.add('hidden');
  showResult();
}

function showResult() {
  ui.resultPlaceholder.classList.add('hidden');
  ui.resultContent.classList.remove('hidden', 'enter');
  void ui.resultContent.offsetWidth; // restart the entrance
  ui.resultContent.classList.add('enter');
  ui.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── SQL Syntax highlighting ───────────────────────────────────────────────

const SQL_KW = /^(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AS|AND|OR|NOT|IN|IS|NULL|ORDER|BY|GROUP|HAVING|LIMIT|OFFSET|UNION|ALL|DISTINCT|INSERT|UPDATE|DELETE|SET|VALUES|CREATE|TABLE|INDEX|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|ASC|DESC)$/i;
const SQL_FN = /^(count|sum|avg|min|max|coalesce|ifnull|upper|lower|substr|length|round|abs|cast|strftime|date|datetime)$/i;
const SQL_NUM = /^\d+(\.\d+)?$/;
const SQL_OP  = /^([=<>!]+|[+\-*/%])$/;

function highlightSQL(sql) {
  const tokens = [];
  // The last alternative takes any other character, so nothing in the query is dropped from view.
  const tokenRegex = /('[^']*'|"[^"]*"|\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+(?:\.\d+)?\b|[=<>!+\-*/%]+|\s+|[().,;]|[\s\S])/g;

  let match;
  while ((match = tokenRegex.exec(sql)) !== null) {
    const token = match[0];
    if (token.startsWith("'") || token.startsWith('"')) {
      tokens.push(`<span class="sql-str">${escHtml(token)}</span>`);
    } else if (SQL_KW.test(token)) {
      tokens.push(`<span class="sql-kw">${token.toUpperCase()}</span>`);
    } else if (SQL_FN.test(token)) {
      tokens.push(`<span class="sql-fn">${token}</span>`);
    } else if (SQL_NUM.test(token)) {
      tokens.push(`<span class="sql-num">${token}</span>`);
    } else if (SQL_OP.test(token)) {
      tokens.push(`<span class="sql-op">${escHtml(token)}</span>`);
    } else {
      tokens.push(escHtml(token));
    }
  }

  return tokens.length > 0 ? tokens.join('') : escHtml(sql);
}

// ── Copy SQL ──────────────────────────────────────────────────────────────

async function copySQL() {
  if (!state.lastSql) return;
  try {
    await navigator.clipboard.writeText(state.lastSql); // the query exactly as the model wrote it
    flashCopyLabel('Copied');
  } catch {
    flashCopyLabel('Copy failed');
  }
}

function flashCopyLabel(text) {
  ui.copyLabel.textContent = text;
  setTimeout(() => { ui.copyLabel.textContent = 'Copy'; }, 1800);
}

// ── History ───────────────────────────────────────────────────────────────

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch { return []; }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(0, MAX_HISTORY)));
  } catch {}
}

function addToHistory(entry) {
  state.history.unshift({ ...entry, ts: Date.now() });
  if (state.history.length > MAX_HISTORY) state.history.pop();
  saveHistory();
  renderHistory();
}

function renderHistory() {
  if (state.history.length === 0) {
    ui.historyList.innerHTML = '<p class="empty">Questions you run appear here</p>';
    return;
  }

  const total = state.history.length;
  const rows  = state.history.map((item, idx) => `
    <tr>
      <td class="num col-num">${total - idx}</td>
      <td class="cell-mono col-db">${escHtml(item.db_id)}</td>
      <td><button type="button" class="history-q" data-idx="${idx}" title="Load this question">${escHtml(item.question)}</button></td>
      <td class="cell-sql col-sql"><span class="clip">${escHtml(item.sql || '')}</span></td>
      <td class="cell-marks">${attemptMarks(item.attempts ?? 1, Boolean(item.error))}</td>
      <td class="num">${escHtml(String(item.elapsed))} s</td>
    </tr>`).join('');

  ui.historyList.innerHTML = `
    <div class="table-wrap flush">
      <table class="table history-table">
        <thead>
          <tr>
            <th scope="col" class="num col-num">#</th>
            <th scope="col" class="col-db">Database</th>
            <th scope="col">Question</th>
            <th scope="col" class="col-sql">SQL</th>
            <th scope="col">Attempts</th>
            <th scope="col" class="num">Time</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  ui.historyList.querySelectorAll('.history-q').forEach(btn => {
    btn.addEventListener('click', () => {
      const item = state.history[+btn.dataset.idx];
      if (!item) return;
      ui.dbSelect.value      = item.db_id;
      ui.questionInput.value = item.question;
      ui.questionInput.dispatchEvent(new Event('input'));
      ui.questionInput.focus();
    });
  });
}

// ── Stats ─────────────────────────────────────────────────────────────────

function updateStats() {
  ui.statQueries.textContent = state.sessionQueries;
  ui.statDbs.textContent = state.dbCount || '—';

  if (state.sessionQueries > 0) {
    const rate = Math.round((state.sessionSuccess / state.sessionQueries) * 100);
    ui.statSuccess.textContent = `${rate}%`;
  } else {
    ui.statSuccess.textContent = '—';
  }

  if (state.sessionTimes.length > 0) {
    const avg = state.sessionTimes.reduce((a, b) => a + b, 0) / state.sessionTimes.length;
    ui.statAvgTime.innerHTML = `${avg.toFixed(2)}<span class="unit">s</span>`;
  } else {
    ui.statAvgTime.textContent = '—';
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────

function setStatus(variant, text) {
  ui.statusDot.className    = variant ? `status-dot ${variant}` : 'status-dot';
  ui.statusText.textContent = text;
}

function setLoading(loading) {
  ui.submitBtn.disabled   = loading;
  ui.submitBtn.classList.toggle('loading', loading);
  ui.btnLabel.textContent = loading ? 'Running…' : 'Run query';
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Init ──────────────────────────────────────────────────────────────────
boot();
