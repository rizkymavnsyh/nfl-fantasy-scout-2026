'use strict';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (n, d = 1) => (n == null || Number.isNaN(Number(n)) ? '—' : Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (n, d = 1) => fmt(n * 100, d) + '%';

const TEAM_COLORS = {
  ARI: '#97233F', ATL: '#A71930', BAL: '#241773', BUF: '#00338D', CAR: '#0085CA', CHI: '#0B162A', CIN: '#FB4F14', CLE: '#311D00',
  DAL: '#003594', DEN: '#FB4F14', DET: '#0076B6', GB: '#203731', HOU: '#03202F', IND: '#002C5F', JAX: '#006778', KC: '#E31837',
  LV: '#000000', LAC: '#0080C6', LA: '#003594', MIA: '#008E97', MIN: '#4F2683', NE: '#002244', NO: '#D3BC8D', NYG: '#0B2265',
  NYJ: '#125740', PHI: '#004C54', PIT: '#FFB612', SF: '#AA0000', SEA: '#002244', TB: '#D50A0A', TEN: '#4B92DB', WAS: '#5A1414',
};
const teamColor = (team) => TEAM_COLORS[team] || '#0b2545';

const PAGE_SIZE = 15;
const RANGE_MAX = 45;
const state = {
  data: null, archive: false, tab: 'rankings', position: 'ALL', search: '', sort: 'prediction',
  pool: 'fantasy', game: 'all', page: 0, selected: null, roster: new Set(),
};

const availabilityOK = (p, now = Date.now()) => p.lineup_eligible === true && !!p.availability_as_of_utc
  && now - Date.parse(p.availability_as_of_utc) >= 0 && now - Date.parse(p.availability_as_of_utc) <= 48 * 3600000;
const isUpcoming = (p, now = Date.now()) => !p.kickoff_utc || Date.parse(p.kickoff_utc) > now;

function optimizeLineup(roster) {
  const by = { QB: [], RB: [], WR: [], TE: [] };
  for (const p of roster) by[p.position]?.push(p);
  for (const ps of Object.values(by)) ps.sort((a, b) => b.prediction - a.prediction);
  if (!by.QB.length || by.RB.length < 2 || by.WR.length < 2 || !by.TE.length) return null;
  let best = null;
  for (const flex of roster.filter((p) => p.position !== 'QB')) {
    const selected = [{ slot: 'QB', p: by.QB[0] }];
    for (const [pos, n] of [['RB', 2], ['WR', 2], ['TE', 1]]) {
      by[pos].filter((p) => p.id !== flex.id).slice(0, n).forEach((p, i) => selected.push({ slot: n > 1 ? pos + (i + 1) : pos, p }));
    }
    if (selected.length !== 6) continue;
    selected.push({ slot: 'FLEX', p: flex });
    const total = selected.reduce((s, r) => s + r.p.prediction, 0);
    if (!best || total > best.total) best = { selected, total };
  }
  return best;
}

// Exposed for src/test_lineup.cjs, which runs this file without a DOM.
window.FantasyScout = { optimizeLineup, isUpcoming, availabilityOK };

// Archived pages replay the moment the snapshot was published: after the forecast and roster checks.
const clock = () => (state.archive
  ? Math.max(Date.parse(state.data.forecast_snapshot.as_of_utc), Date.parse(state.data.availability?.checked_at_utc || 0) || 0)
  : Date.now());
const player = (id) => state.data.players.find((p) => p.id === id);
const kickoff = (iso, opts = { weekday: 'short', hour: 'numeric', minute: '2-digit' }) => (iso ? new Date(iso).toLocaleString('en-US', opts) : '');

function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), 3200);
}

function saveRoster() {
  try { localStorage.setItem('fantasy-scout-roster', JSON.stringify([...state.roster])); } catch { /* storage unavailable */ }
  $('roster-count').textContent = state.roster.size;
}

function setTab(name) {
  state.tab = name;
  document.querySelectorAll('.panel').forEach((el) => { el.hidden = el.id !== `tab-${name}`; });
  document.querySelectorAll('.tab').forEach((el) => el.classList.toggle('is-active', el.dataset.tab === name));
  if (name === 'lineup') renderRoster();
  if (name === 'compare') renderCompare();
}

/* ---------- Status helpers ---------- */

function injuryFlag(p) {
  const s = (p.injury_status || '').toLowerCase();
  if (s === 'out') return 'Out';
  if (s === 'doubtful') return 'Doubtful';
  if (s === 'questionable') return 'Q';
  if (!availabilityOK(p, clock())) return 'Check';
  return '';
}

function availabilityText(p) {
  const injury = { 'Not published': 'not yet published', 'Not listed': 'not listed', 'Conflicting reports': 'conflicting reports' }[p.injury_status] || p.injury_status || 'unavailable';
  return `${p.availability_note || 'Roster not checked'}. Week ${p.week} injury report: ${injury}.`;
}

function gameLines() {
  const lines = {};
  for (const p of state.data.players) {
    if (!p.game_id || p.implied_total == null) continue;
    const g = (lines[p.game_id] ||= {});
    g[p.team] = { implied: p.implied_total, spread: p.spread, total: p.total_line };
  }
  return lines;
}

function lineText(f, lines) {
  const g = lines[f.id];
  if (!g || !g[f.home] || !g[f.away]) return 'Line not posted';
  const total = g[f.home].total ?? g[f.home].implied + g[f.away].implied;
  const fav = g[f.home].spread >= 0 ? f.home : f.away;
  const spread = Math.abs(g[f.home].spread);
  return `${spread === 0 ? 'PK' : fav + ' −' + fmt(spread)} · O/U ${fmt(total)}`;
}

/* ---------- Ticker ---------- */

function renderTicker() {
  const lines = gameLines();
  const now = clock();
  const games = state.data.fixtures.filter((f) => isUpcoming(f, now));
  $('ticker').innerHTML = games.map((f) => `
    <button class="game ${state.game === f.id ? 'is-on' : ''}" data-game="${esc(f.id)}" title="Show players in this game">
      <span class="game-time">${f.week !== state.data.forecast_snapshot.next_week ? 'W' + f.week + ' · ' : ''}${esc(kickoff(f.kickoff_utc))}</span>
      <span class="game-teams">${esc(f.away)} @ ${esc(f.home)}</span>
      <span class="game-line">${esc(lineText(f, lines))}</span>
    </button>`).join('');
  $('ticker').querySelectorAll('[data-game]').forEach((b) => b.addEventListener('click', () => {
    state.game = state.game === b.dataset.game ? 'all' : b.dataset.game;
    $('game-filter').value = state.game;
    state.page = 0;
    setTab('rankings');
    renderTicker();
    renderBoard(true);
  }));
  $('game-filter').innerHTML = '<option value="all">All games</option>' + games.map((f) => `<option value="${esc(f.id)}">W${f.week} · ${esc(f.away)} @ ${esc(f.home)}</option>`).join('');
  $('game-filter').value = games.some((f) => f.id === state.game) ? state.game : 'all';
}

/* ---------- Rankings ---------- */

function filteredPlayers() {
  const now = clock();
  const q = state.search.trim().toLowerCase();
  const ps = state.data.players.filter((p) => (state.pool === 'all' || p.fantasy_pool !== false)
    && isUpcoming(p, now)
    && (state.position === 'ALL' || p.position === state.position)
    && (state.game === 'all' || p.game_id === state.game)
    && (!q || `${p.name} ${p.team} ${p.position}`.toLowerCase().includes(q)));
  const key = state.sort;
  return ps.sort((a, b) => {
    if (key === 'name') return a.name.localeCompare(b.name);
    if (key === 'delta') return (b.prediction - b.average) - (a.prediction - a.average);
    return (b[key] ?? -1) - (a[key] ?? -1);
  });
}

function rangeBar(p) {
  const x = (v) => Math.min(100, Math.max(0, v / RANGE_MAX * 100));
  return `<div class="range" title="${fmt(p.low)}–${fmt(p.high)} PPR">
    <span class="range-track"></span>
    <span class="range-band" style="left:${x(p.low)}%;width:${x(p.high) - x(p.low)}%"></span>
    <span class="range-dot" style="left:${x(p.prediction)}%"></span>
  </div>`;
}

function renderBoard(selectFirst = false) {
  const ps = filteredPlayers();
  const pages = Math.max(1, Math.ceil(ps.length / PAGE_SIZE));
  state.page = Math.max(0, Math.min(state.page, pages - 1));
  const start = state.page * PAGE_SIZE;
  const visible = ps.slice(start, start + PAGE_SIZE);
  if (selectFirst && ps.length) state.selected = ps[0].id;
  $('page-info').textContent = ps.length ? `${start + 1}–${Math.min(start + PAGE_SIZE, ps.length)} of ${ps.length} players` : 'No players';
  $('prev-page').disabled = state.page === 0;
  $('next-page').disabled = start + PAGE_SIZE >= ps.length;
  $('rank-body').innerHTML = visible.length ? visible.map((p, i) => {
    const flag = injuryFlag(p);
    return `<tr class="${state.selected === p.id ? 'is-selected' : ''}" data-id="${esc(p.id)}" tabindex="0">
      <td><input type="checkbox" data-roster="${esc(p.id)}" aria-label="Add ${esc(p.name)} to roster" ${state.roster.has(p.id) ? 'checked' : ''}></td>
      <td class="rank">${start + i + 1}</td>
      <td><div class="who"><span class="team-bar" style="background:${teamColor(p.team)}"></span><span>
        <span class="who-name">${esc(p.name)}${flag ? `<span class="flag">${flag}</span>` : ''}</span>
        <span class="who-sub">${esc(p.position)} · ${esc(p.team)}</span></span></div></td>
      <td class="muted">${p.home ? 'vs' : '@'} ${esc(p.opponent)}</td>
      <td class="num"><span class="proj">${fmt(p.prediction)}</span></td>
      <td class="col-range">${rangeBar(p)}</td>
      <td class="num muted">${fmt(p.average)}</td>
      <td class="num muted col-hide">${fmt(p.implied_total)}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="8" class="empty">No players match. Try another name, team or position.</td></tr>';
  $('rank-body').querySelectorAll('tr[data-id]').forEach((row) => {
    const pick = () => { state.selected = row.dataset.id; renderBoard(); renderCard(); };
    row.addEventListener('click', (e) => { if (e.target.type !== 'checkbox') pick(); });
    row.addEventListener('keydown', (e) => { if (e.target.type !== 'checkbox' && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); pick(); } });
  });
  $('rank-body').querySelectorAll('[data-roster]').forEach((el) => el.addEventListener('change', () => toggleRoster(el.dataset.roster, el.checked)));
  if (selectFirst) renderCard();
}

function historyChart(p) {
  const hist = p.history || [];
  const W = 324, H = 132, base = 104, top = 12;
  const max = Math.max(10, p.high, ...hist.map((h) => h.points)) * 1.08;
  const y = (v) => base - Math.max(0, v) / max * (base - top);
  const n = hist.length + 1;
  const step = (W - 20) / n;
  const bw = Math.min(30, step * .62);
  const cx = (i) => 10 + step * i + step / 2;
  const bars = hist.map((h, i) => `
    <rect x="${cx(i) - bw / 2}" y="${y(h.points)}" width="${bw}" height="${base - y(h.points)}" fill="#c7d3e3"/>
    <text x="${cx(i)}" y="${y(h.points) - 4}" text-anchor="middle" font-size="10" fill="#5d6675">${fmt(h.points, 0)}</text>
    <text x="${cx(i)}" y="${base + 15}" text-anchor="middle" font-size="10" fill="#8a93a1">W${h.week}</text>`).join('');
  const px = cx(hist.length);
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Recent PPR points and week ${p.week} projection">
    <line x1="4" x2="${W - 4}" y1="${base}" y2="${base}" stroke="#b9bec7"/>
    ${bars}
    <line x1="${px}" x2="${px}" y1="${y(p.high)}" y2="${y(p.low)}" stroke="#c8102e" stroke-width="1.5"/>
    <line x1="${px - 6}" x2="${px + 6}" y1="${y(p.high)}" y2="${y(p.high)}" stroke="#c8102e" stroke-width="1.5"/>
    <line x1="${px - 6}" x2="${px + 6}" y1="${y(p.low)}" y2="${y(p.low)}" stroke="#c8102e" stroke-width="1.5"/>
    <rect x="${px - bw / 2}" y="${y(p.prediction)}" width="${bw}" height="${base - y(p.prediction)}" fill="#c8102e" opacity=".9"/>
    <text x="${px}" y="${base + 15}" text-anchor="middle" font-size="10" font-weight="600" fill="#c8102e">W${p.week}</text>
  </svg>`;
}

function usage(p) {
  const items = p.position === 'QB'
    ? [['Pass yds / game', fmt(p.passing_yards)], ['Rush yds / game', fmt(p.rushing_yards)]]
    : p.position === 'RB'
      ? [['Carries / game', fmt(p.carries)], ['Targets / game', fmt(p.targets)]]
      : [['Targets / game', fmt(p.targets)], ['Rec yds / game', fmt(p.receiving_yards)]];
  items.push(['Snap share', p.prior_snap_pct == null ? '—' : pct(p.prior_snap_pct, 0)]);
  items.push(['Team implied total', fmt(p.implied_total)]);
  items.push(['5-game average', fmt(p.average)]);
  items.push([`Opp. allows to ${p.position}`, fmt(p.opponent_allowed)]);
  return `<dl class="kv">${items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>`;
}

function cardHTML(p, { button = true, pick = '' } = {}) {
  const ok = availabilityOK(p, clock());
  return `<div class="card-top" style="border-top-color:${pick ? '#c8102e' : teamColor(p.team)}">
      ${pick ? `<span class="pick-tag">${pick}</span>` : ''}
      <div class="card-meta">${esc(p.position)} · ${esc(p.team)}</div>
      <h3>${esc(p.name)}</h3>
      <div class="card-game">Week ${p.week} · ${p.home ? 'vs' : '@'} ${esc(p.opponent)} · ${esc(kickoff(p.kickoff_utc))}</div>
    </div>
    <div class="card-proj"><strong>${fmt(p.prediction)}</strong><span>80% range<b>${fmt(p.low)} – ${fmt(p.high)}</b></span></div>
    <div class="card-section"><p class="label">Recent games → projection</p>${historyChart(p)}</div>
    <div class="card-section"><p class="label">Pre-game inputs · last 3 games</p>${usage(p)}</div>
    <div class="card-section"><p class="status ${ok ? '' : 'is-warn'}">${esc(availabilityText(p))} ${esc(p.prior_injury_note || '')}.${ok ? '' : ' Excluded from the lineup optimizer.'}</p></div>
    ${button ? `<div class="card-section"><button class="btn" id="card-add">${state.roster.has(p.id) ? 'Remove from roster' : 'Add to roster'}</button></div>` : ''}`;
}

function renderCard() {
  const p = player(state.selected);
  if (!p) { $('player-card').innerHTML = '<p class="empty">Select a player.</p>'; return; }
  $('player-card').innerHTML = cardHTML(p);
  $('card-add').addEventListener('click', () => toggleRoster(p.id, !state.roster.has(p.id)));
}

function toggleRoster(id, on) {
  if (on) state.roster.add(id); else state.roster.delete(id);
  saveRoster();
  renderBoard();
  renderCard();
  if (state.tab === 'lineup') renderRoster();
  else $('depth-chart').innerHTML = '<p class="empty">Roster changed. Press “Optimize lineup” to recalculate.</p>';
}

/* ---------- Start / Sit ---------- */

function versusChart(a, b) {
  const W = 420, H = 92, L = 10, R = 10;
  const max = Math.ceil(Math.max(a.high, b.high, 20) / 5) * 5;
  const x = (v) => L + Math.max(0, v) / max * (W - L - R);
  const row = (p, yy, color) => `
    <line x1="${x(p.low)}" x2="${x(p.high)}" y1="${yy}" y2="${yy}" stroke="${color}" stroke-width="8" opacity=".25"/>
    <line x1="${x(p.prediction)}" x2="${x(p.prediction)}" y1="${yy - 9}" y2="${yy + 9}" stroke="${color}" stroke-width="3"/>
    <text x="${x(p.prediction)}" y="${yy - 13}" text-anchor="middle" font-size="11" font-weight="600" fill="${color}">${esc(p.name.split(' ').slice(-1)[0])} ${fmt(p.prediction)}</text>`;
  const ticks = Array.from({ length: max / 5 + 1 }, (_, i) => i * 5).map((t) => `
    <line x1="${x(t)}" x2="${x(t)}" y1="74" y2="78" stroke="#b9bec7"/>
    <text x="${x(t)}" y="89" text-anchor="middle" font-size="9" fill="#8a93a1">${t}</text>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Projection ranges">
    <line x1="${L}" x2="${W - R}" y1="74" y2="74" stroke="#b9bec7"/>${ticks}
    ${row(a, 30, '#0b2545')}${row(b, 58, '#c8102e')}
  </svg>`;
}

function renderCompare() {
  const a = player($('compare-a').value), b = player($('compare-b').value);
  if (!a || !b) return;
  if (a.id === b.id) { $('compare-result').innerHTML = '<p class="empty">Pick two different players.</p>'; return; }
  const now = clock();
  const best = a.prediction >= b.prediction ? a : b;
  const other = best === a ? b : a;
  const diff = best.prediction - other.prediction;
  const overlap = Math.max(a.low, b.low) <= Math.min(a.high, b.high);
  const sameWeek = a.week === b.week;
  const bothOK = availabilityOK(a, now) && availabilityOK(b, now);
  const lastName = best.name.split(' ').slice(-1)[0];
  const tag = diff < .05 ? '' : diff < 1.5 ? 'Lean' : 'Start';
  const headline = !sameWeek ? 'Different weeks' : !bothOK ? 'Check availability first' : !tag ? 'Coin flip' : tag === 'Lean' ? `Slight lean: ${lastName}` : `Start ${lastName}`;
  const body = [
    !sameWeek ? 'These players project for different weeks (one team is on bye), so they don’t compete for the same slot.' : '',
    sameWeek && diff >= .05 ? `${esc(best.name)} projects ${fmt(diff)} points higher.` : '',
    overlap ? 'Their 80% ranges overlap heavily, so treat this as a lean, not a lock.' : 'Their ranges barely overlap, which is a clearer edge than usual.',
    a.position !== b.position ? 'Different positions: make sure both are eligible for the slot.' : '',
  ].filter(Boolean).join(' ');
  $('compare-result').innerHTML = `
    <div class="verdict">
      <div><p class="label">Verdict</p><h3>${esc(headline)}</h3><p>${body}</p></div>
      <div class="versus-chart">${versusChart(a, b)}</div>
    </div>
    <div class="compare-cards">
      ${[a, b].map((p) => `<aside class="player-card">${cardHTML(p, { button: false, pick: sameWeek && bothOK && tag && best === p ? tag : '' })}</aside>`).join('')}
    </div>`;
}

/* ---------- Lineup ---------- */

function renderRoster() {
  const roster = [...state.roster].map(player).filter(Boolean);
  const counts = { QB: 0, RB: 0, WR: 0, TE: 0 };
  roster.forEach((p) => { counts[p.position] += 1; });
  $('roster-summary').textContent = `${roster.length} on roster · QB ${counts.QB} · RB ${counts.RB} · WR ${counts.WR} · TE ${counts.TE}`;
  $('roster-list').innerHTML = roster.length
    ? roster.map((p) => `<button class="chip" data-remove="${esc(p.id)}" title="Remove"><b>${esc(p.position)}</b>${esc(p.name)} ×</button>`).join('')
    : '<p class="muted">Empty. Tick players in Rankings.</p>';
  $('roster-list').querySelectorAll('[data-remove]').forEach((el) => el.addEventListener('click', () => toggleRoster(el.dataset.remove, false)));
}

function runOptimizer() {
  const week = state.data.metrics.prediction_week;
  const now = clock();
  const roster = [...state.roster].map(player).filter((p) => p && isUpcoming(p, now) && availabilityOK(p, now) && p.week === week);
  const best = optimizeLineup(roster);
  if (!best) {
    $('depth-chart').innerHTML = `<p class="empty">Not enough eligible Week ${week} players. You need at least 1 QB, 2 RB, 2 WR, 1 TE and one more RB/WR/TE.</p>`;
    toast('Roster incomplete for Week ' + week + '.');
    return;
  }
  const chosen = new Set(best.selected.map((r) => r.p.id));
  const bench = roster.filter((p) => !chosen.has(p.id)).sort((a, b) => b.prediction - a.prediction);
  $('depth-chart').innerHTML = best.selected.map(({ slot, p }) => `
    <div class="slot">
      <span class="slot-pos">${slot}</span>
      <span class="team-bar" style="background:${teamColor(p.team)}"></span>
      <span class="slot-name">${esc(p.name)}<small>${esc(p.position)} · ${esc(p.team)} ${p.home ? 'vs' : '@'} ${esc(p.opponent)} · range ${fmt(p.low)}–${fmt(p.high)}</small></span>
      <span class="slot-pts">${fmt(p.prediction)}</span>
    </div>`).join('')
    + `<div class="slot-total"><span>Projected total · Week ${week}</span><strong>${fmt(best.total)}</strong></div>`
    + `<div class="bench">${bench.length ? 'Bench: ' + bench.map((p) => `<b>${esc(p.name)}</b> ${fmt(p.prediction)}`).join(' · ') : 'Everyone on the roster starts.'}</div>`;
}

function loadSampleRoster() {
  const week = state.data.metrics.prediction_week;
  const now = clock();
  state.roster.clear();
  for (const [pos, count] of [['QB', 2], ['RB', 4], ['WR', 4], ['TE', 2]]) {
    state.data.players.filter((p) => p.position === pos && p.week === week && p.fantasy_pool !== false && availabilityOK(p, now))
      .slice(4, 4 + count).forEach((p) => state.roster.add(p.id));
  }
  saveRoster();
  renderRoster();
  renderBoard();
  renderCard();
  runOptimizer();
  toast('Sample roster loaded. Swap in your own players from Rankings.');
}

/* ---------- Model report ---------- */

function renderScorecard() {
  const r = state.data.report;
  const t = r.tolerance;
  const cells = [
    ['MAE', fmt(r.test.mae, 2), `baseline ${fmt(r.baseline.mae, 2)}`],
    ['RMSE', fmt(r.test.rmse, 2), `baseline ${fmt(r.baseline.rmse, 2)}`],
    ['R²', fmt(r.test.r2, 3), `baseline ${fmt(r.baseline.r2, 3)}`],
    ['Within 5 points', pct(t.model.within_5), `baseline ${pct(t.baseline.within_5)}`],
  ];
  $('scorecard').innerHTML = cells.map(([k, v, s]) => `<div><dt>${k}</dt><dd>${v}</dd><small>${s}</small></div>`).join('');
}

function dotPlot(rows, { min, max, marker, markerLabel, extraLabel, width = 520 }) {
  const rowH = 38, L = 150, R = extraLabel ? 110 : 50, T = 24;
  const H = T + rows.length * rowH + 24;
  const x = (v) => L + (v - min) / (max - min) * (width - L - R);
  const ticks = [];
  for (let v = min; v <= max + 1e-9; v += .1) ticks.push(Math.round(v * 10) / 10);
  const body = rows.map((r, i) => {
    const yy = T + i * rowH + rowH / 2;
    const color = r.highlight ? '#c8102e' : '#0b2545';
    return `<text x="0" y="${yy - 2}" font-size="13" font-weight="600" fill="#111827">${esc(r.label)}</text>
      <text x="0" y="${yy + 13}" font-size="10" fill="#8a93a1">${esc(r.sub || '')}</text>
      <line x1="${L}" x2="${x(r.value)}" y1="${yy}" y2="${yy}" stroke="${color}" stroke-width="1" opacity=".3"/>
      <circle cx="${x(r.value)}" cy="${yy}" r="6" fill="${color}"/>
      <text x="${x(r.value) + 11}" y="${yy + 4}" font-size="12" font-weight="600" fill="${color}">${fmt(r.value, 3)}</text>
      ${r.extra ? `<text x="${width - 4}" y="${yy + 4}" text-anchor="end" font-size="12" fill="${r.extraBad ? '#c8102e' : '#5d6675'}">${esc(r.extra)}</text>` : ''}`;
  }).join('') + (extraLabel ? `<text x="${width - 4}" y="${T - 12}" text-anchor="end" font-size="10" font-weight="600" fill="#5d6675">${esc(extraLabel)}</text>` : '');
  const grid = ticks.map((v) => `<line x1="${x(v)}" x2="${x(v)}" y1="${T - 4}" y2="${H - 22}" stroke="#eceef1"/>
    <text x="${x(v)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#8a93a1">${fmt(v)}</text>`).join('');
  const mk = marker == null ? '' : `<line x1="${x(marker)}" x2="${x(marker)}" y1="${T - 10}" y2="${H - 22}" stroke="#8a93a1" stroke-dasharray="4 3"/>
    <text x="${x(marker)}" y="${T - 12}" text-anchor="end" font-size="10" fill="#5d6675">${esc(markerLabel)}</text>`;
  return `<svg class="chart" viewBox="0 0 ${width} ${H}" role="img">${grid}${mk}${body}</svg>`;
}

function renderVersions() {
  const r = state.data.report;
  const worstRmse = Math.max(...r.versions.map((v) => v.rmse));
  const rows = r.versions.map((v, i) => ({
    label: v.version, sub: v.note, value: v.mae, highlight: i === r.versions.length - 1,
    extra: fmt(v.rmse, 3), extraBad: v.rmse === worstRmse,
  }));
  const lo = Math.floor(Math.min(...rows.map((d) => d.value)) * 10) / 10 - .1;
  const hi = Math.ceil(Math.max(r.baseline.mae, ...rows.map((d) => d.value)) * 10) / 10;
  $('chart-versions').innerHTML = dotPlot(rows, { min: lo, max: hi, marker: r.baseline.mae, markerLabel: `5-game avg ${fmt(r.baseline.mae, 2)}`, extraLabel: 'RMSE' })
    + '<p class="footnote">V2 had the lowest MAE by projecting low across the board (average 5.2 points vs 6.6 actual), so big games missed badly and RMSE was the worst. V6 averages 6.7. From V3 on, a version is only promoted if it improves both. V4 failed that test and was never shipped.</p>';
}

function renderCalibration() {
  const r = state.data.report;
  const W = 520, H = 340, L = 40, B = 30, T = 10, Rm = 10, max = 40;
  const x = (v) => L + Math.min(max, Math.max(0, v)) / max * (W - L - Rm);
  const y = (v) => H - B - Math.min(max, Math.max(0, v)) / max * (H - B - T);
  const ticks = [0, 10, 20, 30, 40];
  const grid = ticks.map((t) => `<line x1="${x(0)}" x2="${x(max)}" y1="${y(t)}" y2="${y(t)}" stroke="#eceef1"/>
    <text x="${L - 6}" y="${y(t) + 3}" text-anchor="end" font-size="10" fill="#8a93a1">${t}</text>
    <text x="${x(t)}" y="${H - B + 14}" text-anchor="middle" font-size="10" fill="#8a93a1">${t}</text>`).join('');
  const dots = r.scatter.map(([p, a]) => `<circle cx="${x(p)}" cy="${y(a)}" r="2.2" fill="#0b2545" opacity=".18"/>`).join('');
  const path = r.calibration.map((c, i) => `${i ? 'L' : 'M'}${x(c.predicted)},${y(c.actual)}`).join(' ');
  const pts = r.calibration.map((c) => `<circle cx="${x(c.predicted)}" cy="${y(c.actual)}" r="3.5" fill="#c8102e"/>`).join('');
  $('chart-calibration').innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Projected versus actual points">
    ${grid}
    <line x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}" stroke="#8a93a1" stroke-dasharray="4 3"/>
    ${dots}<path d="${path}" fill="none" stroke="#c8102e" stroke-width="2"/>${pts}
    <text x="${x(max)}" y="${H - 2}" text-anchor="end" font-size="10" fill="#5d6675">Projected PPR →</text>
    <text x="${L + 4}" y="${T + 10}" font-size="10" fill="#5d6675">Actual PPR</text>
  </svg>`;
}

function renderImportance() {
  const items = state.data.report.importance;
  const W = 520, rowH = 26, L = 250, R = 46;
  const H = items.length * rowH + 8;
  const max = Math.max(...items.map((d) => d.share));
  const x = (v) => v / max * (W - L - R);
  $('chart-importance').innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Feature importance">
    ${items.map((d, i) => {
      const yy = i * rowH + 4;
      const color = d.new ? '#c8102e' : '#0b2545';
      return `<text x="${L - 10}" y="${yy + 15}" text-anchor="end" font-size="12" fill="#111827">${esc(d.label)}</text>
        <rect x="${L}" y="${yy + 4}" width="${Math.max(1, x(d.share))}" height="15" fill="${color}" opacity="${d.new ? 1 : .8}"/>
        <text x="${L + x(d.share) + 6}" y="${yy + 15}" font-size="11" fill="#5d6675">${pct(d.share)}</text>`;
    }).join('')}
  </svg>`;
}

function renderPositions() {
  const rows = Object.entries(state.data.report.by_position);
  const W = 520, rowH = 54, L = 60, R = 30, T = 30;
  const H = T + rows.length * rowH + 20;
  const all = rows.flatMap(([, r]) => [r.model, r.baseline]);
  const min = Math.floor(Math.min(...all)) - .5, max = Math.ceil(Math.max(...all)) + .5;
  const x = (v) => L + (v - min) / (max - min) * (W - L - R);
  const ticks = [];
  for (let v = Math.ceil(min); v <= max; v += 1) ticks.push(v);
  $('chart-positions').innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="MAE by position">
    <circle cx="${L}" cy="10" r="5" fill="#b9bec7"/><text x="${L + 10}" y="14" font-size="11" fill="#5d6675">5-game average</text>
    <circle cx="${L + 120}" cy="10" r="5" fill="#0b2545"/><text x="${L + 130}" y="14" font-size="11" fill="#5d6675">Model</text>
    ${ticks.map((v) => `<line x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${H - 20}" stroke="#eceef1"/><text x="${x(v)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="#8a93a1">${v}</text>`).join('')}
    ${rows.map(([pos, r], i) => {
      const yy = T + i * rowH + rowH / 2;
      const cut = (r.baseline - r.model) / r.baseline;
      return `<text x="0" y="${yy + 5}" font-family="Barlow Condensed, Arial Narrow, sans-serif" font-size="20" font-weight="700" fill="#111827">${pos}</text>
        <line x1="${x(r.model)}" x2="${x(r.baseline)}" y1="${yy}" y2="${yy}" stroke="#b9bec7" stroke-width="3"/>
        <circle cx="${x(r.baseline)}" cy="${yy}" r="7" fill="#b9bec7"/>
        <circle cx="${x(r.model)}" cy="${yy}" r="7" fill="#0b2545"/>
        <text x="${x(r.model)}" y="${yy - 12}" text-anchor="middle" font-size="11" font-weight="600" fill="#0b2545">${fmt(r.model, 2)}</text>
        <text x="${x(r.baseline)}" y="${yy - 12}" text-anchor="middle" font-size="11" fill="#5d6675">${fmt(r.baseline, 2)}</text>
        <text x="${x(r.baseline) + 14}" y="${yy + 4}" font-size="11" fill="#2f5d3a">−${pct(cut, 0)}</text>`;
    }).join('')}
  </svg>`;
}

function renderModel() {
  const r = state.data.report;
  renderScorecard();
  renderVersions();
  renderCalibration();
  renderImportance();
  renderPositions();
  const folds = r.validation.folds.map((f) => `${f.validation_season}: ${fmt(f.mae, 3)}`).join(', ');
  $('protocol-text').textContent = `Rows are never shuffled. Validation MAE by season: ${folds}. The model used for this week's projections is refit on every completed game before the snapshot. Interval coverage on test: ${pct(r.interval.coverage_test)} (target 80%).`;
  const ex = state.data.test_examples;
  $('audit-note').textContent = `${ex.length} most recent test rows`;
  $('audit-body').innerHTML = ex.map((e) => `<tr><td>${esc(e.name)}</td><td>${esc(e.position)}</td><td>${e.season} / W${e.week}</td>
    <td class="num">${fmt(e.prediction)}</td><td class="num">${fmt(e.target)}</td><td class="num muted">${fmt(Math.abs(e.prediction - e.target))}</td></tr>`).join('');
  const weights = state.data.metrics.cv_candidates.find((c) => c.model === r.selected_model)?.weights || {};
  const parts = Object.entries(weights).map(([name, w]) => `${fmt(w * 100, 0)}% ${name.includes('hurdle') ? 'two-stage hurdle model (chance of scoring × points if scoring)' : 'LightGBM regressor'}`);
  $('method-model').textContent = `${r.feature_count} pre-game features. Final blend: ${parts.join(', ')}. A new version replaces the old one only if it lowers both MAE and RMSE on 2023–24 validation.`;
}

/* ---------- Init ---------- */

async function loadPayload() {
  const embedded = $('portfolio-data');
  if (embedded) return JSON.parse(embedded.textContent);
  const res = await fetch('data.json');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function renderIntro() {
  const d = state.data, m = d.metrics, snap = d.forecast_snapshot;
  $('week-label').textContent = `2026 · Week ${snap.next_week}`;
  const asOf = new Date(snap.as_of_utc).toLocaleString('en-US', { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
  $('snapshot-text').textContent = `Data through Week ${snap.latest_complete_week} · snapshot ${asOf} · static page, refreshed manually`;
  $('stat-mae').textContent = fmt(m.test.mae, 2);
  $('stat-lift').innerHTML = `<span class="good">−${fmt(m.mae_improvement_pct)}%</span>`;
  $('stat-n').textContent = fmt(m.test.n, 0);
  const pool = d.players.filter((p) => p.fantasy_pool !== false).length;
  $('stat-players').textContent = pool;
  $('stat-players-note').textContent = `of ${d.players.length} candidates`;
  if (d.availability) {
    $('rankings-note').textContent = `${d.availability.eligible} of ${d.availability.total} candidates passed the active-roster check at snapshot time. “Q” = questionable, “Check” = roster or report not confirmed. Fantasy-relevant = recent snap share or touches, decided before the game. Bye-week teams show their next game.`;
  }
  if (state.archive) {
    $('archive-banner').hidden = false;
    $('archive-banner').textContent = `Archived snapshot. Week ${snap.next_week} has kicked off since this page was published, so you're seeing the projections exactly as they stood on ${asOf}.`;
  }
}

function bindControls() {
  document.querySelectorAll('.tab').forEach((el) => el.addEventListener('click', () => setTab(el.dataset.tab)));
  document.querySelector('.wordmark').addEventListener('click', (e) => { e.preventDefault(); setTab('rankings'); window.scrollTo({ top: 0, behavior: 'smooth' }); });
  $('position-filters').querySelectorAll('button').forEach((el) => el.addEventListener('click', () => {
    state.position = el.dataset.pos;
    state.page = 0;
    $('position-filters').querySelectorAll('button').forEach((b) => b.classList.toggle('is-on', b === el));
    renderBoard(true);
  }));
  $('search').addEventListener('input', (e) => { state.search = e.target.value; state.page = 0; renderBoard(true); });
  $('sort').addEventListener('change', (e) => { state.sort = e.target.value; state.page = 0; renderBoard(true); });
  $('pool').addEventListener('change', (e) => { state.pool = e.target.value; state.page = 0; renderBoard(true); });
  $('game-filter').addEventListener('change', (e) => { state.game = e.target.value; state.page = 0; renderTicker(); renderBoard(true); });
  $('prev-page').addEventListener('click', () => { state.page -= 1; renderBoard(); });
  $('next-page').addEventListener('click', () => { state.page += 1; renderBoard(); });
  $('compare-a').addEventListener('change', renderCompare);
  $('compare-b').addEventListener('change', renderCompare);
  $('optimize').addEventListener('click', runOptimizer);
  $('demo-roster').addEventListener('click', loadSampleRoster);
  $('clear-roster').addEventListener('click', () => {
    state.roster.clear();
    saveRoster();
    renderRoster();
    renderBoard();
    renderCard();
    $('depth-chart').innerHTML = '<p class="empty">Roster cleared. Tick players in Rankings.</p>';
  });
}

function fillCompare() {
  const now = clock();
  const ps = state.data.players.filter((p) => isUpcoming(p, now) && p.fantasy_pool !== false).sort((a, b) => a.name.localeCompare(b.name));
  const options = ps.map((p) => `<option value="${esc(p.id)}">${esc(p.name)} · ${p.position} ${esc(p.team)}</option>`).join('');
  $('compare-a').innerHTML = options;
  $('compare-b').innerHTML = options;
  const wr = state.data.players.filter((p) => p.position === 'WR' && isUpcoming(p, now)).sort((a, b) => b.prediction - a.prediction);
  if (wr.length > 12) { $('compare-a').value = wr[10].id; $('compare-b').value = wr[11].id; }
}

async function init() {
  try {
    state.data = await loadPayload();
    const players = state.data.players;
    state.archive = !players.some((p) => isUpcoming(p));
    try {
      const saved = JSON.parse(localStorage.getItem('fantasy-scout-roster') || '[]');
      state.roster = new Set(saved.filter((id) => player(id)));
    } catch { /* storage unavailable */ }
    $('roster-count').textContent = state.roster.size;
    renderIntro();
    bindControls();
    renderTicker();
    fillCompare();
    renderBoard(true);
    renderModel();
    $('loading').hidden = true;
    $('app').hidden = false;
  } catch (err) {
    $('loading').textContent = 'Could not load data.json. Serve this folder over HTTP (for example: python -m http.server) instead of opening the file directly.';
    console.error(err);
  }
}

init();
