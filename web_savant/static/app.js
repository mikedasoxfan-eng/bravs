(() => {
  const q = document.getElementById('q');
  const results = document.getElementById('results');
  const yearSel = document.getElementById('year');
  const card = document.getElementById('card');
  const empty = document.getElementById('empty');

  let currentPlayer = null;
  let latestCardReq = 0;

  // Populate default year range 2015–2025
  for (let y = 2025; y >= 2015; y--) {
    const opt = document.createElement('option');
    opt.value = y;
    opt.textContent = y;
    yearSel.appendChild(opt);
  }

  // ------- Search -------
  let searchTimer = null;
  q.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const val = q.value.trim();
    if (val.length < 2) { results.classList.add('hidden'); return; }
    searchTimer = setTimeout(() => doSearch(val), 160);
  });
  q.addEventListener('focus', () => {
    if (q.value.trim().length >= 2) doSearch(q.value.trim());
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-wrap')) results.classList.add('hidden');
  });

  async function doSearch(qry) {
    try {
      const r = await fetch('/api/search?q=' + encodeURIComponent(qry));
      const hits = await r.json();
      results.innerHTML = '';
      if (!hits.length) {
        const el = document.createElement('div');
        el.className = 'hit';
        el.innerHTML = `<span class="name" style="color: var(--ink-400)">No matches</span>`;
        results.appendChild(el);
      } else {
        hits.forEach(h => {
          const el = document.createElement('div');
          el.className = 'hit';
          const firstYr = h.years[h.years.length - 1];
          const lastYr = h.years[0];
          const span = (firstYr === lastYr) ? `${firstYr}` : `${firstYr}–${lastYr}`;
          el.innerHTML = `<span class="name">${escapeHtml(h.name)}</span><span class="yrs">${span}</span>`;
          el.addEventListener('click', () => selectPlayer(h));
          results.appendChild(el);
        });
      }
      results.classList.remove('hidden');
    } catch (err) {
      console.error(err);
    }
  }

  function selectPlayer(h) {
    currentPlayer = h;
    q.value = h.name;
    results.classList.add('hidden');
    yearSel.innerHTML = '';
    h.years.forEach(y => {
      const opt = document.createElement('option');
      opt.value = y; opt.textContent = y;
      yearSel.appendChild(opt);
    });
    loadCard();
  }

  yearSel.addEventListener('change', loadCard);

  // ------- Card load / render -------
  async function loadCard() {
    if (!currentPlayer) return;
    const year = yearSel.value;
    const reqId = ++latestCardReq;
    try {
      const r = await fetch(`/api/player/${currentPlayer.id}/${year}`);
      if (reqId !== latestCardReq) return; // stale
      const data = await r.json();
      renderCard(data);
    } catch (err) {
      console.error(err);
    }
  }

  function renderCard(data) {
    empty.classList.add('hidden');
    card.classList.remove('hidden');
    document.getElementById('pname').textContent = data.name;
    document.getElementById('pmeta').textContent =
      `${data.year} — ${data.kind === 'pitcher' ? 'Pitcher' : 'Position Player'}`;

    const sections = {
      value: data.value || [],
      batting: data.batting || [],
      pitching: data.pitching || [],
      fielding: data.fielding || [],
      running: data.running || [],
      catching: data.catching || [],
    };

    for (const [k, rows] of Object.entries(sections)) {
      const sec = document.getElementById(`sec-${k}`);
      const bars = sec.querySelector('.bars');
      bars.innerHTML = '';
      if (!rows.length) { sec.classList.add('hidden'); continue; }
      sec.classList.remove('hidden');
      rows.forEach((row, i) => {
        const el = renderBar(row);
        el.style.setProperty('--idx', i);
        bars.appendChild(el);
      });
    }
  }

  function renderBar(r) {
    const el = document.createElement('div');
    el.className = 'bar-row';

    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = r.label;

    const track = document.createElement('div');
    track.className = 'track';

    if (r.percentile !== null && r.percentile !== undefined) {
      const pct = Math.max(0, Math.min(100, r.percentile));
      const color = pctColor(pct);

      const fill = document.createElement('div');
      fill.className = 'fill';
      fill.style.width = pct + '%';
      fill.style.background = color;
      track.appendChild(fill);

      const chip = document.createElement('div');
      chip.className = 'pct-chip';
      chip.style.left = pct + '%';
      chip.style.background = color;
      chip.style.color = chipTextColor(pct);
      chip.textContent = pct;
      track.appendChild(chip);
    } else {
      const chip = document.createElement('div');
      chip.className = 'pct-chip na';
      chip.style.left = '50%';
      chip.textContent = '—';
      track.appendChild(chip);
    }

    const val = document.createElement('div');
    val.className = 'value' + (r.value === null ? ' na' : '');
    val.textContent = r.value_fmt;

    el.appendChild(label);
    el.appendChild(track);
    el.appendChild(val);
    return el;
  }

  // Intensity scales with distance from 50. Near 50 = very light/muted.
  // Near 0 = deep blue. Near 100 = deep red.
  function pctColor(p) {
    const dist = Math.abs(p - 50) / 50;         // 0 at mid, 1 at extremes
    const lightness = 80 - dist * 38;            // 80% → 42%
    const saturation = 20 + dist * 48;           // 20% → 68%
    const hue = p < 50 ? 212 : 351;              // slate blue / cardinal red
    return `hsl(${hue} ${saturation}% ${lightness}%)`;
  }

  function chipTextColor(p) {
    // Chip is more readable with dark text when fill is pale.
    const dist = Math.abs(p - 50) / 50;
    return dist < 0.35 ? 'var(--ink-900)' : '#ffffff';
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }
})();
