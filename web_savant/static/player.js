(() => {
  // --- Search typeahead ---
  const q = document.getElementById('q');
  const results = document.getElementById('results');
  const yearSel = document.getElementById('year');
  let currentPlayer = null;
  let searchTimer = null;

  if (q) {
    q.addEventListener('input', () => {
      clearTimeout(searchTimer);
      const v = q.value.trim();
      if (v.length < 2) { results.classList.add('hidden'); return; }
      searchTimer = setTimeout(() => doSearch(v), 160);
    });
    q.addEventListener('focus', () => {
      if (q.value.trim().length >= 2) doSearch(q.value.trim());
    });
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.search-box')) results.classList.add('hidden');
    });
  }

  async function doSearch(qry) {
    const r = await fetch('/api/search?q=' + encodeURIComponent(qry));
    const hits = await r.json();
    results.innerHTML = '';
    if (!hits.length) {
      results.innerHTML = `<div class="hit"><span class="name" style="color:var(--ink-400)">No matches</span></div>`;
    } else {
      hits.forEach(h => {
        const el = document.createElement('div');
        el.className = 'hit';
        const a = h.years[h.years.length - 1];
        const b = h.years[0];
        const sp = (a === b) ? `${a}` : `${a}–${b}`;
        el.innerHTML = `<span class="name">${escapeHtml(h.name)}</span><span class="yrs">${sp}</span>`;
        el.addEventListener('click', () => {
          currentPlayer = h;
          q.value = h.name;
          results.classList.add('hidden');
          navigateToPlayer(h);
        });
        results.appendChild(el);
      });
    }
    results.classList.remove('hidden');
  }

  function navigateToPlayer(h) {
    // Pick the latest year the player has
    const y = yearSel.value || h.years[0];
    location.href = `/player/${h.id}/${y}`;
  }

  if (yearSel) {
    yearSel.addEventListener('change', () => {
      const m = location.pathname.match(/\/player\/(\d+)/);
      if (m) {
        location.href = `/player/${m[1]}/${yearSel.value}`;
      }
    });
  }

  // --- Percentile bars ---
  document.querySelectorAll('.bars[data-rows]').forEach(barsEl => {
    const rows = JSON.parse(barsEl.getAttribute('data-rows'));
    rows.forEach(r => barsEl.appendChild(renderBar(r)));
  });

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

  // --- Career sparklines ---
  const arc = document.getElementById('career-arc');
  if (arc) {
    const career = JSON.parse(arc.getAttribute('data-career'));
    career.forEach(c => arc.appendChild(renderSpark(c)));
  }

  function renderSpark(c) {
    const row = document.createElement('div');
    row.className = 'spark-row';

    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = c.label;

    const svgWrap = document.createElement('div');
    const pts = c.series.filter(p => p.v !== null);
    if (pts.length === 0) {
      svgWrap.innerHTML = `<span class="mono" style="color:var(--ink-400);font-size:11px">No data</span>`;
    } else {
      const w = 340, h = 32, pad = 2;
      const minV = Math.min(...pts.map(p => p.v));
      const maxV = Math.max(...pts.map(p => p.v));
      const range = maxV - minV || 1;
      const yr0 = Math.min(...pts.map(p => p.year));
      const yr1 = Math.max(...pts.map(p => p.year));
      const yrRange = Math.max(1, yr1 - yr0);
      const coords = pts.map(p => {
        const x = pad + ((p.year - yr0) / yrRange) * (w - pad * 2);
        const y = h - pad - ((p.v - minV) / range) * (h - pad * 2);
        return [x, y];
      });
      const path = coords.map(([x, y], i) => (i === 0 ? `M${x.toFixed(1)} ${y.toFixed(1)}` : `L${x.toFixed(1)} ${y.toFixed(1)}`)).join(' ');
      const circles = coords.map(([x, y], i) => {
        const last = i === coords.length - 1;
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${last ? 2.8 : 1.6}" fill="${last ? 'var(--great)' : 'var(--ink-400)'}"/>`;
      }).join('');
      svgWrap.innerHTML = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <path d="${path}" fill="none" stroke="var(--ink-700)" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round"/>
        ${circles}
      </svg>`;
    }

    const val = document.createElement('div');
    val.className = 'value';
    val.textContent = (c.latest === null || c.latest === undefined)
      ? '—'
      : format(c.fmt, c.latest);

    row.appendChild(label);
    row.appendChild(svgWrap);
    row.appendChild(val);
    return row;
  }

  function format(fmt, v) {
    // Minimal {:.Nf} / {:.0f} handler
    const m = fmt.match(/\{:\.(\d+)f\}/);
    if (m) return Number(v).toFixed(parseInt(m[1], 10));
    return String(v);
  }

  function pctColor(p) {
    const dist = Math.abs(p - 50) / 50;
    const lightness = 80 - dist * 38;
    const saturation = 20 + dist * 48;
    const hue = p < 50 ? 212 : 351;
    return `hsl(${hue} ${saturation}% ${lightness}%)`;
  }
  function chipTextColor(p) {
    const dist = Math.abs(p - 50) / 50;
    return dist < 0.35 ? 'var(--ink-900)' : '#ffffff';
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }
})();
