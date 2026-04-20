(() => {
  const q = document.getElementById('cmp-q');
  const results = document.getElementById('cmp-results');
  const chips = document.getElementById('chips');
  const idsInput = document.getElementById('ids-input');
  const form = document.getElementById('cmp-form');
  if (!q) return;

  let picked = [];
  const initIds = idsInput.value.split(',').filter(Boolean);

  // Hydrate chip list from URL-provided ids + player-head text
  document.querySelectorAll('.player-head .nm a').forEach((a, i) => {
    const id = initIds[i];
    if (id) {
      picked.push({ id: parseInt(id, 10), name: a.textContent.trim() });
    }
  });
  renderChips();

  let timer = null;
  q.addEventListener('input', () => {
    clearTimeout(timer);
    const v = q.value.trim();
    if (v.length < 2) { results.classList.add('hidden'); return; }
    timer = setTimeout(() => doSearch(v), 160);
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#cmp-q, #cmp-results')) results.classList.add('hidden');
  });

  async function doSearch(qry) {
    const r = await fetch('/api/search?q=' + encodeURIComponent(qry));
    const hits = await r.json();
    results.innerHTML = '';
    if (!hits.length) {
      results.innerHTML = `<div class="hit"><span class="name" style="color:var(--ink-400)">No matches</span></div>`;
    } else {
      hits.forEach(h => {
        if (picked.some(p => p.id === h.id)) return;
        const el = document.createElement('div');
        el.className = 'hit';
        el.innerHTML = `<span class="name">${escapeHtml(h.name)}</span><span class="yrs">${h.years[h.years.length-1]}–${h.years[0]}</span>`;
        el.addEventListener('click', () => {
          if (picked.length >= 5) return;
          picked.push({ id: h.id, name: h.name });
          renderChips();
          q.value = '';
          results.classList.add('hidden');
        });
        results.appendChild(el);
      });
    }
    results.classList.remove('hidden');
  }

  function renderChips() {
    chips.innerHTML = '';
    if (!picked.length) {
      chips.innerHTML = '<span style="color:var(--ink-400)">No players selected</span>';
    } else {
      picked.forEach((p, i) => {
        const c = document.createElement('span');
        c.className = 'chip';
        c.innerHTML = `${escapeHtml(p.name)} <button type="button" aria-label="Remove" style="background:none;border:none;color:var(--ink-400);cursor:pointer;padding:0 2px;font-size:14px;line-height:1">×</button>`;
        c.querySelector('button').addEventListener('click', () => {
          picked.splice(i, 1);
          renderChips();
        });
        chips.appendChild(c);
      });
    }
    idsInput.value = picked.map(p => p.id).join(',');
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
})();
