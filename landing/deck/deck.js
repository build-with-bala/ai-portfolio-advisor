/* AI Portfolio Advisor deck.
   Part 1: DATA (every number, copied from reports/metrics.json and docs/REPORT.md).
   Part 2: charts, drawn from DATA as inline SVG / HTML in their final state.
   Part 3: the deck engine (steps, transitions, overview, print). If part 3 fails, the page
           stays a scrolling stack of fully revealed slides. No external script is needed. */
(function () {
'use strict';

/* ------------------------------------------------------------------ 1. DATA */
var DATA = {
  panel: { first: '2018-02-19', last: '2026-10-08' },
  features: {
    candidates: ['RSI_14','MACD_signal','Boll_BW','ATR_14','Stoch_K','OBV_delta','Williams_R','ret_5','ret_21','vol_21',
                 'fracdiff_close','ret_63','mom_126_21','vol_63','dist_52w_high','volume_surge','rel_ret_21','mkt_ret_21','mkt_vol_21'],
    selected: ['mkt_vol_21','fracdiff_close','rel_ret_21','volume_surge','mom_126_21','Stoch_K','vol_63','ret_63','OBV_delta','Boll_BW']
  },
  purgeTradingDays: 31,
  folds: [
    { fold: 1, train_rows: 6205,  val_start: '2019-04-09', val_end: '2020-06-03' },
    { fold: 2, train_rows: 22645, val_start: '2020-06-04', val_end: '2021-07-14' },
    { fold: 3, train_rows: 39165, val_start: '2021-07-15', val_end: '2022-08-30' },
    { fold: 4, train_rows: 55685, val_start: '2022-09-01', val_end: '2023-10-17' },
    { fold: 5, train_rows: 72448, val_start: '2023-10-18', val_end: '2024-12-09' }
  ],
  holdout: { start: '2025-01-23', end: '2026-09-09', rows: 24413, coverage: 0.824, target: 0.80 },
  candidates: [
    { name: 'Naive baseline', pinball: 0.02395 },
    { name: 'Ridge', pinball: null, note: 'Point forecast only, so no quantile loss. MAE 0.0731, naive 0.0722.' },
    { name: 'LightGBM, notebook settings', pinball: 0.02648 },
    { name: 'LightGBM, tuned', pinball: 0.02406, chosen: true }
  ],
  backtest: { /* hold-out, first of 21 start days: value of 1.00 invested, one point per 21 trading days */
    start: 'Jan 2025', end: 'Aug 2026',
    strategy: [1.0004,1.0427,1.0899,1.1838,1.1850,1.0811,1.0586,1.0353,1.0949,1.0854,1.0776,0.9405,0.9218,0.8219,0.9766,0.9756,0.9394,1.0353,1.0278,0.9720],
    benchmark: [0.9697,1.0010,1.0203,1.0527,1.0856,1.0471,1.0482,1.0542,1.1138,1.1021,1.0949,1.0846,1.0965,1.0049,1.0914,1.0626,1.0822,1.1075,1.1062,1.0449]
  },
  shap: [ /* mean |SHAP| on the median model; direction = rank correlation of feature value with its SHAP */
    { feature: 'mkt_vol_21', value: 0.004452, direction: 0.301 },
    { feature: 'mom_126_21', value: 0.002527, direction: -0.680 },
    { feature: 'fracdiff_close', value: 0.002360, direction: -0.544 },
    { feature: 'vol_63', value: 0.001336, direction: 0.351 },
    { feature: 'ret_63', value: 0.001190, direction: -0.621 }
  ],
  regimes: [ /* hold-out rank IC by market volatility regime */
    { group: 'Calm', ic: -0.079 }, { group: 'Normal', ic: 0.076 }, { group: 'Turbulent', ic: 0.142 }
  ]
};

/* ------------------------------------------------------------------ helpers */
var SVGNS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, kids) { return make(document.createElementNS(SVGNS, tag), attrs, kids); }
function H(tag, attrs, kids) { return make(document.createElement(tag), attrs, kids); }
function make(el, attrs, kids) {
  for (var k in (attrs || {})) {
    if (k === 'text') el.textContent = attrs[k];
    else if (k === 'style') el.style.cssText = attrs[k];
    else el.setAttribute(k, attrs[k]);
  }
  (kids || []).forEach(function (c) { if (c) el.appendChild(c); });
  return el;
}
function day(s) { return Date.parse(s + 'T00:00:00Z') / 864e5; }
function charts() {
  var draw = { fan: fan, chips: chips, folds: folds, candidates: candidates, coverage: coverage, backtest: backtest, shap: shap, regimes: regimes };
  document.querySelectorAll('[data-chart]').forEach(function (el) {
    try { draw[el.getAttribute('data-chart')](el); } catch (e) { if (window.console) console.error('chart failed', el.getAttribute('data-chart'), e); }
  });
}

/* ------------------------------------------------------------------ 2. charts */

/* Title motif: a price path, then the P10 to P90 band opening from today. An illustration, labelled as one. */
function fan(el) {
  var seed = 11; function rnd() { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; }
  var X0 = 1090, Y0 = 716, X1 = 1560, pts = [], y = 800, n = 44;
  for (var i = 0; i <= n; i++) {
    var x = -10 + (X0 + 10) * i / n, target = 800 + (Y0 - 800) * i / n;
    y += (rnd() - 0.5) * 46 + (target - y) * 0.22;
    y = Math.max(694, Math.min(822, y));
    if (i === n) y = Y0;
    pts.push([x, y]);
  }
  var hist = 'M' + pts.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L');
  var band = 'M' + X0 + ' ' + Y0 + ' C' + (X0 + 190) + ' ' + (Y0 - 60) + ' ' + (X1 - 200) + ' 372 ' + X1 + ' 340 L' + X1 + ' 852 C' + (X1 - 200) + ' 836 ' + (X0 + 190) + ' ' + (Y0 + 40) + ' ' + X0 + ' ' + Y0 + ' Z';
  var mid = 'M' + X0 + ' ' + Y0 + ' C' + (X0 + 190) + ' ' + (Y0 - 14) + ' ' + (X1 - 200) + ' 606 ' + X1 + ' 596';
  function label(yy, big, small, d) {
    return S('g', { class: 'rv', style: '--d:' + d + 'ms' }, [
      S('text', { x: X1 + 26, y: yy + 6, 'font-size': 36, 'font-weight': 700, fill: '#141936', text: big }),
      S('text', { x: X1 + 26, y: yy + 42, 'font-size': 26, fill: '#5B5F77', text: small })
    ]);
  }
  var svg = S('svg', { viewBox: '0 0 1920 1080', width: 1920, height: 1080, role: 'img', 'aria-label': 'Illustration: a price history, then a forecast range that widens over 21 trading days, with a median line inside it.' }, [
    S('defs', {}, [S('clipPath', { id: 'fanclip' }, [S('rect', { class: 'gx', style: '--d:1500ms', x: X0, y: 300, width: X1 - X0 + 4, height: 600 })])]),
    S('line', { class: 'rv', style: '--d:1300ms', x1: X0, y1: 318, x2: X0, y2: 902, stroke: '#5B5F77', 'stroke-width': 2, 'stroke-dasharray': '3 9' }),
    S('text', { class: 'rv', style: '--d:1300ms', x: X0, y: 300, 'text-anchor': 'middle', 'font-size': 26, fill: '#5B5F77', text: 'Today' }),
    S('g', { 'clip-path': 'url(#fanclip)' }, [S('path', { d: band, fill: '#F2A51A' })]),
    S('path', { class: 'draw', style: '--d:150ms', pathLength: 1, d: hist, fill: 'none', stroke: '#141936', 'stroke-width': 5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }),
    S('path', { class: 'draw', style: '--d:2300ms', pathLength: 1, d: mid, fill: 'none', stroke: '#3B46C4', 'stroke-width': 6, 'stroke-linecap': 'round' }),
    S('circle', { class: 'pop', style: '--d:1400ms', cx: X0, cy: Y0, r: 11, fill: '#141936', stroke: '#F6F2E9', 'stroke-width': 4 }),
    label(340, 'Optimistic', 'P90', 2700), label(596, 'Median', 'P50', 2850), label(816, 'Pessimistic', 'P10', 3000),
    S('g', { class: 'rv', style: '--d:3200ms' }, [
      S('path', { d: 'M' + X0 + ' 912 v14 H' + X1 + ' v-14', fill: 'none', stroke: '#141936', 'stroke-width': 3 }),
      S('text', { x: (X0 + X1) / 2, y: 964, 'text-anchor': 'middle', 'font-size': 26, fill: '#5B5F77', text: '21 trading days ahead (illustration)' })
    ])
  ]);
  el.appendChild(svg);
}

/* 19 candidate features; the 10 that survive selection stay lit. */
function chips(el) {
  var keep = {}; DATA.features.selected.forEach(function (f) { keep[f] = 1; });
  DATA.features.candidates.forEach(function (f, i) {
    el.appendChild(H('span', { class: 'chip ' + (keep[f] ? 'keep' : 'drop'), style: '--d:' + (i * 45) + 'ms', text: f,
      title: f + (keep[f] ? ': kept' : ': dropped as redundant or weak') }));
  });
  el.setAttribute('role', 'img');
  el.setAttribute('aria-label', '19 candidate features. Kept: ' + DATA.features.selected.join(', ') + '.');
}

/* Purged walk-forward folds on a calendar axis, to scale. */
function folds(el) {
  var W = 1728, LEFT = 124, RIGHT = 1716, t0 = day(DATA.panel.first), t1 = day(DATA.panel.last);
  function x(d) { return LEFT + (RIGHT - LEFT) * (d - t0) / (t1 - t0); }
  var purge = DATA.purgeTradingDays * 365 / 252;            /* 31 trading days in calendar days */
  var f1 = DATA.folds[0], trainStart = day(f1.val_start) - purge - (f1.train_rows / 60) * 365 / 252;
  var top = 50, rh = 30, gap = 8, kids = [];
  var defs = S('defs', {}, [S('pattern', { id: 'hatch', width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' },
    [S('rect', { width: 7, height: 7, fill: '#FFFDF8' }), S('rect', { width: 3, height: 7, fill: '#141936' })])]);
  kids.push(defs);
  /* legend */
  var lx = LEFT;
  [['Train', '#C9CDF1', '#3B46C4'], ['Validate', '#F2A51A', '#B9770A'], ['Hold-out', '#3B46C4', '#3B46C4']].forEach(function (l) {
    kids.push(S('rect', { class: 'rv', x: lx, y: 8, width: 30, height: 22, rx: 3, fill: l[1], stroke: l[2], 'stroke-width': 2 }));
    kids.push(S('text', { class: 'rv', x: lx + 40, y: 28, text: l[0] }));
    lx += 40 + l[0].length * 13 + 44;
  });
  /* year grid */
  var bottom = top + 6 * (rh + gap);
  for (var yr = 2019; yr <= 2026; yr++) {
    var gx = x(day(yr + '-01-01'));
    kids.push(S('line', { class: 'rv', x1: gx, y1: top - 6, x2: gx, y2: bottom, stroke: '#E2DBCB', 'stroke-width': 2 }));
    kids.push(S('text', { class: 'rv mut', x: gx + 8, y: bottom + 24, text: String(yr) }));
  }
  kids.push(S('line', { class: 'rv', x1: LEFT, y1: bottom, x2: RIGHT, y2: bottom, stroke: '#141936', 'stroke-width': 2 }));
  var purgeG = S('g', { 'data-f': 2, 'data-fx': 'group' }), holdG = S('g', { 'data-f': 3, 'data-fx': 'group' });
  DATA.folds.forEach(function (f, i) {
    var yy = top + i * (rh + gap), vs = day(f.val_start), ve = day(f.val_end), d = 150 + i * 130;
    kids.push(S('text', { class: 'rv', style: '--d:' + d + 'ms', x: 0, y: yy + 24, text: 'Fold ' + f.fold }));
    kids.push(S('rect', { class: 'gx', style: '--d:' + d + 'ms', x: x(trainStart), y: yy, width: x(vs - purge) - x(trainStart), height: rh, rx: 3, fill: '#C9CDF1', stroke: '#3B46C4', 'stroke-width': 2 },
      [S('title', { text: 'Fold ' + f.fold + ' train: ' + f.train_rows.toLocaleString('en-IN') + ' rows' })]));
    kids.push(S('rect', { class: 'pop', style: '--d:' + (d + 500) + 'ms', x: x(vs), y: yy, width: x(ve) - x(vs), height: rh, rx: 3, fill: '#F2A51A', stroke: '#B9770A', 'stroke-width': 2 },
      [S('title', { text: 'Fold ' + f.fold + ' validate: ' + f.val_start + ' to ' + f.val_end })]));
    purgeG.appendChild(S('rect', { class: 'pop', style: '--d:' + (i * 90) + 'ms', x: x(vs - purge) + 1, y: yy - 4, width: x(vs) - x(vs - purge) - 2, height: rh + 8, fill: 'url(#hatch)', stroke: '#141936', 'stroke-width': 2 },
      [S('title', { text: '31 trading days removed before validation' })]));
  });
  /* warm-up note and purge note sit in the empty space of the first rows */
  kids.push(S('text', { class: 'rv mut', style: '--d:300ms', x: x(trainStart) - 8, y: top + 24, 'text-anchor': 'end', text: 'warm-up' }));
  purgeG.appendChild(S('g', { class: 'rv', style: '--d:450ms' }, [
    S('rect', { x: x(day('2020-08-10')), y: top + 2, width: 30, height: rh - 4, fill: 'url(#hatch)', stroke: '#141936', 'stroke-width': 2 }),
    S('text', { x: x(day('2020-08-10')) + 42, y: top + 24, 'font-weight': 650, text: '31 trading days purged before every validation block: each label looks 21 days ahead' })
  ]));
  /* hold-out */
  var hy = top + 5 * (rh + gap), hs = day(DATA.holdout.start), he = day(DATA.holdout.end);
  holdG.appendChild(S('text', { class: 'rv', x: 0, y: hy + 24, text: 'Hold-out' }));
  holdG.appendChild(S('rect', { class: 'pop', x: x(hs - purge) + 1, y: hy - 4, width: x(hs) - x(hs - purge) - 2, height: rh + 8, fill: 'url(#hatch)', stroke: '#141936', 'stroke-width': 2 }));
  holdG.appendChild(S('rect', { class: 'gx', style: '--d:150ms', x: x(hs), y: hy, width: x(he) - x(hs), height: rh, rx: 3, fill: '#3B46C4' },
    [S('title', { text: 'Hold-out: ' + DATA.holdout.start + ' to ' + DATA.holdout.end })]));
  holdG.appendChild(S('text', { class: 'rv', style: '--d:500ms', x: x(hs - purge) - 14, y: hy + 24, 'text-anchor': 'end', 'font-weight': 650,
    text: DATA.holdout.rows.toLocaleString('en-IN') + ' stock-days, never used for selection or tuning' }));
  kids.push(purgeG); kids.push(holdG);
  el.appendChild(S('svg', { viewBox: '0 0 ' + W + ' ' + (bottom + 34), width: W, height: bottom + 34, role: 'img',
    'aria-label': 'Timeline of five expanding training folds from 2018 to 2024, each followed by a 31-trading-day purge gap and a validation block, then a hold-out period from January 2025 to September 2026.' }, kids));
}

/* Candidates: one bar each on a shared zero-based axis. */
function candidates(el) {
  var max = 0.03;
  DATA.candidates.forEach(function (c, i) {
    var row = H('div', { class: 'cand' + (c.chosen ? ' chosen' : '') });
    var name = H('div', { class: 'name', text: c.name });
    if (c.chosen) name.appendChild(H('span', { class: 'pick', text: 'chosen' }));
    row.appendChild(name);
    if (c.pinball == null) row.appendChild(H('div', { class: 'na rv', style: '--d:' + (i * 120) + 'ms', text: c.note }));
    else {
      row.appendChild(H('div', { class: 'track' }, [H('div', { class: 'bar gx', style: 'width:' + (c.pinball / max * 100).toFixed(2) + '%;--d:' + (i * 120) + 'ms', title: c.name + ': ' + c.pinball.toFixed(5) })]));
      row.appendChild(H('div', { class: 'val', 'data-count': c.pinball, 'data-dec': 5, text: c.pinball.toFixed(5) }));
    }
    el.appendChild(row);
  });
}

/* Coverage meter: share of real outcomes inside the P10 to P90 range, against the 80% it should be. */
function coverage(el) {
  var c = DATA.holdout;
  el.appendChild(H('div', { class: 'gauge', role: 'img', 'aria-label': 'Coverage ' + (c.coverage * 100).toFixed(1) + ' percent against a target of 80 percent.' }, [
    H('div', { class: 'num rv' }, [H('span', { 'data-count': (c.coverage * 100).toFixed(1), 'data-dec': 1, text: (c.coverage * 100).toFixed(1) }), document.createTextNode('%')]),
    H('div', { class: 'meter' }, [
      H('div', { class: 'track' }, [H('div', { class: 'fill gx', style: 'width:' + (c.coverage * 100) + '%;--d:200ms' })]),
      H('div', { class: 'mark rv', style: 'left:' + (c.target * 100) + '%;--d:700ms' }, [H('span', { text: '80% target' })]),
      H('div', { class: 'ends' }, [H('span', { text: '0%' }), H('span', { text: '100%' })])
    ])
  ]));
}

/* Backtest: value of 1.00 invested, strategy against the equal-weight benchmark. One axis, two series. */
function backtest(el) {
  var W = 560, Hh = 268, L = 56, R = 548, T = 12, B = 222, lo = 0.8, hi = 1.2, b = DATA.backtest, n = b.strategy.length;
  function x(i) { return L + (R - L) * i / (n - 1); }
  function y(v) { return B - (B - T) * (v - lo) / (hi - lo); }
  function path(a) { return 'M' + a.map(function (v, i) { return x(i).toFixed(1) + ' ' + y(v).toFixed(1); }).join(' L'); }
  var kids = [S('defs', {}, [S('clipPath', { id: 'btclip' }, [S('rect', { class: 'gx', style: '--d:250ms', x: L - 10, y: 0, width: R - L + 22, height: Hh })])])];
  [0.8, 1.0, 1.2].forEach(function (v) {
    kids.push(S('line', { x1: L, x2: R, y1: y(v), y2: y(v), stroke: v === 1 ? '#141936' : '#E2DBCB', 'stroke-width': 2 }));
    kids.push(S('text', { x: L - 12, y: y(v) + 8, 'text-anchor': 'end', text: v.toFixed(1) }));
  });
  kids.push(S('text', { x: L, y: B + 34, text: b.start }));
  kids.push(S('text', { x: R, y: B + 34, 'text-anchor': 'end', text: b.end }));
  var g = S('g', { 'clip-path': 'url(#btclip)' }, [
    S('path', { d: path(b.benchmark), fill: 'none', stroke: '#141936', 'stroke-width': 3.5, 'stroke-dasharray': '10 7', 'stroke-linejoin': 'round' }),
    S('path', { d: path(b.strategy), fill: 'none', stroke: '#3B46C4', 'stroke-width': 4.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }),
    S('circle', { cx: x(n - 1), cy: y(b.benchmark[n - 1]), r: 7, fill: '#141936', stroke: '#F6F2E9', 'stroke-width': 3 }),
    S('circle', { cx: x(n - 1), cy: y(b.strategy[n - 1]), r: 7, fill: '#3B46C4', stroke: '#F6F2E9', 'stroke-width': 3 })
  ]);
  for (var i = 0; i < n; i++) {
    g.appendChild(S('rect', { x: x(i) - 12, y: T, width: 24, height: B - T, fill: 'transparent' },
      [S('title', { text: 'Period ' + (i + 1) + ': top-5 picks ' + b.strategy[i].toFixed(2) + ', all stocks ' + b.benchmark[i].toFixed(2) })]));
  }
  kids.push(g);
  kids.push(S('text', { x: (L + R) / 2, y: B + 34, 'text-anchor': 'middle', text: 'value of 1.00 invested' }));
  el.appendChild(S('svg', { viewBox: '0 0 ' + W + ' ' + Hh, width: W, height: Hh, role: 'img',
    'aria-label': 'Line chart, January 2025 to August 2026. The top-5 picks end at 0.97 of the starting value; holding all stocks equally ends at 1.04.' }, kids));
}

/* Diverging rows shared by SHAP and the regime chart. */
function divergingRows(el, rows, max, zero, fmt, cap) {
  el.appendChild(H('div', { class: 'axis-cap', style: 'display:block', text: cap }));
  rows.forEach(function (r, i) {
    var w = Math.abs(r.len) / max * (r.side < 0 ? zero : 100 - zero);
    el.appendChild(H('div', { class: 'drow', style: '--zero:' + zero + '%' }, [
      H('div', { class: 'name', text: r.name }),
      H('div', { class: 'track' }, [H('div', { class: 'bar gx ' + (r.side < 0 ? 'neg' : 'pos'), style: 'width:' + w.toFixed(2) + '%;--d:' + (150 + i * 110) + 'ms', title: r.name + ': ' + fmt(r) })]),
      H('div', { class: 'val', text: fmt(r) })
    ]));
  });
}
function shap(el) {
  var rows = DATA.shap.map(function (s) { return { name: s.feature, len: s.value, side: s.direction < 0 ? -1 : 1, v: s.value }; });
  divergingRows(el, rows, 0.0046, 50, function (r) { return r.v.toFixed(4); }, 'A higher value pushes it  ← down | up →');
  el.setAttribute('role', 'img');
  el.setAttribute('aria-label', 'Mean absolute SHAP of the five largest drivers: ' + DATA.shap.map(function (s) { return s.feature + ' ' + s.value.toFixed(4); }).join(', ') + '.');
}
function regimes(el) {
  var rows = DATA.regimes.map(function (r) { return { name: r.group, len: r.ic, side: r.ic < 0 ? -1 : 1, v: r.ic }; });
  /* same units either side of zero: zero sits at 36%, so 64% of the track = 0.15 */
  var zero = 36, max = 0.15;
  el.appendChild(H('div', { class: 'axis-cap', style: 'display:block', text: 'Rank IC by market regime, unseen data' }));
  rows.forEach(function (r, i) {
    var w = Math.abs(r.v) / max * (100 - zero);
    el.appendChild(H('div', { class: 'drow', style: '--zero:' + zero + '%;height:46px' }, [
      H('div', { class: 'name', text: r.name }),
      H('div', { class: 'track' }, [H('div', { class: 'bar gx ' + (r.side < 0 ? 'neg' : 'pos'), style: 'width:' + w.toFixed(2) + '%;--d:' + (150 + i * 110) + 'ms', title: r.name + ' markets: rank IC ' + r.v })]),
      H('div', { class: 'val', text: (r.v > 0 ? '+' : '−') + Math.abs(r.v).toFixed(3) })
    ]));
  });
  el.setAttribute('role', 'img');
  el.setAttribute('aria-label', 'Rank IC: calm markets minus 0.079, normal plus 0.076, turbulent plus 0.142.');
}

/* ------------------------------------------------------------------ 3. engine */
var root = document.documentElement;
var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
var N = slides.length;

charts();
slides.forEach(function (s, i) {
  s.appendChild(H('div', { class: 'foot' }, [
    H('span', { text: i === 0 ? '' : 'AI Portfolio Advisor' }),
    H('span', { text: (i + 1) + ' / ' + N })
  ]));
  s._fr = Array.prototype.slice.call(s.querySelectorAll('[data-f]'));
  s._max = s._fr.reduce(function (m, el) { return Math.max(m, +el.getAttribute('data-f')); }, 0);
});

var q = new URLSearchParams(location.search);
var PRINT = q.has('print');
var EXPORT = parseInt(q.get('export'), 10);
var REDUCED = q.has('static') || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
var ANIM = false, cur = 0, step = 0, busy = false, overview = false;

function ready() {
  var done = function () { root.setAttribute('data-ready', '1'); };
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(done, done); else done();
}

if (PRINT) { root.classList.add('print'); ready(); return; }

try {
  root.classList.add('js');
  if (EXPORT >= 1 && EXPORT <= N) {
    root.classList.add('export');
    slides[EXPORT - 1].classList.add('active');
    ready();
    return;
  }
  ANIM = !REDUCED && typeof document.body.animate === 'function';
  if (ANIM) root.classList.add('anim');
  start();
  ready();
} catch (e) {
  root.classList.remove('js', 'anim', 'overview');
  if (window.console) console.error('deck engine failed; showing all slides', e);
}

function countUp(el) {
  var final = el.textContent, to = parseFloat(el.getAttribute('data-count')), dec = +(el.getAttribute('data-dec') || 0), sep = el.hasAttribute('data-sep');
  if (!isFinite(to)) return;
  var t0 = null, dur = 1000, id = (el._cid = (el._cid || 0) + 1);
  function fmt(v) { return sep ? Math.round(v).toLocaleString('en-US') : v.toFixed(dec); }
  el.textContent = fmt(0);
  function tick(t) {
    if (el._cid !== id) return;
    if (t0 === null) t0 = t;
    var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 4);
    el.textContent = p < 1 ? fmt(to * e) : final;
    if (p < 1) requestAnimationFrame(tick);
  }
  setTimeout(function () { requestAnimationFrame(tick); }, 220);
  setTimeout(function () { if (el._cid === id) el.textContent = final; }, 1700);   /* never leave a half-counted number */
}

function apply(s, st) {
  var all = !ANIM || overview, focus = s.hasAttribute('data-focus');
  s._fr.forEach(function (el) {
    var f = +el.getAttribute('data-f'), on = all || f <= st, was = el.classList.contains('on');
    if (on && !was) {
      el.classList.add('on');
      if (ANIM && !overview) Array.prototype.forEach.call(el.querySelectorAll('[data-count]'), function (c) { if (c.closest('[data-f]') === el) countUp(c); });
    } else if (!on && was) el.classList.remove('on');
    var until = +(el.getAttribute('data-until') || f);
    el.classList.toggle('past', !all && focus && st < s._max && f > 0 && until < st);
  });
  for (var k = 1; k <= s._max; k++) s.classList.toggle('ge' + k, all || st >= k);
}
function clear(s) { s._fr.forEach(function (el) { el.classList.remove('on', 'past'); }); }

function chrome() {
  var bar = document.getElementById('progress');
  if (!bar.children.length) for (var i = 0; i < N; i++) bar.appendChild(H('i', {}, [H('b')]));
  Array.prototype.forEach.call(bar.children, function (seg, i) {
    var m = slides[i]._max, v = i < cur ? 1 : i > cur ? 0 : (ANIM ? (step + 1) / (m + 1) : 1);
    seg.firstChild.style.transform = 'scaleX(' + v + ')';
    seg.classList.toggle('cur', i === cur);
  });
  var h = '#/' + (cur + 1) + (ANIM && step ? '/' + step : '');
  if (location.hash !== h) { try { history.replaceState(null, '', h); } catch (e) { location.hash = h; } }
  document.title = slides[cur].getAttribute('data-title') + ' | AI Portfolio Advisor';
}

function swap(i, st, dir, animate) {
  var old = slides[cur], nw = slides[i];
  if (old !== nw) {
    old.classList.remove('active');
    if (animate) {
      old.classList.add('leaving');
      var a = old.animate([{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'translateX(' + (-70 * dir) + 'px) scale(.985)' }],
        { duration: 280, easing: 'cubic-bezier(.5,0,.8,.3)', fill: 'forwards' });
      var end = function () { old.classList.remove('leaving'); a.cancel(); if (old !== slides[cur]) clear(old); };
      a.onfinish = end; a.oncancel = function () { old.classList.remove('leaving'); };
      nw.animate([{ opacity: 0, transform: 'translateX(' + (90 * dir) + 'px)' }, { opacity: 1, transform: 'none' }],
        { duration: 640, easing: 'cubic-bezier(.2,.7,.1,1)', delay: 240, fill: 'backwards' });
    } else clear(old);
    nw.classList.add('active');
    if (ANIM && st === 0) { clear(nw); void nw.offsetWidth; }   /* so step 0 plays its entrance */
  }
  cur = i; step = st;
  apply(nw, st);
  chrome();
}

function go(i, st, dir) {
  i = Math.max(0, Math.min(N - 1, i));
  st = Math.max(0, Math.min(slides[i]._max, st || 0));
  if (!ANIM) st = slides[i]._max;
  if (busy) return;
  var crossing = ANIM && !overview && i !== cur && dir > 0 && slides[i].getAttribute('data-section') !== slides[cur].getAttribute('data-section');
  if (crossing) {
    /* a marigold band carries the section name across the stage; the slide changes underneath it */
    var sw = document.getElementById('sweep');
    sw.firstChild.textContent = slides[i].getAttribute('data-section');
    busy = true;
    sw.animate([
      { transform: 'translateX(-101%)', offset: 0 }, { transform: 'translateX(0)', offset: 0.3, easing: 'linear' },
      { transform: 'translateX(0)', offset: 0.72, easing: 'cubic-bezier(.6,0,.9,.4)' }, { transform: 'translateX(101%)', offset: 1 }
    ], { duration: 1500, easing: 'cubic-bezier(.2,.7,.1,1)' });
    setTimeout(function () { busy = false; swap(i, st, dir, false); }, 520);
    return;
  }
  swap(i, st, dir, ANIM && !overview && i !== cur);
}
function next() { hideHint(); if (ANIM && step < slides[cur]._max) go(cur, step + 1, 1); else if (cur < N - 1) go(cur + 1, 0, 1); }
function prev() { hideHint(); if (ANIM && step > 0) go(cur, step - 1, -1); else if (cur > 0) go(cur - 1, slides[cur - 1]._max, -1); }

function fromHash() {
  var m = /^#\/?(\d+)(?:\/(\d+))?/.exec(location.hash);
  return m ? [parseInt(m[1], 10) - 1, parseInt(m[2] || '0', 10)] : [0, 0];
}
function hideHint() { var h = document.getElementById('hint'); if (h) h.classList.add('gone'); }

function setOverview(on) {
  overview = on;
  root.classList.toggle('overview', on);
  if (on) { sizeOverview(); slides.forEach(function (s) { apply(s, s._max); }); slides[cur].scrollIntoView({ block: 'center' }); }
  else { slides.forEach(function (s, i) { if (i !== cur) clear(s); }); apply(slides[cur], step); document.getElementById('deck').scrollTop = 0; }
}
function sizeOverview() {
  var w = document.getElementById('deck').clientWidth, cols = w >= 1000 ? 3 : w >= 600 ? 2 : 1;
  root.style.setProperty('--cols', cols);
  root.style.setProperty('--oz', Math.max(0.05, (w - 40 - (cols - 1) * 20) / cols / 1920));
}

function start() {
  var h = fromHash();
  cur = Math.max(0, Math.min(N - 1, h[0]));
  step = ANIM ? Math.max(0, Math.min(slides[cur]._max, h[1])) : slides[cur]._max;
  slides[cur].classList.add('active');
  chrome();
  /* first paint with fragments hidden, then reveal: the title draws itself */
  requestAnimationFrame(function () { requestAnimationFrame(function () { apply(slides[cur], step); }); });

  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var k = e.key;
    if (k === 'o' || k === 'O' || k === 'g' || k === 'G') { setOverview(!overview); e.preventDefault(); return; }
    if (k === 'Escape' && overview) { setOverview(false); return; }
    if (k === 'f' || k === 'F') { if (document.fullscreenElement) document.exitFullscreen(); else if (root.requestFullscreen) root.requestFullscreen(); return; }
    if (overview) {
      if (k === 'ArrowRight') { cur = Math.min(N - 1, cur + 1); } else if (k === 'ArrowLeft') { cur = Math.max(0, cur - 1); }
      else if (k === 'Enter' || k === ' ') { e.preventDefault(); step = 0; slides.forEach(function (s, i) { s.classList.toggle('active', i === cur); }); setOverview(false); chrome(); return; }
      else return;
      slides.forEach(function (s, i) { s.classList.toggle('active', i === cur); });
      slides[cur].scrollIntoView({ block: 'nearest' });
      return;
    }
    if (k === 'ArrowRight' || k === ' ' || k === 'PageDown' || k === 'Enter' || k === 'ArrowDown') { e.preventDefault(); next(); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'Backspace' || k === 'ArrowUp') { e.preventDefault(); prev(); }
    else if (k === 'Home') { e.preventDefault(); go(0, 0, -1); }
    else if (k === 'End') { e.preventDefault(); go(N - 1, slides[N - 1]._max, 1); }
  });

  var deck = document.getElementById('deck'), touch = null, swiped = 0;
  deck.addEventListener('click', function (e) {
    if (overview) {
      var s = e.target.closest ? e.target.closest('.slide') : null;
      if (s) { var i = slides.indexOf(s); slides.forEach(function (x, j) { x.classList.toggle('active', j === i); }); cur = i; step = 0; setOverview(false); chrome(); }
      return;
    }
    if (Date.now() - swiped < 400) return;
    if (e.target.closest && e.target.closest('a,button')) return;
    if (e.clientX / window.innerWidth < 0.3) prev(); else next();
  });
  deck.addEventListener('touchstart', function (e) { var t = e.changedTouches[0]; touch = { x: t.clientX, y: t.clientY }; }, { passive: true });
  deck.addEventListener('touchend', function (e) {
    if (!touch || overview) return;
    var t = e.changedTouches[0], dx = t.clientX - touch.x, dy = t.clientY - touch.y; touch = null;
    if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy) * 1.3) { swiped = Date.now(); if (dx < 0) next(); else prev(); }
  }, { passive: true });

  window.addEventListener('hashchange', function () {
    var h = fromHash();
    if (h[0] !== cur || (ANIM && h[1] !== step)) { busy = false; go(h[0], h[1], h[0] >= cur ? 1 : -1); }
  });
  window.addEventListener('resize', function () { if (overview) sizeOverview(); });
  window.addEventListener('beforeprint', function () { root.classList.remove('js', 'anim', 'overview'); root.classList.add('print'); });
  window.addEventListener('afterprint', function () { root.classList.remove('print'); root.classList.add('js'); if (ANIM) root.classList.add('anim'); });
  setTimeout(hideHint, 7000);
}
})();
