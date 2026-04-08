"""
visualize_circuit_supernode.py
─────────────────────────────────────────────────────────────────────────────
Enhanced circuit graph visualization with supernode overlay support.

Loads:
  - subgraph/austin_plt.pt      (raw circuit graph)
  - supernode_map.json          (output of structure_grouping.py)

Features:
  - Toggle button: show/hide supernode overlay
  - When ON:  raw nodes colored by their supernode cluster
              supernode convex-hull regions drawn behind nodes
              sidebar shows supernode membership
  - When OFF: original node coloring (node_inf gradient)
  - Supernode map loaded from JSON — no hardcoding

Usage:
  python visualize_circuit_supernode.py \\
      --file subgraph/austin_plt.pt \\
      --supernode supernode_map.json \\
      --out circuit_supernode.html
"""

import json
import argparse
import torch


def parse_layer(nid):
    if nid.startswith('E'): return 0
    if nid.startswith('27'): return 27
    return int(nid.split('_')[0])


def parse_activation(attr_node):
    a = attr_node['activation']
    if a is not None: return float(a)
    if attr_node.get('is_target_logit', False):
        return float(attr_node.get('token_prob', 0)) * 100
    inf = attr_node.get('influence', 0) or 0
    return float(inf) * 100


def load_snapshot(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def prepare_graph_data(raw):
    kept_ids  = raw['kept_ids']
    adj       = raw['pruned_adj'].clone().float()
    attr      = raw['attr']
    node_inf  = raw['node_inf'].float()
    node_rel  = raw['node_rel'].float()
    logit_idx = len(kept_ids) - 1

    for i, nid in enumerate(kept_ids):
        inf = attr[nid].get('influence')
        if inf is not None:
            adj[i, logit_idx] = float(inf)

    layers = [parse_layer(n) for n in kept_ids]
    acts   = [parse_activation(attr[n]) for n in kept_ids]

    nodes = []
    for i, nid in enumerate(kept_ids):
        a = attr[nid]
        layer = layers[i]
        is_emb = nid.startswith('E')
        is_log = a.get('is_target_logit', False)
        if is_emb:   ntype = 'embedding'
        elif is_log: ntype = 'logit'
        else:        ntype = 'transcoder'
        nodes.append({
            'id':         nid,
            'idx':        i,
            'layer':      layer,
            'clerp':      a['clerp'],
            'activation': acts[i],
            'influence':  float(a.get('influence') or 0),
            'node_inf':   float(node_inf[i]),
            'node_rel':   float(node_rel[i]),
            'type':       ntype,
            'ctx_idx':    a.get('ctx_idx', 0),
        })

    act_tensor = torch.tensor(acts, dtype=torch.float32)
    flow_mat   = act_tensor.unsqueeze(1) * adj
    N          = len(kept_ids)
    logit_id   = kept_ids[logit_idx]

    edges = []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            w    = adj[i, j].item()
            flow = flow_mat[i, j].item()
            if w == 0.0: continue
            edges.append({
                'source':   kept_ids[i],
                'target':   kept_ids[j],
                'weight':   w,
                'flow':     flow,
                'abs_flow': abs(flow),
                'to_logit': kept_ids[j] == logit_id,
            })

    print(f"  Total edges:    {len(edges)}")
    print(f"  Edges to logit: {sum(1 for e in edges if e['to_logit'])}")
    return {'nodes': nodes, 'edges': edges}


# ─── Build node→supernode lookup ──────────────────────────────────────────────

def build_supernode_lookup(supernode_map: dict) -> dict:
    """Returns {node_id: supernode_name}"""
    lookup = {}
    for sn_name, members in supernode_map.items():
        for nid in members:
            lookup[nid] = sn_name
    return lookup


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Circuit Graph — Austin · Supernode View</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg:        #060910;
  --bg2:       #0b1020;
  --border:    #1a2540;
  --text:      #c8d4e8;
  --dim:       #3a4a60;
  --accent:    #3d8eff;
  --gold:      #f5a623;
  --red:       #f43f5e;
  --green:     #10b981;
  --purple:    #8b5cf6;
}

* { box-sizing:border-box; margin:0; padding:0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  overflow: hidden;
}

#app { display:flex; height:100vh; width:100vw; }

/* ── Sidebar ── */
#sidebar {
  width: 300px;
  min-width: 300px;
  background: var(--bg2);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#sidebar-header {
  padding: 18px 16px 14px;
  border-bottom: 1px solid var(--border);
}

#sidebar-header h2 {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 13px;
  letter-spacing: .08em;
  color: var(--accent);
  margin-bottom: 3px;
}

#sidebar-header p {
  font-size: 9px;
  color: var(--dim);
  letter-spacing: .06em;
}

/* ── Supernode toggle button ── */
#sn-toggle-wrap {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}

#sn-toggle {
  width: 100%;
  padding: 8px 12px;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--dim);
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: .08em;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s;
}

#sn-toggle:hover {
  border-color: var(--accent);
  color: var(--accent);
}

#sn-toggle.active {
  background: rgba(61,142,255,0.08);
  border-color: var(--accent);
  color: var(--accent);
}

#sn-toggle .dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--dim);
  flex-shrink: 0;
  transition: background 0.2s;
}

#sn-toggle.active .dot { background: var(--accent); }

/* ── Controls ── */
#controls {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.control-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 9px;
  color: var(--dim);
}

.control-row label { min-width: 78px; letter-spacing:.04em; }

input[type=range] {
  flex: 1;
  accent-color: var(--accent);
  height: 2px;
}

.val-display {
  min-width: 28px;
  text-align: right;
  color: var(--accent);
  font-size: 9px;
}

/* ── Legend ── */
#legend {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
}

#legend h3 {
  font-size: 8px;
  letter-spacing: .12em;
  color: var(--dim);
  text-transform: uppercase;
  margin-bottom: 7px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 4px;
  font-size: 9px;
  color: var(--dim);
}

.legend-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }

/* ── Supernode legend (shown when active) ── */
#sn-legend {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  display: none;
  max-height: 200px;
  overflow-y: auto;
}

#sn-legend.visible { display: block; }

#sn-legend h3 {
  font-size: 8px;
  letter-spacing: .12em;
  color: var(--dim);
  text-transform: uppercase;
  margin-bottom: 7px;
}

.sn-legend-item {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 4px;
  font-size: 9px;
  color: var(--dim);
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 3px;
  transition: background 0.15s;
}

.sn-legend-item:hover { background: rgba(255,255,255,0.04); color: var(--text); }
.sn-legend-item .sn-dot { width:8px; height:8px; border-radius:2px; flex-shrink:0; }
.sn-legend-item .sn-count { margin-left:auto; opacity:.5; font-size:8px; }

/* ── Node info panel ── */
#node-info {
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
}

#node-info h3 {
  font-size: 8px;
  letter-spacing: .12em;
  color: var(--dim);
  text-transform: uppercase;
  margin-bottom: 10px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 9px;
}

.info-label { color: var(--dim); }
.info-value { color: var(--text); }

.info-clerp {
  font-size: 10px;
  color: var(--accent);
  margin-bottom: 10px;
  line-height: 1.5;
  padding: 8px 10px;
  background: rgba(61,142,255,0.06);
  border-radius: 3px;
  border-left: 2px solid var(--accent);
}

.info-sn-badge {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 3px;
  font-size: 8px;
  letter-spacing: .05em;
  margin-bottom: 8px;
}

.bar-container { margin-top: 4px; margin-bottom: 5px; }
.bar-label { font-size: 8px; color: var(--dim); margin-bottom: 2px; }
.bar-track { height: 3px; background: rgba(255,255,255,0.05); border-radius:2px; overflow:hidden; }
.bar-fill  { height: 100%; border-radius:2px; transition: width 0.3s; }

/* ── Canvas ── */
#canvas { flex:1; position:relative; overflow:hidden; }
svg { width:100%; height:100%; }

.node circle { cursor:pointer; transition: r 0.15s; }
.node:hover circle { filter: brightness(1.3); }
.node text { pointer-events:none; font-family:'JetBrains Mono', monospace; }

.link { fill:none; stroke-linecap:round; }
.layer-line { stroke-dasharray:2,8; }

/* Supernode hull regions */
.sn-hull {
  fill-opacity: 0.06;
  stroke-width: 1.5;
  stroke-dasharray: none;
  stroke-opacity: 0.35;
  pointer-events: none;
  transition: fill-opacity 0.3s;
}

.sn-hull:hover { fill-opacity: 0.12; }

.sn-label {
  pointer-events: none;
  font-family: 'Syne', sans-serif;
  font-size: 8px;
  font-weight: 600;
  letter-spacing: .06em;
  opacity: 0;
  transition: opacity 0.3s;
}

.sn-label.visible { opacity: 0.55; }

#title-overlay {
  position: absolute;
  top: 14px; left: 14px;
  font-size: 8px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--dim);
  pointer-events: none;
  font-family: 'JetBrains Mono', monospace;
}

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:2px; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">

    <div id="sidebar-header">
      <h2>Circuit Graph</h2>
      <p>Gemma-2-2B &middot; Austin prediction &middot; supernode overlay</p>
    </div>

    <div id="sn-toggle-wrap">
      <button id="sn-toggle" onclick="toggleSupernode()">
        <span class="dot"></span>
        <span id="sn-toggle-label">SHOW SUPERNODES</span>
      </button>
    </div>

    <div id="controls">
      <div class="control-row">
        <label>Flow thresh</label>
        <input type="range" id="flow-thresh" min="0" max="500" value="10" step="5">
        <span class="val-display" id="flow-val">10</span>
      </div>
      <div class="control-row">
        <label>Node size</label>
        <input type="range" id="node-size" min="3" max="20" value="8" step="1">
        <span class="val-display" id="size-val">8</span>
      </div>
      <div class="control-row">
        <label>Edge opacity</label>
        <input type="range" id="edge-opacity" min="5" max="100" value="30" step="5">
        <span class="val-display" id="opacity-val">30</span>
      </div>
    </div>

    <!-- Normal legend (hidden when SN active) -->
    <div id="legend">
      <h3>Legend</h3>
      <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Embedding</div>
      <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Transcoder low inf</div>
      <div class="legend-item"><div class="legend-dot" style="background:#3d8eff"></div>Transcoder high inf</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f43f5e"></div>Logit output</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f5a623"></div>Edge &rarr; Logit</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>Suppression (&minus;)</div>
    </div>

    <!-- Supernode legend (shown when SN active) -->
    <div id="sn-legend">
      <h3>Supernodes</h3>
      <div id="sn-legend-items"></div>
    </div>

    <div id="node-info">
      <h3>Node inspector</h3>
      <p style="font-size:9px;color:var(--dim)">Click a node to inspect</p>
    </div>

  </div>

  <div id="canvas">
    <div id="title-overlay">prompt: "Fact: The capital of the state containing Dallas is"</div>
    <svg id="svg"></svg>
  </div>
</div>

<script>
const GRAPH_DATA    = __GRAPH_DATA__;
const SUPERNODE_MAP = __SUPERNODE_MAP__;

// ── Supernode color palette (distinct, readable on dark bg) ──────────────────
const SN_PALETTE = [
  '#3d8eff','#f5a623','#10b981','#f43f5e','#8b5cf6',
  '#06b6d4','#ec4899','#84cc16','#f97316','#14b8a6',
  '#a78bfa','#fb7185',
];

// Build lookup: node_id → supernode name
const node2sn = {};
const snNames = Object.keys(SUPERNODE_MAP);
snNames.forEach(sn => {
  SUPERNODE_MAP[sn].forEach(nid => { node2sn[nid] = sn; });
});

// Assign color per supernode (EMB/LOGIT get special colors)
const snColor = {};
let ci = 0;
snNames.forEach(sn => {
  if (sn === 'SN_EMB')   { snColor[sn] = '#f5a623'; return; }
  if (sn === 'SN_LOGIT') { snColor[sn] = '#f43f5e'; return; }
  snColor[sn] = SN_PALETTE[ci++ % SN_PALETTE.length];
});

// ── State ────────────────────────────────────────────────────────────────────
let supernodeActive   = false;
let currentFlowThresh = 10;
let currentNodeSize   = 8;
let currentOpacity    = 0.30;
let highlightedSN     = null;

// ── Toggle ───────────────────────────────────────────────────────────────────
function toggleSupernode() {
  supernodeActive = !supernodeActive;
  const btn   = document.getElementById('sn-toggle');
  const label = document.getElementById('sn-toggle-label');
  const leg   = document.getElementById('legend');
  const snLeg = document.getElementById('sn-legend');

  if (supernodeActive) {
    btn.classList.add('active');
    label.textContent = 'HIDE SUPERNODES';
    leg.style.display  = 'none';
    snLeg.classList.add('visible');
  } else {
    btn.classList.remove('active');
    label.textContent = 'SHOW SUPERNODES';
    leg.style.display  = 'block';
    snLeg.classList.remove('visible');
    highlightedSN = null;
  }
  render();
}

// ── Build supernode legend items ──────────────────────────────────────────────
function buildSNLegend() {
  const container = document.getElementById('sn-legend-items');
  container.innerHTML = '';
  snNames.forEach(sn => {
    const members = SUPERNODE_MAP[sn];
    const color   = snColor[sn];
    const div     = document.createElement('div');
    div.className = 'sn-legend-item';
    div.innerHTML = `
      <span class="sn-dot" style="background:${color}"></span>
      <span>${sn}</span>
      <span class="sn-count">${members.length}n</span>`;
    div.addEventListener('click', () => {
      highlightedSN = (highlightedSN === sn) ? null : sn;
      renderSNHighlight();
    });
    container.appendChild(div);
  });
}

buildSNLegend();

// ── D3 setup ─────────────────────────────────────────────────────────────────
const svg    = d3.select('#svg');
const width  = () => document.getElementById('canvas').clientWidth;
const height = () => document.getElementById('canvas').clientHeight;

const allLayers  = [...new Set(GRAPH_DATA.nodes.map(d => d.layer))].sort((a,b)=>a-b);
const layerNodes = {};
GRAPH_DATA.nodes.forEach(d => {
  if (!layerNodes[d.layer]) layerNodes[d.layer] = [];
  layerNodes[d.layer].push(d);
});
const nodeLayerMap = {};
GRAPH_DATA.nodes.forEach(d => { nodeLayerMap[d.id] = d.layer; });

const maxFlow = d3.max(GRAPH_DATA.edges.map(e => e.abs_flow)) || 1;

function computePositions() {
  const W = width(), H = height();
  const layerX = d3.scalePoint().domain(allLayers).range([80, W-80]).padding(0.3);
  GRAPH_DATA.nodes.forEach(d => {
    const group = layerNodes[d.layer];
    const idx   = group.indexOf(d);
    const n     = group.length;
    const margin = 50;
    const step   = (H - 2*margin) / Math.max(n, 1);
    d.x = layerX(d.layer);
    d.y = n === 1 ? H/2 : margin + idx*step + step/2;
  });
}

// ── Color helpers ─────────────────────────────────────────────────────────────
const colorNodeDefault = d => {
  if (d.type === 'embedding') return '#f59e0b';
  if (d.type === 'logit')     return '#f43f5e';
  return d3.interpolateRgb('#10b981', '#3d8eff')(d.node_inf);
};

const colorNodeSN = d => {
  const sn = node2sn[d.id];
  return sn ? snColor[sn] : '#334155';
};

// ── Convex hull per supernode ─────────────────────────────────────────────────
function computeHulls(pos) {
  const hulls = {};
  snNames.forEach(sn => {
    const pts = SUPERNODE_MAP[sn]
      .filter(nid => pos[nid])
      .map(nid => [pos[nid].x, pos[nid].y]);
    if (pts.length < 1) return;
    if (pts.length === 1) {
      // Single point — draw a small circle approximation
      const r = currentNodeSize * 2.5;
      const [cx, cy] = pts[0];
      hulls[sn] = `M${cx-r},${cy} A${r},${r} 0 1,0 ${cx+r},${cy} A${r},${r} 0 1,0 ${cx-r},${cy}`;
      return;
    }
    if (pts.length === 2) {
      // Two points — pad to a capsule
      const r = currentNodeSize * 2.2;
      pts.push([pts[0][0]+0.1, pts[0][1]+r]);
    }
    try {
      const hull = d3.polygonHull(pts);
      if (!hull) return;
      // Inflate hull outward
      const centroid = d3.polygonCentroid(hull);
      const pad = currentNodeSize * 2.8;
      const padded = hull.map(([x, y]) => {
        const dx = x - centroid[0], dy = y - centroid[1];
        const dist = Math.sqrt(dx*dx + dy*dy) || 1;
        return [x + dx/dist*pad, y + dy/dist*pad];
      });
      hulls[sn] = `M${padded.map(p=>p.join(',')).join('L')}Z`;
    } catch(e) {}
  });
  return hulls;
}

// ── Main render ───────────────────────────────────────────────────────────────
function render() {
  svg.selectAll('*').remove();
  const W = width(), H = height();
  svg.attr('viewBox', `0 0 ${W} ${H}`);
  computePositions();

  const layerX = d3.scalePoint().domain(allLayers).range([80, W-80]).padding(0.3);
  const pos    = {};
  GRAPH_DATA.nodes.forEach(d => { pos[d.id] = {x: d.x, y: d.y}; });

  // ── Defs (arrowheads) ──
  const defs = svg.append('defs');
  [{id:'fwd',color:'#3d8eff'},{id:'bwd',color:'#f87171'},
   {id:'neg',color:'#8b5cf6'},{id:'logit',color:'#f5a623'}].forEach(({id,color}) => {
    defs.append('marker')
      .attr('id',`arrow-${id}`).attr('viewBox','0 -4 8 8')
      .attr('refX',8).attr('refY',0)
      .attr('markerWidth',5).attr('markerHeight',5)
      .attr('orient','auto')
      .append('path').attr('d','M0,-4L8,0L0,4')
      .attr('fill',color).attr('opacity',0.8);
  });

  // ── Layer guide lines ──
  allLayers.forEach(l => {
    const x = layerX(l);
    svg.append('line').attr('class','layer-line')
      .attr('x1',x).attr('y1',22).attr('x2',x).attr('y2',H-8)
      .attr('stroke','#1a2540').attr('stroke-width',1);
    svg.append('text').attr('x',x).attr('y',14)
      .attr('text-anchor','middle').attr('fill','#253550')
      .attr('font-size',9).attr('font-family','JetBrains Mono')
      .text(l===0?'EMB':l===27?'LOGIT':`L${l}`);
  });

  // ── Supernode hulls (drawn before edges and nodes) ──
  if (supernodeActive) {
    const hulls   = computeHulls(pos);
    const hullG   = svg.append('g').attr('id','hulls');
    const labelG  = svg.append('g').attr('id','hull-labels');

    snNames.forEach(sn => {
      if (!hulls[sn]) return;
      const color = snColor[sn];
      const dim   = highlightedSN && highlightedSN !== sn;

      hullG.append('path')
        .attr('class', 'sn-hull')
        .attr('d', hulls[sn])
        .attr('fill', color)
        .attr('stroke', color)
        .attr('fill-opacity', dim ? 0.01 : 0.07)
        .attr('stroke-opacity', dim ? 0.08 : 0.38);

      // Label at centroid of members
      const members = SUPERNODE_MAP[sn].filter(nid => pos[nid]);
      if (members.length === 0) return;
      const cx = members.reduce((s,n)=>s+pos[n].x,0)/members.length;
      const cy = members.reduce((s,n)=>s+pos[n].y,0)/members.length - currentNodeSize*3.5;

      labelG.append('text')
        .attr('class','sn-label visible')
        .attr('x', cx).attr('y', cy)
        .attr('text-anchor','middle')
        .attr('fill', color)
        .attr('opacity', dim ? 0.1 : 0.5)
        .text(sn.replace('SN_',''));
    });
  }

  // ── Edges ──
  const edgeG      = svg.append('g').attr('id','edges-regular');
  const logitEdgeG = svg.append('g').attr('id','edges-logit');

  GRAPH_DATA.edges.forEach(e => {
    const src = pos[e.source];
    const tgt = pos[e.target];
    if (!src || !tgt) return;

    const isToLogit = e.to_logit === true;
    const isNeg     = e.flow < 0;
    const srcL      = nodeLayerMap[e.source] || 0;
    const tgtL      = nodeLayerMap[e.target] || 0;
    const isBwd     = srcL > tgtL && !isToLogit;

    if (!isToLogit && e.abs_flow < currentFlowThresh) return;

    const strokeW = Math.max(0.3, (e.abs_flow / maxFlow) * 5);
    const opacity = isToLogit
      ? Math.max(0.12, (e.abs_flow / maxFlow) * currentOpacity * 3)
      : currentOpacity * (isNeg ? 0.4 : 1.0);

    // In supernode mode, color edges by source supernode
    let color, arrowId;
    if (supernodeActive && !isToLogit) {
      const sn = node2sn[e.source];
      color   = sn ? snColor[sn] : '#3d8eff';
      arrowId = isBwd ? 'bwd' : 'fwd';
    } else {
      color   = isToLogit ? '#f5a623' : isNeg ? '#8b5cf6' : isBwd ? '#f87171' : '#3d8eff';
      arrowId = isToLogit ? 'logit'   : isNeg ? 'neg'     : isBwd ? 'bwd'     : 'fwd';
    }

    const dx   = tgt.x - src.x;
    const dy   = tgt.y - src.y;
    const dist = Math.sqrt(dx*dx + dy*dy) + 1;
    const curv = isToLogit ? 0 : (isBwd ? -35 : 25);
    const cx   = (src.x+tgt.x)/2 - dy*curv/dist;
    const cy2  = (src.y+tgt.y)/2 + dx*curv/dist;

    const g = isToLogit ? logitEdgeG : edgeG;
    g.append('path')
      .attr('class','link')
      .datum(e)
      .attr('d',`M${src.x},${src.y} Q${cx},${cy2} ${tgt.x},${tgt.y}`)
      .attr('stroke', color)
      .attr('stroke-width', strokeW)
      .attr('opacity', opacity)
      .attr('marker-end',`url(#arrow-${arrowId})`);
  });

  // ── Nodes ──
  const nodeG   = svg.append('g').attr('id','nodes');
  const nodeEls = nodeG.selectAll('.node')
    .data(GRAPH_DATA.nodes).enter()
    .append('g').attr('class','node')
    .attr('transform', d => `translate(${d.x},${d.y})`)
    .style('cursor','pointer')
    .on('click',      (ev,d) => showNodeInfo(d))
    .on('mouseenter', (ev,d) => highlightNode(d, true))
    .on('mouseleave', (ev,d) => highlightNode(d, false));

  nodeEls.append('circle')
    .attr('r', d => {
      const b = currentNodeSize;
      if (d.type==='embedding') return b*1.15;
      if (d.type==='logit')     return b*1.65;
      return b*(0.5 + 0.9*d.node_inf);
    })
    .attr('fill', d => supernodeActive ? colorNodeSN(d) : colorNodeDefault(d))
    .attr('stroke', d => {
      if (d.type==='logit') return '#f5a623';
      if (supernodeActive) {
        const sn = node2sn[d.id];
        return sn ? snColor[sn] : 'none';
      }
      return 'none';
    })
    .attr('stroke-width', d => d.type==='logit' ? 2 : (supernodeActive ? 0.8 : 0))
    .attr('opacity', d => d.type==='transcoder' ? 0.82 : 1.0);

  nodeEls.append('text')
    .attr('dy', d => {
      const r = currentNodeSize*(d.type==='logit' ? 1.65 : 0.5+0.9*d.node_inf);
      return -(r+3);
    })
    .attr('text-anchor','middle')
    .attr('fill','#6a7f9a').attr('font-size',7)
    .text(d => {
      if (d.type==='embedding') return d.clerp.replace(/Emb:\s*"?\s*/,'').replace('"','').trim();
      if (d.type==='logit')     return 'Austin';
      const c = d.clerp;
      return c.length>12 ? c.slice(0,11)+'…' : c;
    });
}

// ── Highlight on hover ────────────────────────────────────────────────────────
function highlightNode(d, on) {
  d3.selectAll('.link').attr('opacity', function(e) {
    if (!e) return on ? 0.02 : currentOpacity;
    const isLogit   = e.to_logit === true;
    const baseLogit = Math.max(0.12, (e.abs_flow/maxFlow)*currentOpacity*3);
    if (!on) return isLogit ? baseLogit : currentOpacity;
    const connected = e.source===d.id || e.target===d.id;
    return connected ? 0.92 : (isLogit ? 0.06 : 0.02);
  });
}

// ── Supernode highlight from legend click ─────────────────────────────────────
function renderSNHighlight() {
  if (!supernodeActive) return;
  // Just re-render hulls with dim state
  svg.select('#hulls').remove();
  svg.select('#hull-labels').remove();

  const pos = {};
  GRAPH_DATA.nodes.forEach(d => { pos[d.id] = {x: d.x, y: d.y}; });
  const hulls  = computeHulls(pos);
  const hullG  = svg.insert('g','#edges-regular').attr('id','hulls');
  const labelG = svg.insert('g','#edges-regular').attr('id','hull-labels');

  snNames.forEach(sn => {
    if (!hulls[sn]) return;
    const color = snColor[sn];
    const dim   = highlightedSN && highlightedSN !== sn;

    hullG.append('path')
      .attr('class','sn-hull')
      .attr('d', hulls[sn])
      .attr('fill', color)
      .attr('stroke', color)
      .attr('fill-opacity', dim ? 0.01 : 0.1)
      .attr('stroke-opacity', dim ? 0.05 : 0.5);

    const members = SUPERNODE_MAP[sn].filter(nid => pos[nid]);
    if (members.length === 0) return;
    const cx = members.reduce((s,n)=>s+pos[n].x,0)/members.length;
    const cy = members.reduce((s,n)=>s+pos[n].y,0)/members.length - currentNodeSize*3.5;

    labelG.append('text')
      .attr('class','sn-label visible')
      .attr('x', cx).attr('y', cy)
      .attr('text-anchor','middle')
      .attr('fill', color)
      .attr('opacity', dim ? 0.08 : 0.6)
      .text(sn.replace('SN_',''));
  });

  // Also dim node circles not in highlighted SN
  if (highlightedSN) {
    const snMembers = new Set(SUPERNODE_MAP[highlightedSN] || []);
    d3.selectAll('.node circle').attr('opacity', d =>
      snMembers.has(d.id) ? 1.0 : 0.15
    );
  } else {
    d3.selectAll('.node circle').attr('opacity', d =>
      d.type==='transcoder' ? 0.82 : 1.0
    );
  }
}

// ── Node info panel ───────────────────────────────────────────────────────────
function showNodeInfo(d) {
  const panel    = document.getElementById('node-info');
  const maxAct   = Math.max(...GRAPH_DATA.nodes.map(n => n.activation));
  const outEdges = GRAPH_DATA.edges.filter(e => e.source===d.id);
  const inEdges  = GRAPH_DATA.edges.filter(e => e.target===d.id);
  const toLogit  = outEdges.find(e => e.to_logit);
  const fl       = toLogit ? toLogit.flow : null;
  const sn       = node2sn[d.id];
  const snCol    = sn ? snColor[sn] : null;

  const snBadge = sn
    ? `<div class="info-sn-badge" style="background:${snCol}22;color:${snCol};border:1px solid ${snCol}55">${sn}</div>`
    : '';

  panel.innerHTML = `
    <h3>Node inspector</h3>
    ${snBadge}
    <div class="info-clerp">${d.clerp}</div>
    <div class="info-row"><span class="info-label">ID</span><span class="info-value" style="font-size:8px">${d.id}</span></div>
    <div class="info-row"><span class="info-label">Layer</span><span class="info-value">${d.layer===0?'EMB':d.layer===27?'LOGIT':d.layer}</span></div>
    <div class="info-row"><span class="info-label">ctx_idx</span><span class="info-value">${d.ctx_idx}</span></div>
    <div class="bar-container">
      <div class="bar-label">Activation: ${d.activation.toFixed(2)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.activation/maxAct*100).toFixed(1)}%;background:#f59e0b"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">Influence: ${d.influence.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.influence*100).toFixed(1)}%;background:#f5a623"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">Flow→logit: ${fl!==null?fl.toFixed(2):'n/a'}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${fl!==null?Math.min(100,Math.abs(fl)/40*100).toFixed(1):0}%;background:#f43f5e"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">node_inf: ${d.node_inf.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.node_inf*100).toFixed(1)}%;background:#3d8eff"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">node_rel: ${d.node_rel.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.node_rel*100).toFixed(1)}%;background:#10b981"></div></div>
    </div>
    <div style="margin-top:8px;font-size:9px;color:var(--dim)">${outEdges.length} out &middot; ${inEdges.length} in</div>`;
}

// ── Slider controls ───────────────────────────────────────────────────────────
document.getElementById('flow-thresh').addEventListener('input', function() {
  currentFlowThresh = +this.value;
  document.getElementById('flow-val').textContent = this.value;
  render();
});
document.getElementById('node-size').addEventListener('input', function() {
  currentNodeSize = +this.value;
  document.getElementById('size-val').textContent = this.value;
  render();
});
document.getElementById('edge-opacity').addEventListener('input', function() {
  currentOpacity = this.value/100;
  document.getElementById('opacity-val').textContent = this.value;
  render();
});
window.addEventListener('resize', render);

render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',      type=str, default='subgraph/austin_plt.pt')
    parser.add_argument('--supernode', type=str, default='supernode_map.json')
    parser.add_argument('--out',       type=str, default='circuit_supernode.html')
    args = parser.parse_args()

    print(f"Loading {args.file}...")
    raw  = load_snapshot(args.file)
    data = prepare_graph_data(raw)
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Edges: {len(data['edges'])}")

    print(f"Loading supernode map: {args.supernode}...")
    with open(args.supernode, 'r') as f:
        supernode_map = json.load(f)
    print(f"  Supernodes: {len(supernode_map)}")

    graph_json = json.dumps(data,          indent=None, separators=(',', ':'))
    sn_json    = json.dumps(supernode_map, indent=None, separators=(',', ':'))

    html = HTML_TEMPLATE \
        .replace('__GRAPH_DATA__',    graph_json) \
        .replace('__SUPERNODE_MAP__', sn_json)

    with open(args.out, 'w') as f:
        f.write(html)

    print(f"Saved → {args.out}")
    print(f"Open:  open {args.out}")


if __name__ == '__main__':
    main()