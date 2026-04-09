"""
Generate circuit graph visualization as HTML with D3.js
Usage: python visualize_circuit.py --file subgraph/austin_plt.pt
"""

import json
import argparse
import torch

def parse_layer(nid):
    if nid.startswith('E'): return 0
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
            'id': nid, 'idx': i, 'layer': layer,
            'clerp': a['clerp'],
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


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Circuit Graph — Austin</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0a0e1a; color:#e0e6f0; font-family:'Courier New',monospace; overflow:hidden; }
#app { display:flex; height:100vh; width:100vw; }
#sidebar { width:280px; min-width:280px; background:#0d1220; border-right:1px solid #1e2d4a; display:flex; flex-direction:column; overflow:hidden; }
#sidebar-header { padding:16px; border-bottom:1px solid #1e2d4a; }
#sidebar-header h2 { font-size:11px; letter-spacing:.15em; text-transform:uppercase; color:#4a9eff; margin-bottom:4px; }
#sidebar-header p { font-size:10px; color:#4a5568; }
#controls { padding:12px 16px; border-bottom:1px solid #1e2d4a; display:flex; flex-direction:column; gap:8px; }
.control-row { display:flex; align-items:center; gap:8px; font-size:10px; color:#8899aa; }
.control-row label { min-width:80px; }
input[type=range] { flex:1; accent-color:#4a9eff; height:2px; }
.val-display { min-width:32px; text-align:right; color:#4a9eff; font-size:10px; }
#legend { padding:12px 16px; border-bottom:1px solid #1e2d4a; }
#legend h3 { font-size:10px; letter-spacing:.1em; color:#4a5568; text-transform:uppercase; margin-bottom:8px; }
.legend-item { display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:10px; color:#8899aa; }
.legend-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
#node-info { flex:1; overflow-y:auto; padding:12px 16px; }
#node-info h3 { font-size:10px; letter-spacing:.1em; color:#4a5568; text-transform:uppercase; margin-bottom:10px; }
.info-row { display:flex; justify-content:space-between; margin-bottom:5px; font-size:10px; }
.info-label { color:#4a5568; } .info-value { color:#e0e6f0; }
.info-clerp { font-size:11px; color:#4a9eff; margin-bottom:10px; line-height:1.4; padding:8px; background:#1a2535; border-radius:4px; border-left:2px solid #4a9eff; }
#canvas { flex:1; position:relative; overflow:hidden; }
svg { width:100%; height:100%; }
.node circle { cursor:pointer; }
.node:hover circle { opacity:1 !important; }
.node text { pointer-events:none; font-family:'Courier New',monospace; }
.link { fill:none; stroke-linecap:round; }
.layer-line { stroke-dasharray:3,6; }
#title-overlay { position:absolute; top:16px; left:16px; font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:#2a3a5a; pointer-events:none; }
.bar-container { margin-top:4px; margin-bottom:6px; }
.bar-label { font-size:9px; color:#4a5568; margin-bottom:2px; }
.bar-track { height:4px; background:#1a2535; border-radius:2px; overflow:hidden; }
.bar-fill { height:100%; border-radius:2px; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#0d1220; }
::-webkit-scrollbar-thumb { background:#1e2d4a; border-radius:2px; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div id="sidebar-header">
      <h2>Circuit Graph</h2>
      <p>Gemma-2-2B · Austin prediction</p>
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
    <div id="legend">
      <h3>Legend</h3>
      <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Embedding</div>
      <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Transcoder low inf</div>
      <div class="legend-item"><div class="legend-dot" style="background:#4a9eff"></div>Transcoder high inf</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f43f5e"></div>Logit output</div>
      <div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div>Edge → Logit</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>Suppression (−)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f87171"></div>Backward edge</div>
    </div>
    <div id="node-info">
      <h3>Node inspector</h3>
      <p style="font-size:10px;color:#4a5568">Click a node to inspect</p>
    </div>
  </div>
  <div id="canvas">
    <div id="title-overlay">prompt: "Fact: The capital of the state containing Dallas is"</div>
    <svg id="svg"></svg>
  </div>
</div>

<script>
const GRAPH_DATA = __GRAPH_DATA__;

const colorNode = d => {
  if (d.type === 'embedding') return '#f59e0b';
  if (d.type === 'logit')     return '#f43f5e';
  return d3.interpolateRgb('#10b981', '#4a9eff')(d.node_inf);
};

const svg    = d3.select('#svg');
const width  = () => document.getElementById('canvas').clientWidth;
const height = () => document.getElementById('canvas').clientHeight;

const allLayers = [...new Set(GRAPH_DATA.nodes.map(d => d.layer))].sort((a,b)=>a-b);
const layerNodes = {};
GRAPH_DATA.nodes.forEach(d => {
  if (!layerNodes[d.layer]) layerNodes[d.layer] = [];
  layerNodes[d.layer].push(d);
});
const nodeLayerMap = {};
GRAPH_DATA.nodes.forEach(d => { nodeLayerMap[d.id] = d.layer; });

// Precompute maxFlow once
const maxFlow = d3.max(GRAPH_DATA.edges.map(e => e.abs_flow)) || 1;

function computePositions() {
  const W = width(), H = height();
  const layerX = d3.scalePoint().domain(allLayers).range([80, W-80]).padding(0.3);
  GRAPH_DATA.nodes.forEach(d => {
    const group  = layerNodes[d.layer];
    const idx    = group.indexOf(d);
    const n      = group.length;
    const margin = 50;
    const step   = (H - 2*margin) / Math.max(n, 1);
    d.x = layerX(d.layer);
    d.y = n === 1 ? H/2 : margin + idx*step + step/2;
  });
}

let currentFlowThresh = 10;
let currentNodeSize   = 8;
let currentOpacity    = 0.30;

function render() {
  svg.selectAll('*').remove();
  const W = width(), H = height();
  svg.attr('viewBox', `0 0 ${W} ${H}`);
  computePositions();

  const layerX = d3.scalePoint().domain(allLayers).range([80, W-80]).padding(0.3);

  // Arrowhead markers
  const defs = svg.append('defs');
  [{id:'fwd',color:'#4a9eff'},{id:'bwd',color:'#f87171'},
   {id:'neg',color:'#8b5cf6'},{id:'logit',color:'#fbbf24'}].forEach(({id,color}) => {
    defs.append('marker')
      .attr('id',`arrow-${id}`).attr('viewBox','0 -4 8 8')
      .attr('refX',8).attr('refY',0)
      .attr('markerWidth',5).attr('markerHeight',5)
      .attr('orient','auto')
      .append('path').attr('d','M0,-4L8,0L0,4')
      .attr('fill',color).attr('opacity',0.85);
  });

  // Layer guide lines
  allLayers.forEach(l => {
    const x = layerX(l);
    svg.append('line').attr('class','layer-line')
      .attr('x1',x).attr('y1',22).attr('x2',x).attr('y2',H-8)
      .attr('stroke','#1a2535').attr('stroke-width',1);
    svg.append('text').attr('x',x).attr('y',14)
      .attr('text-anchor','middle').attr('fill','#2a3a5a')
      .attr('font-size',9).attr('font-family','Courier New')
      .text(l===0?'EMB':l===27?'LOGIT':`L${l}`);
  });

  // Position lookup
  const pos = {};
  GRAPH_DATA.nodes.forEach(d => { pos[d.id] = {x:d.x, y:d.y}; });

  // Two layers: regular edges underneath, logit edges on top
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

    // Logit edges always shown; others filtered by threshold
    if (!isToLogit && e.abs_flow < currentFlowThresh) return;

    // Uniform scale across all edges
    const strokeW = Math.max(0.3, (e.abs_flow / maxFlow) * 5);
    const opacity = isToLogit
      ? Math.max(0.15, (e.abs_flow / maxFlow) * currentOpacity * 3)
      : currentOpacity * (isNeg ? 0.5 : 1.0);

    // Color and arrow type
    const color   = isToLogit ? '#fbbf24'
                  : isNeg     ? '#8b5cf6'
                  : isBwd     ? '#f87171'
                  :             '#4a9eff';
    const arrowId = isToLogit ? 'logit'
                  : isNeg     ? 'neg'
                  : isBwd     ? 'bwd'
                  :             'fwd';

    const dx   = tgt.x - src.x;
    const dy   = tgt.y - src.y;
    const dist = Math.sqrt(dx*dx + dy*dy) + 1;
    const curv = isToLogit ? 0 : (isBwd ? -35 : 25);
    const cx   = (src.x+tgt.x)/2 - dy*curv/dist;
    const cy   = (src.y+tgt.y)/2 + dx*curv/dist;

    const g = isToLogit ? logitEdgeG : edgeG;
    g.append('path')
      .attr('class','link')
      .datum(e)
      .attr('d',`M${src.x},${src.y} Q${cx},${cy} ${tgt.x},${tgt.y}`)
      .attr('stroke',color)
      .attr('stroke-width',strokeW)
      .attr('opacity',opacity)
      .attr('marker-end',`url(#arrow-${arrowId})`);
  });

  // Nodes on top
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
      if (d.type==='embedding') return b*1.1;
      if (d.type==='logit')     return b*1.6;
      return b*(0.5 + 0.9*d.node_inf);
    })
    .attr('fill', colorNode)
    .attr('stroke', d => d.type==='logit' ? '#fbbf24' : 'none')
    .attr('stroke-width', d => d.type==='logit' ? 2.5 : 0)
    .attr('opacity', d => d.type==='transcoder' ? 0.80 : 1.0);

  nodeEls.append('text')
    .attr('dy', d => {
      const r = currentNodeSize*(d.type==='logit' ? 1.6 : 0.5+0.9*d.node_inf);
      return -(r+3);
    })
    .attr('text-anchor','middle')
    .attr('fill','#8899aa').attr('font-size',8)
    .text(d => {
      if (d.type==='embedding') return d.clerp.replace(/Emb:\s*"?\s*/,'').replace('"','').trim();
      if (d.type==='logit')     return 'Austin';
      const c = d.clerp;
      return c.length>12 ? c.slice(0,11)+'…' : c;
    });
}

function highlightNode(d, on) {
  d3.selectAll('.link').attr('opacity', function(e) {
    if (!e) return on ? 0.02 : currentOpacity;
    const isLogit   = e.to_logit === true;
    const baseLogit = Math.max(0.15, (e.abs_flow/maxFlow)*currentOpacity*3);
    if (!on) return isLogit ? baseLogit : currentOpacity;
    const connected = e.source===d.id || e.target===d.id;
    return connected ? 0.95 : (isLogit ? 0.08 : 0.02);
  });
}

function showNodeInfo(d) {
  const panel    = document.getElementById('node-info');
  const maxAct   = Math.max(...GRAPH_DATA.nodes.map(n => n.activation));
  const outEdges = GRAPH_DATA.edges.filter(e => e.source===d.id);
  const inEdges  = GRAPH_DATA.edges.filter(e => e.target===d.id);
  const toLogit  = outEdges.find(e => e.to_logit);
  const fl       = toLogit ? toLogit.flow : null;

  panel.innerHTML = `
    <h3>Node inspector</h3>
    <div class="info-clerp">${d.clerp}</div>
    <div class="info-row"><span class="info-label">ID</span><span class="info-value" style="font-size:9px">${d.id}</span></div>
    <div class="info-row"><span class="info-label">Layer</span><span class="info-value">${d.layer===0?'EMB':d.layer===27?'LOGIT':d.layer}</span></div>
    <div class="info-row"><span class="info-label">ctx_idx</span><span class="info-value">${d.ctx_idx}</span></div>
    <div class="bar-container">
      <div class="bar-label">Activation: ${d.activation.toFixed(2)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.activation/maxAct*100).toFixed(1)}%;background:#f59e0b"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">Influence: ${d.influence.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.influence*100).toFixed(1)}%;background:#fbbf24"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">Flow→logit: ${fl!==null?fl.toFixed(2):'n/a'}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${fl!==null?Math.min(100,Math.abs(fl)/40*100).toFixed(1):0}%;background:#f43f5e"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">node_inf: ${d.node_inf.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.node_inf*100).toFixed(1)}%;background:#4a9eff"></div></div>
    </div>
    <div class="bar-container">
      <div class="bar-label">node_rel: ${d.node_rel.toFixed(4)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(d.node_rel*100).toFixed(1)}%;background:#10b981"></div></div>
    </div>
    <div style="margin-top:10px;font-size:10px;color:#4a5568">${outEdges.length} out · ${inEdges.length} in</div>`;
}

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
    parser.add_argument('--file', type=str, default='subgraph/austin_plt.pt')
    parser.add_argument('--out',  type=str, default='circuit_graph.html')
    args = parser.parse_args()

    print(f"Loading {args.file}...")
    raw  = load_snapshot(args.file)
    data = prepare_graph_data(raw)
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Edges: {len(data['edges'])}")

    graph_json = json.dumps(data, indent=None, separators=(',', ':'))
    html       = HTML_TEMPLATE.replace('__GRAPH_DATA__', graph_json)

    with open(args.out, 'w') as f:
        f.write(html)

    print(f"Saved → {args.out}")
    print(f"Open:  open {args.out}")

if __name__ == '__main__':
    main()