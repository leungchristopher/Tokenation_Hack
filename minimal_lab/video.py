"""High-resolution video composition from the live simulator and its decision graph."""
import textwrap

import cv2
import numpy as np

WIDTH, HEIGHT = 2560, 1440


def compose(robot, graph, action, detail, assay, landscape=None):
    canvas = np.full((HEIGHT, WIDTH, 3), (243, 246, 244), dtype=np.uint8)

    def text(value, x, y, size=.65, colour=(48, 58, 50)):
        # OpenCV's built-in font is ASCII; keep units legible.
        value = str(value).replace('µ', 'u').replace('±', '+/-').encode('ascii', 'replace').decode()
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)

    def box(x, y, w, label, colour, h=34):
        cv2.rectangle(canvas, (x, y), (x+w, y+h), colour, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (165, 180, 169), 1)
        text(label[:65], x+8, y+23, .49)

    nodes = (graph or {}).get('nodes', [])
    decisions = [n for n in nodes if n['kind'] == 'decision']
    current = decisions[-1] if decisions else None
    text(assay, 28, 44, 1.0)
    text('LIVE ROBOT EXECUTION', 28, 88, .7)
    text('GROWING REASONING GRAPH', 1320, 88, .8)
    if current:
        text(f"Iteration {current.get('iteration', current['round'])} | preparation {current['round']} | "
             f"{current.get('phase', 'proposal')} | decision {current['id']}", 28, 125, .65)
        for j, line in enumerate(textwrap.wrap(current['reason'], width=108)[:4]):
            text(line, 28, 161+27*j, .59)
    else:
        text('Retrieving evidence and preparing the first decision...', 28, 140, .65)
    # This image is the current MjData render, never a replay of another run.
    canvas[275:995,160:1120] = cv2.resize(robot,(960,720))
    if landscape is not None:
        canvas[1040:1430,:1280] = landscape
    text(f"{action} | {detail.get('role', '')} | {detail.get('well', '')}", 28, 1025, .67)
    if 'volume_ul' in detail:
        text(f"Requested transfer: {detail['volume_ul']:.4g} uL", 790, 1025, .6)
    text('MuJoCo motion; liquid and assay outcomes are modelled', 28, 262, .46)
    cv2.line(canvas, (1295, 100), (1295, 1400), (180, 193, 183), 2)
    sources = {n['id']:n for n in nodes if n['kind']=='source'}
    prior = next((n for n in reversed(nodes) if n['kind']=='prior'), None)
    active_ids = (current or {}).get('source_ids', [])
    text('EVIDENCE -> STARTING PRIOR -> ADAPTIVE SEARCH', 1320, 124, .62)
    for j, sid in enumerate(active_ids[:2]):
        paper = sources.get(sid, {})
        y = 146+j*69
        cv2.rectangle(canvas, (1320,y), (2520,y+60), (240,231,216), -1)
        text(f"[{sid}] {paper.get('title', 'Retrieved source')[:117]}",1329,y+23,.5)
        text(paper.get('url','')[:130],1329,y+47,.45)
    if len(active_ids)>2:
        text('Additional source tags: '+', '.join(active_ids[2:]),1320,295,.45)
    if prior:
        text(f"[{prior['id']}] Literature-derived starting hypothesis | uncertainty: transfer unverified",1320,322,.53)
        if active_ids:
            cv2.arrowedLine(canvas,(2505,278),(2505,317),(120,100,150),2,tipLength=.2)
    else:
        text('No validated numerical prior: no literature attribution claimed.',1320,322,.53)
    cv2.rectangle(canvas,(1320,339),(2520,414),(218,237,251),-1)
    if current and current.get('latent_sd') is not None:
        text(f"CURRENT PREDICTION: {current['predicted_mean']:.2f} | model SD {current['latent_sd']:.2f}",1330,368,.65)
    else:
        text('CURRENT PREDICTION: unknown before measurements',1330,368,.65)
    text('Model SD = uncertainty about recipe mean, uncalibrated. Observed SD = replicate scatter.',1330,396,.49)
    # Compact preparation repeats into a recipe decision group; preserve every node ID
    # in the interactive graph and frame/action timeline.
    groups = {}
    for d in decisions:
        key = ('confirmation',d['selected']) if d.get('phase')=='confirmation' else ('search',d.get('iteration',d['round']))
        groups.setdefault(key,[]).append(d)
    entries = list(groups.values())
    rows = max(1, (len(entries)+1)//2)
    spacing = min(69, 850/rows)
    positions = {}
    for i, group in enumerate(entries):
        d=group[0]
        col,row = divmod(i, rows)
        x,y = 1320+col*610, int(437+row*spacing)
        selected=d['selected']
        current_group = current is not None and any(n['id']==current['id'] for n in group)
        colour = (196,231,203) if current_group else (228,237,231)
        box(x,y,590,f"{d['id']}  I{d.get('iteration',d['round'])} C{selected}  {d.get('phase','proposal')}",colour,h=int(min(62,spacing-3)))
        observations=[n for n in nodes if n['kind']=='observation' and n['candidate']==selected]
        values=[n['value'] for n in observations if n['value'] is not None]
        tags='prior '+','.join(d.get('source_ids',[])) if d.get('source_ids') else 'data only'
        if values:
            sd=float(np.std(values,ddof=1)) if len(values)>1 else None
            scatter=f"SD {sd:.2f}" if sd is not None else 'SD unknown'
            label=f"n={len(values)} mean {np.mean(values):.2f} {scatter} | {tags}"
        else:
            label='Awaiting observation | '+tags
        if spacing>=45:
            text(label[:84],x+8,y+46,.43)
        positions[group[-1]['id']]=(x,y)
        if row:
            cv2.arrowedLine(canvas,(x-6,int(y-spacing+28)),(x-6,y+16),(123,148,130),1,tipLength=.15)
    if current and prior and current['id'] in positions:
        x,y=positions[current['id']]
        cv2.arrowedLine(canvas,(2510,323),(min(x+575,2510),y),(145,113,156),1,tipLength=.025)
    text(f"{len(decisions)} preparations | {len(entries)} decision groups | deferred alternatives stay open",1320,1323,.56)
    result=next((n for n in reversed(nodes) if n['kind']=='recommendation'),None)
    if result and result.get('mean') is not None:
        text(f"Confirmed C{result['candidate']}: mean {result['mean']:.2f}; "
             f"SE {result['standard_error']:.2f}" if result.get('standard_error') is not None
             else f"Confirmed C{result['candidate']}: uncertainty not yet estimable",1320,1364,.64)
    else:
        text('Source arrows denote a prior hypothesis, not proof of a measured drug effect.',1320,1364,.49)
    text('Full source quotations, uncertainty and deferred-branch edges: graph.html',1320,1396,.47)
    return canvas
