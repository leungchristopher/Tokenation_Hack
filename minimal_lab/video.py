"""High-resolution video composition from the live simulator and its decision graph."""
import textwrap

import cv2
import numpy as np

WIDTH, HEIGHT = 2560, 1440


def compose(robot, graph, action, detail, assay):
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
    canvas[290:1250, :1280] = robot
    text(f"{action} | {detail.get('role', '')} | {detail.get('well', '')}", 28, 1293, .85)
    if 'volume_ul' in detail:
        text(f"Requested transfer: {detail['volume_ul']:.4g} uL", 28, 1332, .7)
    text('Motion: MuJoCo | Liquids, incubation and survival: separate simulation model', 28, 1386, .51)
    cv2.line(canvas, (1295, 100), (1295, 1400), (180, 193, 183), 2)
    sources = [n for n in nodes if n['kind'] == 'source']
    priors = [n for n in nodes if n['kind'] == 'prior']
    text(f"{len(sources)} retrieved sources | {len(priors)} cited priors | deferred options remain eligible", 1320, 128, .56)
    text('Decision / proposal                         Deferred options              Observation', 1320, 166, .56)
    positions = {}
    spacing = min(47, 1110 / max(len(decisions), 1))
    for i, d in enumerate(decisions):
        y = int(195+i*spacing)
        positions[d['id']] = (1330, y)
        label = f"{d['id']} I{d.get('iteration', d['round'])} C{d['selected']} {d.get('phase', 'proposal')}"
        box(1330, y, 430, label, (220, 237, 224))
        if i:
            cv2.line(canvas, (1315, y-int(spacing)+17), (1315, y+17), (103, 135, 110), 1)
            cv2.arrowedLine(canvas, (1315, y+17), (1330, y+17), (103, 135, 110), 1, tipLength=.3)
        alternatives = d.get('alternatives', {})
        if alternatives:
            cv2.line(canvas, (1760, y+17), (1790, y+17), (140, 151, 142), 1)
            box(1790, y, 280, 'defer '+', '.join('C'+c for c in alternatives), (235, 235, 231))
        observation = next((n for n in nodes if n['kind']=='observation' and n['round']==d['round']), None)
        if observation:
            value = observation.get('value')
            label = f"{observation['id']} {value:.3f}%" if value is not None else f"{observation['id']} failed"
            cv2.arrowedLine(canvas, (2070, y+17), (2110, y+17), (103, 135, 110), 1, tipLength=.3)
            box(2110, y, 385, label, (239, 227, 210))
    result = next((n for n in reversed(nodes) if n['kind']=='recommendation'), None)
    if result and result['mean'] is not None:
        text(f"Recommended C{result['candidate']}: {result['mean']:.3f}% mean survival; "
             f"{result['repeats']} preparations", 1320, 1380, .66)
    else:
        text('Sources, quotations and full alternative reasons are retained in graph.html.', 1320, 1380, .5)
    return canvas
