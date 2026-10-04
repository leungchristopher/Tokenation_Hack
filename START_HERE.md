# Inquiry live demo

Unzip, open a terminal in this directory, and run:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    python -m minimal_lab.live --budget 6

Open http://127.0.0.1:8765 and click Run experiment loop.

Windows activation: .venv\Scripts\activate

For literature-informed selection, set AMASS_API_KEY and your model provider's credentials, then run:

    python -m minimal_lab.live --budget 6 --model YOUR_INSPECT_MODEL_ID

The default needs no API keys. It uses numerical BO, the real upstream MuJoCo motion implementation, a separately assumed volume-error model and the UPO-ABTS response table. The interface shows live robot frames, the decision graph, experiment results and the best observed protocol. Click a graph node to inspect its reasons, deferred alternatives, reported delivery and citations. Save protocol downloads the recommendation. Inspect logs and full graph outputs are written under logs/live/.

Contents
- minimal_lab/loop.py: domain-independent optimiser and decision record.
- minimal_lab/lab.py: protocol, motion orchestration and dataset black-box adapter.
- minimal_lab/live.py and live.html: local demo server, rendering and interface.
- minimal_lab/task.py: Inspect runner/scorer and optional Amass/LLM calls.
- lab_sim/ and harness/: unchanged upstream robotics implementation and assets.
- example/: exported graph, recommendation and last rendered frame from the verified run.

Validation
Seven focused tests passed. A real HTTP/Inspect run completed three MuJoCo experiments with zero invalid results and 246 rendered frames. The result download endpoint returned HTTP 200. Live Amass/LLM requests were not tested because credentials were unavailable. The browser UI has not yet completed screenshot-based verification.

Status of additional requests
This is the runnable checkpoint, not the finished visual/data revision. Matching leungchristopher.com, the explicit four-part epistemology view, and adapters for the nanohybrid Source Data and DECREASE datasets remain unfinished. This download contains the UPO demo only.

Model limits
MuJoCo models motion, not liquid. Pipetting error is assumed and labelled. Concentration-to-volume conversions are simulation assumptions because this CSV omits stock recipes and concentration units. pH and temperature are ideal external settings. Nearest-row lookup uses a sparse table. The final result is the best observed proxy, not a verified optimum or a validated wet-lab protocol.
