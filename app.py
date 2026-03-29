"""
app.py — Hugging Face Space Entry Point
=========================================
Gradio interface that runs the Hybrid RL→LLM→Explorer experiment headlessly.
Game frames rendered on the left, live coloured log on the right.
Works without a display (rgb_array mode) — perfect for HF Spaces.

NOTE: Ollama is NOT available on HF Spaces infrastructure.
      The fallback rule engine automatically activates, so the full
      pipeline still demonstrates correctly without a local LLM.
"""

import queue
import shutil
import threading
import time
import re

import numpy as np
import gradio as gr
from PIL import Image

# ── Patch stdout so we capture all prints ──────────────────
import sys

LOG_Q   = queue.Queue()
FRAME_Q = queue.Queue(maxsize=2)   # keep only latest frames

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

class _Capture:
    def __init__(self, orig):
        self.orig = orig
        self.buf  = ""
    def write(self, s):
        self.orig.write(s)
        self.orig.flush()
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            clean = ANSI_RE.sub('', line).strip()
            if clean:
                LOG_Q.put(clean)
        if '\r' in self.buf:
            clean = ANSI_RE.sub('', self.buf).strip()
            if clean:
                LOG_Q.put(clean)
            self.buf = ""
    def flush(self): self.orig.flush()
    def isatty(self): return False

sys.stdout = _Capture(sys.__stdout__)

# ── Headless display shim (replaces display.py on HF Spaces) ──
class HeadlessDisplay:
    def __init__(self): self._phase = ""
    def set_phase(self, t): self._phase = t; LOG_Q.put(f"▶ {t}")
    def render_frame(self, rgb=None):
        if rgb is not None:
            try:
                FRAME_Q.put_nowait(rgb)
            except queue.Full:
                try: FRAME_Q.get_nowait()
                except queue.Empty: pass
                FRAME_Q.put_nowait(rgb)
    def wait(self, secs): time.sleep(secs)
    def cleanup(self): pass

# ─────────────────────────────────────────────────────────
# Experiment runner (background thread)
# ─────────────────────────────────────────────────────────

_running = False
_thread  = None

def run_experiment_thread():
    global _running
    try:
        import importlib, environments
        import gymnasium as gym
        from rl_core           import DQNAgent, preprocess_obs
        from reflection_engine import analyze_failure_log, verify_rule
        from memory_hub        import MemoryHub
        from planner_agent     import OnlineExplorerAgent

        # Inject headless display as the global `display` in run_experiment
        import run_experiment as exp_mod
        disp = HeadlessDisplay()
        exp_mod.display = disp

        memory  = MemoryHub()
        agent_a = DQNAgent()

        def make_env(name):
            return gym.make(name, render_mode="rgb_array")

        # ── Phase 1: Lava Room ──────────────────────────────
        disp.set_phase("PHASE 1 — Lava Room (RL Exploration)")
        env = make_env("MiniGrid-LavaRoom-v0")
        LOG_Q.put("─"*48)
        LOG_Q.put("[Agent A] Entering MiniGrid-LavaRoom-v0")
        LOG_Q.put("─"*48)
        hazard_1 = exp_mod.run_learning_phase("MiniGrid-LavaRoom-v0", agent_a, memory)

        # ── Phase 2: Sand Room ──────────────────────────────
        disp.set_phase("PHASE 2 — Sand Room (RL Exploration)")
        hazard_2 = exp_mod.run_learning_phase("MiniGrid-QuicksandRoom-v0", agent_a, memory)

        LOG_Q.put(f"Conclusion: must avoid {hazard_1} and {hazard_2}.")

        # ── Phase 3: Final Exam ─────────────────────────────
        disp.set_phase("PHASE 3 — Final Exam")
        exp_mod.run_final_exam("MiniGrid-CombinedTesting-v0", memory)

        LOG_Q.put("Experiment complete! ✓")

    except Exception as e:
        LOG_Q.put(f"[ERROR] {e}")
    finally:
        _running = False


# ──────────────────────────────────────────
# Colour a log line (HTML span)
# ──────────────────────────────────────────

def colour_line(line):
    t = line.lower()
    if "truth confirmed" in t or "flawless" in t or "confirmed" in t:
        c = "#3fb950"; b = True
    elif "✓" in line or "safe" in t or "stepping" in t:
        c = "#3fb950"; b = False
    elif "💀" in t or "death" in t or "refuted" in t or "failed" in t:
        c = "#f85149"; b = True
    elif "✗" in line or "danger" in t or "dead end" in t or "backtrack" in t:
        c = "#f85149"; b = False
    elif "phase" in t or "▶" in t or "===" in t:
        c = "#a371f7"; b = True
    elif "rule" in t or "stored" in t or "memory" in t or "verification" in t:
        c = "#58a6ff"; b = False
    elif "reflection" in t or "llm" in t or "asking" in t or "trial" in t:
        c = "#d29922"; b = False
    elif "loading" in t or "warning" in t or "bert" in t or "unexpected" in t:
        c = "#6e7681"; b = False
    else:
        c = "#c9d1d9"; b = False
    tag = f'<b style="color:{c}">{line}</b>' if b else f'<span style="color:{c}">{line}</span>'
    return tag


# ───────────────────────────────────────────
# Gradio streaming generator
# ───────────────────────────────────────────

_log_html_lines = []
_BLANK_FRAME = np.zeros((256, 256, 3), dtype=np.uint8)

def start_and_stream():
    global _running, _thread, _log_html_lines

    if _running:
        yield gr.update(), "<p style='color:#d29922'>Already running…</p>"
        return

    _running = True
    _log_html_lines = []

    # drain old queues
    while not LOG_Q.empty():
        try: LOG_Q.get_nowait()
        except: pass
    while not FRAME_Q.empty():
        try: FRAME_Q.get_nowait()
        except: pass

    _thread = threading.Thread(target=run_experiment_thread, daemon=True)
    _thread.start()

    last_frame = _BLANK_FRAME
    idle_ticks = 0

    while _running or not LOG_Q.empty() or not FRAME_Q.empty():
        # Grab latest frame
        try:
            while True:
                last_frame = FRAME_Q.get_nowait()
        except queue.Empty:
            pass

        # Grab all pending log lines
        got_log = False
        for _ in range(20):
            try:
                line = LOG_Q.get_nowait()
                _log_html_lines.append(colour_line(line))
                got_log = True
            except queue.Empty:
                break

        img = Image.fromarray(last_frame.astype(np.uint8))

        log_body = "<br>".join(_log_html_lines[-80:])
        html = f"""
        <div style="font-family:Monaco,monospace;font-size:12px;
                    background:#0d1117;padding:12px;border-radius:8px;
                    height:460px;overflow-y:auto;line-height:1.6;">
          {log_body}
          <div id='bottom'></div>
        </div>
        <script>document.getElementById('bottom').scrollIntoView();</script>
        """

        yield img, html
        time.sleep(0.12)

        if not got_log:
            idle_ticks += 1
            if idle_ticks > 60 and not _running:
                break
        else:
            idle_ticks = 0

    yield Image.fromarray(_BLANK_FRAME), "<br>".join(
        ['<b style="color:#3fb950">Experiment finished ✓</b>'] + _log_html_lines[-5:])


# ───────────────────────────────────────────
# Gradio UI
# ───────────────────────────────────────────

CSS = """
body { background: #0d1117 !important; }
.gr-button { font-family: Monaco, monospace !important; }
#title { text-align:center; font-family: Monaco, monospace; color:#58a6ff; }
#subtitle { text-align:center; font-family: Monaco, monospace; color:#6e7681; font-size:14px; }
"""

with gr.Blocks(css=CSS, title="Hybrid RL→LLM→Explorer") as demo:

    gr.HTML("""
    <h1 id='title'>🧠 Hybrid RL → LLM → Explorer</h1>
    <p id='subtitle'>Exp-1 · Semantic Safety Rule Transfer via Vector DB<br>
    RL agent dies → LLM distills rule → Explorer navigates safely</p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            frame_out = gr.Image(
                label="Game View (rgb_array)",
                show_label=True,
                type="pil",
                height=460,
                value=Image.fromarray(_BLANK_FRAME),
            )
        with gr.Column(scale=1):
            log_out = gr.HTML(
                label="Live Log",
                value="<p style='color:#6e7681;font-family:Monaco'>Press ▶ Run to start…</p>",
            )

    with gr.Row():
        run_btn = gr.Button("▶  Run Experiment", variant="primary", size="lg")

    gr.HTML("""
    <div style='font-family:Monaco;font-size:11px;color:#6e7681;text-align:center;margin-top:8px'>
    ⚠ No Ollama on HF Spaces — fallback safety rules activate automatically.
    Clone locally and run <code>python run_experiment.py</code> for full LLM experience.
    </div>
    """)

    run_btn.click(
        fn=start_and_stream,
        inputs=[],
        outputs=[frame_out, log_out],
    )

if __name__ == "__main__":
    demo.launch()
