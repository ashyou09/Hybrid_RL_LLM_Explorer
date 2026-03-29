"""
monitor.py — Live Experiment Dashboard
=======================================
Run this instead of run_experiment.py.

It opens a dark terminal-style window showing all output
(Ollama, deaths, rules, ChromaDB) with color highlights.
The MiniGrid game window opens separately as usual.

Position this window next to the game for a full picture.
"""

import re
import subprocess
import threading
import tkinter as tk


ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text):
    return ANSI_ESCAPE.sub('', text)


class ExperimentMonitor:

    def __init__(self, root):
        self.root    = root
        self.process = None
        self.root.title("🧠  Experiment Monitor — Hybrid RL → LLM → Explorer")
        self.root.configure(bg="#0d1117")
        self.root.geometry("680x860")
        self.root.resizable(True, True)
        self._build_ui()

    # ─────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────

    def _build_ui(self):
        # ── Header bar ──
        header = tk.Frame(self.root, bg="#161b22", pady=12)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="🧠  Hybrid RL → LLM → Explorer   |   Live Log",
            font=("Monaco", 13, "bold"),
            bg="#161b22", fg="#58a6ff",
        ).pack()

        self.status_var = tk.StringVar(value="● Idle — press Run to start")
        self.status_lbl = tk.Label(
            header,
            textvariable=self.status_var,
            font=("Monaco", 10),
            bg="#161b22", fg="#6e7681",
        )
        self.status_lbl.pack(pady=(2, 0))

        # ── Legend bar ──
        legend = tk.Frame(self.root, bg="#0d1117", pady=4)
        legend.pack(fill=tk.X, padx=12)

        for color, label in [
            ("#3fb950", "✓ Safe"),
            ("#f85149", "✗ Danger / Death"),
            ("#d29922", "⚠ LLM / Warning"),
            ("#58a6ff", "ℹ  Memory / Rule"),
            ("#a371f7", "◈ Phase Header"),
        ]:
            tk.Label(legend, text=label, font=("Monaco", 9),
                     bg="#0d1117", fg=color).pack(side=tk.LEFT, padx=6)

        # ── Log text area ──
        txt_frame = tk.Frame(self.root, bg="#0d1117")
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 0))

        self.text = tk.Text(
            txt_frame,
            bg="#0d1117", fg="#c9d1d9",
            font=("Monaco", 11),
            insertbackground="#58a6ff",
            selectbackground="#264f78",
            wrap=tk.WORD,
            padx=10, pady=8,
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=0,
        )
        sb = tk.Scrollbar(txt_frame, command=self.text.yview,
                          bg="#21262d", troughcolor="#0d1117",
                          activebackground="#30363d")
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Colour tags
        bold11 = ("Monaco", 11, "bold")
        norm11 = ("Monaco", 11)
        self.text.tag_configure("success", foreground="#3fb950", font=norm11)
        self.text.tag_configure("confirmed", foreground="#3fb950", font=bold11)
        self.text.tag_configure("danger",  foreground="#f85149", font=norm11)
        self.text.tag_configure("death",   foreground="#f85149", font=bold11)
        self.text.tag_configure("warning", foreground="#d29922", font=norm11)
        self.text.tag_configure("info",    foreground="#58a6ff", font=norm11)
        self.text.tag_configure("phase",   foreground="#a371f7", font=bold11)
        self.text.tag_configure("header",  foreground="#e3b341", font=bold11)
        self.text.tag_configure("dim",     foreground="#6e7681", font=norm11)
        self.text.tag_configure("rule",    foreground="#79c0ff", font=norm11)

        # ── Button bar ──
        btn_frame = tk.Frame(self.root, bg="#0d1117", pady=10)
        btn_frame.pack(fill=tk.X, padx=12)

        self.run_btn = tk.Button(
            btn_frame, text="▶  Run Experiment",
            command=self.run_experiment,
            bg="#238636", fg="white",
            font=("Monaco", 11, "bold"),
            relief=tk.FLAT, padx=16, pady=7,
            cursor="hand2", activebackground="#2ea043",
        )
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.clear_btn = tk.Button(
            btn_frame, text="⌫  Clear",
            command=self.clear_log,
            bg="#21262d", fg="#c9d1d9",
            font=("Monaco", 11),
            relief=tk.FLAT, padx=12, pady=7,
            cursor="hand2", activebackground="#30363d",
        )
        self.clear_btn.pack(side=tk.LEFT)

        tk.Label(
            btn_frame,
            text="(game window opens automatically)",
            font=("Monaco", 9), bg="#0d1117", fg="#6e7681"
        ).pack(side=tk.RIGHT)

    # ─────────────────────────────────────
    # Tag selection per line
    # ─────────────────────────────────────

    def _tag_for(self, line):
        if "✓ TRUTH CONFIRMED" in line or "✓ FLAWLESS" in line:
            return "confirmed"
        if "✓" in line or "Safe" in line or "stepping" in line:
            return "success"
        if "💀" in line or "Death" in line:
            return "death"
        if "✗ TRUTH REFUTED" in line or "✗ FAILED" in line:
            return "danger"
        if "✗" in line or "Danger" in line or "Dead end" in line or "Backtracking" in line:
            return "danger"
        if "Respawning" in line or "Cleanup" in line or "Warning" in line or "Loading" in line:
            return "dim"
        if "Rule learned" in line or "Derived Rule" in line or "Stored rule" in line:
            return "rule"
        if "LLM" in line or "Reflection" in line or "Memory Hub" in line or "Verification" in line or "PASSED" in line:
            return "info"
        if "PHASE" in line or "═" in line or "FINAL EXAM" in line or "Exp‑1" in line or "=====" in line:
            return "phase"
        if "─" in line or "╔" in line or "╚" in line or "║" in line or "Conclusion" in line:
            return "header"
        if "Trial" in line or "→" in line:
            return "warning"
        return None

    # ─────────────────────────────────────
    # Append a line to the text widget
    # ─────────────────────────────────────

    def _append(self, raw):
        # Strip carriage returns and ANSI codes
        line = strip_ansi(raw).rstrip('\r\n')
        if not line:
            # Still insert blank line for spacing
            self.text.configure(state=tk.NORMAL)
            self.text.insert(tk.END, "\n")
            self.text.see(tk.END)
            self.text.configure(state=tk.DISABLED)
            return

        tag = self._tag_for(line)
        self.text.configure(state=tk.NORMAL)
        if tag:
            self.text.insert(tk.END, line + "\n", tag)
        else:
            self.text.insert(tk.END, line + "\n")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    # ─────────────────────────────────────
    # Experiment runner
    # ─────────────────────────────────────

    def run_experiment(self):
        if self.process and self.process.poll() is None:
            return  # already running

        self.run_btn.configure(state=tk.DISABLED, text="⏳  Running…", bg="#1a6127")
        self.status_var.set("● Running — game window loading…")
        self.status_lbl.configure(fg="#d29922")
        self.clear_log()

        def _thread():
            import os
            cwd = os.path.dirname(os.path.abspath(__file__))

            self.process = subprocess.Popen(
                ["python3", "run_experiment.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd,
            )

            for line in self.process.stdout:
                self.root.after(0, self._append, line)

            self.process.wait()
            self.root.after(0, self._on_done, self.process.returncode)

        threading.Thread(target=_thread, daemon=True).start()

    def _on_done(self, code):
        self._append("")
        self._append("─" * 55)
        self._append(f"  Experiment finished  (exit code {code})")
        self._append("─" * 55)
        self.run_btn.configure(state=tk.NORMAL, text="▶  Run Experiment", bg="#238636")
        self.status_var.set("● Done — click Run to go again")
        self.status_lbl.configure(fg="#3fb950")

    def clear_log(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)


# ─────────────────────────────────────
# Entry point
# ─────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = ExperimentMonitor(root)
    root.mainloop()
