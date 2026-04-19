"""
display.py — Unified Side-by-Side Display
==========================================
Single pygame window: game on the left, live coloured log on the right.
"""

import sys
import re
import pygame
import time as _time

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# Colour palette (GitHub Dark)
BG        = (13, 17, 23)
PANEL_BG  = (22, 27, 34)
DIV_CLR   = (48, 54, 61)
C_DEFAULT = (201, 209, 217)
C_GREEN   = (63, 185, 80)
C_RED     = (248, 81, 73)
C_YELLOW  = (210, 153, 34)
C_BLUE    = (88, 166, 255)
C_PURPLE  = (163, 113, 247)
C_DIM     = (110, 118, 129)
C_GOLD    = (227, 179, 65)


class UnifiedDisplay:
    GAME_W  = 520
    LOG_W   = 560
    HEIGHT  = 520
    TOTAL_W = GAME_W + 3 + LOG_W
    LINE_H  = 17

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Hybrid RL > LLM > Explorer")
        self.screen = pygame.display.set_mode((self.TOTAL_W, self.HEIGHT))
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Monaco", 12)
        self.bold   = pygame.font.SysFont("Monaco", 12, bold=True)
        self.pfont  = pygame.font.SysFont("Monaco", 14, bold=True)
        self.log_lines  = []
        self.scroll     = 0
        self._last_frame = None
        self._phase      = ""
        self._alive      = True
        self.overlay_visits = {}  # (x,y) -> count
        self.grid_size      = 7
        self._orig_out   = sys.stdout
        sys.stdout = _Tee(self)

    def set_phase(self, t):
        self._phase = t

    def render_frame(self, game_rgb=None, overlay_visits=None, grid_size=None):
        if not self._alive:
            return
        if game_rgb is not None:
            self._last_frame = game_rgb
        if overlay_visits is not None:
            self.overlay_visits = overlay_visits
        if grid_size is not None:
            self.grid_size = grid_size
            
        self._pump()
        self._draw()

    def wait(self, secs):
        end = _time.time() + secs
        while _time.time() < end and self._alive:
            self._pump()
            self._draw()
            self.clock.tick(60)

    def cleanup(self):
        sys.stdout = self._orig_out
        self._alive = False
        pygame.quit()

    def add_log(self, raw):
        clean = ANSI_RE.sub('', raw).rstrip('\r\n')
        if not clean:
            return
        self.log_lines.append((clean, _colour(raw, clean)))
        if len(self.log_lines) > 300:
            self.log_lines = self.log_lines[-300:]
        vis = (self.HEIGHT - 56) // self.LINE_H
        self.scroll = max(0, len(self.log_lines) - vis)

    def _pump(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.cleanup()
                raise SystemExit(0)
            if ev.type == pygame.MOUSEWHEEL:
                vis = (self.HEIGHT - 56) // self.LINE_H
                self.scroll = max(0, self.scroll - ev.y * 3)
                self.scroll = min(self.scroll, max(0, len(self.log_lines) - vis))

    def _draw(self):
        if not self._alive:
            return
        self.screen.fill(BG)

        # Left: game
        if self._last_frame is not None:
            h, w = self._last_frame.shape[:2]
            surf = pygame.image.frombuffer(
                self._last_frame.tobytes(), (w, h), 'RGB')
            scaled = pygame.transform.smoothscale(
                surf, (self.GAME_W, self.HEIGHT))
            self.screen.blit(scaled, (0, 0))
        else:
            t = self.font.render("Waiting for game...", True, C_DIM)
            self.screen.blit(t, (self.GAME_W // 2 - 70, self.HEIGHT // 2))

        # Phase overlay
        if self._phase:
            ov = pygame.Surface((self.GAME_W, 28), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 160))
            self.screen.blit(ov, (0, 0))
            self.screen.blit(
                self.pfont.render(self._phase, True, C_GOLD), (10, 5))

        # --- Grid Overlay (Penalties) ---
        if self._last_frame is not None and self.overlay_visits:
            # Calculate cell size
            cell_w = self.GAME_W / self.grid_size
            cell_h = self.HEIGHT / self.grid_size
            
            for (x, y), count in self.overlay_visits.items():
                if count > 1: # Only show penalty if revisited
                    # MiniGrid coordinates (x is column, y is row)
                    # Screen coordinates: (x * cell_w, y * cell_h)
                    penalty_text = f"-{(count-1) * 3}"
                    txt_surf = self.font.render(penalty_text, True, (255, 255, 255))
                    
                    # Create a sub-surface for the number so it's more visible
                    px = int(x * cell_w + cell_w//2 - txt_surf.get_width()//2)
                    py = int(y * cell_h + cell_h//2 - txt_surf.get_height()//2)
                    
                    # Draw a tiny shadow/bg for readability
                    pygame.draw.rect(self.screen, (0, 0, 0, 120), 
                                     (px-2, py-2, txt_surf.get_width()+4, txt_surf.get_height()+4))
                    self.screen.blit(txt_surf, (px, py))

        # Divider
        pygame.draw.rect(self.screen, DIV_CLR,
                         (self.GAME_W, 0, 3, self.HEIGHT))

        # Right: log header
        lx = self.GAME_W + 3
        pygame.draw.rect(self.screen, PANEL_BG, (lx, 0, self.LOG_W, 38))
        self.screen.blit(
            self.bold.render("Live Experiment Log", True, C_BLUE),
            (lx + 10, 10))

        # Legend
        cx = lx + 6
        for clr, lbl in [(C_GREEN, "Safe"), (C_RED, "Danger"),
                          (C_YELLOW, "LLM"), (C_BLUE, "Rule"),
                          (C_PURPLE, "Phase")]:
            r = self.font.render(lbl, True, clr)
            self.screen.blit(r, (cx, 42))
            cx += r.get_width() + 14

        # Log text
        vis = (self.HEIGHT - 56) // self.LINE_H
        end = min(self.scroll + vis, len(self.log_lines))
        y = 56
        for i in range(self.scroll, end):
            txt, clr = self.log_lines[i]
            if len(txt) > 70:
                txt = txt[:67] + "..."
            self.screen.blit(self.font.render(txt, True, clr), (lx + 8, y))
            y += self.LINE_H

        pygame.display.flip()


def _colour(raw, clean):
    if '\033[92m' in raw:
        return C_GREEN
    if '\033[91m' in raw:
        return C_RED
    t = clean.lower()
    if 'truth confirmed' in t or 'flawless' in t:
        return C_GREEN
    if 'safe' in t or 'stepping' in t:
        return C_GREEN
    if 'death' in t:
        return C_RED
    if 'refuted' in t or 'failed' in t or 'danger' in t:
        return C_RED
    if 'dead end' in t or 'backtrack' in t:
        return C_RED
    if 'loading' in t or 'warning' in t or 'cleanup' in t:
        return C_DIM
    if 'respawn' in t or 'bert' in t or 'key' in t:
        return C_DIM
    if 'rule' in t or 'stored' in t:
        return C_BLUE
    if 'reflection' in t or 'memory' in t or 'verification' in t or 'passed' in t:
        return C_BLUE
    if 'phase' in t or 'final exam' in t or '===' in t:
        return C_PURPLE
    if 'conclusion' in t or 'entering' in t:
        return C_GOLD
    if 'trial' in t:
        return C_YELLOW
    return C_DEFAULT


class _Tee:
    def __init__(self, disp):
        self.d = disp
        self.orig = disp._orig_out
        self.buf = ""

    def write(self, s):
        self.orig.write(s)
        self.orig.flush()
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            self.d.add_log(line)
        if '\r' in self.buf and '\n' not in self.buf:
            self.d.add_log(self.buf)
            self.buf = ""

    def flush(self):
        self.orig.flush()

    def isatty(self):
        return False
