"""
display.py — Unified Side-by-Side Display + Pre-Experiment Config Screen
=========================================================================
Single pygame window:
  • Before the experiment: shows a configuration panel where the researcher
    can set Phase 1 min-deaths and Phase 3 max-episodes, then click START.
  • During the experiment: game on the left, live coloured log on the right.
"""

import sys
import re
import pygame
import time as _time

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# ── Colour palette (GitHub Dark) ──────────────────────────────
BG        = (13,  17,  23)
PANEL_BG  = (22,  27,  34)
CARD_BG   = (30,  36,  46)
DIV_CLR   = (48,  54,  61)
C_DEFAULT = (201, 209, 217)
C_GREEN   = (63,  185, 80)
C_RED     = (248, 81,  73)
C_YELLOW  = (210, 153, 34)
C_BLUE    = (88,  166, 255)
C_PURPLE  = (163, 113, 247)
C_DIM     = (110, 118, 129)
C_GOLD    = (227, 179, 65)
C_BTN     = (35,  134, 54)   # green start button
C_BTN_HOV = (46,  160, 67)   # hovered
C_MINUS   = (60,  70,  90)
C_MINUS_H = (80,  95, 120)


# ══════════════════════════════════════════════════════════════
#  Config screen helper – a simple +/- stepper widget
# ══════════════════════════════════════════════════════════════

class _Stepper:
    """Renders a  [−]  value  [+]  row and handles click events."""

    BTN_W  = 32
    BTN_H  = 32
    VAL_W  = 64

    def __init__(self, label: str, value: int, min_val: int, max_val: int):
        self.label   = label
        self.value   = value
        self.min_val = min_val
        self.max_val = max_val
        # rects filled in during draw()
        self.rect_minus: pygame.Rect | None = None
        self.rect_plus:  pygame.Rect | None = None

    def draw(self, surf: pygame.Surface, font, bold,
             cx: int, y: int, hover_minus: bool, hover_plus: bool):
        """Draw centred at x=cx, returning (rect_minus, rect_plus)."""
        total_w = self.BTN_W + 8 + self.VAL_W + 8 + self.BTN_W
        lx = cx - total_w // 2

        # Label above
        lbl = bold.render(self.label, True, C_DEFAULT)
        surf.blit(lbl, (cx - lbl.get_width() // 2, y - 26))

        # [−] button
        minus_rect = pygame.Rect(lx, y, self.BTN_W, self.BTN_H)
        pygame.draw.rect(surf, C_MINUS_H if hover_minus else C_MINUS,
                         minus_rect, border_radius=6)
        m = bold.render("−", True, C_DEFAULT)
        surf.blit(m, (minus_rect.centerx - m.get_width()//2,
                      minus_rect.centery - m.get_height()//2))

        # Value box
        vx = lx + self.BTN_W + 8
        val_rect = pygame.Rect(vx, y, self.VAL_W, self.BTN_H)
        pygame.draw.rect(surf, CARD_BG, val_rect, border_radius=6)
        pygame.draw.rect(surf, DIV_CLR, val_rect, 1, border_radius=6)
        v = bold.render(str(self.value), True, C_GOLD)
        surf.blit(v, (val_rect.centerx - v.get_width()//2,
                      val_rect.centery - v.get_height()//2))

        # [+] button
        px = vx + self.VAL_W + 8
        plus_rect = pygame.Rect(px, y, self.BTN_W, self.BTN_H)
        pygame.draw.rect(surf, C_MINUS_H if hover_plus else C_MINUS,
                         plus_rect, border_radius=6)
        p = bold.render("+", True, C_GREEN)
        surf.blit(p, (plus_rect.centerx - p.get_width()//2,
                      plus_rect.centery - p.get_height()//2))

        self.rect_minus = minus_rect
        self.rect_plus  = plus_rect
        return minus_rect, plus_rect

    def handle_click(self, pos):
        if self.rect_minus and self.rect_minus.collidepoint(pos):
            self.value = max(self.min_val, self.value - 1)
        if self.rect_plus and self.rect_plus.collidepoint(pos):
            self.value = min(self.max_val, self.value + 1)

class _Toggle:
    """A simple checkbox toggle switch widget."""
    def __init__(self, label: str, value: bool):
        self.label = label
        self.value = value
        self.rect = None

    def draw(self, surf: pygame.Surface, font, cx: int, y: int, hover: bool):
        lbl = font.render(self.label, True, C_DEFAULT)
        total_w = 24 + 8 + lbl.get_width()
        lx = cx - total_w // 2
        
        self.rect = pygame.Rect(lx, y - 12, 24, 24)
        pygame.draw.rect(surf, C_MINUS_H if hover else C_MINUS, self.rect, border_radius=6)
        if self.value:
            pygame.draw.rect(surf, C_GREEN, self.rect.inflate(-8, -8), border_radius=4)
            
        surf.blit(lbl, (lx + 32, y - lbl.get_height() // 2))

    def handle_click(self, pos):
        if self.rect and self.rect.collidepoint(pos):
            self.value = not self.value


# ══════════════════════════════════════════════════════════════
#  Main display class
# ══════════════════════════════════════════════════════════════

class UnifiedDisplay:
    GAME_W  = 520
    LOG_W   = 560
    HEIGHT  = 520
    TOTAL_W = GAME_W + 3 + LOG_W
    LINE_H  = 17

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Hybrid RL › LLM › Explorer — Exp-1")
        self.screen = pygame.display.set_mode((self.TOTAL_W, self.HEIGHT))
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Monaco", 12)
        self.bold   = pygame.font.SysFont("Monaco", 13, bold=True)
        self.pfont  = pygame.font.SysFont("Monaco", 14, bold=True)
        self.hfont  = pygame.font.SysFont("Monaco", 20, bold=True)
        self.log_lines  = []
        self.scroll     = 0
        self._last_frame = None
        self._phase      = ""
        self._alive      = True
        self.overlay_visits = {}
        self.grid_size      = 7
        self._orig_out   = sys.stdout
        sys.stdout = _Tee(self)

    # ──────────────────────────────────────────────────────────
    #  PRE-EXPERIMENT CONFIG SCREEN
    # ──────────────────────────────────────────────────────────

    def show_config_screen(self) -> dict:
        """Block until the researcher configures and clicks START.

        Returns:
            {
              "min_deaths":      int,   # Phase 1: deaths to collect
              "max_p3_episodes": int,   # Phase 3: max episodes
            }
        """
        W, H = self.TOTAL_W, self.HEIGHT
        cx   = W // 2

        stepper_p1 = _Stepper(
            label="Deaths to Collect",
            value=3, min_val=1, max_val=20,
        )
        stepper_p1_lava = _Stepper(
            label="Phase 1 Lava Tiles",
            value=2, min_val=1, max_val=25,
        )
        stepper_p3 = _Stepper(
            label="Phase 3 Episodes",
            value=3, min_val=1, max_val=50,
        )
        stepper_p3_lava = _Stepper(
            label="Phase 3 Lava Tiles",
            value=10, min_val=1, max_val=25,
        )
        toggle_auto = _Toggle(
            label="Auto-Increase Lava (+1 per death)",
            value=True
        )

        start_rect = pygame.Rect(cx - 130, H - 110, 260, 52)

        running = True
        while running and self._alive:
            mx, my = pygame.mouse.get_pos()
            hover_start  = start_rect.collidepoint(mx, my)
            hover_m1 = stepper_p1.rect_minus.collidepoint(mx, my) if stepper_p1.rect_minus else False
            hover_p1 = stepper_p1.rect_plus.collidepoint(mx, my)  if stepper_p1.rect_plus  else False
            hover_p1_mlava = stepper_p1_lava.rect_minus.collidepoint(mx, my) if stepper_p1_lava.rect_minus else False
            hover_p1_plava = stepper_p1_lava.rect_plus.collidepoint(mx, my)  if stepper_p1_lava.rect_plus  else False
            hover_m3 = stepper_p3.rect_minus.collidepoint(mx, my) if stepper_p3.rect_minus else False
            hover_p3 = stepper_p3.rect_plus.collidepoint(mx, my)  if stepper_p3.rect_plus  else False
            hover_p3_mlava = stepper_p3_lava.rect_minus.collidepoint(mx, my) if stepper_p3_lava.rect_minus else False
            hover_p3_plava = stepper_p3_lava.rect_plus.collidepoint(mx, my)  if stepper_p3_lava.rect_plus  else False
            hover_tog = toggle_auto.rect.collidepoint(mx, my) if toggle_auto.rect else False

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.cleanup()
                    raise SystemExit(0)
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    stepper_p1.handle_click(ev.pos)
                    stepper_p1_lava.handle_click(ev.pos)
                    stepper_p3.handle_click(ev.pos)
                    stepper_p3_lava.handle_click(ev.pos)
                    toggle_auto.handle_click(ev.pos)
                    if hover_start:
                        running = False

            # ── Draw config screen ─────────────────────────────
            self.screen.fill(BG)

            # Title bar
            pygame.draw.rect(self.screen, PANEL_BG, (0, 0, W, 64))
            pygame.draw.line(self.screen, DIV_CLR, (0, 64), (W, 64), 1)
            t1 = self.hfont.render("Exp-1: Hybrid RL  →  LLM  →  Smart Explorer", True, C_BLUE)
            t2 = self.font.render("Configure the experiment, then click START", True, C_DIM)
            self.screen.blit(t1, (cx - t1.get_width() // 2, 12))
            self.screen.blit(t2, (cx - t2.get_width() // 2, 44))

            # ── Phase 1 card ───────────────────────────────────
            card1 = pygame.Rect(cx - 260, 95, 520, 150)
            pygame.draw.rect(self.screen, CARD_BG, card1, border_radius=10)
            pygame.draw.rect(self.screen, DIV_CLR,  card1, 1, border_radius=10)

            ph1_lbl = self.pfont.render("① PHASE 1  —  RL Learning", True, C_GOLD)
            self.screen.blit(ph1_lbl, (cx - ph1_lbl.get_width()//2, 108))
            desc1 = self.font.render(
                "Agent explores with -3/step. Learns by dying: records which tile killed it.", True, C_DIM)
            self.screen.blit(desc1, (cx - desc1.get_width()//2, 130))

            stepper_p1.draw(self.screen, self.font, self.bold,
                            cx - 120, 185, hover_m1, hover_p1)
            stepper_p1_lava.draw(self.screen, self.font, self.bold,
                            cx + 120, 185, hover_p1_mlava, hover_p1_plava)
            toggle_auto.draw(self.screen, self.font, cx, 222, hover_tog)

            # ── Phase 3 card ───────────────────────────────────
            card3 = pygame.Rect(cx - 260, 260, 520, 135)
            pygame.draw.rect(self.screen, CARD_BG, card3, border_radius=10)
            pygame.draw.rect(self.screen, DIV_CLR,  card3, 1, border_radius=10)

            ph3_lbl = self.pfont.render("③ PHASE 3  —  Smart Explorer (LLM Rules Active)", True, C_PURPLE)
            self.screen.blit(ph3_lbl, (cx - ph3_lbl.get_width()//2, 258))
            desc3 = self.font.render(
                "Uses rules from Phase 1. Lava is RANDOM each episode — rules must generalize.", True, C_DIM)
            self.screen.blit(desc3, (cx - desc3.get_width()//2, 295))

            stepper_p3.draw(self.screen, self.font, self.bold,
                            cx - 120, 350, hover_m3, hover_p3)
            stepper_p3_lava.draw(self.screen, self.font, self.bold,
                            cx + 120, 350, hover_p3_mlava, hover_p3_plava)


            # ── Legend strip ──────────────────────────────────
            legend = self.font.render(
                "Each step costs  -3   |   Lava = instant death  -10   |   Goal = +10   |   Auto-death tile penalty > -60",
                True, C_DIM,
            )
            self.screen.blit(legend, (cx - legend.get_width()//2, 415))

            # ── START button ──────────────────────────────────
            btn_col = C_BTN_HOV if hover_start else C_BTN
            pygame.draw.rect(self.screen, btn_col, start_rect, border_radius=10)
            pygame.draw.rect(self.screen, C_GREEN,  start_rect, 2, border_radius=10)
            btn_t = self.hfont.render("▶  START EXPERIMENT", True, (255, 255, 255))
            self.screen.blit(btn_t, (start_rect.centerx - btn_t.get_width()//2,
                                     start_rect.centery - btn_t.get_height()//2))

            pygame.display.flip()
            self.clock.tick(60)

        return {
            "min_deaths":      stepper_p1.value,
            "p1_lava":         stepper_p1_lava.value,
            "auto_inc_lava":   toggle_auto.value,
            "max_p3_episodes": stepper_p3.value,
            "p3_lava":         stepper_p3_lava.value,
        }


    # ──────────────────────────────────────────────────────────
    #  Standard display helpers
    # ──────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────
    #  Internal rendering
    # ──────────────────────────────────────────────────────────

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

        # Left: game frame
        if self._last_frame is not None:
            h, w = self._last_frame.shape[:2]
            surf   = pygame.image.frombuffer(self._last_frame.tobytes(), (w, h), 'RGB')
            scaled = pygame.transform.smoothscale(surf, (self.GAME_W, self.HEIGHT))
            self.screen.blit(scaled, (0, 0))
        else:
            t = self.font.render("Waiting for game…", True, C_DIM)
            self.screen.blit(t, (self.GAME_W // 2 - 70, self.HEIGHT // 2))

        # Phase banner overlay
        if self._phase:
            ov = pygame.Surface((self.GAME_W, 28), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 160))
            self.screen.blit(ov, (0, 0))
            self.screen.blit(self.pfont.render(self._phase, True, C_GOLD), (10, 5))

        # Grid penalty overlay
        if self._last_frame is not None and self.overlay_visits:
            cell_w = self.GAME_W / self.grid_size
            cell_h = self.HEIGHT  / self.grid_size

            for (x, y), value in self.overlay_visits.items():
                pen = float(value)
                if pen >= 0:
                    continue

                abs_pen = abs(pen)
                if abs_pen >= 45:
                    txt_colour = (255, 80,  80)
                elif abs_pen >= 21:
                    txt_colour = (255, 165, 0)
                else:
                    txt_colour = (220, 220, 100)

                label    = f"{int(pen)}"
                txt_surf = self.font.render(label, True, txt_colour)
                px = int(x * cell_w + cell_w // 2 - txt_surf.get_width()  // 2)
                py = int(y * cell_h + cell_h // 2 - txt_surf.get_height() // 2)
                pygame.draw.rect(self.screen, (0, 0, 0),
                                 (px-2, py-2, txt_surf.get_width()+4, txt_surf.get_height()+4))
                self.screen.blit(txt_surf, (px, py))

        # Divider
        pygame.draw.rect(self.screen, DIV_CLR, (self.GAME_W, 0, 3, self.HEIGHT))

        # Right panel – header
        lx = self.GAME_W + 3
        pygame.draw.rect(self.screen, PANEL_BG, (lx, 0, self.LOG_W, 38))
        self.screen.blit(self.bold.render("Live Experiment Log", True, C_BLUE), (lx + 10, 10))

        # Colour legend
        cx_l = lx + 6
        for clr, lbl in [(C_GREEN, "Safe"), (C_RED, "Death"),
                         (C_YELLOW, "LLM"), (C_BLUE, "Rule"), (C_PURPLE, "Phase")]:
            r = self.font.render(lbl, True, clr)
            self.screen.blit(r, (cx_l, 42))
            cx_l += r.get_width() + 14

        # Log text (scrollable)
        vis = (self.HEIGHT - 56) // self.LINE_H
        end = min(self.scroll + vis, len(self.log_lines))
        y   = 56
        for i in range(self.scroll, end):
            txt, clr = self.log_lines[i]
            if len(txt) > 70:
                txt = txt[:67] + "…"
            self.screen.blit(self.font.render(txt, True, clr), (lx + 8, y))
            y += self.LINE_H

        pygame.display.flip()


# ──────────────────────────────────────────────────────────────
#  Log line colour classifier
# ──────────────────────────────────────────────────────────────

def _colour(raw, clean):
    if '\033[92m' in raw:
        return C_GREEN
    if '\033[91m' in raw:
        return C_RED
    t = clean.lower()
    if 'success' in t or 'goal' in t or 'truth confirmed' in t or 'flawless' in t:
        return C_GREEN
    if 'safe' in t or 'stepping into' in t:
        return C_GREEN
    if 'death' in t or 'auto-death' in t or '💀' in t or '🔴' in t:
        return C_RED
    if 'rule blocked' in t or '⛔' in t or 'refuted' in t or 'danger' in t:
        return C_RED
    if 'failed' in t or 'dead end' in t or 'backtrack' in t:
        return C_RED
    if 'loading' in t or 'warning' in t or 'cleanup' in t:
        return C_DIM
    if 'rule' in t or 'stored' in t or 'memory' in t:
        return C_BLUE
    if 'llm' in t or 'synthesiz' in t or 'reflection' in t:
        return C_YELLOW
    if 'learning log' in t or '📝' in t:
        return C_PURPLE
    if 'phase' in t or '===' in t:
        return C_PURPLE
    return C_DEFAULT


# ──────────────────────────────────────────────────────────────
#  stdout tee (prints to terminal + log panel simultaneously)
# ──────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, disp):
        self.d    = disp
        self.orig = disp._orig_out
        self.buf  = ""

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
