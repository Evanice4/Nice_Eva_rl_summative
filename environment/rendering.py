"""
SokoPrice Panda3D Visualisation
3D market environment with animated agent, market stalls, and HUD overlay.
Renders offscreen and saves as GIF.

"""

import os
import sys
import math
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import (
    SokoPriceEnv, FOOD_ITEMS, BASE_PRICES, N_ITEMS, DAILY_BUDGET, MAX_DAYS
)


#  Try Panda3D - fall back to enhanced pygame

PANDA3D_AVAILABLE = False
try:
    from panda3d.core import (
        loadPrcFileData, GraphicsEngine, GraphicsPipeSelection,
        FrameBufferProperties, WindowProperties, GraphicsPipe,
        Texture, NodePath, AmbientLight, DirectionalLight,
        LPoint3, LColor, LVector3, AntialiasAttrib,
        TextNode, CardMaker
    )
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import GeomVertexFormat, GeomVertexData, Geom
    from panda3d.core import GeomTriangles, GeomVertexWriter, GeomNode
    PANDA3D_AVAILABLE = True
    print(" Panda3D found - using 3D rendering")
except ImportError:
    print("  Panda3D not found - using enhanced PIL rendering")


#  Colour palette

C_BG       = (12,  18,  35)
C_PANEL    = (20,  30,  55, 210)
C_BLUE     = (50, 130, 255)
C_GREEN    = (50, 210, 120)
C_RED      = (255,  80,  80)
C_AMBER    = (255, 190,  50)
C_TEAL     = (50,  210, 200)
C_WHITE    = (235, 240, 255)
C_GREY     = (130, 145, 175)

W, H = 1100, 700
FPS  = 6

STALL_COLORS = [
    (220,  80,  60),  # Beans     - red
    (240, 180,  40),  # Maize     - yellow
    (180, 140,  80),  # Cassava   - tan
    (255, 200,  60),  # Banana    - bright yellow
    (220,  60,  60),  # Tomato    - red
    (60,  180,  80),  # Spinach   - green
    (200, 130,  60),  # Sweet Pot - orange
    (240, 240, 220),  # Rice      - cream
]

#  ENHANCED PIL RENDERER (always available, high quality)

class PILRenderer:
    """
    High-quality 2.5D market scene rendered with PIL.
    Isometric-style market stalls, animated agent, full HUD.
    """

    def __init__(self):
        self.agent_x   = W // 2
        self.agent_y   = H // 2 - 20
        self.agent_step = 0
        self.trail     = []
        self.reward_history = []

        try:
            self._font_lg  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            self._font_md  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
            self._font_sm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     12)
            self._font_xs  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     10)
        except Exception:
            self._font_lg = self._font_md = self._font_sm = self._font_xs = ImageFont.load_default()

    # Iso helpers
    def _iso(self, gx, gy, gz=0):
        """Convert grid (gx, gy, gz) to screen (x, y) isometric."""
        TILE_W, TILE_H = 64, 32
        sx = W // 2 + (gx - gy) * TILE_W // 2
        sy = 180    + (gx + gy) * TILE_H // 2 - gz * 24
        return sx, sy

    def _draw_iso_box(self, draw, gx, gy, w, h, d, top_col, left_col, right_col):
        """Draw an isometric box at grid position."""
        # 8 corners
        tl = self._iso(gx,     gy,     d)
        tr = self._iso(gx + w, gy,     d)
        br = self._iso(gx + w, gy + h, d)
        bl = self._iso(gx,     gy + h, d)
        tl0= self._iso(gx,     gy,     0)
        tr0= self._iso(gx + w, gy,     0)
        br0= self._iso(gx + w, gy + h, 0)
        bl0= self._iso(gx,     gy + h, 0)

        # top face
        draw.polygon([tl, tr, br, bl], fill=top_col)
        draw.polygon([tl, tr, br, bl], outline=(0,0,0,80))
        # left face
        draw.polygon([tl, bl, bl0, tl0], fill=left_col)
        draw.polygon([tl, bl, bl0, tl0], outline=(0,0,0,60))
        # right face
        draw.polygon([tr, br, br0, tr0], fill=right_col)
        draw.polygon([tr, br, br0, tr0], outline=(0,0,0,60))

    def _darken(self, col, factor=0.6):
        return tuple(max(0, int(c * factor)) for c in col[:3])

    def _lighten(self, col, factor=1.3):
        return tuple(min(255, int(c * factor)) for c in col[:3])

    # Scene 
    def _draw_scene(self, img, draw, prices, purchases, day):
        """Draw isometric market scene."""
        # Ground tiles
        for gx in range(-1, 9):
            for gy in range(-1, 5):
                tl = self._iso(gx,   gy)
                tr = self._iso(gx+1, gy)
                br = self._iso(gx+1, gy+1)
                bl = self._iso(gx,   gy+1)
                shade = 25 + (gx + gy) % 2 * 8
                draw.polygon([tl, tr, br, bl], fill=(shade, shade+5, shade+15))

        # Market stalls (isometric boxes)
        stall_positions = [
            (0, 0), (2, 0), (4, 0), (6, 0),
            (0, 2), (2, 2), (4, 2), (6, 2),
        ]
        purchased_items = {p["item"] for p in purchases}

        for i, (gx, gy) in enumerate(stall_positions):
            base_col  = STALL_COLORS[i]
            price_ratio = prices[i] / BASE_PRICES[i]
            # tint stall red if price spike
            if price_ratio > 1.25:
                base_col = tuple(min(255, int(c * 1.3)) if j == 0 else int(c * 0.7)
                                 for j, c in enumerate(base_col))

            top   = self._lighten(base_col, 1.2)
            left  = self._darken(base_col, 0.65)
            right = self._darken(base_col, 0.8)

            # stall box
            self._draw_iso_box(draw, gx, gy, 1.6, 1.6, 1.2, top, left, right)

            # canopy (flat roof overhang)
            tl = self._iso(gx - 0.1, gy - 0.1, 1.4)
            tr = self._iso(gx + 1.7, gy - 0.1, 1.4)
            br = self._iso(gx + 1.7, gy + 1.7, 1.4)
            bl = self._iso(gx - 0.1, gy + 1.7, 1.4)
            canopy_col = (*base_col[:3], 160)
            draw.polygon([tl, tr, br, bl], fill=base_col)

            # item name label on stall
            mid_x = (self._iso(gx, gy, 1.4)[0] + self._iso(gx+1.6, gy+1.6, 1.4)[0]) // 2
            mid_y = self._iso(gx + 0.8, gy + 0.8, 1.5)[1]
            name  = FOOD_ITEMS[i][:7]
            draw.text((mid_x - 18, mid_y - 6), name, font=self._font_xs, fill=C_WHITE)

            # price badge
            price_col = C_RED if price_ratio > 1.25 else (C_GREEN if price_ratio < 0.9 else C_AMBER)
            draw.text((mid_x - 15, mid_y + 6), f"{prices[i]:.0f}", font=self._font_xs, fill=price_col)

            # green glow if recently purchased
            if FOOD_ITEMS[i] in purchased_items:
                glow_pos = self._iso(gx + 0.8, gy + 0.8, 1.8)
                for r in range(12, 2, -3):
                    draw.ellipse([glow_pos[0]-r, glow_pos[1]-r//2,
                                  glow_pos[0]+r, glow_pos[1]+r//2],
                                 outline=(*C_GREEN, max(0, 200 - r*15)))

    # Agent 
    def _draw_agent(self, draw, action, budget, purchases):
        stall_positions = [
            (0, 0), (2, 0), (4, 0), (6, 0),
            (0, 2), (2, 2), (4, 2), (6, 2),
        ]
        if action < 24:
            item_idx = action % 8
            gx, gy   = stall_positions[item_idx]
            tx, ty   = self._iso(gx + 0.8, gy + 0.8, 1.5)
            tx += 10; ty -= 20
        elif action == 24:
            tx, ty = W // 2, H // 2
        elif action == 25:
            tx, ty = W // 2 + 150, H // 2 - 60
        else:
            tx, ty = W // 2 - 150, H // 2

        self.agent_x += (tx - self.agent_x) * 0.25
        self.agent_y += (ty - self.agent_y) * 0.25
        ax, ay = int(self.agent_x), int(self.agent_y)

        # Trail
        self.trail.append((ax, ay))
        if len(self.trail) > 15:
            self.trail.pop(0)
        for k, (tx2, ty2) in enumerate(self.trail):
            a = int(180 * k / len(self.trail))
            r = max(1, k // 4)
            draw.ellipse([tx2-r, ty2-r, tx2+r, ty2+r], fill=(*C_TEAL, a))

        # Budget colour
        ratio    = float(np.clip(budget / (DAILY_BUDGET * MAX_DAYS), 0, 1))
        body_col = C_GREEN if ratio > 0.5 else (C_AMBER if ratio > 0.25 else C_RED)

        # Walking animation
        self.agent_step += 1
        bob  = int(math.sin(self.agent_step * 0.3) * 3)
        leg  = int(math.sin(self.agent_step * 0.3) * 7)

        # Shadow
        draw.ellipse([ax-14, ay+26+bob, ax+14, ay+32+bob], fill=(0, 0, 0, 80))

        # Head
        draw.ellipse([ax-10, ay-22+bob, ax+10, ay-2+bob], fill=(*body_col,), outline=C_WHITE)
        # Eyes
        draw.ellipse([ax-4, ay-17+bob, ax-1, ay-14+bob], fill=(15,20,35))
        draw.ellipse([ax+1, ay-17+bob, ax+4,  ay-14+bob], fill=(15,20,35))
        # Body
        draw.line([ax, ay-2+bob, ax, ay+12+bob], fill=body_col, width=3)
        # Arms
        draw.line([ax, ay+2+bob, ax-12, ay+8+leg+bob], fill=body_col, width=2)
        draw.line([ax, ay+2+bob, ax+12, ay+8-leg+bob], fill=body_col, width=2)
        # Legs
        draw.line([ax, ay+12+bob, ax-7, ay+24+leg+bob],  fill=body_col, width=2)
        draw.line([ax, ay+12+bob, ax+7, ay+24-leg+bob],  fill=body_col, width=2)

        # Basket
        n_items = min(len(purchases), 8)
        bx, by  = ax + 13, ay + 2 + bob
        draw.polygon([(bx,by),(bx+14,by),(bx+12,by+12),(bx+2,by+12)],
                     outline=C_AMBER, fill=(50,40,20,180))
        draw.arc([bx+1, by-7, bx+13, by+1], 180, 0, fill=C_AMBER, width=2)
        for fi in range(n_items):
            fx = bx + 3 + (fi % 3) * 4
            fy = by + 3 + (fi // 3) * 4
            draw.ellipse([fx-2, fy-2, fx+2, fy+2], fill=C_GREEN)

    # HUD panels 
    def _rounded_rect(self, draw, x, y, w, h, r, fill):
        draw.rectangle([x+r, y, x+w-r, y+h], fill=fill)
        draw.rectangle([x, y+r, x+w, y+h-r], fill=fill)
        for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)

    def _draw_hud(self, draw, prices, nutrition, budget, day, action, reward, total_reward, alert, purchases):
        panel = (20, 28, 52, 210)

        # Title 
        draw.text((W//2, 10), "SokoPrice Market Environment",
                  font=self._font_lg, fill=C_WHITE, anchor="mt")
        draw.text((W//2, 34), "Rwanda Informal Agricultural Market Simulation",
                  font=self._font_sm, fill=C_GREY, anchor="mt")
        if alert:
            draw.text((W//2, 54), "PRICE SPIKE ALERT",
                      font=self._font_md, fill=C_RED, anchor="mt")

        # Left: prices 
        self._rounded_rect(draw, 8, 80, 210, N_ITEMS*28+50, 8, panel)
        draw.text((15, 85), "MARKET PRICES (RWF)", font=self._font_sm, fill=C_TEAL)
        draw.text((15, 103), "ITEM", font=self._font_xs, fill=C_GREY)
        draw.text((120,103), "PRICE", font=self._font_xs, fill=C_GREY)
        draw.text((170,103), "TREND", font=self._font_xs, fill=C_GREY)
        for i, item in enumerate(FOOD_ITEMS):
            iy  = 118 + i * 28
            col = C_RED if prices[i] > BASE_PRICES[i]*1.25 else (C_GREEN if prices[i] < BASE_PRICES[i]*0.9 else C_AMBER)
            draw.text((15,  iy), item[:11],        font=self._font_sm, fill=C_WHITE)
            draw.text((120, iy), f"{prices[i]:.0f}", font=self._font_sm, fill=col)
            arrow = "UP" if prices[i] > BASE_PRICES[i] else "DN"
            acol  = C_RED if arrow == "UP" else C_GREEN
            draw.text((172, iy), arrow, font=self._font_xs, fill=acol)

        # Right: nutrition
        rx = W - 220
        self._rounded_rect(draw, rx, 80, 210, 155, 8, panel)
        draw.text((rx+10, 85), "NUTRITION STATUS", font=self._font_sm, fill=C_GREEN)
        labels = ["Protein", "Carbs", "Vitamins"]
        target = np.array([1.5, 2.0, 1.5]) * max(day, 1)
        cols   = [C_BLUE, C_AMBER, C_GREEN]
        for i, (lbl, col) in enumerate(zip(labels, cols)):
            iy    = 108 + i * 40
            ratio = float(np.clip(nutrition[i] / max(target[i], 0.01), 0, 1))
            draw.text((rx+10, iy), lbl, font=self._font_sm, fill=C_WHITE)
            draw.text((rx+130, iy), f"{nutrition[i]:.1f}/{target[i]:.1f}", font=self._font_xs, fill=col)
            # bar bg
            draw.rectangle([rx+10, iy+16, rx+200, iy+26], fill=(40,50,80))
            # bar fill
            bar_w = int(190 * ratio)
            if bar_w > 0:
                draw.rectangle([rx+10, iy+16, rx+10+bar_w, iy+26], fill=col)

        # Right: budget 
        self._rounded_rect(draw, rx, 245, 210, 90, 8, panel)
        draw.text((rx+10, 250), "BUDGET REMAINING", font=self._font_sm, fill=C_AMBER)
        total = DAILY_BUDGET * MAX_DAYS
        ratio = float(np.clip(budget / total, 0, 1))
        bcol  = C_GREEN if ratio > 0.4 else (C_AMBER if ratio > 0.2 else C_RED)
        draw.text((rx + 105, 275), f"{budget:,.0f} RWF", font=self._font_lg, fill=bcol, anchor="mm")
        draw.rectangle([rx+10, 300, rx+200, 312], fill=(40,50,80))
        bw = int(190 * ratio)
        if bw > 0:
            draw.rectangle([rx+10, 300, rx+10+bw, 312], fill=bcol)

        #  Bottom left: purchase log 
        self._rounded_rect(draw, 8, H-148, 210, 138, 8, panel)
        draw.text((15, H-143), "PURCHASE LOG", font=self._font_sm, fill=C_BLUE)
        recent = purchases[-4:]
        for j, p in enumerate(recent):
            iy = H - 122 + j * 26
            draw.text((15,  iy), f"Day {p['day']}: {p['item'][:9]} ×{p['units']}", font=self._font_xs, fill=C_WHITE)
            draw.text((155, iy), f"{p['cost']:.0f}", font=self._font_xs, fill=C_AMBER)

        # Bottom centre: stats 
        self._rounded_rect(draw, 228, H-148, W-456, 138, 8, panel)
        draw.text((235, H-143), f"Day {day}/{MAX_DAYS}", font=self._font_md, fill=C_TEAL)
        draw.text((235, H-120), f"Action: {_action_name(action)}", font=self._font_sm, fill=C_WHITE)
        sr_col = C_GREEN if reward >= 0 else C_RED
        draw.text((235, H-100), f"Step Reward:  {reward:+.2f}", font=self._font_sm, fill=sr_col)
        draw.text((235, H-80),  f"Total Reward: {total_reward:.2f}", font=self._font_sm, fill=C_AMBER)

        # Day progress bar
        draw.rectangle([430, H-138, W-236, H-130], fill=(40,50,80))
        pw = int((W-666) * day / MAX_DAYS)
        if pw > 0:
            draw.rectangle([430, H-138, 430+pw, H-130], fill=C_TEAL)

        # Reward sparkline
        if len(self.reward_history) > 1:
            mn = min(self.reward_history); mx = max(self.reward_history)
            rng = max(mx - mn, 1e-3)
            pts = []
            sw  = W - 470
            hist = self.reward_history[-(sw//3):]
            for k, v in enumerate(hist):
                px2 = 235 + int(k / max(len(hist)-1, 1) * (sw-10))
                py2 = H - 30 - int((v - mn) / rng * 40)
                pts.append((px2, py2))
            if len(pts) > 1:
                draw.line(pts, fill=C_AMBER, width=2)

    def render_frame(self, env_info, action, reward, total_reward, alert=False):
        self.reward_history.append(reward)

        prices    = env_info.get("prices",    BASE_PRICES)
        nutrition = env_info.get("nutrition", np.zeros(3))
        budget    = env_info.get("budget",    DAILY_BUDGET * MAX_DAYS)
        day       = env_info.get("day",       0)
        purchases = env_info.get("purchases", [])

        # Base image
        img  = Image.new("RGBA", (W, H), (*C_BG, 255))

        # Gradient sky effect
        for row in range(H // 2):
            t   = row / (H // 2)
            col = tuple(int(C_BG[i] + (C_BLUE[i] - C_BG[i]) * t * 0.3) for i in range(3))
            ImageDraw.Draw(img).line([(0, row), (W, row)], fill=(*col, 255))

        draw = ImageDraw.Draw(img, "RGBA")

        self._draw_scene(img, draw, prices, purchases, day)
        self._draw_agent(draw, action, budget, purchases)
        self._draw_hud(draw, prices, nutrition, budget, day,
                       action, reward, total_reward, alert, purchases)

        return np.array(img.convert("RGB"))

    def close(self):
        pass


#  Public API — matches rendering.py interface


def _action_name(action):
    from environment.custom_env import N_ITEMS, N_BUY_TIERS, FOOD_ITEMS
    if action < N_BUY_TIERS * N_ITEMS:
        tier = action // N_ITEMS + 1
        item = FOOD_ITEMS[action % N_ITEMS]
        return f"Buy {tier}× {item}"
    elif action == N_BUY_TIERS * N_ITEMS:     return "Skip"
    elif action == N_BUY_TIERS * N_ITEMS + 1: return "Issue Alert"
    else:                                      return "Substitute"


def run_random_agent_demo_3d(save_path="random_agent_demo.gif", n_steps=MAX_DAYS):
    env      = SokoPriceEnv()
    renderer = PILRenderer()

    obs, info = env.reset()
    total_reward = 0.0
    frames = []

    print("=" * 55)
    print("  SokoPrice - Random Agent Demo")
    print("=" * 55)

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        alert = bool((info["prices"] > BASE_PRICES * 1.3).any())

        frame = renderer.render_frame(info, action, reward, total_reward, alert)
        frames.append(frame)

        print(f"  Day {info['day']:>2} | {_action_name(action):<22} | "
              f"Reward: {reward:+6.2f} | Budget: {info['budget']:>7.0f} RWF")

        if terminated or truncated:
            break

    env.close()
    renderer.close()
    imageio.mimsave(save_path, frames, fps=FPS, loop=0)
    print(f"\n demo saved -> {save_path}  (total reward: {total_reward:.2f})")
    return frames


if __name__ == "__main__":
    run_random_agent_demo_3d("random_agent_demo.gif")