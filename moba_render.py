from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROLE_COLORS = {
    "Tank": (120, 73, 47),
    "Fighter": (206, 137, 55),
    "Assassin": (113, 38, 138),
    "Marksman": (52, 105, 189),
    "Support": (52, 153, 114),
}


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_health_bar(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, frac: float, fill: Tuple[int, int, int]):
    frac = max(0.0, min(1.0, frac))
    draw.rounded_rectangle((x, y, x + width, y + 8), radius=4, fill=(32, 36, 40))
    draw.rounded_rectangle((x, y, x + int(width * frac), y + 8), radius=4, fill=fill)


def render_moba_frame(obs_dict: Dict, grid_size: int, scale: int = 44, hud_height: int = 92) -> np.ndarray:
    width = grid_size * scale
    height = grid_size * scale + hud_height
    image = Image.new("RGB", (width, height), (32, 66, 42))
    draw = ImageDraw.Draw(image)

    lane_color = (176, 154, 120)
    lane_edge = (208, 192, 156)
    grid_line = (59, 92, 62)
    red_base = (115, 30, 30)
    blue_base = (25, 45, 115)
    stone = (103, 110, 118)
    header = (15, 20, 28)

    draw.rectangle((0, 0, width, hud_height), fill=header)

    lanes = obs_dict["lanes"]
    for lane_y in lanes.values():
        top = hud_height + lane_y * scale
        draw.rounded_rectangle((0, top, width, top + scale), radius=12, fill=lane_color, outline=lane_edge, width=2)
        for x in range(0, width, scale):
            draw.line((x + scale // 4, top + scale // 2, x + 3 * scale // 4, top + scale // 2), fill=(235, 228, 204), width=2)

    for x in range(grid_size + 1):
        draw.line((x * scale, hud_height, x * scale, height), fill=grid_line, width=1)
    for y in range(grid_size + 1):
        draw.line((0, hud_height + y * scale, width, hud_height + y * scale), fill=grid_line, width=1)

    base_pad = scale
    draw.rounded_rectangle((0, hud_height + height - hud_height - 4 * scale, 4 * scale, height), radius=22, fill=red_base)
    draw.rounded_rectangle((width - 4 * scale, hud_height, width, hud_height + 4 * scale), radius=22, fill=blue_base)

    for ox, oy in obs_dict.get("obstacles", []):
        left = ox * scale + scale // 6
        top = hud_height + oy * scale + scale // 6
        draw.rounded_rectangle((left, top, left + 2 * scale // 3, top + 2 * scale // 3), radius=8, fill=stone)

    for dx, dy in obs_dict.get("dynamic_obstacles", []):
        cx = dx * scale + scale // 2
        cy = hud_height + dy * scale + scale // 2
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(219, 93, 52), outline=(248, 172, 124), width=2)

    for tower in obs_dict["towers"]:
        tx, ty = tower["pos"]
        cx = tx * scale + scale // 2
        cy = hud_height + ty * scale + scale // 2
        alive = tower["hp"] > 0
        color = (178, 52, 52) if tower["team"] == "red" else (58, 87, 188)
        ring = (220, 127, 127, 80) if tower["team"] == "red" else (126, 156, 240, 80)
        if alive:
            draw.ellipse((cx - scale, cy - scale, cx + scale, cy + scale), outline=ring[:3], width=2)
            draw.rounded_rectangle((cx - 12, cy - 18, cx + 12, cy + 18), radius=6, fill=color, outline=(240, 235, 228), width=2)
            draw.rectangle((cx - 5, cy - 28, cx + 5, cy - 12), fill=color)
            _draw_health_bar(draw, cx - 18, cy - 40, 36, tower["hp"] / max(1.0, tower["max_hp"]), (96, 235, 118))
        else:
            draw.line((cx - 16, cy - 16, cx + 16, cy + 16), fill=(70, 70, 70), width=4)
            draw.line((cx + 16, cy - 16, cx - 16, cy + 16), fill=(70, 70, 70), width=4)

    for key, color in [("red_core_pos", (218, 74, 74)), ("blue_core_pos", (83, 132, 238))]:
        x, y = obs_dict[key]
        cx = x * scale + scale // 2
        cy = hud_height + y * scale + scale // 2
        draw.polygon(
            [(cx, cy - 18), (cx + 14, cy), (cx, cy + 18), (cx - 14, cy)],
            fill=color,
            outline=(249, 241, 232),
        )

    # Make fights obvious in exported media by drawing combat links between nearby enemies.
    for rid in obs_dict["red_team"]:
        if rid in obs_dict["respawn_timers"]:
            continue
        rx, ry = obs_dict["agent_positions"][rid]
        rcx = rx * scale + scale // 2
        rcy = hud_height + ry * scale + scale // 2
        for bid in obs_dict["blue_team"]:
            if bid in obs_dict["respawn_timers"]:
                continue
            bx, by = obs_dict["agent_positions"][bid]
            dist = abs(rx - bx) + abs(ry - by)
            if dist <= 3:
                bcx = bx * scale + scale // 2
                bcy = hud_height + by * scale + scale // 2
                draw.line((rcx, rcy, bcx, bcy), fill=(255, 208, 120), width=3)
                draw.ellipse((rcx - 4, rcy - 4, rcx + 4, rcy + 4), fill=(255, 235, 170))
                draw.ellipse((bcx - 4, bcy - 4, bcx + 4, bcy + 4), fill=(255, 235, 170))

    for aid, (ax, ay) in obs_dict["agent_positions"].items():
        role = obs_dict["agent_roles"][aid]
        hp = obs_dict["agent_health"][aid]
        respawn = obs_dict["respawn_timers"].get(aid, 0)
        cx = ax * scale + scale // 2
        cy = hud_height + ay * scale + scale // 2
        team_fill = (214, 88, 88) if aid in obs_dict["red_team"] else (88, 118, 214)
        role_fill = ROLE_COLORS.get(role, team_fill)
        outline = (245, 239, 230)
        halo = (255, 210, 210) if aid in obs_dict["red_team"] else (205, 220, 255)

        if respawn > 0:
            draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), outline=(255, 214, 92), width=3)
            continue

        draw.ellipse((cx - 20, cy - 20, cx + 20, cy + 20), fill=halo, outline=team_fill, width=2)

        if role == "Tank":
            draw.rounded_rectangle((cx - 18, cy - 18, cx + 18, cy + 18), radius=7, fill=role_fill, outline=outline, width=2)
        elif role == "Assassin":
            draw.polygon([(cx, cy - 19), (cx + 17, cy), (cx, cy + 19), (cx - 17, cy)], fill=role_fill, outline=outline)
        elif role == "Marksman":
            draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=role_fill, outline=outline, width=2)
            draw.line((cx - 21, cy, cx + 21, cy), fill=outline, width=2)
        elif role == "Support":
            draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=role_fill, outline=outline, width=2)
            draw.line((cx, cy - 12, cx, cy + 12), fill=outline, width=2)
            draw.line((cx - 12, cy, cx + 12, cy), fill=outline, width=2)
        else:
            draw.polygon([(cx, cy - 19), (cx + 17, cy + 14), (cx - 17, cy + 14)], fill=role_fill, outline=outline)

        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=team_fill)
        max_hp = 180.0 if role == "Tank" else 130.0 if role == "Fighter" else 95.0 if role == "Assassin" else 90.0 if role == "Marksman" else 110.0
        _draw_health_bar(draw, cx - 20, cy - 32, 40, hp / max_hp, (98, 241, 127))
        label = f"{'R' if aid in obs_dict['red_team'] else 'B'}{aid}"
        draw.text((cx - 11, cy + 19), label, fill=(250, 245, 238), font=_font(11))

    title_font = _font(30)
    body_font = _font(18)
    red_towers_destroyed = sum(1 for tower in obs_dict["towers"] if tower["team"] == "blue" and tower["hp"] <= 0)
    blue_towers_destroyed = sum(1 for tower in obs_dict["towers"] if tower["team"] == "red" and tower["hp"] <= 0)
    draw.text((22, 18), "Clash Demo: 5v5 MOBA Arena", fill=(247, 244, 238), font=title_font)
    draw.text((22, 54), f"Step {obs_dict['step']}", fill=(209, 216, 224), font=body_font)
    draw.text((220, 54), f"Red Score {obs_dict['red_score']}", fill=(235, 150, 150), font=body_font)
    draw.text((390, 54), f"Blue Score {obs_dict['blue_score']}", fill=(150, 180, 242), font=body_font)
    draw.text((520, 54), f"Red Towers {red_towers_destroyed}", fill=(235, 150, 150), font=body_font)
    draw.text((680, 54), f"Blue Towers {blue_towers_destroyed}", fill=(150, 180, 242), font=body_font)
    draw.text((840, 54), f"Red Core HP {obs_dict['red_core_hp']:.0f}", fill=(235, 150, 150), font=body_font)
    draw.text((1060, 54), f"Blue Core HP {obs_dict['blue_core_hp']:.0f}", fill=(150, 180, 242), font=body_font)

    legend_y = 18
    legend_x = width - 280
    for idx, role in enumerate(["Tank", "Fighter", "Assassin", "Marksman", "Support"]):
        draw.rounded_rectangle((legend_x, legend_y + idx * 14, legend_x + 12, legend_y + 10 + idx * 14), radius=3, fill=ROLE_COLORS[role])
        draw.text((legend_x + 18, legend_y - 2 + idx * 14), role, fill=(230, 234, 239), font=_font(12))

    return np.asarray(image, dtype=np.uint8)
