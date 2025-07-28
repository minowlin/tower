# minimal_sim.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from numbers import Integral
import math

# These will get bound when you do the interactive launch
state: Dict[str,Any]
build: pd.DataFrame
vis: pd.DataFrame

# ---------- Contracts (no imputation) ----------
ROOM_REQ = {
    "build_id","room_name","room_class","level","cost",
    "yen_reward","track_reward","reward_repeat","slots"
}
VISITOR_REQ = {"visitor","desired_room_class","desired_level"}

TRACK_FOR_CLASS = {
    "Hospitality": "Population",
    "Culture": "Influence",
    "Entertainment": "Interest",
    # Commerce/Service → no track by default
}

def load_build_catalog_strict(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = ROOM_REQ - set(df.columns)
    if missing:
        raise ValueError(f"build CSV missing required columns: {sorted(missing)}")
    # basic type coercions (fail if not convertible)
    df["level"] = df["level"].astype(int)
    df["cost"] = df["cost"].astype(int)
    df["yen_reward"] = df["yen_reward"].astype(int)
    df["track_reward"] = pd.to_numeric(df["track_reward"])
    df["reward_repeat"] = df["reward_repeat"].astype(str)
    df["slots"] = df["slots"].astype(int)
    return df

def load_visitors_strict(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = VISITOR_REQ - set(df.columns)
    if missing:
        raise ValueError(f"visitor CSV missing required columns: {sorted(missing)}")
    df["desired_level"] = df["desired_level"].astype(int)
    return df

# ---------- State ----------
def new_state(start_yen: int = 300) -> Dict[str,Any]:
    return {
        "week": 0,
        "yen": start_yen,
        "population": 0.0,
        "influence": 0.0,
        "interest": 0.0,
        "satisfaction_history": [],
        "rooms": [],  # list of room instances (dicts)
        "log": []
    }

def make_room_instance(row: pd.Series, floor: int) -> Dict[str,Any]:
    """Create a placeable instance from one catalog row."""
    is_repeat = row["reward_repeat"].lower().startswith("rep")
    return {
        "floor": floor,
        "room_name": row["room_name"],
        "room_class": row["room_class"],
        "level": int(row["level"]),
        "yen_reward": int(row["yen_reward"]),
        "cost": int(row["cost"]),
        "track_reward_amount": float(row["track_reward"]),
        "track_reward_key": TRACK_FOR_CLASS.get(row["room_class"]),
        "repeat": is_repeat,
        "slots_total": int(row["slots"]),
        "slots_remaining": int(row["slots"]) if not is_repeat else int(row["slots"]),  # for repeat, this is weekly capacity
        "weekly_served": 0,
        "active": True,
        "window_weeks": int(row["window_weeks"]) if "window_weeks" in row else 4,
        "window_progress": 0,  # 0..window_weeks-1
    }

# ---------- Mechanics ----------
def start_of_week(state: Dict[str,Any]) -> None:
    state["week"] += 1
    for r in state["rooms"]:
        if r["repeat"]:
            # Backfill defaults for old saves:
            r.setdefault("window_weeks", 1)
            r.setdefault("window_progress", 0)

            ww = max(1, int(r["window_weeks"]))
            # Reset ONLY when a window completes
            if r["window_progress"] >= ww - 1:
                r["weekly_served"] = 0
                r["window_progress"] = 0
            else:
                r["window_progress"] += 1

            # keep slots_remaining in sync for display
            r["slots_remaining"] = max(0, int(r["slots_total"]) - int(r["weekly_served"]))
        else:
            # "once" rooms unchanged
            pass

def place_room(state: Dict[str,Any], catalog: pd.DataFrame, sel: str|int) -> Optional[Dict[str,Any]]:
    """Select by build_id (int) or substring of room_name (str). Costs are handled outside."""
    row = None
    if isinstance(sel, Integral) or (isinstance(sel, float) and sel.is_integer()) \
            or (isinstance(sel, str) and sel.strip().lstrip('#').isdigit()):
        bid = int(str(sel).strip().lstrip('#'))  # handles 101, 101.0, "101", "#101"
        m = catalog[pd.to_numeric(catalog["build_id"], errors="coerce") == bid]
        if not m.empty:
            row = m.iloc[0]
    if row is None and isinstance(sel, str):
        m = catalog[catalog["room_name"].str.contains(sel, case=False, na=False)]
        if not m.empty: row = m.iloc[0]
    if row is None: return None
    floor = (max([r["floor"] for r in state["rooms"]] or [0]) + 1)
    inst = make_room_instance(row, floor)
    state["rooms"].append(inst)
    state["yen"] -= int(row["cost"])
    return inst

def match_quality_penalized(match_level: int, desired_level: int, refuse_overshoot: bool = False) -> float:
    if refuse_overshoot and match_level > desired_level:
        return 0.0
    miss = abs(match_level - desired_level)
    return 1 / (1 + miss)  # 1 level over → 0.5, 2 over → 0.333

def choose_best_room(state: Dict[str,Any], desired_class: str, desired_level: int,
                     refuse_overshoot: bool = False) -> Optional[int]:
    """Return index into state['rooms'] or None."""
    cand = []
    for i, r in enumerate(state["rooms"]):
        if not r["active"]: continue
        if r["room_class"].lower() != desired_class.lower(): continue
        # capacity check
        if r["repeat"]:
            if r["weekly_served"] >= r["slots_total"]: continue
        else:
            if r["slots_remaining"] <= 0: continue
        q = match_quality_penalized(r["level"], desired_level, refuse_overshoot)
        cand.append((q, -r["level"], i))  # prefer higher quality, then lower level
    if not cand: return None
    cand.sort(reverse=True)
    if cand[0][0] <= 0.0:  # best quality is zero → treat as no match
        return None
    return cand[0][2]

def serve_room(room: Dict[str,Any], desired_level: int, refuse_overshoot: bool = False) -> Tuple[int, Dict[str,float], float]:
    q = match_quality_penalized(room["level"], desired_level, refuse_overshoot)
    yen = int(round(room["yen_reward"] * q))
    track_delta: Dict[str,float] = {}
    if room.get("track_reward_key") and room.get("track_reward_amount", 0):
        track_delta[room["track_reward_key"]] = track_delta.get(room["track_reward_key"], 0.0) + (room["track_reward_amount"] * q)
    # capacity updates
    if room["repeat"]:
        room["weekly_served"] += 1
        room["slots_remaining"] = max(0, room["slots_total"] - room["weekly_served"])
    else:
        room["slots_remaining"] -= 1
        if room["slots_remaining"] <= 0:
            room["active"] = False
    return yen, track_delta, q

# ---------- Weekly loop ----------
def run_week(
    state: Dict[str, Any],
    build: pd.DataFrame,
    vis: pd.DataFrame,
    build_choice: str|int|None = None,
    draws: int|None = None,
    refuse_overshoot: bool = False
) -> Dict[str,Any]:
    """
    Uses global `state`, `build`, and `vis`.
    If `draws` is None, computes floor(1 + week/4).
    """
    # Compute default draws if not provided
    if draws is None:
        draws = math.floor(1 + (state["week"] / 4))
    start_of_week(state)

    built = None
    built_cost = 0
    if build_choice is not None:
        # validate affordability
        sel_row = None
        if isinstance(build_choice, int) or (isinstance(build_choice, str) and build_choice.isdigit()):
            m = build[build["build_id"] == int(build_choice)]
            if not m.empty: sel_row = m.iloc[0]
        if sel_row is None and isinstance(build_choice, str):
            m = build[build["room_name"].str.contains(build_choice, case=False, na=False)]
            if not m.empty: sel_row = m.iloc[0]
        if sel_row is None:
            raise ValueError(f"No catalog match for choice: {build_choice}")
        if state["yen"] < int(sel_row["cost"]):
            raise ValueError(f"Not enough Yen: need {int(sel_row['cost'])}, have {state['yen']}")
        built_cost = int(sel_row["cost"])
        built_inst = place_room(state, build, sel_row["build_id"])
        built = built_inst["room_name"]

    # Draw visitors (no month unlocks here; you can filter upstream)
    visitors = vis.sample(n=draws, replace=True, random_state=100 + state["week"])

    total_demand = 0
    quality_weighted = 0.0
    yen_gain = 0
    tracks_gain: Dict[str,float] = {}
    assigns = []

    for _, v in visitors.iterrows():
        desired_class = v["desired_room_class"]
        desired_level = int(v["desired_level"])
        total_demand += desired_level

        ridx = choose_best_room(state, desired_class, desired_level, refuse_overshoot)
        if ridx is None:
            assigns.append({"visitor": v["visitor"], "desired_class": desired_class, "desired_level": desired_level,
                            "matched_room": None, "quality": 0.0, "yen": 0})
            continue

        room = state["rooms"][ridx]
        yen, delta, q = serve_room(room, desired_level, refuse_overshoot)
        yen_gain += yen
        for k,val in delta.items():
            tracks_gain[k] = tracks_gain.get(k, 0.0) + val
        quality_weighted += q * desired_level
        assigns.append({"visitor": v["visitor"], "desired_class": desired_class, "desired_level": desired_level,
                        "matched_room": room["room_name"], "quality": q, "yen": yen})

    satisfaction = (quality_weighted / total_demand) if total_demand else 1.0

    # apply gains
    state["yen"] += yen_gain
    state["population"] += tracks_gain.get("Population", 0.0)
    state["influence"] += tracks_gain.get("Influence", 0.0)
    state["interest"] += tracks_gain.get("Interest", 0.0)
    state["satisfaction_history"].append(float(satisfaction))

    # --- Upkeep deduction (e.g. 5% per week of each active repeat room's build cost) ---
    UPKEEP_RATE = 0.05
    upkeep = sum(
        r["cost"] for r in state["rooms"]
        if r["repeat"] and r["active"]
    ) * UPKEEP_RATE
    upkeep = int(round(upkeep))
    state["yen"] -= upkeep

    # log row
    state["log"].append({
        "week": state["week"],
        "built": built, "built_cost": built_cost,
        "yen_earned": yen_gain, "yen_balance": state["yen"],
        "upkeep": upkeep,
        "satisfaction": round(satisfaction, 3),
        "tracks_gain": tracks_gain
    })

    result = {
        "assignments": assigns,
        "summary": {
            "week": state["week"],
            "built": built, "built_cost": built_cost,
            "yen_earned": yen_gain,
            "upkeep": upkeep,
            "yen_balance": state["yen"],
            "satisfaction": satisfaction, **{f"{k}_gain": v for k, v in tracks_gain.items()}
        }
    }

    return result

    # fire callbacks
    show_summary(result["summary"])
    show_assignments(pd.DataFrame(result["assignments"]))
    show_rooms(pd.DataFrame(state["rooms"]))
    show_totals(state)

# ---------- Interface ----------

def show_summary(s):
    print("\n— WEEK SUMMARY —")
    for k,v in s.items(): print(f"{k:12}: {v}")

def show_assignments(df):
    print("\n— ASSIGNMENTS —")
    print(df)

def show_rooms(df):
    print("\n— TOWER ROOMS —")
    print(df[["floor","room_name","level","slots_remaining","active"]])

def show_totals(state):
    print("\n— TOTALS —")
    print(f"Weeks played  : {state['week']}")
    print(f"Yen balance   : {state['yen']}")
    print(f"Population    : {state['population']:.2f}")
    print(f"Influence     : {state['influence']:.2f}")
    print(f"Interest      : {state['interest']:.2f}")
    avg_s = sum(state['satisfaction_history']) / len(state['satisfaction_history']) if state['satisfaction_history'] else 0
    print(f"Avg satisfaction: {avg_s:.3f}")


if __name__ == "__main__":
    import argparse, code

    parser = argparse.ArgumentParser(
        description="Launch interactive sim shell with build, vis, state pre-loaded"
    )
    parser.add_argument("--build-csv",   default="build library.csv")
    parser.add_argument("--visitor-csv", default="visitor_table_basic.csv")
    parser.add_argument("--start-yen",   type=int, default=300)
    parser.add_argument("--draws",       type=int, default=1)
    parser.add_argument("--refuse",      action="store_true",
                        help="disable overshoot matches")
    args = parser.parse_args()

    # Load data and initialize
    build = load_build_catalog_strict(args.build_csv)
    vis   = load_visitors_strict(args.visitor_csv)
    state = new_state(args.start_yen)

    banner = """
Interactive Visitor Deck Sim
----------------------------
Variables pre-loaded for you:
  build   -> build catalog DataFrame
  vis     -> visitor table DataFrame
  state   -> simulation state dict

Helper available:
  run_week(state, build, vis, build_choice, draws, refuse_overshoot)
  show_summary(state["log"][-1])  # to reprint last summary
  show_totals(state)

Example:
>>> out = run_week(state, build, vis, build_choice=None, draws=args.draws, refuse_overshoot=args.refuse)
>>> show_totals(state)
"""
    # start interactive shell
    code.interact(banner=banner, local=globals())

# --------- Graphic render (Pillow) ----------
from PIL import Image, ImageDraw

_PALETTE = {
    "Hospitality": (76, 175, 80),     # green
    "Commerce":    (66, 133, 244),    # blue
    "Entertainment": (255, 152, 0),   # orange
    "Culture":     (156, 39, 176),    # purple
    "Service":     (158, 158, 158),   # gray
}

def _lighten(rgb, factor=0.5):
    r,g,b = rgb
    return (int(r + (255-r)*factor), int(g + (255-g)*factor), int(b + (255-b)*factor))

def render_tower_image(state, tile_w=180, tile_h=70, margin=20):
    rooms = list(state.get("rooms", []))
    floors = sorted({r["floor"] for r in rooms}) or [1]
    by_floor = {}
    for r in rooms:
        by_floor.setdefault(r["floor"], []).append(r)
    for f in by_floor:
        by_floor[f].sort(key=lambda x: (x["room_class"], x["room_name"]))

    max_cols = max((len(v) for v in by_floor.values()), default=1)
    left_g, right_g, top_g, bot_g = 60, 20, 20, 30
    W = left_g + max_cols*tile_w + right_g
    H = top_g + len(floors)*tile_h + bot_g

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # elevator shaft
    shaft_x0, shaft_x1 = 20, left_g - 10
    draw.rectangle([shaft_x0, top_g, shaft_x1, H - bot_g], outline=(200,200,200), width=2, fill=(245,245,245))

    # draw floors bottom-up
    n = len(floors)
    for idx, f in enumerate(sorted(floors)):
        y = top_g + (n - 1 - idx) * tile_h
        # floor tick
        draw.line([(shaft_x0, y+tile_h-1), (shaft_x1, y+tile_h-1)], fill=(210,210,210), width=1)
        draw.text((5, y + tile_h/2 - 6), f"{f}", fill=(0,0,0))

        row = by_floor.get(f, [])
        for j, r in enumerate(row):
            x0 = left_g + j*tile_w + 5
            y0 = y + 5
            x1 = x0 + tile_w - 10
            y1 = y0 + tile_h - 10

            base = _PALETTE.get(r["room_class"], (180,180,180))
            fill = base if r.get("active", True) else _lighten(base, 0.6)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(90,90,90), width=2)

            # label
            name = f"{r['room_name']} (L{r['level']})"
            draw.text((x0+6, y0+6), name[:24], fill=(0,0,0))

            # capacity bar & text
            if r.get("repeat", False):
                cap = max(1, int(r.get("slots_total", 1)))
                used = int(r.get("weekly_served", 0))
                txt = f"{used}/{cap} served"
                frac = min(1.0, used / cap)
            else:
                total = max(1, int(r.get("slots_total", 1)))
                rem = max(0, int(r.get("slots_remaining", total)))
                txt = f"{rem}/{total} left"
                frac = 1 - min(1.0, (total - rem) / total)
            # bar
            bx0, by0 = x0+6, y1-14
            bx1 = x0+6 + int((x1 - (x0+6)) * frac)
            draw.rectangle([x0+6, y1-14, x1-6, y1-8], fill=(230,230,230), outline=(150,150,150))
            draw.rectangle([x0+6, y1-14, bx1,   y1-8], fill=(100,100,100))
            draw.text((x0+6, y1-26), txt, fill=(0,0,0))

    return img
