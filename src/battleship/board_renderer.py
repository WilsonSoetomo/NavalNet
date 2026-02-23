"""
Matplotlib-based board renderer for Battleship.
Produces visualization of both boards for TensorBoard logging.

Board layout (2-panel for end-of-game, 2-panel for step-by-step):
  Left  = Agent's Defense Board  (agent's ships + opponent's shots)
  Right = Agent's Attack Board   (agent's shots on opponent's grid, with ships revealed)

Cell states (from game_engine.CellState):
  0 = WATER, 1 = SHIP, 2 = HIT, 3 = MISS, 4 = SUNK
"""

import matplotlib

matplotlib.use("Agg")

import io
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .constants import GRID_SIZE, HORIZONTAL

# ── Colour palette ──────────────────────────────────────────────────
_WATER = "#1a3a5c"
_SHIP = "#4a90d9"
_MISS = "#c8ddf0"
_HIT = "#e85d3a"
_SUNK = "#8b1a1a"
_GRID_LINE = "#2a4a6c"
_UNKNOWN = "#1a3a5c"
_BG = "#0d1b2a"


def _cell_color(value: int, is_defense: bool) -> str:
    """Map a CellState int to a hex colour."""
    if value == 0:  # WATER
        return _WATER
    if value == 1:  # SHIP (only visible on defense board)
        return _SHIP if is_defense else _WATER
    if value == 2:  # HIT
        return _HIT
    if value == 3:  # MISS
        return _MISS
    if value == 4:  # SUNK
        return _SUNK
    return _WATER


def _draw_board(
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    is_defense: bool,
    ships: list[dict] | None = None,
    show_ship_outlines: bool = True,
) -> None:
    """Draw a single 10x10 board on the given axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=10)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor(_BG)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            color = _cell_color(int(grid[r, c]), is_defense)
            rect = mpatches.FancyBboxPatch(
                (c - 0.48, r - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=_GRID_LINE, linewidth=0.8,
            )
            ax.add_patch(rect)

            val = int(grid[r, c])
            if val == 2:  # HIT
                ax.plot(c, r, "x", color="white", markersize=14, markeredgewidth=3)
            elif val == 3:  # MISS
                ax.plot(c, r, "o", color="white", markersize=6,
                        markerfacecolor="none", markeredgewidth=1.5)
            elif val == 4:  # SUNK
                ax.plot(c, r, "x", color="#ff6666", markersize=14, markeredgewidth=3)

    if show_ship_outlines and ships:
        for ship in ships:
            sr, sc = ship["row"], ship["col"]
            length = ship["length"]
            if ship["orientation"] == HORIZONTAL:
                w, h = length, 1
            else:
                w, h = 1, length

            sunk = ship.get("sunk", False)
            outline = mpatches.FancyBboxPatch(
                (sc - 0.45, sr - 0.45), w - 0.1, h - 0.1,
                boxstyle="round,pad=0.05",
                facecolor="none",
                edgecolor="#ffffff" if not sunk else "#ff4444",
                linewidth=2.5,
                linestyle="-" if not sunk else "--",
            )
            ax.add_patch(outline)

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels([str(i) for i in range(GRID_SIZE)], fontsize=9, color="#aaa")
    ax.set_yticklabels([str(i) for i in range(GRID_SIZE)], fontsize=9, color="#aaa")
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


def render_game_boards(
    board_state: dict,
    episode: int | None = None,
    step: int | None = None,
    result: str | None = None,
) -> plt.Figure:
    """
    Render both boards side-by-side as a matplotlib Figure.
    Shows opponent ship outlines on the attack board so viewers
    can see the true positions the agent is hunting for.

    Args:
        board_state: dict from BattleshipEnv.get_full_board_state()
        episode: Episode number (for title)
        step: Shot number (for title)
        result: "WIN", "LOSS", or None

    Returns:
        matplotlib Figure (caller should close it after use)
    """
    fig, (ax_def, ax_atk) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(_BG)

    # Build suptitle
    parts = []
    if episode is not None:
        parts.append(f"Episode {episode}")
    if step is not None:
        parts.append(f"Shot {step}")
    if result:
        parts.append(result)

    agent_hits = int((board_state["opponent_grid"] == 2).sum() +
                     (board_state["opponent_grid"] == 4).sum())
    agent_misses = int((board_state["opponent_grid"] == 3).sum())
    opp_hits = int((board_state["agent_grid"] == 2).sum() +
                   (board_state["agent_grid"] == 4).sum())
    opp_misses = int((board_state["agent_grid"] == 3).sum())

    total_shots = agent_hits + agent_misses
    accuracy = (agent_hits / total_shots * 100) if total_shots > 0 else 0

    stats_line = f"Agent: {agent_hits}H/{agent_misses}M ({accuracy:.0f}% acc)  |  Opponent: {opp_hits}H/{opp_misses}M"
    parts.append(stats_line)

    suptitle = " | ".join(parts[:3]) if len(parts) >= 3 else " | ".join(parts)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", color="white", y=0.97)

    # Left: Agent's Defense Board
    _draw_board(
        ax_def,
        board_state["agent_grid"],
        "Agent Defense Board",
        is_defense=True,
        ships=board_state.get("agent_ships"),
        show_ship_outlines=True,
    )

    # Right: Opponent's Board (attack view + reveal ship positions)
    _draw_board(
        ax_atk,
        board_state["opponent_grid"],
        "Opponent Board (ships revealed)",
        is_defense=False,
        ships=board_state.get("opponent_ships"),
        show_ship_outlines=True,
    )

    # Stats subtitle
    fig.text(
        0.5, 0.02, stats_line,
        ha="center", va="bottom", fontsize=11, color="#aabbcc",
        fontfamily="monospace",
    )

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=_WATER, edgecolor=_GRID_LINE, label="Water"),
        mpatches.Patch(facecolor=_SHIP, edgecolor=_GRID_LINE, label="Ship"),
        mpatches.Patch(facecolor=_MISS, edgecolor=_GRID_LINE, label="Miss"),
        mpatches.Patch(facecolor=_HIT, edgecolor=_GRID_LINE, label="Hit"),
        mpatches.Patch(facecolor=_SUNK, edgecolor=_GRID_LINE, label="Sunk"),
    ]
    fig.legend(
        handles=legend_items, loc="lower center", ncol=5,
        fontsize=10, frameon=False,
        labelcolor="white", handlelength=1.5, handletextpad=0.5,
        bbox_to_anchor=(0.5, 0.05),
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.93])
    return fig


def figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Convert a matplotlib Figure to a numpy array (H, W, 3) for TensorBoard.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf).convert("RGB")
    arr = np.array(img)
    buf.close()
    return arr
