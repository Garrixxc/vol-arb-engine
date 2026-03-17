"""
backtest/report.py

Generates a professional tearsheet: P&L curve, drawdown,
Greeks exposure, and trade distribution charts.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

from backtest.metrics import full_metrics_report


DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GREEN    = "#3fb950"
RED      = "#f85149"
BLUE     = "#58a6ff"
GOLD     = "#e3b341"
GREY     = "#8b949e"
WHITE    = "#e6edf3"


def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.spines["bottom"].set_color(GREY)
    ax.spines["left"].set_color(GREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, color=WHITE, fontsize=9, fontweight="bold", pad=6)


def generate_tearsheet(
    portfolio_df: pd.DataFrame,
    trades_df:    pd.DataFrame,
    event_log:    pd.DataFrame,
    capital:      float = 100_000,
    output_path:  str   = "tearsheet.png",
    strategy_name: str  = "Vol Arb Engine — Delta-Hedged SVI Mispricing",
) -> dict:
    """
    Generate a full tearsheet PNG and return metrics dict.
    """
    metrics = full_metrics_report(portfolio_df, trades_df, capital, verbose=True)

    if portfolio_df.empty:
        print("[Report] No data to plot.")
        return metrics

    dates   = pd.to_datetime(portfolio_df["date"] if "date" in portfolio_df.columns
                             else range(len(portfolio_df)))
    cum_pnl = portfolio_df["cum_pnl"].values
    day_pnl = portfolio_df["day_pnl"].values

    # Drawdown series
    peak    = np.maximum.accumulate(cum_pnl)
    dd      = cum_pnl - peak

    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    fig.suptitle(strategy_name, color=WHITE, fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ── 1. Cumulative P&L (wide) ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _style_ax(ax1, "Cumulative P&L")
    color = GREEN if cum_pnl[-1] >= 0 else RED
    ax1.plot(dates, cum_pnl, color=color, lw=1.5, zorder=3)
    ax1.fill_between(dates, cum_pnl, 0,
                     where=cum_pnl >= 0, alpha=0.15, color=GREEN)
    ax1.fill_between(dates, cum_pnl, 0,
                     where=cum_pnl <  0, alpha=0.15, color=RED)
    ax1.axhline(0, color=GREY, lw=0.7, ls="--", alpha=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.tick_params(axis='x', labelrotation=30)

    # ── 2. Key stats panel ────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.set_facecolor(PANEL_BG)
    ax_stats.axis("off")
    ax_stats.set_title("Key Metrics", color=WHITE, fontsize=9,
                        fontweight="bold", pad=6)

    stats = [
        ("Total P&L",     f"${metrics.get('total_pnl',0):,.0f}"),
        ("Ann. Return",   f"{metrics.get('ann_return_pct',0):.1f}%"),
        ("Sharpe",        f"{metrics.get('sharpe',0):.3f}"),
        ("Sortino",       f"{metrics.get('sortino',0):.3f}"),
        ("Max DD",        f"${metrics.get('max_dd_pnl',0):,.0f}"),
        ("Hit Rate",      f"{metrics.get('hit_rate_pct',0):.1f}%"),
        ("Profit Factor", f"{metrics.get('profit_factor',0):.2f}"),
        ("Trades",        f"{metrics.get('n_trades',0)}"),
    ]
    for j, (label, value) in enumerate(stats):
        y = 0.92 - j * 0.115
        ax_stats.text(0.05, y, label, color=GREY,   fontsize=8.5,
                      transform=ax_stats.transAxes)
        col = GREEN if "P&L" in label or "Return" in label else WHITE
        if "DD" in label and metrics.get('max_dd_pnl',0) < 0:
            col = RED
        ax_stats.text(0.95, y, value, color=col, fontsize=8.5,
                      fontweight="bold", ha="right",
                      transform=ax_stats.transAxes)

    # ── 3. Drawdown ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    _style_ax(ax2, "Drawdown")
    ax2.fill_between(dates, dd, 0, color=RED, alpha=0.4)
    ax2.plot(dates, dd, color=RED, lw=1.0)
    ax2.axhline(0, color=GREY, lw=0.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.tick_params(axis='x', labelrotation=30)

    # ── 4. Daily P&L distribution ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    _style_ax(ax3, "Daily P&L Distribution")
    pos_pnl = day_pnl[day_pnl >= 0]
    neg_pnl = day_pnl[day_pnl <  0]
    bins = np.linspace(day_pnl.min(), day_pnl.max(), 30)
    ax3.hist(pos_pnl, bins=bins, color=GREEN, alpha=0.7, label="Profit")
    ax3.hist(neg_pnl, bins=bins, color=RED,   alpha=0.7, label="Loss")
    ax3.axvline(day_pnl.mean(), color=GOLD, lw=1.5, ls="--",
                label=f"Mean ${day_pnl.mean():,.0f}")
    ax3.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=WHITE, framealpha=0.5)
    ax3.tick_params(colors=GREY, labelsize=7)

    # ── 5. Open positions over time ───────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _style_ax(ax4, "Open Positions")
    if "n_positions" in portfolio_df.columns:
        ax4.bar(dates, portfolio_df["n_positions"].values,
                color=BLUE, alpha=0.7, width=0.8)
        ax4.set_ylim(0, portfolio_df["n_positions"].max() + 1)
    ax4.tick_params(axis='x', labelrotation=30)

    # ── 6. Portfolio vega exposure ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _style_ax(ax5, "Portfolio Net Vega")
    if "port_vega" in portfolio_df.columns:
        vega = portfolio_df["port_vega"].values
        ax5.plot(dates, vega, color=GOLD, lw=1.2)
        ax5.fill_between(dates, vega, 0,
                         where=vega >= 0, color=GREEN, alpha=0.2)
        ax5.fill_between(dates, vega, 0,
                         where=vega <  0, color=RED,   alpha=0.2)
        ax5.axhline(0, color=GREY, lw=0.5, ls="--")
    ax5.tick_params(axis='x', labelrotation=30)

    # ── 7. Trade P&L scatter ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    _style_ax(ax6, "Trade P&L vs Entry Mispricing")
    if not trades_df.empty and "mispricing_entry_bps" in trades_df.columns:
        closed = trades_df[trades_df["exit_date"] != "OPEN"].dropna(
            subset=["mispricing_entry_bps", "total_pnl"]
        )
        if len(closed) > 0:
            colors = [GREEN if p > 0 else RED for p in closed["total_pnl"]]
            ax6.scatter(closed["mispricing_entry_bps"], closed["total_pnl"],
                        c=colors, alpha=0.7, s=40, zorder=3)
            ax6.axhline(0, color=GREY, lw=0.5, ls="--")
            ax6.axvline(0, color=GREY, lw=0.5, ls="--")
            ax6.set_xlabel("Entry Mispricing (bps)", color=GREY, fontsize=7)
            ax6.set_ylabel("Trade P&L ($)", color=GREY, fontsize=7)
            # Fit line
            if len(closed) > 3:
                z = np.polyfit(closed["mispricing_entry_bps"],
                               closed["total_pnl"], 1)
                x_line = np.linspace(closed["mispricing_entry_bps"].min(),
                                     closed["mispricing_entry_bps"].max(), 50)
                ax6.plot(x_line, np.polyval(z, x_line),
                         color=GOLD, lw=1.2, ls="--", alpha=0.8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"\n[Report] Tearsheet saved → {output_path}")
    return metrics
