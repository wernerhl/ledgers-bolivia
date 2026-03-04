"""
LEDGERS OF THE SELF-EMPLOYED
Publication-Quality Illustrations
Four core figures for the paper
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── load data ───────────────────────────────────────────────
pl = pd.read_csv("/home/claude/ledgers_analytics.csv")

MIN_WAGE_HOURLY = 2362 / (4.33 * 8 * 5)   # 13.64 Bs/hr
IVA_RATE        = 0.13
WEEKS_PER_YEAR  = 52

# Derived columns needed
pl["net_per_hour_adj"]  = pl["net_income_adjusted"] / pl["total_hours_weekly"].replace(0, np.nan)
pl["iva_weekly"]        = pl["total_revenue_diary"] * IVA_RATE
pl["net_income_post_iva"] = pl["net_income_adjusted"] - pl["iva_weekly"]

# Sector palette and labels
SECTOR_LABELS = {
    "admin":       "Admin & Apoyo",
    "comercio":    "Comercio",
    "gastronomia": "Gastronomía",
    "manufactura": "Manufactura",
    "servicios":   "Servicios",
    "transporte":  "Transporte",
}
PALETTE = {
    "comercio":    "#2E86AB",
    "manufactura": "#E8821A",
    "transporte":  "#27AE60",
    "gastronomia": "#C0392B",
    "servicios":   "#8E44AD",
    "admin":       "#17A589",
}
TYPOLOGY_COLORS = {
    "I_Viable":              "#27AE60",
    "II_Precaria":           "#F39C12",
    "III_Atrapada_deuda":    "#E74C3C",
    "IV_Riesgo_critico":     "#7D3C98",
}

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "color": "#1a1a2e"}
FONT_LABEL  = {"fontsize": 9,  "color": "#333333"}
FONT_TICK   = {"fontsize": 8,  "color": "#444444"}
FONT_ANNOT  = {"fontsize": 8,  "color": "#222222"}

FIGDIR = "/home/claude/figures"
import os; os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e8e8e8",
    "grid.linewidth":   0.6,
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "font.family":      "DejaVu Sans",
})

# ════════════════════════════════════════════════════════════
# FIGURE 1 — THE PROFIT REVERSAL
# Three panels: (A) share positive by sector conv vs adj
#               (B) waterfall of median profit components
#               (C) full distribution shift
# ════════════════════════════════════════════════════════════

def fig_profit_reversal():
    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38,
                   left=0.06, right=0.97, top=0.88, bottom=0.18)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    fig.suptitle(
        "Figure 1 — The Profit Reversal: From Universal Profitability to Structural Loss",
        **FONT_TITLE, y=0.97
    )

    sectors = ["comercio","manufactura","transporte","gastronomia","servicios","admin"]
    labels  = [SECTOR_LABELS[s] for s in sectors]
    colors  = [PALETTE[s] for s in sectors]

    # ── Panel A: % profitable conventional vs adjusted ──────
    pct_conv = [100.0] * len(sectors)   # by construction
    pct_adj  = [pl[pl["sector"]==s]["net_income_adjusted"].gt(0).mean()*100 for s in sectors]

    x = np.arange(len(sectors))
    w = 0.35
    bars_c = ax0.bar(x - w/2, pct_conv, w, color=colors, alpha=0.3,
                     label="Convencional", edgecolor="white", linewidth=0.5)
    bars_a = ax0.bar(x + w/2, pct_adj,  w, color=colors, alpha=0.95,
                     label="Ajustado", edgecolor="white", linewidth=0.5)

    # annotate adjusted bars
    for bar, v in zip(bars_a, pct_adj):
        ax0.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                 f"{v:.0f}%", ha="center", va="bottom", **FONT_ANNOT, fontweight="bold")

    # reference line
    ax0.axhline(100, color="#aaaaaa", lw=1, ls="--", zorder=0)
    ax0.axhline(50,  color="#dddddd", lw=0.8, ls=":", zorder=0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha="right", **FONT_TICK)
    ax0.set_ylabel("% firmas con ingreso positivo", **FONT_LABEL)
    ax0.set_ylim(0, 118)
    ax0.set_title("A. ¿Quién resulta rentable?", fontsize=10, fontweight="bold", pad=8)

    conv_patch = mpatches.Patch(facecolor="#aaaaaa", alpha=0.4, label="Convencional (100% todas)")
    adj_patch  = mpatches.Patch(facecolor="#555555", alpha=0.9, label="Ajustado (imputa trabajo + dep.)")
    ax0.legend(handles=[conv_patch, adj_patch], fontsize=7, loc="lower right",
               framealpha=0.9, edgecolor="#cccccc")

    # ── Panel B: Waterfall of median profit components ──────
    rev    = pl["total_revenue_diary"].median()
    cogs   = pl["cogs"].median()
    opex   = pl["total_op_exp_conventional"].median() - pl["interest_paid"].median()
    intpd  = pl["interest_paid"].median()
    labor  = pl["labor_imputed"].median()
    dep    = pl["depreciation_imputed"].median()
    net_c  = pl["net_income_conventional"].median()
    net_a  = pl["net_income_adjusted"].median()

    cats   = ["Ingreso\nbruto", "− COGS", "− Gastos\noper.", "− Interés",
              "= Utilidad\nConvenc.", "− Trabajo\nimputado", "− Deprec.",
              "= Utilidad\nAjustada"]
    vals   = [rev, -cogs, -opex, -intpd, net_c, -labor, -dep, net_a]
    cumval = [rev, rev-cogs, rev-cogs-opex, rev-cogs-opex-intpd,
              net_c, net_c-labor, net_c-labor-dep, net_a]
    bottoms= [0, rev-cogs, rev-cogs-opex, rev-cogs-opex-intpd,
              0, net_c, net_c-labor, 0]

    bar_colors = []
    for i, v in enumerate(vals):
        if i in [0, 4, 7]:   # totals
            bar_colors.append("#2c3e50" if i==0 else ("#27AE60" if v>0 else "#C0392B"))
        else:
            bar_colors.append("#e74c3c" if v < 0 else "#27AE60")

    bar_heights = [abs(v) for v in vals]
    for i, (cat, h, bot, col) in enumerate(zip(cats, bar_heights, bottoms, bar_colors)):
        ax1.bar(i, h, bottom=bot, color=col, alpha=0.85, width=0.6,
                edgecolor="white", linewidth=0.8)
        v = vals[i]
        ypos = bot + h + 8 if v >= 0 else bot - 8
        va   = "bottom" if v >= 0 else "top"
        sign = "" if i == 0 else ("+" if v > 0 else "−")
        ax1.text(i, ypos, f"{sign}Bs {abs(v):,.0f}",
                 ha="center", va=va, fontsize=7.5, fontweight="bold",
                 color=col if i not in [0] else "#1a1a2e")

    # connector lines between waterfall steps
    for i in range(len(vals)-1):
        if i not in [3, 6]:   # don't connect into totals
            y_end = bottoms[i] + bar_heights[i]
            ax1.plot([i+0.31, i+0.69], [y_end, y_end],
                     color="#bbbbbb", lw=0.8, ls="--", zorder=5)

    ax1.axhline(0, color="#333333", lw=1, zorder=10)
    ax1.set_xticks(range(len(cats)))
    ax1.set_xticklabels(cats, fontsize=7.5)
    ax1.set_ylabel("Bs / semana (mediana)", **FONT_LABEL)
    ax1.set_title("B. Cascada: cómo se construye la brecha contable", fontsize=10, fontweight="bold", pad=8)
    ax1.set_xlim(-0.6, len(cats)-0.4)

    # ── Panel C: Distribution shift ──────────────────────────
    conv_vals = pl["net_income_conventional"].clip(-200, 4000)
    adj_vals  = pl["net_income_adjusted"].clip(-2000, 4000)

    bins = np.linspace(-2000, 3500, 55)
    ax2.hist(conv_vals, bins=bins, alpha=0.45, color="#2E86AB", label="Convencional",
             edgecolor="white", linewidth=0.3)
    ax2.hist(adj_vals,  bins=bins, alpha=0.65, color="#C0392B", label="Ajustado",
             edgecolor="white", linewidth=0.3)

    ax2.axvline(0, color="#1a1a2e", lw=1.5, zorder=10)
    ax2.axvline(conv_vals.median(), color="#2E86AB", lw=1.5, ls="--", alpha=0.9,
                label=f"Mediana conv. Bs {conv_vals.median():,.0f}")
    ax2.axvline(adj_vals.median(),  color="#C0392B", lw=1.5, ls="--", alpha=0.9,
                label=f"Mediana adj. Bs {adj_vals.median():,.0f}")

    # shade loss region
    ax2.axvspan(-2000, 0, alpha=0.05, color="#C0392B", zorder=0)
    ax2.text(-1000, ax2.get_ylim()[1]*0.6 if ax2.get_ylim()[1]>0 else 10,
             "PÉRDIDA", ha="center", fontsize=8, color="#C0392B", alpha=0.7,
             fontweight="bold", rotation=90)

    # pct annotations
    pct_pos_conv = (pl["net_income_conventional"] > 0).mean()
    pct_pos_adj  = (pl["net_income_adjusted"] > 0).mean()
    ax2.text(0.97, 0.96, f"Convencional: {pct_pos_conv*100:.0f}% positivas",
             transform=ax2.transAxes, ha="right", fontsize=8,
             color="#2E86AB", fontweight="bold")
    ax2.text(0.97, 0.90, f"Ajustado: {pct_pos_adj*100:.0f}% positivas",
             transform=ax2.transAxes, ha="right", fontsize=8,
             color="#C0392B", fontweight="bold")

    ax2.set_xlabel("Ingreso neto Bs / semana", **FONT_LABEL)
    ax2.set_ylabel("N° firmas", **FONT_LABEL)
    ax2.set_title("C. Distribución del ingreso neto", fontsize=10, fontweight="bold", pad=8)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    path = f"{FIGDIR}/fig1_profit_reversal.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ Figure 1 saved")
    return path


# ════════════════════════════════════════════════════════════
# FIGURE 2 — THE DISGUISED UNEMPLOYMENT TAXONOMY
# Scatter: adjusted income vs interest burden
# Color = typology, shape = gender, size = hours
# ════════════════════════════════════════════════════════════

def fig_taxonomy():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             gridspec_kw={"wspace": 0.38,
                                          "left":0.07,"right":0.96,
                                          "top":0.88,"bottom":0.14})
    fig.suptitle(
        "Figure 2 — Taxonomy of Disguised Unemployment: Adjusted Profit vs Debt Burden",
        **FONT_TITLE, y=0.97
    )

    ax0, ax1 = axes

    # ── Panel A: main scatter ────────────────────────────────
    x_col = "interest_burden_pct"
    y_col = "net_income_adjusted"
    df = pl.copy()
    df["ib_plot"]  = df[x_col].clip(0, 40)
    df["ni_plot"]  = df[y_col].clip(-2500, 1500)
    df["hrs_plot"] = (df["total_hours_weekly"].clip(20, 80) - 20) / 60 * 120 + 20

    # quadrant boundaries
    ax0.axhline(0,  color="#555555", lw=1.2, zorder=5)
    ax0.axvline(10, color="#555555", lw=1.2, ls="--", zorder=5)

    # region shading
    ax0.fill_betweenx([-2600, 0],  0, 10, alpha=0.04, color="#F39C12")   # II Precaria
    ax0.fill_betweenx([-2600, 0], 10, 45, alpha=0.05, color="#7D3C98")   # IV Riesgo
    ax0.fill_betweenx([0, 1600],  0, 10, alpha=0.04, color="#27AE60")    # I Viable
    ax0.fill_betweenx([0, 1600], 10, 45, alpha=0.04, color="#E74C3C")    # III Atrapada

    type_map = {
        "I_Viable":              ("#27AE60", "o", "I — Viable"),
        "II_Precaria":           ("#F39C12", "s", "II — Precaria\n(desempleo encubierto puro)"),
        "III_Atrapada_deuda":    ("#E74C3C", "^", "III — Atrapada en deuda"),
        "IV_Riesgo_critico":     ("#7D3C98", "D", "IV — Riesgo crítico\n(desempleo encubierto + deuda)"),
    }

    handles = []
    for typ, (col, mrk, lbl) in type_map.items():
        sub = df[df["typology"] == typ]
        sc  = ax0.scatter(sub["ib_plot"], sub["ni_plot"],
                          c=col, marker=mrk, s=sub["hrs_plot"],
                          alpha=0.65, linewidths=0.3, edgecolors="white",
                          zorder=6)
        handles.append(mpatches.Patch(facecolor=col, label=f"{lbl} (N={len(sub)})"))

    # region labels
    ax0.text(1, 1350,   "I — VIABLE\nempresario genuino",
             fontsize=8, color="#27AE60", fontweight="bold", ha="left", va="top")
    ax0.text(11, 1350,  "III — ATRAPADA EN DEUDA",
             fontsize=8, color="#E74C3C", fontweight="bold", ha="left", va="top")
    ax0.text(1, -2300,  "II — PRECARIA\ndesempleo encubierto puro",
             fontsize=8, color="#E8821A", fontweight="bold", ha="left", va="bottom")
    ax0.text(11, -2300, "IV — RIESGO CRÍTICO\ndesempleo + trampa",
             fontsize=8, color="#7D3C98", fontweight="bold", ha="left", va="bottom")

    # mw threshold line
    mw_weekly = MIN_WAGE_HOURLY * pl["total_hours_weekly"].median()
    ax0.axhline(-mw_weekly, color="#2c3e50", lw=1, ls=":", alpha=0.6, zorder=4)
    ax0.text(38, -mw_weekly - 50, "−salario mínimo equivalente",
             ha="right", fontsize=7, color="#2c3e50", alpha=0.8)

    ax0.set_xlabel("Carga de interés (% ingreso bruto)", **FONT_LABEL)
    ax0.set_ylabel("Ingreso neto ajustado (Bs/semana)", **FONT_LABEL)
    ax0.set_title("A. Espacio ingreso–deuda: cuatro tipos de firma", fontsize=10, fontweight="bold", pad=8)
    ax0.set_xlim(-0.5, 42)
    ax0.set_ylim(-2550, 1620)

    # size legend
    for hrs, lbl in [(30, "30 hrs"), (55, "55 hrs"), (75, "75 hrs")]:
        sz = (hrs - 20) / 60 * 120 + 20
        ax0.scatter([], [], s=sz, color="#888888", alpha=0.5, label=lbl)
    leg1 = ax0.legend(handles=handles, fontsize=7.5, loc="upper right",
                      framealpha=0.95, edgecolor="#cccccc", title="Tipología",
                      title_fontsize=8)
    ax0.add_artist(leg1)

    # ── Panel B: treemap-style sector × typology bars ────────
    order = ["I_Viable","II_Precaria","III_Atrapada_deuda","IV_Riesgo_critico"]
    typology_colors = [TYPOLOGY_COLORS[t] for t in order]
    type_labels = ["I\nViable", "II\nPrecaria", "III\nAtrapada", "IV\nRiesgo\ncrítico"]

    sect_order = ["gastronomia","admin","servicios","comercio","manufactura","transporte"]
    for i, sec in enumerate(sect_order):
        sub = pl[pl["sector"]==sec]
        left = 0
        for typ, col in zip(order, typology_colors):
            n   = (sub["typology"]==typ).sum()
            pct = n / len(sub) * 100
            ax1.barh(i, pct, left=left, color=col, height=0.65,
                     edgecolor="white", linewidth=0.8, alpha=0.88)
            if pct > 7:
                ax1.text(left + pct/2, i, f"{pct:.0f}%",
                         ha="center", va="center", fontsize=7.5,
                         color="white", fontweight="bold")
            left += pct

    ax1.set_yticks(range(len(sect_order)))
    ax1.set_yticklabels([SECTOR_LABELS[s] for s in sect_order], fontsize=9)
    ax1.set_xlabel("% de firmas en el sector", **FONT_LABEL)
    ax1.set_xlim(0, 100)
    ax1.set_title("B. Distribución de tipologías por sector", fontsize=10, fontweight="bold", pad=8)
    ax1.grid(axis="x", color="#e8e8e8", lw=0.6)
    ax1.grid(axis="y", visible=False)

    type_patches = [mpatches.Patch(facecolor=TYPOLOGY_COLORS[t], label=lbl, alpha=0.88)
                    for t, lbl in zip(order, type_labels)]
    ax1.legend(handles=type_patches, loc="lower right", fontsize=8,
               framealpha=0.95, edgecolor="#cccccc", ncol=2,
               title="Tipología", title_fontsize=8)

    path = f"{FIGDIR}/fig2_taxonomy.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ Figure 2 saved")
    return path


# ════════════════════════════════════════════════════════════
# FIGURE 3 — UPDATED LEWIS DIAGRAM
# Classical Lewis diagram but with accounting correction:
# shows that the informal "wage" is below zero when
# all costs are properly counted
# ════════════════════════════════════════════════════════════

def fig_lewis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"wspace":0.42,
                                          "left":0.07,"right":0.96,
                                          "top":0.88,"bottom":0.12})
    fig.suptitle(
        "Figure 3 — The Lewis Diagram with Accounting Correction:\n"
        "The Informal 'Wage' is Below Zero",
        **FONT_TITLE, y=0.99
    )
    ax0, ax1 = axes

    # ── Panel A: Classical Lewis + accounting correction ─────
    L = np.linspace(0, 1, 300)   # labour force share

    # Formal sector: rising marginal product
    L_formal_max = 0.30
    MP_formal = np.where(L <= L_formal_max,
                         4.0 + 6.0 * (L / L_formal_max),
                         np.nan)

    # Informal sector wage (conventional): flat at ~subsistence
    w_conv   = 2.0   # conventional "wage" = survey-implied profit/hour
    # Informal sector wage (adjusted): below zero
    w_adj    = -0.3  # adjusted wage = negative hourly return

    # Minimum wage
    w_min    = 1.0   # normalized

    # Shade areas
    ax0.fill_between(L[:int(0.30*300)],
                     MP_formal[:int(0.30*300)], w_min,
                     where=MP_formal[:int(0.30*300)] >= w_min,
                     alpha=0.12, color="#27AE60", label="Renta formal")
    ax0.fill_between([0.30, 1.0], w_conv, w_adj,
                     alpha=0.10, color="#F39C12", label="Brecha contable informal")
    ax0.fill_between([0.30, 1.0], w_adj, 0,
                     alpha=0.12, color="#C0392B", label="Zona pérdida ajustada")

    # Formal MP curve
    l_f = L[L <= L_formal_max]
    mp_f = 4.0 + 6.0 * (l_f / L_formal_max)
    ax0.plot(l_f, mp_f, color="#2E86AB", lw=2.2, label="Producto marginal formal")

    # Wage lines
    ax0.axhline(w_min,  color="#2c3e50", lw=1.4, ls="-",  label=f"Salario mínimo ($w^*$)")
    ax0.axhline(w_conv, color="#F39C12", lw=1.8, ls="--", label="'Salario' informal (convencional)")
    ax0.axhline(w_adj,  color="#C0392B", lw=1.8, ls="-.", label="Retorno/hora informal (ajustado)")
    ax0.axhline(0,      color="#888888", lw=0.8, ls=":")

    # Turning point
    ax0.axvline(0.30, color="#2E86AB", lw=1.0, ls=":", alpha=0.7)

    # Labels
    ax0.text(0.15, 4.2, "Sector\nFormal", ha="center", fontsize=9,
             fontweight="bold", color="#2E86AB")
    ax0.text(0.65, 4.2, "Sector\nInformal", ha="center", fontsize=9,
             fontweight="bold", color="#C0392B")
    ax0.annotate("", xy=(0.30, 2.0), xytext=(0.45, 2.0),
                 arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
    ax0.text(0.35, 2.15, "Punto de\nquiebre Lewis", ha="center", fontsize=7.5,
             color="#555555")
    ax0.text(0.68, w_conv + 0.12, "Encuesta (EH): ingresos positivos",
             fontsize=7.5, color="#E8821A", style="italic")
    ax0.text(0.68, w_adj - 0.25, "Diario contable: retorno negativo",
             fontsize=7.5, color="#C0392B", style="italic")

    ax0.set_xlabel("Fuerza laboral (fracción)", **FONT_LABEL)
    ax0.set_ylabel("Retorno horario (normalizado)", **FONT_LABEL)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(-1.2, 5.5)
    ax0.set_title("A. Diagrama Lewis con corrección contable", fontsize=10, fontweight="bold", pad=8)
    ax0.legend(fontsize=7.5, loc="upper right", framealpha=0.9,
               edgecolor="#cccccc", ncol=1)

    # ── Panel B: Empirical hourly returns distribution ───────
    hr_adj = pl["net_per_hour_adj"].clip(-60, 40).dropna()

    bins = np.linspace(-60, 40, 45)
    counts, edges = np.histogram(hr_adj, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    colors_bar = ["#C0392B" if c < 0 else
                  ("#F39C12" if c < MIN_WAGE_HOURLY else "#27AE60")
                  for c in centers]

    ax1.bar(centers, counts, width=(bins[1]-bins[0])*0.88,
            color=colors_bar, alpha=0.85, edgecolor="white", linewidth=0.3)

    ax1.axvline(0,                 color="#1a1a2e", lw=1.5, zorder=10)
    ax1.axvline(MIN_WAGE_HOURLY,   color="#2c3e50", lw=1.5, ls="--", zorder=10,
                label=f"Salario mínimo = Bs {MIN_WAGE_HOURLY:.1f}/hr")
    ax1.axvline(hr_adj.median(),   color="#E8821A", lw=1.8, ls="-", zorder=10,
                label=f"Mediana = Bs {hr_adj.median():.2f}/hr")

    # shaded regions with labels
    ymax = counts.max()
    ax1.fill_betweenx([0, ymax*1.05], -62, 0,
                      alpha=0.05, color="#C0392B", zorder=0)
    ax1.fill_betweenx([0, ymax*1.05], 0, MIN_WAGE_HOURLY,
                      alpha=0.05, color="#F39C12", zorder=0)
    ax1.fill_betweenx([0, ymax*1.05], MIN_WAGE_HOURLY, 42,
                      alpha=0.05, color="#27AE60", zorder=0)

    pct_neg     = (hr_adj < 0).mean() * 100
    pct_sub_min = ((hr_adj >= 0) & (hr_adj < MIN_WAGE_HOURLY)).mean() * 100
    pct_above   = (hr_adj >= MIN_WAGE_HOURLY).mean() * 100

    ax1.text(-30, ymax*0.88, f"{pct_neg:.0f}%\nretorno\nnegativo",
             ha="center", fontsize=9, color="#C0392B", fontweight="bold")
    ax1.text(6,   ymax*0.88, f"{pct_sub_min:.0f}%\nbajo\nmínimo",
             ha="center", fontsize=9, color="#E8821A", fontweight="bold")
    ax1.text(25,  ymax*0.88, f"{pct_above:.0f}%\nsobre\nmínimo",
             ha="center", fontsize=9, color="#27AE60", fontweight="bold")

    ax1.set_xlabel("Retorno por hora ajustado (Bs/hora)", **FONT_LABEL)
    ax1.set_ylabel("N° firmas", **FONT_LABEL)
    ax1.set_title("B. Distribución empírica del retorno horario ajustado", fontsize=10,
                  fontweight="bold", pad=8)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    ax1.set_xlim(-62, 42)

    path = f"{FIGDIR}/fig3_lewis.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ Figure 3 saved")
    return path


# ════════════════════════════════════════════════════════════
# FIGURE 4 — TAX INCIDENCE CASCADE
# IVA = 13% applied to the distribution of adjusted income
# Shows: who was already gone, who gets pushed under,
#        who survives, and the regressive incidence path
# ════════════════════════════════════════════════════════════

def fig_tax_cascade():
    fig = plt.figure(figsize=(16, 7))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.40,
                   left=0.06, right=0.97, top=0.88, bottom=0.14)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    fig.suptitle(
        "Figure 4 — IVA Tax Incidence Cascade: Who Pays, Who Exits, Who Survives",
        **FONT_TITLE, y=0.97
    )

    iva = IVA_RATE
    df  = pl.copy()
    df["iva_weekly"]        = df["total_revenue_diary"] * iva
    df["net_post_iva"]      = df["net_income_adjusted"] - df["iva_weekly"]

    # Classification
    df["status_pre"]  = np.where(df["net_income_adjusted"] > 0, "viable_pre",  "loss_pre")
    df["status_post"] = np.where(df["net_post_iva"]         > 0, "viable_post", "loss_post")

    n_total          = len(df)
    n_already_loss   = (df["status_pre"]  == "loss_pre").sum()
    n_newly_pushed   = ((df["status_pre"] == "viable_pre") & (df["status_post"] == "loss_post")).sum()
    n_survive        = (df["status_post"] == "viable_post").sum()

    # ── Panel A: Stacked flow / cascade diagram ──────────────
    categories = ["Pre-IVA\nPérdida\n(desempleo)", "Pre-IVA\nViable",
                  "Post-IVA\nNueva pérdida", "Post-IVA\nSobrevive"]
    values     = [n_already_loss, len(df)-n_already_loss,
                  n_newly_pushed, n_survive]
    pcts       = [v/n_total*100 for v in values]
    bar_colors = ["#C0392B","#27AE60","#E74C3C","#2E86AB"]

    bars = ax0.bar([0, 1, 3, 4], pcts, color=bar_colors, alpha=0.88,
                   width=0.65, edgecolor="white", linewidth=0.8)

    for bar, p, n in zip(bars, pcts, values):
        ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                 f"{p:.0f}%\n(N={n})", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold",
                 color=bar.get_facecolor())

    # arrow from viable to new loss
    ax0.annotate("", xy=(3, pcts[2]+2), xytext=(1, pcts[1]+2),
                 arrowprops=dict(arrowstyle="-|>", color="#888888",
                                 lw=1.5, connectionstyle="arc3,rad=-0.25"))
    ax0.text(2.0, max(pcts[1],pcts[2])+8, f"IVA empuja\n{n_newly_pushed} firmas\na pérdida",
             ha="center", fontsize=8, color="#E74C3C", fontweight="bold")

    ax0.set_xticks([0, 1, 3, 4])
    ax0.set_xticklabels(categories, fontsize=8)
    ax0.set_ylabel("% del total de firmas", **FONT_LABEL)
    ax0.set_ylim(0, max(pcts)*1.35)
    ax0.set_title("A. Efecto del IVA sobre la distribución\nde rentabilidad", fontsize=10,
                  fontweight="bold", pad=8)
    ax0.axhline(0, color="#888888", lw=0.5)

    # ── Panel B: Scatter pre vs post IVA income ──────────────
    clip = 2000
    x = df["net_income_adjusted"].clip(-clip, clip)
    y = df["net_post_iva"].clip(-clip, clip)

    # color by what happened
    c_map = {
        ("loss_pre",  "loss_post"):   "#C0392B",   # already lost
        ("viable_pre","loss_post"):   "#E74C3C",   # newly pushed under
        ("viable_pre","viable_post"): "#27AE60",   # survived
        ("viable_pre","loss_post"):   "#F39C12",
    }
    status_colors = []
    for _, row in df.iterrows():
        if row["status_pre"] == "loss_pre":
            status_colors.append("#C0392B")
        elif row["status_post"] == "loss_post":
            status_colors.append("#F39C12")
        else:
            status_colors.append("#27AE60")

    ax1.scatter(x, y, c=status_colors, alpha=0.55, s=22,
                edgecolors="white", linewidths=0.3, zorder=5)

    # 45-degree line
    lim = clip
    ax1.plot([-lim, lim], [-lim, lim], "#888888", lw=0.8, ls="--", zorder=3,
             label="Sin cambio (sin IVA)")
    ax1.axhline(0, color="#1a1a2e", lw=1.0, zorder=4)
    ax1.axvline(0, color="#1a1a2e", lw=1.0, zorder=4)

    # IVA line: y = x - iva*revenue (slope < 1)
    x_line = np.linspace(-clip, clip, 100)
    med_rev = df["total_revenue_diary"].median()
    ax1.plot(x_line, x_line - df["iva_weekly"].median(), "#E74C3C",
             lw=1.2, ls="-.", alpha=0.7, label=f"Desplazamiento IVA (mediana Bs {df['iva_weekly'].median():.0f}/sem)")

    # legend patches
    p1 = mpatches.Patch(color="#C0392B", alpha=0.7, label=f"Ya en pérdida pre-IVA (N={n_already_loss})")
    p2 = mpatches.Patch(color="#F39C12", alpha=0.7, label=f"Empujadas por IVA (N={n_newly_pushed})")
    p3 = mpatches.Patch(color="#27AE60", alpha=0.7, label=f"Sobreviven post-IVA (N={n_survive})")
    ax1.legend(handles=[p1,p2,p3], fontsize=7.5, loc="upper left",
               framealpha=0.92, edgecolor="#cccccc")

    ax1.set_xlabel("Ingreso ajustado pre-IVA (Bs/sem)", **FONT_LABEL)
    ax1.set_ylabel("Ingreso ajustado post-IVA (Bs/sem)", **FONT_LABEL)
    ax1.set_title("B. Cada punto = una firma:\ndesplazamiento post-IVA", fontsize=10,
                  fontweight="bold", pad=8)
    ax1.set_xlim(-clip-50, clip+50)
    ax1.set_ylim(-clip-50, clip+50)

    # ── Panel C: Tax burden as % of revenue by typology ──────
    df["iva_burden_rev_pct"] = df["iva_weekly"] / df["total_revenue_diary"].replace(0,np.nan) * 100  # = 13% always
    df["iva_burden_ni_pct"]  = df["iva_weekly"] / df["net_income_adjusted"].replace(0,np.nan).abs() * 100

    # effective burden = IVA / adjusted net income
    # for loss-making firms, burden is infinite (or negative denominator)
    # more interesting: IVA / gross revenue = always 13%
    # but IVA / conventional profit varies enormously → shows regressivity

    df["iva_burden_conv_pct"] = df["iva_weekly"] / df["net_income_conventional"].replace(0,np.nan) * 100

    typ_order  = ["I_Viable","II_Precaria","IV_Riesgo_critico"]
    typ_labels = ["I — Viable", "II — Precaria\n(desempleo)", "IV — Riesgo\ncrítico"]
    typ_colors = [TYPOLOGY_COLORS[t] for t in typ_order]

    for i, (typ, lbl, col) in enumerate(zip(typ_order, typ_labels, typ_colors)):
        sub = df[df["typology"]==typ]["iva_burden_conv_pct"].clip(0, 300)
        parts = ax2.violinplot([sub.dropna()], positions=[i], widths=0.6,
                               showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.7)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
        for part in ["cbars","cmaxes","cmins"]:
            parts[part].set_color("#888888"); parts[part].set_linewidth(1)
        med = sub.median()
        ax2.text(i, med + 5, f"Bs {df[df['typology']==typ]['iva_weekly'].median():.0f}/sem\n= {med:.0f}% utilidad",
                 ha="center", fontsize=7.5, color=col, fontweight="bold")

    ax2.axhline(13, color="#2c3e50", lw=1.2, ls="--", label="IVA = 13% del ingreso bruto")
    ax2.axhline(100, color="#C0392B", lw=0.8, ls=":", alpha=0.6, label="Carga = utilidad completa")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(typ_labels, fontsize=9)
    ax2.set_ylabel("IVA como % de la utilidad convencional", **FONT_LABEL)
    ax2.set_ylim(0, 280)
    ax2.set_title("C. Incidencia regresiva: IVA / utilidad\npor tipología", fontsize=10,
                  fontweight="bold", pad=8)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    path = f"{FIGDIR}/fig4_tax_cascade.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ Figure 4 saved")
    return path


# ════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating paper illustrations...")
    fig_profit_reversal()
    fig_taxonomy()
    fig_lewis()
    fig_tax_cascade()
    print(f"\nAll figures saved to {FIGDIR}/")
    import glob
    for f in sorted(glob.glob(f"{FIGDIR}/*.png")):
        size = os.path.getsize(f) // 1024
        print(f"  {os.path.basename(f):<40} {size:>4} KB")
