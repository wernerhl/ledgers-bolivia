"""
reproduce_ledgers_v3.py
=======================
V3 CORRECTIONS: unified Tobin's Q (full opening book value, no
depreciation add-back, N=276), extended shadow-wage grid + Mincer
Panel B, gender column (4), epsilon decomposition, balance table.
Optional inputs: diary_shadow_wages.csv, eh_selfemployed_balance.csv
=======================
Replication code for:
  "Ledgers of the Self-Employed: Accounting for the Invisible Firm"
  Werner Hernani-Limarino (2026)
  Target: Journal of Development Economics

Produces:
  MAIN TEXT:
    Tables 1-9
    Figures 1-8
  
  APPENDIX:
    Tables A1-A4
    Figures B1-B12

Requirements: pandas, numpy, scipy, matplotlib, seaborn, statsmodels

Usage:
  python reproduce_ledgers_v2.py
  (run from directory containing the three CSV data files)

Outputs written to ./output/
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def out(fname):
    return os.path.join(OUT_DIR, fname)

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

# Constants
SHADOW_WAGE  = 13.64  # Bs/hour (minimum wage)
DISC_RATE    = 0.18   # Informal sector hurdle rate
WEEKS_YEAR   = 52
POVERTY_LINE = 254    # Bs/week per capita

SECTOR_ORDER = ["Comercio","Manufactura","Transporte","Gastronomia","Construccion","Serv_Profesionales"]
SECTOR_LABEL = {
    "Comercio": "Commerce",
    "Manufactura": "Manufacturing",
    "Transporte": "Transport",
    "Gastronomia": "Food service",
    "Construccion": "Construction",
    "Serv_Profesionales": "Prof. services"
}
SECTOR_LABEL_ES = {
    "Comercio": "Comercio",
    "Manufactura": "Manufactura",
    "Transporte": "Transporte",
    "Gastronomia": "Gastronomía",
    "Construccion": "Construcción",
    "Serv_Profesionales": "Serv. Prof."
}

# Colors
COLORS_TYPE = {
    "I_Viable": "#27ae60",
    "II_Precaria": "#e67e22", 
    "III_Atrapada_deuda": "#8e44ad",
    "IV_Zombie": "#c0392b"
}
COLORS_SECTOR = {
    "Comercio": "#3498db",
    "Manufactura": "#e74c3c",
    "Transporte": "#2ecc71",
    "Gastronomia": "#9b59b6",
    "Construccion": "#f39c12",
    "Serv_Profesionales": "#1abc9c"
}
COLOR_FEMALE = "#c0392b"
COLOR_MALE = "#2980b9"
COLOR_EH = "#95a5a6"
COLOR_DIARY_CONV = "#3498db"
COLOR_DIARY_ADJ = "#e74c3c"

# ─── LOAD DATA ──────────────────────────────────────────────────────────────
print("="*70)
print("LEDGERS OF THE SELF-EMPLOYED: Replication Script v2")
print("="*70)
print("\nLoading data...")

d5 = pd.read_csv(os.path.join(DATA_DIR, "diary_firms_500.csv"))
da = pd.read_csv(os.path.join(DATA_DIR, "diary_accounts_500.csv"))
dt = pd.read_csv(os.path.join(DATA_DIR, "diary_transactions_500.csv"))

print(f"  Firms:        {d5.shape[0]:,} observations, {d5.shape[1]} variables")
print(f"  Accounts:     {da.shape[0]:,} observations, {da.shape[1]} variables")
print(f"  Transactions: {dt.shape[0]:,} observations, {dt.shape[1]} variables")

# Merge for convenience - select only columns from da not already in d5
da_cols = ['firm_id', 'cf_operating', 'labor_imputed', 'depreciation_imputed', 
           'interest_paid', 'gross_profit', 'typology_ledger', 'hourly_return_adj',
           'roa_annualized', 'interest_burden_pct', 'eh_yi_net_weekly', 'cogs',
           'total_revenue_weekly']
da_cols = [c for c in da_cols if c in da.columns]
df = d5.merge(da[da_cols], on='firm_id', how='left', suffixes=('', '_da'))

# Derived variables
df['female'] = (df['hombre'] == 0).astype(int)
df['loss_making'] = (df['net_income_adjusted'] <= 0).astype(int)
df['viable'] = (df['net_income_adjusted'] > 0).astype(int)
df['poor'] = (df['p0'] == 'Pobre').astype(int)
# V3 FIX: do NOT alias conventional_net_weekly (wrong column pair);
# d5['net_income_conventional'] is the audited series (median 1,149; identity closes to Bs 0.01)

print(f"\n  Key statistics:")
print(f"    Loss-making firms: {df['loss_making'].sum()} ({100*df['loss_making'].mean():.1f}%)")
print(f"    Female-owned:      {df['female'].sum()} ({100*df['female'].mean():.1f}%)")
print(f"    With active debt:  {df['has_debt'].sum()} ({100*df['has_debt'].mean():.1f}%)")

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def ols_hc3(y, X):
    """OLS with HC3 heteroskedasticity-robust standard errors."""
    coef, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = len(y), X.shape[1]
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(hat).clip(max=0.9999)
    e2 = (resid / (1 - h)) ** 2
    meat = (X.T * e2) @ X
    vcov = np.linalg.inv(X.T @ X) @ meat @ np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(vcov))
    t = coef / np.where(se > 0, se, 1e-12)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    return coef, se, t, r2, n

def stars(tv):
    """Significance stars from t-value."""
    a = abs(tv)
    return "***" if a > 2.576 else ("**" if a > 1.96 else ("*" if a > 1.645 else ""))

def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for proportion."""
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - spread), min(1, center + spread)

def format_pct(x):
    return f"{100*x:.1f}%"

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEXT TABLES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("MAIN TEXT TABLES")
print("─"*70)

# ─── TABLE 1: Descriptive Statistics by Sector ──────────────────────────────
print("\nTable 1: Descriptive Statistics by Sector...")
rows_t1 = []
for s in SECTOR_ORDER:
    sub = df[df["sector"] == s]
    rows_t1.append({
        "Sector": SECTOR_LABEL[s],
        "N": len(sub),
        "Female (%)": round(100 * sub["female"].mean(), 1),
        "Poor (%)": round(100 * sub["poor"].mean(), 1),
        "Indebted (%)": round(100 * sub["has_debt"].mean(), 1),
        "Hours/wk": round(sub["tothrs"].median(), 0),
        "Revenue (Bs/wk)": round(sub["total_revenue_weekly"].median(), 0),
        "Loss-making (%)": round(100 * sub["loss_making"].mean(), 1)
    })
# Total row
rows_t1.append({
    "Sector": "Total",
    "N": len(df),
    "Female (%)": round(100 * df["female"].mean(), 1),
    "Poor (%)": round(100 * df["poor"].mean(), 1),
    "Indebted (%)": round(100 * df["has_debt"].mean(), 1),
    "Hours/wk": round(df["tothrs"].median(), 0),
    "Revenue (Bs/wk)": round(df["total_revenue_weekly"].median(), 0),
    "Loss-making (%)": round(100 * df["loss_making"].mean(), 1)
})
pd.DataFrame(rows_t1).to_csv(out("table1_descriptive.csv"), index=False)

# ─── TABLE 2: Transaction Records by Module ─────────────────────────────────
print("Table 2: Transaction Records by Module...")
active = dt[dt["module"] != "DIA_INACTIVO"]
module_stats = active.groupby("module").agg(
    N=("tx_id", "count"),
    Total_Bs=("amount_bs", "sum"),
    Median_Bs=("amount_bs", "median")
).reset_index()
module_stats["Pct_N"] = (100 * module_stats["N"] / len(active)).round(1)
module_stats = module_stats.sort_values("N", ascending=False)
module_stats.to_csv(out("table2_modules.csv"), index=False)

# ─── TABLE 3: Weekly Income Statement by Sector ─────────────────────────────
print("Table 3: Income Statement by Sector...")
is_vars = ["total_revenue_weekly", "cogs", "gross_profit", 
           "labor_imputed", "depreciation_imputed", "interest_paid",
           "net_income_conventional", "net_income_adjusted"]
is_labels = ["Gross revenue", "COGS", "Gross profit",
             "Imputed labour", "Depreciation", "Interest paid",
             "Net income (conv.)", "Net income (adj.)"]

rows_t3 = []
for var, lbl in zip(is_vars, is_labels):
    row = {"Item": lbl}
    for s in SECTOR_ORDER:
        sub = da[da["sector"] == s]
        row[SECTOR_LABEL[s]] = round(sub[var].median(), 0)
    row["All"] = round(da[var].median(), 0)
    rows_t3.append(row)
pd.DataFrame(rows_t3).to_csv(out("table3_income_statement.csv"), index=False)

# ─── TABLE 4: Firm Typology (2x2) ───────────────────────────────────────────
print("Table 4: Firm Typology...")
da["viable"] = (da["net_income_adjusted"] > 0).astype(int)
da["high_burden"] = (da["interest_burden_pct"] >= 10).astype(int)

n_I = ((da["viable"]==1) & (da["high_burden"]==0)).sum()
n_II = ((da["viable"]==0) & (da["high_burden"]==0)).sum()
n_III = ((da["viable"]==1) & (da["high_burden"]==1)).sum()
n_IV = ((da["viable"]==0) & (da["high_burden"]==1)).sum()
N = len(da)

typology_df = pd.DataFrame({
    "Type": ["I (Viable)", "II (Precarious)", "III (Debt-trapped)", "IV (Critical)"],
    "Condition": ["π_adj > 0, burden < 10%", "π_adj ≤ 0, burden < 10%", 
                  "π_adj > 0, burden ≥ 10%", "π_adj ≤ 0, burden ≥ 10%"],
    "N": [n_I, n_II, n_III, n_IV],
    "Pct": [round(100*n_I/N,1), round(100*n_II/N,1), round(100*n_III/N,1), round(100*n_IV/N,1)]
})
typology_df.to_csv(out("table4_typology.csv"), index=False)
print(f"  Type I: {n_I} ({100*n_I/N:.1f}%), Type II: {n_II} ({100*n_II/N:.1f}%), "
      f"Type III: {n_III} ({100*n_III/N:.1f}%), Type IV: {n_IV} ({100*n_IV/N:.1f}%)")

# ─── TABLE 5: Financial Ratios ──────────────────────────────────────────────
print("Table 5: Financial Ratios...")
ratio_vars = ["gross_margin_pct", "net_margin_adj_pct", "roa_annualized",
              "hourly_return_adj", "cf_operating", "interest_burden_pct"]
ratio_labels = ["Gross margin (%)", "Adj. net margin (%)", "ROA annualised (%)",
                "Hourly return (Bs/hr)", "Operating CF (Bs/wk)", "Interest burden (%)"]

rows_t5 = []
for var, lbl in zip(ratio_vars, ratio_labels):
    col = da[var].replace([np.inf, -np.inf], np.nan).dropna()
    rows_t5.append({
        "Ratio": lbl,
        "Median": round(col.median(), 2),
        "P25": round(col.quantile(0.25), 2),
        "P75": round(col.quantile(0.75), 2),
        "Mean": round(col.mean(), 2),
        "SD": round(col.std(), 2)
    })
pd.DataFrame(rows_t5).to_csv(out("table5_ratios.csv"), index=False)

# ─── TABLE 6: Going-Concern Valuation ───────────────────────────────────────
print("Table 6: Going-Concern Valuation...")
viable_mask = da["net_income_adjusted"] > 0
viable_a = da[viable_mask]
viable_d = d5[viable_mask.values]

# V3: book value = opening net assets; FCF = adjusted profit (no dep add-back)
book_value = (viable_d["cash_initial"].values + viable_d["inventory_init"].values
              + viable_d["receivables_init"].values + viable_d["tool_value"].values
              - viable_d["payables_init"].values - viable_d["debt_amount"].values)
fcf_annual = viable_a["net_income_adjusted"].values * WEEKS_YEAR

gc_rows = []
for r_ in [0.10, 0.15, 0.18, 0.20, 0.25, 0.30]:
    pv = fcf_annual / r_
    q = np.where(book_value > 0, pv / book_value, np.nan)
    gc_rows.append({"Hurdle rate": f"{int(r_*100)}%",
        "Median Q": round(np.nanmedian(q), 2),
        "Q IQR": f"[{np.nanpercentile(q[~np.isnan(q)],25):.1f}, {np.nanpercentile(q[~np.isnan(q)],75):.1f}]",
        "N": int(np.sum(~np.isnan(q)))})
for mult in [1.0, 1.5, 2.0, 3.0]:
    pv = fcf_annual / DISC_RATE
    q = np.where(book_value > 0, pv / (book_value*mult), np.nan)
    gc_rows.append({"Hurdle rate": f"book x{mult}",
        "Median Q": round(np.nanmedian(q), 2), "Q IQR": "", "N": int(np.sum(~np.isnan(q)))})
pd.DataFrame(gc_rows).to_csv(out("table6_valuation.csv"), index=False)
q18 = np.where(book_value>0, (fcf_annual/DISC_RATE)/book_value, np.nan)
print(f"  Median Tobin's Q (r=18%, corrected): {np.nanmedian(q18):.2f}  N={int(np.sum(~np.isnan(q18)))}")

# ─── TABLE 7: Survey vs Diary Comparison ────────────────────────────────────
print("Table 7: Survey vs Diary Comparison...")
comparison_rows = [
    {"Measure": "Gross income (Bs/wk)", 
     "EH Survey": round(df["yi_tot"].median()/52, 0),
     "Diary Conv.": round(df["total_revenue_weekly"].median(), 0),
     "Diary Adj.": round(df["total_revenue_weekly"].median(), 0)},
    {"Measure": "Net income (Bs/wk)",
     "EH Survey": round(df["yi_net"].median()/52, 0),
     "Diary Conv.": round(df["net_income_conventional"].median(), 0),
     "Diary Adj.": round(df["net_income_adjusted"].median(), 0)},
    {"Measure": "Loss-making (%)",
     "EH Survey": f"{100*(df['yi_net']<=0).mean():.1f}%",
     "Diary Conv.": f"{100*(df['net_income_conventional']<=0).mean():.1f}%",
     "Diary Adj.": f"{100*df['loss_making'].mean():.1f}%"},
    {"Measure": "Active debt (%)",
     "EH Survey": f"{100*(df['ci6']>0).mean():.1f}%",
     "Diary Conv.": f"{100*df['has_debt'].mean():.1f}%",
     "Diary Adj.": f"{100*df['has_debt'].mean():.1f}%"},
    {"Measure": "Median accounting gap (Bs/wk)",
     "EH Survey": "—",
     "Diary Conv.": "—",
     "Diary Adj.": round(df["accounting_gap_weekly"].median(), 0)}
]
pd.DataFrame(comparison_rows).to_csv(out("table7_eh_vs_diary.csv"), index=False)

# ─── TABLE 8: Survey Scorecard ──────────────────────────────────────────────
print("Table 8: Survey Scorecard...")
scorecard = [
    {"Category": "Revenue", "Concept": "Gross cash revenue", "EH Status": "Partial", "Critical": ""},
    {"Category": "Revenue", "Concept": "Credit sales", "EH Status": "No", "Critical": ""},
    {"Category": "Revenue", "Concept": "Own-consumption at cost", "EH Status": "No", "Critical": ""},
    {"Category": "Costs", "Concept": "Raw materials / COGS", "EH Status": "Partial", "Critical": ""},
    {"Category": "Costs", "Concept": "Transport", "EH Status": "Partial", "Critical": ""},
    {"Category": "Costs", "Concept": "Rent", "EH Status": "Partial", "Critical": ""},
    {"Category": "Costs", "Concept": "Utilities", "EH Status": "Partial", "Critical": ""},
    {"Category": "Costs", "Concept": "Hired labour wages", "EH Status": "Partial", "Critical": ""},
    {"Category": "Costs", "Concept": "Owner labour (imputed)", "EH Status": "No", "Critical": "●"},
    {"Category": "Costs", "Concept": "Family labour (imputed)", "EH Status": "No", "Critical": "●"},
    {"Category": "Costs", "Concept": "Depreciation", "EH Status": "No", "Critical": "●"},
    {"Category": "Costs", "Concept": "Informal interest", "EH Status": "No", "Critical": "●"},
    {"Category": "Balance Sheet", "Concept": "Cash on hand", "EH Status": "No", "Critical": ""},
    {"Category": "Balance Sheet", "Concept": "Inventory", "EH Status": "No", "Critical": ""},
    {"Category": "Balance Sheet", "Concept": "Accounts receivable", "EH Status": "No", "Critical": "●"},
    {"Category": "Balance Sheet", "Concept": "Productive assets", "EH Status": "Partial", "Critical": ""},
    {"Category": "Balance Sheet", "Concept": "Accumulated depreciation", "EH Status": "No", "Critical": "●"},
    {"Category": "Balance Sheet", "Concept": "Accounts payable", "EH Status": "No", "Critical": "●"},
    {"Category": "Balance Sheet", "Concept": "Informal debt principal", "EH Status": "No", "Critical": "●"},
    {"Category": "Cash Flow", "Concept": "Operating cash flow", "EH Status": "No", "Critical": ""},
    {"Category": "Cash Flow", "Concept": "Household-firm transfers", "EH Status": "No", "Critical": "●"},
    {"Category": "Cash Flow", "Concept": "Loan disbursements", "EH Status": "No", "Critical": ""},
]
pd.DataFrame(scorecard).to_csv(out("table8_scorecard.csv"), index=False)
n_absent = sum(1 for r in scorecard if r["EH Status"] == "No")
n_critical = sum(1 for r in scorecard if r["Critical"] == "●")
print(f"  Absent from EH: {n_absent}/22, Critical: {n_critical}")

# ─── TABLE 9: Robustness to Shadow Wage ─────────────────────────────────────
print("Table 9: Robustness to Shadow Wage...")
# V3: identity-preserving recomputation (matches published Table, incl. Panel B)
imp_hours = df["imputed_labor_weekly"] / SHADOW_WAGE
other_comp = ((df["net_income_conventional"] - df["net_income_adjusted"])
              - df["imputed_labor_weekly"] - df["depreciation_weekly"])
def loss_share_at(hourly):
    pa = (df["net_income_conventional"] - hourly*imp_hours
          - df["depreciation_weekly"] - other_comp)
    return 100 * (pa <= 0).mean()
robustness_rows = []
for pct in [40,50,60,70,80,90,100,110,120,130,150,192]:
    h = 1049.0/40 if pct == 192 else SHADOW_WAGE*pct/100  # 192% = EH median formal wage
    robustness_rows.append({"Specification": f"{pct}% of minimum",
        "Shadow wage (Bs/hr)": round(h,2), "Loss-making (%)": round(loss_share_at(h),1)})
try:
    swf = pd.read_csv(os.path.join(DATA_DIR, "diary_shadow_wages.csv"))
    fm = d5.merge(swf, on="folio", how="left")
    for col, lab in [("wstar_weekly_s3","Mincer all-salaried (floored)"),
                     ("wstar_weekly_s1","Mincer formal (floored)")]:
        wi = np.maximum(fm[col], 546.0); hr = wi/40.0
        pa = (df["net_income_conventional"].values - hr.values*imp_hours.values
              - df["depreciation_weekly"].values - other_comp.values)
        robustness_rows.append({"Specification": lab, "Shadow wage (Bs/hr)": "person-specific",
            "Loss-making (%)": round(100*(pa<=0).mean(),1)})
except FileNotFoundError:
    print("  [diary_shadow_wages.csv not found: Panel B Mincer rows skipped]")
for k in [0.7, 0.5]:
    pa = df["net_income_adjusted"] + SHADOW_WAGE*df["family_hours_weekly"].fillna(0)*(1-k)
    robustness_rows.append({"Specification": f"Family hours at {int(k*100)}% of owner wage",
        "Shadow wage (Bs/hr)": SHADOW_WAGE, "Loss-making (%)": round(100*(pa<=0).mean(),1)})
pd.DataFrame(robustness_rows).to_csv(out("table9_robustness.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEXT FIGURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("MAIN TEXT FIGURES")
print("─"*70)

# ─── FIGURE 1: Income Distribution Shift ────────────────────────────────────
print("\nFigure 1: Income Distribution Shift...")
fig, ax = plt.subplots(figsize=(8, 5))

# KDE plots
from scipy.stats import gaussian_kde
x_range = np.linspace(-2000, 4000, 500)

y_eh = df["yi_net"] / 52
y_conv = df["net_income_conventional"]
y_adj = df["net_income_adjusted"]

for y, label, color, ls in [
    (y_eh, "EH Survey (recalled)", COLOR_EH, "-"),
    (y_conv, "Diary conventional", COLOR_DIARY_CONV, "--"),
    (y_adj, "Diary adjusted", COLOR_DIARY_ADJ, "-")
]:
    kde = gaussian_kde(y.clip(-2000, 4000))
    ax.plot(x_range, kde(x_range), label=label, color=color, ls=ls, lw=2)

ax.axvline(0, color="black", lw=1, ls=":", alpha=0.7)
ax.axvline(SHADOW_WAGE * 40, color="gray", lw=1, ls="--", alpha=0.5, 
           label=f"w* = {SHADOW_WAGE*40:.0f} Bs/wk")
ax.set_xlabel("Net income (Bs/week)")
ax.set_ylabel("Density")
ax.set_title("Figure 1: The Profit Reversal — Distribution Shift from Survey to Diary")
ax.legend(loc="upper right")
ax.set_xlim(-2000, 4000)
plt.tight_layout()
plt.savefig(out("fig1_distribution_shift.pdf"), bbox_inches="tight")
plt.savefig(out("fig1_distribution_shift.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE 2: Waterfall by Sector ──────────────────────────────────────────
print("Figure 2: Waterfall by Sector...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, s in enumerate(SECTOR_ORDER):
    ax = axes[idx]
    sub = da[da["sector"] == s]
    
    # Median values
    eh_val = (d5[d5["sector"]==s]["yi_net"] / 52).median()
    conv_val = sub["net_income_conventional"].median()
    labor_adj = sub["labor_imputed"].median()
    depr_adj = sub["depreciation_imputed"].median()
    adj_val = sub["net_income_adjusted"].median()
    
    # Waterfall
    bars = ["EH\nSurvey", "Diary\n(conv)", "−Labour", "−Deprec.", "Diary\n(adj)"]
    vals = [eh_val, conv_val, -labor_adj, -depr_adj, adj_val]
    colors = [COLOR_EH, COLOR_DIARY_CONV, "#e74c3c", "#e74c3c", 
              "#27ae60" if adj_val > 0 else "#c0392b"]
    
    x = np.arange(len(bars))
    ax.bar(x, vals, color=colors, edgecolor="white", width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bars, fontsize=8)
    ax.set_ylabel("Bs/week")
    
    loss_pct = 100 * (sub["net_income_adjusted"] <= 0).mean()
    ax.set_title(f"{SECTOR_LABEL[s]}\n({loss_pct:.0f}% loss-making)", fontsize=10)
    
    # Add value labels
    for i, v in enumerate(vals):
        ypos = v + 30 if v >= 0 else v - 60
        ax.text(i, ypos, f"{v:.0f}", ha="center", fontsize=7)

plt.suptitle("Figure 2: From Survey to Adjusted Income — The Accounting Gap by Sector", 
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(out("fig2_waterfall_sector.pdf"), bbox_inches="tight")
plt.savefig(out("fig2_waterfall_sector.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE 3: Firm Typology Scatter ────────────────────────────────────────
print("Figure 3: Firm Typology Scatter...")
fig, ax = plt.subplots(figsize=(9, 7))

for typ, color in COLORS_TYPE.items():
    sub = da[da["typology_ledger"] == typ]
    label = typ.replace("_", " ").replace("Atrapada deuda", "Debt-trapped")
    label = label.replace("Precaria", "Precarious").replace("Zombie", "Critical")
    ax.scatter(sub["net_income_conventional"].clip(-2000, 5000),
               sub["net_income_adjusted"].clip(-2000, 4000),
               color=color, alpha=0.6, s=30, label=f"{label} (n={len(sub)})")

# Reference lines
ax.axhline(0, color="gray", lw=1.5, ls="--", label="π_adj = 0")
ax.axvline(SHADOW_WAGE * 40, color="black", lw=1, ls=":", alpha=0.7, 
           label=f"w* = {SHADOW_WAGE*40:.0f}")
ax.plot([-2000, 5000], [-2000, 5000], color="gray", lw=0.5, ls=":", alpha=0.5)

ax.set_xlabel("Conventional net income (Bs/week)")
ax.set_ylabel("Adjusted net income (Bs/week)")
ax.set_title("Figure 3: Firm Typology — Conventional vs Adjusted Net Income")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(-2000, 5000)
ax.set_ylim(-2000, 4000)

# Quadrant labels
ax.text(3500, 2500, "I\nViable", fontsize=12, ha="center", color=COLORS_TYPE["I_Viable"], alpha=0.7)
ax.text(3500, -1200, "II\nPrecarious", fontsize=12, ha="center", color=COLORS_TYPE["II_Precaria"], alpha=0.7)

plt.tight_layout()
plt.savefig(out("fig3_typology.pdf"), bbox_inches="tight")
plt.savefig(out("fig3_typology.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE 4: Q Distribution ───────────────────────────────────────────────
print("Figure 4: Tobin's Q Distribution...")
fig, ax = plt.subplots(figsize=(8, 5))

# Compute Q for viable firms
viable_mask = da["net_income_adjusted"] > 0
vd = d5.loc[viable_mask.values]
bv = (vd["cash_initial"].values + vd["inventory_init"].values + vd["receivables_init"].values
      + vd["tool_value"].values - vd["payables_init"].values - vd["debt_amount"].values)
fcf = da.loc[viable_mask, "net_income_adjusted"].values * 52
pv = fcf / DISC_RATE
q_vals = np.where(bv > 0, pv / bv, np.nan)
q_vals = q_vals[~np.isnan(q_vals)]
q_vals = q_vals[q_vals < 50]  # Clip outliers for display

ax.hist(q_vals, bins=30, color=COLORS_TYPE["I_Viable"], edgecolor="white", alpha=0.8)
ax.axvline(np.median(q_vals), color="#c0392b", lw=2, ls="--", 
           label=f"Median Q = {np.median(q_vals):.1f}")
ax.axvline(1, color="black", lw=1, ls=":", label="Q = 1 (book value)")

ax.set_xlabel("Tobin's Q (Going-concern PV / Book value)")
ax.set_ylabel("Number of firms")
ax.set_title(f"Figure 4: Distribution of Tobin's Q for Viable Firms (N={len(q_vals)})")
ax.legend()
plt.tight_layout()
plt.savefig(out("fig4_tobins_q.pdf"), bbox_inches="tight")
plt.savefig(out("fig4_tobins_q.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE 5: Accounting Gap Components ────────────────────────────────────
print("Figure 5: Accounting Gap Components...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Gap distribution
ax = axes[0]
gap = df["accounting_gap_weekly"].clip(-500, 3000)
ax.hist(gap, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(gap.median(), color="#c0392b", lw=2, ls="--", 
           label=f"Median = {gap.median():.0f} Bs/wk")
ax.axvline(0, color="black", lw=1, ls=":")
ax.set_xlabel("Accounting gap (Bs/week)")
ax.set_ylabel("Number of firms")
ax.set_title("(a) Gap distribution")
ax.legend()

# (b) Components
ax = axes[1]
components = {
    "Labour\nimputation": df["imputed_labor_weekly"].median(),
    "Depreciation": df["depreciation_weekly"].median(),
    "Interest": da["interest_paid"].median()
}
bars = ax.bar(list(components.keys()), list(components.values()),
              color=["#e74c3c", "#f39c12", "#9b59b6"], edgecolor="white")
ax.set_ylabel("Bs/week (median)")
ax.set_title("(b) Gap components")
for bar, val in zip(bars, components.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
            f"{val:.0f}", ha="center", fontsize=9)

# (c) Direction by sector
ax = axes[2]
sectors = [SECTOR_LABEL[s] for s in SECTOR_ORDER]
overstate_pct = [(df[df["sector"]==s]["accounting_gap_weekly"] > 0).mean() * 100 
                 for s in SECTOR_ORDER]
colors = ["#e74c3c" if p > 50 else "#3498db" for p in overstate_pct]
ax.barh(sectors, overstate_pct, color=colors, edgecolor="white")
ax.axvline(50, color="black", lw=1, ls="--")
ax.set_xlabel("% of firms where EH overstates")
ax.set_title("(c) Overstatement by sector")
ax.set_xlim(0, 100)

plt.suptitle("Figure 5: The Accounting Gap — Magnitude, Components, and Direction", 
             fontweight="bold")
plt.tight_layout()
plt.savefig(out("fig5_gap_decomposition.pdf"), bbox_inches="tight")
plt.savefig(out("fig5_gap_decomposition.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE 6: Debt Detection Failure ───────────────────────────────────────
print("Figure 6: Debt Detection Failure...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Detection rate
ax = axes[0]
eh_debt = (df["ci6"] > 0).sum()
diary_debt = df["has_debt"].sum()
ax.bar(["EH declares\ninterest (ci6>0)", "Diary detects\nactive debt"],
       [100*eh_debt/len(df), 100*diary_debt/len(df)],
       color=[COLOR_EH, COLOR_DIARY_ADJ], edgecolor="white")
ax.set_ylabel("% of firms")
ax.set_title(f"(a) Detection rate: {100*eh_debt/diary_debt:.0f}% of true prevalence")
for i, v in enumerate([eh_debt, diary_debt]):
    ax.text(i, 100*v/len(df) + 2, f"{100*v/len(df):.1f}%", ha="center", fontsize=10)

# (b) Debt source composition
ax = axes[1]
debt_types = df[df["has_debt"]==1]["debt_type"].value_counts()
colors_debt = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#2ecc71"]
ax.pie(debt_types.values, labels=debt_types.index, colors=colors_debt[:len(debt_types)],
       autopct="%1.0f%%", startangle=90)
ax.set_title(f"(b) Debt sources (N={diary_debt})")

# (c) Interest rates by source
ax = axes[2]
debtors = df[df["has_debt"]==1].copy()
debtors["ear"] = ((1 + debtors["monthly_interest_rate"])**12 - 1) * 100
debt_rates = debtors.groupby("debt_type")["ear"].median().sort_values(ascending=False)
ax.barh(debt_rates.index, debt_rates.values, color="#e74c3c", edgecolor="white")
ax.set_xlabel("Effective Annual Rate (%)")
ax.set_title("(c) Median interest rate by source")
ax.axvline(25, color="gray", lw=1, ls="--", label="Formal rate (~25%)")
ax.legend(fontsize=8)

plt.suptitle("Figure 6: The Hidden Debt Economy — What Surveys Miss", fontweight="bold")
plt.tight_layout()
plt.savefig(out("fig6_debt_detection.pdf"), bbox_inches="tight")
plt.savefig(out("fig6_debt_detection.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE 7: Robustness to Shadow Wage ────────────────────────────────────
print("Figure 7: Robustness to Shadow Wage...")
fig, ax = plt.subplots(figsize=(8, 5))

pcts = np.arange(40, 201, 5)
loss_pcts = [loss_share_at(SHADOW_WAGE*p/100) for p in pcts]
ax.plot(pcts, loss_pcts, color="#2980b9", lw=2)
ax.axvline(100, color="#c0392b", lw=1.5, ls="--")
ax.text(101, 12, "SMN\n(baseline 42.8%)", fontsize=8, color="#c0392b")
ax.axvline(192, color="gray", ls="--", lw=1.2)
ax.text(160, 12, "median formal\nwage (71.0%)", fontsize=8, color="gray")
for x, y, lab in [(126, 55.6, "Mincer all-salaried (55.6%)"),
                  (156, 62.8, "Mincer formal (62.8%)")]:
    ax.plot(x, y, "o", color="#c0392b", ms=6)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8,-10), fontsize=8)
ax.set_xlabel("Shadow wage (% of statutory minimum)")
ax.set_ylabel("Loss-making share (%)")
ax.set_title("Figure 7: Sensitivity of Loss-Making Share to Shadow Wage Assumption")
ax.set_xlim(40, 200)
ax.set_ylim(0, 80)
plt.tight_layout()
plt.savefig(out("fig7_robustness.pdf"), bbox_inches="tight")
plt.savefig(out("fig7_robustness.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE 8: Transaction Complexity ───────────────────────────────────────
print("Figure 8: Transaction Complexity...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Transactions by module
ax = axes[0]
active_dt = dt[dt["module"] != "DIA_INACTIVO"]
module_counts = active_dt.groupby("module").size().sort_values(ascending=True)
module_labels = [m.replace("_", "\n").replace("M1A", "Sales\n(cash)").replace("M1B", "Collections")
                 .replace("M2A", "Purchases").replace("M2B", "Expenses").replace("M3", "Wages")
                 .replace("M4A", "Debt\npayments").replace("M4B", "New\nloans")
                 .replace("M5", "Assets").replace("M6", "Own\nconsump.").replace("M7", "HH\ntransfers")
                 for m in module_counts.index]
ax.barh(range(len(module_counts)), module_counts.values, color="#3498db", edgecolor="white")
ax.set_yticks(range(len(module_counts)))
ax.set_yticklabels(module_labels, fontsize=8)
ax.set_xlabel("Number of transactions")
ax.set_title("(a) Transactions by module")

# (b) Intra-week revenue pattern
ax = axes[1]
sales_dt = dt[dt["module"].isin(["M1A_VENTAS", "M1B_COBROS"])]
daily = sales_dt.groupby(["firm_id", "day_name"])["amount_bs"].sum().reset_index()
day_order = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily_med = daily.groupby("day_name")["amount_bs"].median().reindex(day_order)

ax.bar(range(7), daily_med.values, color="#3498db", edgecolor="white")
ax.axhline(daily_med.mean(), color="#c0392b", lw=1.5, ls="--", label="Daily mean")
ax.set_xticks(range(7))
ax.set_xticklabels(day_labels)
ax.set_ylabel("Median daily revenue (Bs)")
ax.set_title("(b) Intra-week revenue pattern")
ax.legend()

# Highlight Friday-Saturday
ax.bar([4, 5], daily_med.values[4:6], color="#e74c3c", edgecolor="white")
fri_sat_share = daily_med.values[4:6].sum() / daily_med.sum() * 100
ax.text(4.5, daily_med.max() + 20, f"Fri-Sat:\n{fri_sat_share:.0f}%", ha="center", fontsize=9)

plt.suptitle(f"Figure 8: Transaction Complexity — {len(active_dt):,} Records from {len(df)} Firms", 
             fontweight="bold")
plt.tight_layout()
plt.savefig(out("fig8_transactions.pdf"), bbox_inches="tight")
plt.savefig(out("fig8_transactions.png"), bbox_inches="tight", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# HETEROGENEITY ANALYSIS (Section 9)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("HETEROGENEITY ANALYSIS")
print("─"*70)

# ─── TABLE 10: Heterogeneity Summary ────────────────────────────────────────
print("\nTable 10: Heterogeneity Summary...")
het_rows = []

# By gender
for gender, label in [(1, "Male"), (0, "Female")]:
    sub = df[df["hombre"] == gender]
    sub_a = da[df["hombre"].values == gender]
    het_rows.append({
        "Dimension": "Gender",
        "Category": label,
        "N": len(sub),
        "Loss-making (%)": round(100 * (sub_a["net_income_adjusted"] <= 0).mean(), 1),
        "Median gap (Bs/wk)": round(sub["accounting_gap_weekly"].median(), 0),
        "Median hourly return": round(sub_a["hourly_return_adj"].median(), 2),
        "Has debt (%)": round(100 * sub["has_debt"].mean(), 1)
    })

# By sector
for s in SECTOR_ORDER:
    sub = df[df["sector"] == s]
    sub_a = da[da["sector"] == s]
    het_rows.append({
        "Dimension": "Sector",
        "Category": SECTOR_LABEL[s],
        "N": len(sub),
        "Loss-making (%)": round(100 * (sub_a["net_income_adjusted"] <= 0).mean(), 1),
        "Median gap (Bs/wk)": round(sub["accounting_gap_weekly"].median(), 0),
        "Median hourly return": round(sub_a["hourly_return_adj"].median(), 2),
        "Has debt (%)": round(100 * sub["has_debt"].mean(), 1)
    })

# By debt status
for debt, label in [(0, "No debt"), (1, "Has debt")]:
    sub = df[df["has_debt"] == debt]
    sub_a = da[df["has_debt"].values == debt]
    het_rows.append({
        "Dimension": "Debt",
        "Category": label,
        "N": len(sub),
        "Loss-making (%)": round(100 * (sub_a["net_income_adjusted"] <= 0).mean(), 1),
        "Median gap (Bs/wk)": round(sub["accounting_gap_weekly"].median(), 0),
        "Median hourly return": round(sub_a["hourly_return_adj"].median(), 2),
        "Has debt (%)": round(100 * sub["has_debt"].mean(), 1)
    })

pd.DataFrame(het_rows).to_csv(out("table10_heterogeneity.csv"), index=False)

# ─── TABLE 11: Gender Regression ────────────────────────────────────────────
print("Table 11: Gender Regression...")
# Prepare data
reg_df = df[["female", "poor", "has_debt", "tothrs", "tool_value", 
             "years_operating", "sector", "hourly_return_adj", 
             "accounting_gap_weekly", "net_income_adjusted"]].copy()
reg_df["log_hours"] = np.log(reg_df["tothrs"].clip(lower=1))
reg_df["log_assets"] = np.log(reg_df["tool_value"].clip(lower=1))
reg_df["log_age"] = np.log(reg_df["years_operating"].clip(lower=1))
reg_df = reg_df.dropna()

# Create sector dummies
sector_dummies = pd.get_dummies(reg_df["sector"], drop_first=True)

# Model 1: Female + sector FE only
X1 = np.column_stack([np.ones(len(reg_df)), reg_df["female"], sector_dummies.values])
y1 = reg_df["hourly_return_adj"].clip(-20, 50).values
c1, s1, t1, r1, n1 = ols_hc3(y1, X1)

# Model 2: Full controls
X2 = np.column_stack([np.ones(len(reg_df)), 
                      reg_df[["female", "poor", "has_debt", "log_hours", "log_assets", "log_age"]].values,
                      sector_dummies.values])
c2, s2, t2, r2, n2 = ols_hc3(y1, X2)

# Model 3: Gap as outcome
y3 = reg_df["accounting_gap_weekly"].clip(-500, 3000).values
c3, s3, t3, r3, n3 = ols_hc3(y3, X2)

# V3 Model 4: Non-labour gap (gap minus w*h line)
nonlab = (df["accounting_gap_weekly"] - df["imputed_labor_weekly"]).clip(-500, 3000)
y4 = nonlab.loc[reg_df.index].values
c4, s4, t4, r4, n4 = ols_hc3(y4, X2)

gender_reg = pd.DataFrame({
    "Variable": ["Female", "Poor", "Has debt", "Log hours", "Log assets", "Log age", "Sector FE", "R²", "N"],
    "(1) Hourly return": [f"{c1[1]:.2f}{stars(t1[1])} ({s1[1]:.2f})", "—", "—", "—", "—", "—", "Yes", f"{r1:.3f}", n1],
    "(2) Hourly return": [f"{c2[1]:.2f}{stars(t2[1])} ({s2[1]:.2f})", f"{c2[2]:.2f}{stars(t2[2])} ({s2[2]:.2f})",
                         f"{c2[3]:.2f}{stars(t2[3])} ({s2[3]:.2f})", f"{c2[4]:.2f}{stars(t2[4])} ({s2[4]:.2f})",
                         f"{c2[5]:.2f}{stars(t2[5])} ({s2[5]:.2f})", f"{c2[6]:.2f}{stars(t2[6])} ({s2[6]:.2f})",
                         "Yes", f"{r2:.3f}", n2],
    "(3) Imputation gap": [f"{c3[1]:.0f}{stars(t3[1])} ({s3[1]:.0f})", f"{c3[2]:.0f}{stars(t3[2])} ({s3[2]:.0f})",
                          f"{c3[3]:.0f}{stars(t3[3])} ({s3[3]:.0f})", f"{c3[4]:.0f}{stars(t3[4])} ({s3[4]:.0f})",
                          f"{c3[5]:.0f}{stars(t3[5])} ({s3[5]:.0f})", f"{c3[6]:.0f}{stars(t3[6])} ({s3[6]:.0f})",
                          "Yes", f"{r3:.3f}", n3],
    "(4) Non-labour gap": [f"{c4[1]:.0f}{stars(t4[1])} ({s4[1]:.0f})", f"{c4[2]:.0f}{stars(t4[2])} ({s4[2]:.0f})",
                          f"{c4[3]:.0f}{stars(t4[3])} ({s4[3]:.0f})", f"{c4[4]:.0f}{stars(t4[4])} ({s4[4]:.0f})",
                          f"{c4[5]:.0f}{stars(t4[5])} ({s4[5]:.0f})", f"{c4[6]:.0f}{stars(t4[6])} ({s4[6]:.0f})",
                          "Yes", f"{r4:.3f}", n4]
})
gender_reg.to_csv(out("table11_gender_reg.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════
# APPENDIX TABLES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("APPENDIX TABLES")
print("─"*70)

# ─── TABLE A1: Sector Profiles ──────────────────────────────────────────────
print("\nTable A1: Sector Profiles...")
sector_profile = []
for s in SECTOR_ORDER:
    sub_d = df[df["sector"] == s]
    sub_a = da[da["sector"] == s]
    
    # Compute Q for viable firms in this sector
    viable_mask = sub_a["net_income_adjusted"] > 0
    if viable_mask.sum() > 0:
        bv = sub_d.loc[viable_mask.values, "tool_value"].values
        fcf = sub_a.loc[viable_mask, "net_income_adjusted"].values * 52
        q_sector = np.nanmedian(np.where(bv > 0, (fcf/DISC_RATE)/bv, np.nan))
    else:
        q_sector = np.nan
    
    sector_profile.append({
        "Sector": SECTOR_LABEL[s],
        "N": len(sub_d),
        "Female (%)": round(100 * sub_d["female"].mean(), 1),
        "Median hours/wk": round(sub_d["tothrs"].median(), 0),
        "Median revenue (Bs/wk)": round(sub_a["total_revenue_weekly"].median(), 0),
        "Median adj. income (Bs/wk)": round(sub_a["net_income_adjusted"].median(), 0),
        "Loss-making (%)": round(100 * (sub_a["net_income_adjusted"] <= 0).mean(), 1),
        "Has debt (%)": round(100 * sub_d["has_debt"].mean(), 1),
        "Median gap (Bs/wk)": round(sub_d["accounting_gap_weekly"].median(), 0),
        "Median Q": round(q_sector, 1) if not np.isnan(q_sector) else "—"
    })
pd.DataFrame(sector_profile).to_csv(out("tableA1_sector_profiles.csv"), index=False)

# ─── TABLE A2: Reliability Ratios ───────────────────────────────────────────
print("Table A2: Reliability Ratios by Sector × Gender...")
reliability = []
for s in SECTOR_ORDER:
    for gender, g_label in [(1, "Male"), (0, "Female")]:
        sub = df[(df["sector"] == s) & (df["hombre"] == gender)]
        sub_a = da[(da["sector"] == s) & (df["hombre"].values == gender)]
        
        if len(sub) < 10:
            continue
        
        # λ = Var(π_adj) / [Var(π_adj) + Var(ε)]
        var_pi = sub_a["net_income_adjusted"].var()
        var_eps = sub["accounting_gap_weekly"].var()
        lambda_val = var_pi / (var_pi + var_eps) if (var_pi + var_eps) > 0 else np.nan
        
        reliability.append({
            "Sector": SECTOR_LABEL[s],
            "Gender": g_label,
            "N": len(sub),
            "Var(π_adj)": round(var_pi, 0),
            "Var(gap)": round(var_eps, 0),
            "λ (reliability)": round(lambda_val, 3)
        })
pd.DataFrame(reliability).to_csv(out("tableA2_reliability.csv"), index=False)

# ─── TABLE A3: Debt Details ─────────────────────────────────────────────────
print("Table A3: Debt Details by Source...")
debtors = df[df["has_debt"] == 1].copy()
debtors["ear"] = ((1 + debtors["monthly_interest_rate"])**12 - 1) * 100

debt_details = []
for dtype in debtors["debt_type"].unique():
    sub = debtors[debtors["debt_type"] == dtype]
    debt_details.append({
        "Source": dtype,
        "N": len(sub),
        "Pct of debtors": round(100 * len(sub) / len(debtors), 1),
        "Median principal (Bs)": round(sub["debt_amount"].median(), 0),
        "Median monthly rate (%)": round(100 * sub["monthly_interest_rate"].median(), 1),
        "Median EAR (%)": round(sub["ear"].median(), 1),
        "Median weekly payment (Bs)": round(sub["weekly_installment"].median(), 0)
    })
pd.DataFrame(debt_details).to_csv(out("tableA3_debt_details.csv"), index=False)

# ─── TABLE A4: Loss-Making Share with CIs ───────────────────────────────────
print("Table A4: Loss-Making Share with Confidence Intervals...")
ci_table = []
for s in SECTOR_ORDER:
    sub_a = da[da["sector"] == s]
    n = len(sub_a)
    p = (sub_a["net_income_adjusted"] <= 0).mean()
    lo, hi = wilson_ci(p, n)
    ci_table.append({
        "Sector": SECTOR_LABEL[s],
        "N": n,
        "Loss-making (%)": round(100 * p, 1),
        "95% CI lower": round(100 * lo, 1),
        "95% CI upper": round(100 * hi, 1)
    })

# Total
p_total = (da["net_income_adjusted"] <= 0).mean()
lo, hi = wilson_ci(p_total, len(da))
ci_table.append({
    "Sector": "Total",
    "N": len(da),
    "Loss-making (%)": round(100 * p_total, 1),
    "95% CI lower": round(100 * lo, 1),
    "95% CI upper": round(100 * hi, 1)
})
pd.DataFrame(ci_table).to_csv(out("tableA4_loss_ci.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════
# APPENDIX FIGURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("APPENDIX FIGURES")
print("─"*70)

# ─── FIGURE B1: Income by Sector (6-panel) ──────────────────────────────────
print("\nFigure B1: Income by Sector (6-panel)...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, s in enumerate(SECTOR_ORDER):
    ax = axes[idx]
    sub = df[df["sector"] == s]
    sub_a = da[da["sector"] == s]
    
    y_eh = sub["yi_net"] / 52
    y_adj = sub_a["net_income_adjusted"]
    
    ax.hist(y_eh.clip(-1000, 3000), bins=25, alpha=0.5, color=COLOR_EH, 
            label=f"EH Survey (med={y_eh.median():.0f})", edgecolor="white")
    ax.hist(y_adj.clip(-1000, 3000), bins=25, alpha=0.5, color=COLOR_DIARY_ADJ,
            label=f"Diary adj. (med={y_adj.median():.0f})", edgecolor="white")
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.set_xlabel("Net income (Bs/week)")
    ax.set_ylabel("Count")
    ax.set_title(f"{SECTOR_LABEL[s]} (N={len(sub)})")
    ax.legend(fontsize=7)

plt.suptitle("Figure B1: Income Distribution by Sector — EH Survey vs Diary Adjusted", 
             fontweight="bold")
plt.tight_layout()
plt.savefig(out("figB1_income_by_sector.pdf"), bbox_inches="tight")
plt.savefig(out("figB1_income_by_sector.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B2: Gender Heterogeneity ────────────────────────────────────────
print("Figure B2: Gender Heterogeneity...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Hourly return by sector × gender
ax = axes[0]
x = np.arange(len(SECTOR_ORDER))
width = 0.35

male_returns = [da[(da["sector"]==s) & (df["hombre"].values==1)]["hourly_return_adj"].median() 
                for s in SECTOR_ORDER]
female_returns = [da[(da["sector"]==s) & (df["hombre"].values==0)]["hourly_return_adj"].median() 
                  for s in SECTOR_ORDER]

ax.bar(x - width/2, male_returns, width, label="Male", color=COLOR_MALE, edgecolor="white")
ax.bar(x + width/2, female_returns, width, label="Female", color=COLOR_FEMALE, edgecolor="white")
ax.axhline(SHADOW_WAGE, color="black", lw=1, ls="--", label=f"Shadow wage ({SHADOW_WAGE})")
ax.axhline(0, color="gray", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([SECTOR_LABEL[s] for s in SECTOR_ORDER], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Median hourly return (Bs/hr)")
ax.set_title("(a) Hourly return by sector × gender")
ax.legend(fontsize=8)

# (b) Accounting gap by gender
ax = axes[1]
male_gap = df[df["hombre"]==1]["accounting_gap_weekly"]
female_gap = df[df["hombre"]==0]["accounting_gap_weekly"]

ax.boxplot([male_gap.clip(-500, 2500), female_gap.clip(-500, 2500)],
           labels=["Male", "Female"], patch_artist=True,
           boxprops=dict(facecolor=COLOR_MALE, alpha=0.6),
           medianprops=dict(color="black", lw=2))
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Accounting gap (Bs/week)")
ax.set_title(f"(b) Gap distribution: Male med={male_gap.median():.0f}, Female med={female_gap.median():.0f}")

plt.suptitle("Figure B2: Gender Heterogeneity in Returns and Measurement Error", fontweight="bold")
plt.tight_layout()
plt.savefig(out("figB2_gender.pdf"), bbox_inches="tight")
plt.savefig(out("figB2_gender.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B3: Debt Economy ────────────────────────────────────────────────
print("Figure B3: Debt Economy by Sector...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Debt prevalence by sector
ax = axes[0]
debt_by_sector = [100 * df[df["sector"]==s]["has_debt"].mean() for s in SECTOR_ORDER]
eh_debt_by_sector = [100 * (df[df["sector"]==s]["ci6"] > 0).mean() for s in SECTOR_ORDER]

x = np.arange(len(SECTOR_ORDER))
width = 0.35
ax.bar(x - width/2, eh_debt_by_sector, width, label="EH declares (ci6>0)", color=COLOR_EH)
ax.bar(x + width/2, debt_by_sector, width, label="Diary detects", color=COLOR_DIARY_ADJ)
ax.set_xticks(x)
ax.set_xticklabels([SECTOR_LABEL[s] for s in SECTOR_ORDER], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("% with active debt")
ax.set_title("(a) Debt detection by sector")
ax.legend()

# (b) Interest rates by sector (for debtors)
ax = axes[1]
debtors = df[df["has_debt"]==1].copy()
debtors["ear"] = ((1 + debtors["monthly_interest_rate"])**12 - 1) * 100
ear_by_sector = [debtors[debtors["sector"]==s]["ear"].median() if len(debtors[debtors["sector"]==s]) > 5 else np.nan 
                 for s in SECTOR_ORDER]
ax.bar(x, ear_by_sector, color="#e74c3c", edgecolor="white")
ax.axhline(25, color="gray", lw=1, ls="--", label="Formal rate (~25%)")
ax.set_xticks(x)
ax.set_xticklabels([SECTOR_LABEL[s] for s in SECTOR_ORDER], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Median EAR (%)")
ax.set_title("(b) Interest rates by sector (debtors only)")
ax.legend()

plt.suptitle("Figure B3: The Hidden Debt Economy by Sector", fontweight="bold")
plt.tight_layout()
plt.savefig(out("figB3_debt_by_sector.pdf"), bbox_inches="tight")
plt.savefig(out("figB3_debt_by_sector.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B4: Gap Decomposition by Sector ─────────────────────────────────
print("Figure B4: Gap Decomposition by Sector...")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(SECTOR_ORDER))
width = 0.6

labor_by_sector = [df[df["sector"]==s]["imputed_labor_weekly"].median() for s in SECTOR_ORDER]
depr_by_sector = [df[df["sector"]==s]["depreciation_weekly"].median() for s in SECTOR_ORDER]
int_by_sector = [da[da["sector"]==s]["interest_paid"].median() for s in SECTOR_ORDER]

ax.bar(x, labor_by_sector, width, label="Labour imputation", color="#e74c3c")
ax.bar(x, depr_by_sector, width, bottom=labor_by_sector, label="Depreciation", color="#f39c12")
ax.bar(x, int_by_sector, width, bottom=[l+d for l,d in zip(labor_by_sector, depr_by_sector)], 
       label="Interest", color="#9b59b6")

ax.set_xticks(x)
ax.set_xticklabels([SECTOR_LABEL[s] for s in SECTOR_ORDER], rotation=20, ha="right")
ax.set_ylabel("Bs/week (median)")
ax.set_title("Figure B4: Accounting Gap Components by Sector")
ax.legend()
plt.tight_layout()
plt.savefig(out("figB4_gap_by_sector.pdf"), bbox_inches="tight")
plt.savefig(out("figB4_gap_by_sector.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B5: Asset Composition ───────────────────────────────────────────
print("Figure B5: Asset Composition by Sector...")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(SECTOR_ORDER))
width = 0.6

cash = [df[df["sector"]==s]["cash_initial"].median() for s in SECTOR_ORDER]
inventory = [df[df["sector"]==s]["inventory_init"].median() for s in SECTOR_ORDER]
receivables = [df[df["sector"]==s]["receivables_init"].median() for s in SECTOR_ORDER]
tools = [df[df["sector"]==s]["tool_value"].median() for s in SECTOR_ORDER]

ax.bar(x, cash, width, label="Cash", color="#3498db")
ax.bar(x, inventory, width, bottom=cash, label="Inventory", color="#2ecc71")
ax.bar(x, receivables, width, bottom=[c+i for c,i in zip(cash, inventory)], label="Receivables", color="#f39c12")
ax.bar(x, tools, width, bottom=[c+i+r for c,i,r in zip(cash, inventory, receivables)], label="Tools/Equipment", color="#9b59b6")

ax.set_xticks(x)
ax.set_xticklabels([SECTOR_LABEL[s] for s in SECTOR_ORDER], rotation=20, ha="right")
ax.set_ylabel("Bs (median)")
ax.set_title("Figure B5: Asset Composition by Sector")
ax.legend()
plt.tight_layout()
plt.savefig(out("figB5_assets.pdf"), bbox_inches="tight")
plt.savefig(out("figB5_assets.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B6: Hours Distribution ──────────────────────────────────────────
print("Figure B6: Hours Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Owner hours by sector
ax = axes[0]
hours_data = [df[df["sector"]==s]["tothrs"].values for s in SECTOR_ORDER]
bp = ax.boxplot(hours_data, labels=[SECTOR_LABEL[s][:8] for s in SECTOR_ORDER],
                patch_artist=True, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp['boxes'], [COLORS_SECTOR[s] for s in SECTOR_ORDER]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(40, color="gray", lw=1, ls="--", label="40 hrs (full-time)")
ax.set_ylabel("Owner hours/week")
ax.set_title("(a) Owner hours by sector")
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

# (b) Family vs owner hours
ax = axes[1]
has_family = df["n_family_workers"] > 0
owner_hrs = df.loc[has_family, "tothrs"]
family_hrs = df.loc[has_family, "family_hours_weekly"]

ax.scatter(owner_hrs, family_hrs, alpha=0.5, color="#3498db", s=20)
ax.plot([0, 80], [0, 80], color="gray", ls="--", lw=1, label="1:1 line")
ax.set_xlabel("Owner hours/week")
ax.set_ylabel("Family hours/week")
ax.set_title(f"(b) Family vs owner hours (N={has_family.sum()} with family workers)")
ax.legend()

plt.suptitle("Figure B6: Labour Input Distribution", fontweight="bold")
plt.tight_layout()
plt.savefig(out("figB6_hours.pdf"), bbox_inches="tight")
plt.savefig(out("figB6_hours.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B7: Cash Flow vs Adjusted Income ────────────────────────────────
print("Figure B7: Cash Flow vs Adjusted Income...")
fig, ax = plt.subplots(figsize=(8, 7))

for typ, color in COLORS_TYPE.items():
    mask = da["typology_ledger"] == typ
    ax.scatter(da.loc[mask, "cf_operating"].clip(-500, 3000),
               da.loc[mask, "net_income_adjusted"].clip(-2000, 3000),
               color=color, alpha=0.6, s=25, label=typ.replace("_", " "))

ax.axhline(0, color="gray", lw=1, ls="--")
ax.axvline(0, color="gray", lw=1, ls="--")
ax.plot([-500, 3000], [-500, 3000], color="black", ls=":", lw=0.5, alpha=0.5)

ax.set_xlabel("Operating cash flow (Bs/week)")
ax.set_ylabel("Adjusted net income (Bs/week)")
ax.set_title("Figure B7: Cash Flow vs Adjusted Income\n(All loss-makers have positive CF)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(out("figB7_cf_vs_adj.pdf"), bbox_inches="tight")
plt.savefig(out("figB7_cf_vs_adj.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B8: Type Distribution by Sector ─────────────────────────────────
print("Figure B8: Type Distribution by Sector...")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(SECTOR_ORDER))
width = 0.6

type_shares = {}
for typ in ["I_Viable", "II_Precaria", "III_Atrapada_deuda", "IV_Zombie"]:
    type_shares[typ] = [100 * (da[da["sector"]==s]["typology_ledger"]==typ).mean() for s in SECTOR_ORDER]

bottom = np.zeros(len(SECTOR_ORDER))
for typ, color in COLORS_TYPE.items():
    label = typ.replace("_", " ").replace("Atrapada deuda", "Debt-trap").replace("Precaria", "Precarious")
    ax.bar(x, type_shares[typ], width, bottom=bottom, label=label, color=color)
    bottom += type_shares[typ]

ax.set_xticks(x)
ax.set_xticklabels([SECTOR_LABEL[s] for s in SECTOR_ORDER], rotation=20, ha="right")
ax.set_ylabel("Share (%)")
ax.set_title("Figure B8: Firm Type Distribution by Sector")
ax.legend(loc="upper right")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(out("figB8_types_by_sector.pdf"), bbox_inches="tight")
plt.savefig(out("figB8_types_by_sector.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B9: Revenue-to-EH Ratio ─────────────────────────────────────────
print("Figure B9: Revenue Ratio Distribution...")
fig, ax = plt.subplots(figsize=(8, 5))

ratio = df["diary_eh_revenue_ratio"].clip(0, 5)
ax.hist(ratio, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(ratio.median(), color="#c0392b", lw=2, ls="--", 
           label=f"Median = {ratio.median():.2f}×")
ax.axvline(1, color="black", lw=1, ls=":", label="Diary = EH")

pct_higher = 100 * (ratio > 1).mean()
ax.text(0.95, 0.95, f"{pct_higher:.0f}% diary > EH", transform=ax.transAxes, 
        ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="white"))

ax.set_xlabel("Diary revenue / EH gross income")
ax.set_ylabel("Number of firms")
ax.set_title("Figure B9: Income Under-declaration — Diary vs EH Revenue Ratio")
ax.legend()
plt.tight_layout()
plt.savefig(out("figB9_revenue_ratio.pdf"), bbox_inches="tight")
plt.savefig(out("figB9_revenue_ratio.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B10: Department Map ─────────────────────────────────────────────
print("Figure B10: Results by Department...")
fig, ax = plt.subplots(figsize=(10, 6))

dept_stats = df.groupby("depto").agg(
    N=("firm_id", "count"),
    Loss_pct=("loss_making", "mean")
).reset_index()
dept_stats = dept_stats.sort_values("N", ascending=True)

colors = ["#c0392b" if p > 0.5 else "#27ae60" for p in dept_stats["Loss_pct"]]
bars = ax.barh(dept_stats["depto"], dept_stats["N"], color=colors, edgecolor="white")

for bar, pct in zip(bars, dept_stats["Loss_pct"]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
            f"{100*pct:.0f}% loss", va="center", fontsize=8)

ax.set_xlabel("Number of firms")
ax.set_title("Figure B10: Sample and Loss-Making Share by Department")
ax.legend([mpatches.Patch(color="#c0392b"), mpatches.Patch(color="#27ae60")],
          [">50% loss-making", "≤50% loss-making"], loc="lower right")
plt.tight_layout()
plt.savefig(out("figB10_departments.pdf"), bbox_inches="tight")
plt.savefig(out("figB10_departments.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B11: Mean Reversion in Error ────────────────────────────────────
print("Figure B11: Mean Reversion in Measurement Error...")
fig, ax = plt.subplots(figsize=(8, 6))

# ε₁ = y_EH - π_conv (cash error)
eps1 = df["yi_net"]/52 - df["net_income_conventional"]
pi_conv = df["net_income_conventional"]

ax.scatter(pi_conv.clip(-500, 3000), eps1.clip(-2000, 5000), alpha=0.4, s=15, color="#3498db")

# Lowess
from scipy.ndimage import uniform_filter1d
sorted_idx = pi_conv.clip(-500, 3000).argsort()
x_sorted = pi_conv.clip(-500, 3000).iloc[sorted_idx].values
y_sorted = eps1.clip(-2000, 5000).iloc[sorted_idx].values
y_smooth = uniform_filter1d(y_sorted, size=50)
ax.plot(x_sorted, y_smooth, color="#c0392b", lw=2, label="Smoothed trend")

ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("Conventional profit π_conv (Bs/week)")
ax.set_ylabel("Cash error ε₁ = y_EH - π_conv (Bs/week)")
ax.set_title("Figure B11: Mean Reversion in Cash Measurement Error")
ax.legend()
plt.tight_layout()
plt.savefig(out("figB11_mean_reversion.pdf"), bbox_inches="tight")
plt.savefig(out("figB11_mean_reversion.png"), bbox_inches="tight", dpi=150)
plt.close()

# ─── FIGURE B12: Reliability by Cell ────────────────────────────────────────
print("Figure B12: Reliability by Sector × Gender...")
fig, ax = plt.subplots(figsize=(10, 6))

rel_df = pd.read_csv(out("tableA2_reliability.csv"))
rel_df = rel_df.sort_values("λ (reliability)", ascending=True)

colors = [COLOR_MALE if g == "Male" else COLOR_FEMALE for g in rel_df["Gender"]]
bars = ax.barh(rel_df["Sector"] + " (" + rel_df["Gender"] + ")", 
               rel_df["λ (reliability)"], color=colors, edgecolor="white")

ax.axvline(0.5, color="gray", lw=1, ls="--", label="λ = 0.5 (half signal)")
ax.set_xlabel("Reliability ratio λ = Var(π_adj) / [Var(π_adj) + Var(ε)]")
ax.set_title("Figure B12: Reliability Ratios by Sector × Gender")
ax.legend([mpatches.Patch(color=COLOR_MALE), mpatches.Patch(color=COLOR_FEMALE)],
          ["Male", "Female"], loc="lower right")
ax.set_xlim(0, 0.5)
plt.tight_layout()
plt.savefig(out("figB12_reliability.pdf"), bbox_inches="tight")
plt.savefig(out("figB12_reliability.png"), bbox_inches="tight", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS FOR TEXT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("KEY STATISTICS FOR TEXT")
print("─"*70)

stats_text = {
    "N_firms": len(df),
    "N_transactions": len(dt[dt["module"] != "DIA_INACTIVO"]),
    "N_departments": df["depto"].nunique(),
    "N_sectors": df["sector"].nunique(),
    "Pct_loss_making": round(100 * df["loss_making"].mean(), 1),
    "Pct_viable": round(100 * df["viable"].mean(), 1),
    "Median_accounting_gap": round(df["accounting_gap_weekly"].median(), 0),
    "Gap_as_pct_of_EH_net": round(100 * df["accounting_gap_weekly"].median() / (df["yi_net"].median()/52), 1),
    "EH_debt_detection": round(100 * (df["ci6"] > 0).sum() / df["has_debt"].sum(), 1),
    "True_debt_prevalence": round(100 * df["has_debt"].mean(), 1),
    "Median_Q": round(np.nanmedian(q_vals), 1),
    "N_Type_I": n_I,
    "N_Type_II": n_II,
    "N_Type_III": n_III,
    "N_Type_IV": n_IV,
    "Median_prestamista_EAR": round(debtors[debtors["debt_type"]=="prestamista_informal"]["ear"].median(), 0),
}

for k, v in stats_text.items():
    print(f"  {k}: {v}")

pd.DataFrame([stats_text]).T.to_csv(out("key_statistics.csv"), header=["Value"])

# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("REPLICATION COMPLETE")
print("="*70)
print(f"\nOutput directory: {OUT_DIR}")
print(f"\nFiles created:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
print("\n" + "="*70)


# ═══════════════════════════════════════════════════════════════════════════
# V3 ADDITIONS: epsilon decomposition, EH censoring, balance table, key stats
# ═══════════════════════════════════════════════════════════════════════════
print("V3: Decomposition, censoring, balance...")
yEH_w = df["yi_net"]/52.0
eps1 = yEH_w - df["net_income_conventional"]
eps2 = df["net_income_conventional"] - df["net_income_adjusted"]
total_gap = yEH_w - df["net_income_adjusted"]
print(f"  eps1 median={eps1.median():.0f}  eps2 median={eps2.median():.0f}  total={total_gap.median():.0f}")
print(f"  EH min weekly={yEH_w.min():.0f}  share<=0={(yEH_w<=0).mean()*100:.1f}%")

try:
    bal = pd.read_csv(os.path.join(DATA_DIR, "eh_selfemployed_balance.csv"))
    eh = bal[(bal["in_diary"]==0) & (bal["is_patron"]==0)].copy()
    def wmean(x,w): m=x.notna(); return np.average(x[m],weights=w[m])
    def wsd(x,w):
        m=x.notna(); mu=np.average(x[m],weights=w[m])
        return np.sqrt(np.average((x[m]-mu)**2,weights=w[m]))
    rows=[("Age (years)", d5["mage"], eh["edad"]),
          ("Schooling (years)", d5["msch"], eh["aestudio"]),
          ("Male (share)", d5["hombre"], eh["hombre"]),
          ("Weekly hours", d5["phrs"], eh["phrs"]),
          ("Enterprise tenure (years)", d5["years_operating"], eh["tenure_years"]),
          ("Household size", d5["hhtotal"], eh["hhtotal"]),
          ("Children under 15", d5["nn0_15"], eh["nn0_15"])]
    out_rows=[]
    for nm,dcol,ecol in rows:
        m1,sd1=dcol.mean(),dcol.std(); m0,sd0=wmean(ecol,eh["factor"]),wsd(ecol,eh["factor"])
        out_rows.append({"Variable":nm,"Diary":round(m1,2),"EH":round(m0,2),
                         "NormDiff":round((m1-m0)/np.sqrt((sd1**2+sd0**2)/2),3)})
    pd.DataFrame(out_rows).to_csv(out("table12_balance.csv"), index=False)
    print(f"  balance table written (EH n={len(eh)})")
except FileNotFoundError:
    print("  [eh_selfemployed_balance.csv not found: balance table skipped]")

ks = {"Median_Q_corrected": round(np.nanmedian(q18),2), "N_valuation": int(np.sum(~np.isnan(q18))),
      "eps1_median": round(eps1.median(),0), "eps2_median": round(eps2.median(),0),
      "total_gap_median": round(total_gap.median(),0),
      "gap_pct_of_pconv": round(100*eps2.median()/df["net_income_conventional"].median(),1),
      "gap_pct_of_yEH": round(100*eps2.median()/yEH_w.median(),1),
      "total_pct_of_yEH": round(100*total_gap.median()/yEH_w.median(),1)}
pd.Series(ks).to_csv(out("key_statistics_v3.csv"))
print("  key_statistics_v3.csv written")
print("REPLICATION COMPLETE (v3)")
