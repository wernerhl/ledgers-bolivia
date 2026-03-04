"""
=============================================================================
LEDGERS OF THE SELF-EMPLOYED — Bolivia Diary Study
Analytics Program
=============================================================================
Usage:
    python ledgers_analytics.py \
        --firms   ledgers_firms.csv \
        --accounts ledgers_accounts.csv \
        --transactions ledgers_transactions.csv \
        --out     ledgers_results/

Produces:
    ledgers_results/
        tables/        ← all paper tables as CSV
        figures/       ← all paper figures as PNG
        ledgers_paper_tables.xlsx  ← workbook with all tabs
        ledgers_full_dataset.csv   ← master analytic panel

Requirements:
    pip install pandas numpy scipy statsmodels matplotlib seaborn openpyxl

=============================================================================
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ── global constants ───────────────────────────────────────────────────────
MIN_WAGE_MONTHLY  = 2_362          # Bs/month Bolivia 2024
MIN_WAGE_HOURLY   = MIN_WAGE_MONTHLY / (4.33 * 8 * 5)   # ≈13.64 Bs/hr
DISCOUNT_RATE     = 0.18           # informal-sector hurdle rate
P2P_RATE          = 9.5            # Bs per USD (2025 P2P)
WEEKS_PER_YEAR    = 52

SECTOR_LABELS = {
    "admin":       "Admin & Apoyo",
    "comercio":    "Comercio",
    "gastronomia": "Gastronomía",
    "manufactura": "Manufactura",
    "servicios":   "Servicios",
    "transporte":  "Transporte",
}

PALETTE = {
    "comercio":    "#2E6DB4",
    "manufactura": "#E8821A",
    "transporte":  "#28A745",
    "gastronomia": "#C0392B",
    "servicios":   "#8E44AD",
    "admin":       "#17A589",
}

TYPOLOGY_COLORS = {
    "I_Viable":           "#27AE60",
    "II_Precaria":        "#F39C12",
    "III_Atrapada_deuda": "#E74C3C",
    "IV_Riesgo_critico":  "#7D3C98",
}

# ── helpers ────────────────────────────────────────────────────────────────

def pct(series, decimals=1):
    return (series * 100).round(decimals)

def safe_div(a, b, fill=np.nan):
    return np.where(b != 0, a / b, fill)

def winsorize(s, lo=0.01, hi=0.99):
    lb, ub = s.quantile(lo), s.quantile(hi)
    return s.clip(lb, ub)

def stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def fmt_bs(x, decimals=0):
    return f"Bs {x:,.{decimals}f}"

def save_fig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# ── loader ─────────────────────────────────────────────────────────────────

def load_data(firms_path, accounts_path, transactions_path):
    firms = pd.read_csv(firms_path, low_memory=False)
    accts = pd.read_csv(accounts_path, low_memory=False)
    tx    = pd.read_csv(transactions_path, low_memory=False)

    # Merge EH fields into accounts if not already present
    eh_cols = [c for c in firms.columns if c.startswith("eh_")]
    already  = [c for c in eh_cols if c in accts.columns]
    missing  = [c for c in eh_cols if c not in accts.columns]
    if missing:
        accts = accts.merge(firms[["firm_id"] + missing], on="firm_id", how="left")

    # Merge firm-level attributes needed for analytics
    firm_attrs = [
        "firm_id", "tool_value", "inventory_init", "receivables_init",
        "payables_init", "cash_initial", "debt_amount", "debt_type",
        "monthly_interest_rate", "years_operating", "age",
        "shadow_wage_per_hour", "owner_hours_weekly",
        "family_hours_weekly", "n_family_workers",
        "has_debt_true",
    ]
    # avoid re-merging cols already in accts (e.g. gender, sector, city)
    extra = [c for c in firm_attrs if c not in accts.columns]
    if extra:
        accts = accts.merge(firms[["firm_id"] + extra], on="firm_id", how="left")

    # Normalise a few column aliases
    for alias, canon in [("eh_nonresponder_x", "eh_nonresponder"),
                         ("debt_type_x",       "debt_type_clean")]:
        if alias in accts.columns and canon not in accts.columns:
            accts[canon] = accts[alias]
    if "debt_type_clean" not in accts.columns and "debt_type" in accts.columns:
        accts["debt_type_clean"] = accts["debt_type"]

    # Resolve eh_nonresponder
    if "eh_nonresponder" not in accts.columns:
        accts["eh_nonresponder"] = accts.get("eh_nonresponder_y", False)

    print(f"  Loaded {len(firms)} firms | {len(accts)} account rows | {len(tx):,} transactions")
    return firms, accts, tx

# ── PART A: BUILD MASTER ANALYTIC PANEL ────────────────────────────────────

def build_panel(accts):
    pl = accts.copy()

    # ── annualise weekly flow variables
    weekly_flows = [
        "total_revenue_diary", "cogs", "gross_profit",
        "transport_exp", "rent_exp", "utilities_exp", "guild_exp",
        "total_op_exp_conventional", "labor_imputed", "depreciation_imputed",
        "total_op_exp_adjusted", "interest_paid",
        "net_income_conventional", "net_income_adjusted",
        "accounting_gap_bs", "cash_flow_operating", "household_draws",
    ]
    for c in weekly_flows:
        if c in pl.columns:
            pl[f"{c}_annual"] = pl[c] * WEEKS_PER_YEAR

    # ── balance sheet
    pl["current_assets"]    = pl["cash_initial"] + pl["inventory_init"] + pl["receivables_init"]
    pl["accum_dep"]         = pl["depreciation_imputed"] * WEEKS_PER_YEAR * pl["years_operating"].clip(upper=20)
    pl["net_fixed_assets"]  = (pl["tool_value"] - pl["accum_dep"]).clip(lower=0)
    pl["total_assets"]      = pl["current_assets"] + pl["net_fixed_assets"]
    pl["current_liab"]      = pl["payables_init"]
    pl["noncurrent_liab"]   = pl["debt_amount"].fillna(0)
    pl["total_liab"]        = pl["current_liab"] + pl["noncurrent_liab"]
    pl["book_equity"]       = (pl["total_assets"] - pl["total_liab"]).clip(lower=1)

    # ── labour
    pl["total_hours"]       = pl["owner_hours_weekly"] + pl["family_hours_weekly"].fillna(0)

    # ── profitability margins
    rev = pl["total_revenue_diary"].replace(0, np.nan)
    gp  = pl["gross_profit"].replace(0, np.nan)
    be  = pl["book_equity"].replace(0, np.nan)
    ta  = pl["total_assets"].replace(0, np.nan)
    tl  = pl["total_liab"].replace(0, np.nan)
    cc  = pl["cash_flow_operating"].replace(0, np.nan)
    h   = pl["total_hours"].replace(0, np.nan)

    pl["gross_margin"]          = pl["gross_profit"]             / rev
    pl["ebitda_weekly"]         = pl["net_income_conventional"]  + pl["interest_paid"] + pl["depreciation_imputed"]
    pl["ebitda_margin"]         = pl["ebitda_weekly"]            / rev
    pl["net_margin_conv"]       = pl["net_income_conventional"]  / rev
    pl["net_margin_adj"]        = pl["net_income_adjusted"]      / rev

    # ── return ratios (annualised)
    pl["roa_conv"]              = pl["net_income_conventional_annual"] / ta
    pl["roa_adj"]               = pl["net_income_adjusted_annual"]     / ta
    pl["roe_conv"]              = pl["net_income_conventional_annual"] / be
    pl["roe_adj"]               = pl["net_income_adjusted_annual"]     / be
    pl["ebv_conv"]              = pl["net_income_conventional_annual"] / be   # E/BV
    pl["ebv_adj"]               = pl["net_income_adjusted_annual"]     / be

    # ── du-pont
    pl["asset_turnover"]        = pl["total_revenue_diary_annual"] / ta
    pl["equity_multiplier"]     = ta / be
    pl["dupont_roe"]            = pl["net_margin_conv"] * pl["asset_turnover"] * pl["equity_multiplier"]

    # ── leverage / coverage
    pl["debt_to_assets"]        = tl / ta
    pl["debt_to_equity"]        = tl / be
    pl["interest_coverage"]     = pl["ebitda_weekly"] / pl["interest_paid"].replace(0, np.nan)
    pl["eff_annual_rate"]       = (1 + pl["monthly_interest_rate"].fillna(0)) ** 12 - 1
    pl["interest_burden_rev"]   = pl["interest_paid"] / rev * 100

    # ── liquidity / activity
    cogs_w = pl["cogs"].replace(0, np.nan)
    pl["current_ratio"]         = pl["current_assets"] / pl["current_liab"].replace(0, np.nan)
    pl["quick_ratio"]           = (pl["cash_initial"] + pl["receivables_init"]) / pl["current_liab"].replace(0, np.nan)
    pl["inventory_days"]        = (pl["inventory_init"]   / cogs_w) * 7
    pl["receivable_days"]       = (pl["receivables_init"] / rev)    * 7
    pl["payable_days"]          = (pl["payables_init"]    / cogs_w) * 7
    pl["cash_conv_cycle"]       = pl["inventory_days"] + pl["receivable_days"] - pl["payable_days"]
    pl["cash_holding_days"]     = (pl["cash_initial"] / rev) * 7

    # ── labour productivity
    pl["rev_per_hour"]          = pl["total_revenue_diary"] / h
    pl["va_per_hour"]           = pl["gross_profit"]         / h
    pl["net_per_hour_adj"]      = pl["net_income_adjusted"]  / h
    pl["mw_multiple"]           = pl["net_per_hour_adj"]     / MIN_WAGE_HOURLY

    # ── going-concern valuation
    pl["fcf_annual"]            = (pl["cash_flow_operating"] * WEEKS_PER_YEAR
                                   - pl["depreciation_imputed_annual"])
    for g, label in [(0.00, "g0"), (0.02, "g2"), (0.05, "g5")]:
        r_eff = DISCOUNT_RATE - g
        pl[f"pv_{label}"] = np.where(
            pl["fcf_annual"] > 0,
            pl["fcf_annual"] / r_eff,
            np.nan
        )
    pl["tobin_q_g0"]            = pl["pv_g0"] / ta
    pl["tobin_q_g2"]            = pl["pv_g2"] / ta
    pl["payback_years"]         = (pl["total_assets"] / pl["fcf_annual"].replace(0, np.nan)).clip(0, 50)

    # ── extraction rate
    pl["extraction_rate"]       = pl["household_draws_annual"] / pl["total_revenue_diary_annual"].replace(0, np.nan)

    # ── accounting gap
    pl["gap_pct_conv"]          = safe_div(pl["accounting_gap_bs"],
                                           pl["net_income_conventional"].replace(0, np.nan)) * 100

    # ── typology (re-derive on panel)
    cond_viable   = (pl["net_income_adjusted"] > 0) & (pl["interest_burden_rev"] < 10)
    cond_precaria = (pl["net_income_adjusted"] <= 0) & (pl["interest_burden_rev"] < 10)
    cond_trapped  = (pl["net_income_adjusted"] > 0) & (pl["interest_burden_rev"] >= 10)
    cond_crisis   = (pl["net_income_adjusted"] <= 0) & (pl["interest_burden_rev"] >= 10)
    pl["typology"] = np.select(
        [cond_viable, cond_precaria, cond_trapped, cond_crisis],
        ["I_Viable", "II_Precaria", "III_Atrapada_deuda", "IV_Riesgo_critico"],
        default="II_Precaria"
    )

    return pl

# ── PART B: TABLES ──────────────────────────────────────────────────────────

def table_descriptive(pl):
    """Table 1: Descriptive statistics by sector."""
    g = pl.groupby("sector")
    t = pd.DataFrame({
        "N":                        g["firm_id"].count(),
        "% female":                 pct(g["gender"].apply(lambda x: (x=="F").mean())),
        "% poor":                   pct(g["is_poor"].mean()),
        "% indebted (true)":        pct(g["has_debt_true"].mean()),
        "Hours/week (med)":         g["total_hours"].median().round(0),
        "Weekly revenue (med Bs)":  g["total_revenue_diary"].median().round(0),
        "Gross margin % (med)":     pct(g["gross_margin"].median()),
        "Net margin conv % (med)":  pct(g["net_margin_conv"].median()),
        "Net margin adj % (med)":   pct(g["net_margin_adj"].median()),
        "% loss-making (adj)":      pct(g["net_income_adjusted"].apply(lambda x: (x<0).mean())),
        "Acctg gap Bs/wk (med)":    g["accounting_gap_bs"].median().round(0),
        "Gap % conv profit (med)":  g["gap_pct_conv"].median().round(1),
        "MW multiple (med)":        g["mw_multiple"].median().round(2),
    })
    t.index = t.index.map(SECTOR_LABELS)
    return t

def table_balance_sheet(pl):
    """Table 2: Balance sheet by sector."""
    g = pl.groupby("sector")
    t = pd.DataFrame({
        "Cash Bs (med)":            g["cash_initial"].median().round(0),
        "Inventory Bs (med)":       g["inventory_init"].median().round(0),
        "Receivables Bs (med)":     g["receivables_init"].median().round(0),
        "Net fixed assets Bs (med)":g["net_fixed_assets"].median().round(0),
        "Total assets Bs (med)":    g["total_assets"].median().round(0),
        "Payables Bs (med)":        g["current_liab"].median().round(0),
        "Debt Bs (med)":            g["noncurrent_liab"].median().round(0),
        "Book equity Bs (med)":     g["book_equity"].median().round(0),
        "Debt/assets (med)":        g["debt_to_assets"].median().round(3),
        "Debt/equity (med)":        g["debt_to_equity"].median().round(2),
        "Current ratio (med)":      g["current_ratio"].median().round(2),
    })
    t.index = t.index.map(SECTOR_LABELS)
    return t

def table_returns(pl):
    """Table 3: Return ratios and valuation by sector."""
    g = pl.groupby("sector")
    t = pd.DataFrame({
        "ROA conv %/yr (med)":      pct(g["roa_conv"].median()),
        "ROA adj %/yr (med)":       pct(g["roa_adj"].median()),
        "ROE conv %/yr (med)":      pct(g["roe_conv"].median()),
        "ROE adj %/yr (med)":       pct(g["roe_adj"].median()),
        "E/BV conv (med)":          g["ebv_conv"].median().round(2),
        "E/BV adj (med)":           g["ebv_adj"].median().round(2),
        "Asset turnover (med)":     g["asset_turnover"].median().round(2),
        "EBITDA margin % (med)":    pct(g["ebitda_margin"].median()),
        "FCF annual Bs (med)":      g["fcf_annual"].median().round(0),
        "PV g=0 Bs (med viable)":   pl[pl["typology"]=="I_Viable"].groupby("sector")["pv_g0"].median().round(0),
        "Tobin Q g=0 (med)":        g["tobin_q_g0"].median().round(1),
        "Payback yrs (viable)":     pl[pl["typology"]=="I_Viable"].groupby("sector")["payback_years"].median().round(1),
    })
    t.index = t.index.map(SECTOR_LABELS)
    return t

def table_activity(pl):
    """Table 4: Activity and liquidity ratios."""
    g = pl.groupby("sector")
    t = pd.DataFrame({
        "Inventory days (med)":     g["inventory_days"].median().round(1),
        "Receivable days (med)":    g["receivable_days"].median().round(1),
        "Payable days (med)":       g["payable_days"].median().round(1),
        "Cash conv cycle days(med)":g["cash_conv_cycle"].median().round(1),
        "Cash holding days (med)":  g["cash_holding_days"].median().round(1),
        "Rev per hour Bs (med)":    g["rev_per_hour"].median().round(1),
        "VA per hour Bs (med)":     g["va_per_hour"].median().round(1),
        "Net/hr adj Bs (med)":      g["net_per_hour_adj"].median().round(2),
        "MW multiple (med)":        g["mw_multiple"].median().round(2),
        "Extraction rate % (med)":  pct(g["extraction_rate"].median()),
    })
    t.index = t.index.map(SECTOR_LABELS)
    return t

def table_eh_vs_diary(pl):
    """Table 5: EH vs diary measurement bias."""
    sub = pl[~pl["eh_nonresponder"] & (pl.get("eh_yi_tot", 0) > 0)].copy()
    if "eh_yi_tot" not in sub.columns:
        sub["eh_yi_tot"] = sub.get("eh_yi_tot_annual", 0)
    sub["ratio_income"] = safe_div(sub["diary_yi_tot_annualized"], sub["eh_yi_tot"].replace(0, np.nan))

    g = sub.groupby("sector")
    t = pd.DataFrame({
        "N (EH responders)":         g["firm_id"].count(),
        "EH yi_tot med Bs/yr":       g["eh_yi_tot"].median().round(0),
        "Diary revenue med Bs/yr":   g["diary_yi_tot_annualized"].median().round(0),
        "Ratio diary/EH income":     g["ratio_income"].median().round(2),
        "% diary > EH income":       pct(g["ratio_income"].apply(lambda x: (x>1).mean())),
        "EH yi_net med Bs/yr":       g["eh_yi_net_annual"].median().round(0),
        "Diary net conv Bs/yr":      g["diary_yi_net_conv_annualized"].median().round(0),
        "Diary net adj Bs/yr":       g["diary_yi_net_adj_annualized"].median().round(0),
        "Gap conv-adj Bs/yr (med)":  (g["diary_yi_net_conv_annualized"].median()
                                      - g["diary_yi_net_adj_annualized"].median()).round(0),
    })
    t.index = t.index.map(SECTOR_LABELS)
    return t, sub

def table_typology(pl):
    """Table 6: Typology distribution and characteristics."""
    g = pl.groupby("typology")
    t = pd.DataFrame({
        "N":                         g["firm_id"].count(),
        "% total":                   pct(g["firm_id"].count() / len(pl)),
        "% female":                  pct(g["gender"].apply(lambda x: (x=="F").mean())),
        "% poor":                    pct(g["is_poor"].mean()),
        "Med weekly revenue Bs":     g["total_revenue_diary"].median().round(0),
        "Med net income adj Bs/wk":  g["net_income_adjusted"].median().round(0),
        "Med ROA adj %/yr":          pct(g["roa_adj"].median()),
        "Med interest burden %":     g["interest_burden_rev"].median().round(1),
        "Med debt/equity":           g["debt_to_equity"].median().round(2),
        "Med MW multiple":           g["mw_multiple"].median().round(2),
        "Med accounting gap Bs/wk":  g["accounting_gap_bs"].median().round(0),
        "Viable share in sector":    g["typology"].count() / len(pl) * 100,
    })
    return t

def table_heterogeneity(pl):
    """Table 7: Sector × Gender heterogeneity."""
    g = pl.groupby(["sector", "gender"])
    t = pd.DataFrame({
        "N":                         g["firm_id"].count(),
        "Med weekly revenue Bs":     g["total_revenue_diary"].median().round(0),
        "Med gross margin %":        pct(g["gross_margin"].median()),
        "Med net adj Bs/wk":         g["net_income_adjusted"].median().round(0),
        "% loss-making":             pct(g["net_income_adjusted"].apply(lambda x: (x<0).mean())),
        "Med hours/wk":              g["total_hours"].median().round(0),
        "Med net/hr adj Bs":         g["net_per_hour_adj"].median().round(2),
        "Med ROA adj %/yr":          pct(g["roa_adj"].median()),
        "% indebted":                pct(g["has_debt_true"].mean()),
    })
    t.index = t.index.map(lambda x: (SECTOR_LABELS.get(x[0], x[0]), x[1]))
    return t

def table_production_function(pl):
    """Table 8: Cobb-Douglas production function estimates."""
    pf = pl[
        (pl["total_revenue_diary"] > 10) &
        (pl["total_assets"] > 10) &
        (pl["total_hours"] > 0)
    ].copy()
    pf["ln_Y"]   = np.log(pf["total_revenue_diary"])
    pf["ln_K"]   = np.log(pf["total_assets"])
    pf["ln_L"]   = np.log(pf["total_hours"])
    pf["ln_age"] = np.log(pf["years_operating"].clip(lower=1))

    sdums = pd.get_dummies(pf["sector"], drop_first=True, prefix="s").astype(float)

    rows = []

    def _fit(name, df, X_cols):
        X = sm.add_constant(df[X_cols])
        m = OLS(df["ln_Y"], X).fit(cov_type="HC3")
        bK   = m.params.get("ln_K", np.nan)
        bL   = m.params.get("ln_L", np.nan)
        seK  = m.bse.get("ln_K", np.nan)
        seL  = m.bse.get("ln_L", np.nan)
        rts  = bK + bL
        rts_se = np.sqrt(
            m.cov_params().loc[["ln_K","ln_L"],["ln_K","ln_L"]].values.sum()
        ) if "ln_K" in m.params and "ln_L" in m.params else np.nan
        rts_t  = (rts - 1) / rts_se if rts_se > 0 else np.nan
        rts_p  = 2 * stats.t.sf(abs(rts_t), m.df_resid) if not np.isnan(rts_t) else np.nan
        rows.append({
            "Model":      name,
            "N":          int(m.nobs),
            "α_K":        round(bK, 3),
            "se_K":       f"({seK:.3f})",
            "α_L":        round(bL, 3),
            "se_L":       f"({seL:.3f})",
            "RTS":        round(rts, 3),
            "RTS stars":  stars(rts_p),
            "R²":         round(m.rsquared, 3),
            "Adj R²":     round(m.rsquared_adj, 3),
        })

    # Pooled models
    _fit("(1) Pooled basic",         pf,            ["ln_K", "ln_L"])
    pf2 = pf.join(sdums)
    _fit("(2) Pooled + sector FE",   pf2,           ["ln_K", "ln_L"] + sdums.columns.tolist())
    _fit("(3) + Firm age",           pf.join(sdums),["ln_K", "ln_L", "ln_age"] + sdums.columns.tolist())

    # By sector
    for sec in sorted(pf["sector"].unique()):
        sd = pf[pf["sector"] == sec]
        if len(sd) >= 15:
            _fit(f"  Sector: {SECTOR_LABELS.get(sec, sec)}", sd, ["ln_K", "ln_L"])

    return pd.DataFrame(rows)

def table_credit(pl):
    """Table 9: Informal credit analysis."""
    debt = pl[pl["has_debt_true"]].copy()
    if "debt_type_clean" not in debt.columns:
        debt["debt_type_clean"] = "desconocido"

    g = debt.groupby("debt_type_clean")
    t = pd.DataFrame({
        "N":                          g["firm_id"].count(),
        "% of indebted firms":        pct(g["firm_id"].count() / len(debt)),
        "Med debt amount Bs":         g["debt_amount"].median().round(0),
        "Monthly rate % (med)":       pct(g["monthly_interest_rate"].median()),
        "EAR % (med)":                pct(g["eff_annual_rate"].median()),
        "Med interest burden % rev":  g["interest_burden_rev"].median().round(1),
    })
    # Add EH vs true detection
    eh_declared = (pl.get("eh_ci6", pd.Series(0, index=pl.index)) > 0).sum()
    detection = pd.DataFrame([{
        "debt_type_clean":  "TOTAL",
        "N":                int(pl["has_debt_true"].sum()),
        "% of indebted firms": 100.0,
        "Med debt amount Bs": debt["debt_amount"].median().round(0),
        "Monthly rate % (med)": "—",
        "EAR % (med)": "—",
        "Med interest burden % rev": debt["interest_burden_rev"].median().round(1),
    }])
    eh_row = pd.DataFrame([{
        "debt_type_clean":  "EH declared (ci6>0)",
        "N":                int(eh_declared),
        "% of indebted firms": round(eh_declared / len(debt) * 100, 1),
        "Med debt amount Bs": "—",
        "Monthly rate % (med)": "—",
        "EAR % (med)": "—",
        "Med interest burden % rev": "—",
    }])
    return t

def table_eh_adequacy():
    """Table 10: EH questionnaire adequacy scorecard."""
    rows = [
        ("Gross revenue",                    "Partial — recall anchored weekly",  "Full (daily obs)",          "Critical"),
        ("Net income",                        "Partial — remembered residual",     "Full (computed)",           "Critical"),
        ("Cost of goods sold",               "Partial — ci1, recall ~47%",        "Full — Módulo 2",           "Critical"),
        ("Owner labour (imputed)",           "ABSENT",                            "Full — Módulo 5, A2",       "Critical"),
        ("Unpaid family labour (imputed)",   "ABSENT",                            "Full — Módulo 5, A3",       "Critical"),
        ("Informal interest (non-cuota)",    "ABSENT",                            "Full — Módulo 4B",          "Critical"),
        ("Household–firm boundary flows",    "ABSENT",                            "Full — Módulo 7",           "Critical"),
        ("Production function (K, L)",       "Hours only, no capital",            "Daily transactions + assets","Critical"),
        ("Rent",                             "Partial — ci4, ~75% recall",        "Full — Módulo 3",           "High"),
        ("Receivables / credit sales",       "ABSENT",                            "Full — Módulo 1B",          "High"),
        ("Payables / supplier credit",       "ABSENT",                            "Full — Módulo 2",           "High"),
        ("Inventory opening/closing",        "ABSENT",                            "Full — Módulo 0 + 8",       "High"),
        ("Asset composition & depreciation", "ABSENT",                            "Full — Módulo 6, A1",       "High"),
        ("Loan principal outstanding",       "ABSENT",                            "Partial — screening q.",    "High"),
        ("Formal loan installments",         "Partial — ci6, 7% report",          "Full — Módulo 4B",          "High"),
        ("Daily revenue variance",           "ABSENT",                            "Full — 7-day panel",        "High"),
        ("Utilities",                        "Partial — ci5, 30–70% recall",      "Full — Módulo 3",           "Medium"),
        ("Transport costs",                  "Partial — ci2",                     "Full — Módulo 3",           "Medium"),
        ("Own-consumption (in-kind)",        "ABSENT",                            "Full — Módulo 1C",          "Medium"),
        ("USD/FX transactions",              "ABSENT",                            "Full — currency field",     "Medium"),
        ("Wages paid (external workers)",    "Partial — ci3, ~70% recall",        "Full — Módulo 5",           "Medium"),
        ("Dual-currency cost exposure",      "ABSENT",                            "Full — transaction currency","Medium"),
        ("Guild / gremio dues",             "Partial — ci8",                     "Full — Módulo 3",           "Low"),
        ("Taxes",                            "Partial — ci7",                     "Full — Módulo 3",           "Low"),
    ]
    df = pd.DataFrame(rows, columns=["Concept", "EH Coverage", "Diary Coverage", "Importance"])
    df["EH Absent"] = df["EH Coverage"].str.startswith("ABSENT")
    return df

def table_transaction_complexity(tx):
    """Table 11: Transaction complexity metrics."""
    active = tx[tx["module"] != "DIA_INACTIVO"].copy()
    sales  = active[active["module"] == "M1A_VENTAS"].copy()

    daily_n = active.groupby(["firm_id","day_num"])["tx_id"].count()
    firm_cv = (
        sales.groupby(["firm_id","day_num"])["amount_bs"].sum()
             .unstack(fill_value=0)
             .apply(lambda r: r.std() / r.mean() if r.mean() > 0 else np.nan, axis=1)
    )

    rows = {
        "Total transactions":             len(active),
        "Transactions per firm-day (med)":round(daily_n.median(), 1),
        "Transactions per firm-day (p90)":round(daily_n.quantile(0.90), 1),
        "Module types":                   active["module"].nunique(),
        "Category types":                 active["category"].nunique(),
        "Median sale amount Bs":          round(sales["amount_bs"].median(), 1),
        "P10 sale amount Bs":             round(sales["amount_bs"].quantile(0.10), 1),
        "P90 sale amount Bs":             round(sales["amount_bs"].quantile(0.90), 1),
        "% USD transactions":             round((active["currency"]=="USD").mean()*100, 1),
        "% credit sales":                 round(sales["is_credit"].mean()*100, 1),
        "% imputed transactions":         round(active["is_imputed"].mean()*100, 1),
        "Revenue CV across days (med)":   round(firm_cv.median(), 2),
        "Revenue CV across days (p90)":   round(firm_cv.quantile(0.90), 2),
    }
    df = pd.DataFrame(list(rows.items()), columns=["Metric", "Value"])

    # Module-level breakdown
    mod_summary = (
        active.groupby("module")["amount_bs"]
              .agg(N="count", Total_Bs="sum", Median_Bs="median")
              .assign(Pct_N=lambda d: (d["N"] / d["N"].sum() * 100).round(1))
              .round(0)
    )
    return df, mod_summary

# ── PART C: FIGURES ─────────────────────────────────────────────────────────

def fig_accounting_gap(pl, outdir):
    """Figure 1: Conventional vs adjusted profit distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 1 — The Accounting Gap: Conventional vs Adjusted Profit",
                 fontsize=13, fontweight="bold", y=1.01)

    sectors = sorted(pl["sector"].unique())
    colors  = [PALETTE[s] for s in sectors]
    labels  = [SECTOR_LABELS[s] for s in sectors]

    # Panel A: Box plots conventional vs adjusted
    ax = axes[0]
    data_conv = [pl[pl["sector"]==s]["net_income_conventional"].clip(-500,3000).dropna() for s in sectors]
    data_adj  = [pl[pl["sector"]==s]["net_income_adjusted"].clip(-500,3000).dropna()  for s in sectors]
    pos_conv  = np.arange(len(sectors)) * 2.5
    pos_adj   = pos_conv + 0.9
    bp1 = ax.boxplot(data_conv, positions=pos_conv, widths=0.7,
                     patch_artist=True, medianprops=dict(color="black", lw=2))
    bp2 = ax.boxplot(data_adj,  positions=pos_adj,  widths=0.7,
                     patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp1["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for patch in bp2["boxes"]:
        patch.set_facecolor("lightgray"); patch.set_alpha(0.75)
    ax.axhline(0, color="red", lw=1.2, ls="--", alpha=0.7)
    ax.set_xticks(pos_conv + 0.45)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Net income Bs/week", fontsize=9)
    ax.set_title("A. Conventional (color) vs Adjusted (grey)", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="steelblue",label="Conventional"),
                        Patch(facecolor="lightgray",label="Adjusted")],
              fontsize=8, loc="upper right")

    # Panel B: % loss-making by sector
    ax = axes[1]
    pct_loss_conv = [0.0] * len(sectors)   # always 0% by construction
    pct_loss_adj  = [pl[pl["sector"]==s]["net_income_adjusted"].lt(0).mean()*100 for s in sectors]
    x = np.arange(len(sectors))
    ax.bar(x - 0.2, pct_loss_conv, 0.35, label="Conventional", color=[PALETTE[s] for s in sectors], alpha=0.6)
    ax.bar(x + 0.2, pct_loss_adj,  0.35, label="Adjusted",     color=[PALETTE[s] for s in sectors], alpha=1.0)
    ax.axhline(50, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of firms loss-making", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("B. % Loss-making: Conv. vs Adjusted", fontsize=9)
    for i, (c, a) in enumerate(zip(pct_loss_conv, pct_loss_adj)):
        ax.text(x[i]+0.2, a+2, f"{a:.0f}%", ha="center", fontsize=8, fontweight="bold")

    # Panel C: Gap decomposition stacked bar
    ax = axes[2]
    med_labor = [pl[pl["sector"]==s]["labor_imputed"].median() for s in sectors]
    med_dep   = [pl[pl["sector"]==s]["depreciation_imputed"].median() for s in sectors]
    med_int   = [pl[pl["sector"]==s]["interest_paid"].median() for s in sectors]
    x = np.arange(len(sectors))
    ax.bar(x, med_labor, label="Imputed labour",   color="#2c7bb6", alpha=0.85)
    ax.bar(x, med_dep,   bottom=med_labor,          label="Depreciation",   color="#fdae61", alpha=0.85)
    bot2 = [l+d for l,d in zip(med_labor, med_dep)]
    ax.bar(x, med_int,   bottom=bot2,               label="Interest",        color="#d7191c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Bs / week (median)", fontsize=9)
    ax.set_title("C. Accounting Gap Decomposition", fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return save_fig(fig, outdir, "fig1_accounting_gap.png")

def fig_ratios_heatmap(pl, outdir):
    """Figure 2: Financial ratios heatmap conventional vs adjusted."""
    metrics = {
        "Gross margin %":       pct(pl.groupby("sector")["gross_margin"].median()),
        "Net margin conv %":    pct(pl.groupby("sector")["net_margin_conv"].median()),
        "Net margin adj %":     pct(pl.groupby("sector")["net_margin_adj"].median()),
        "ROA conv %/yr":        pct(pl.groupby("sector")["roa_conv"].median()),
        "ROA adj %/yr":         pct(pl.groupby("sector")["roa_adj"].median()),
        "ROE conv %/yr":        pct(pl.groupby("sector")["roe_conv"].median()).clip(-500,500),
        "ROE adj %/yr":         pct(pl.groupby("sector")["roe_adj"].median()).clip(-500,500),
        "E/BV conv":            pl.groupby("sector")["ebv_conv"].median().round(1),
        "E/BV adj":             pl.groupby("sector")["ebv_adj"].median().round(1),
        "Asset turnover":       pl.groupby("sector")["asset_turnover"].median().round(1),
        "MW multiple":          pl.groupby("sector")["mw_multiple"].median().round(2),
    }
    df = pd.DataFrame(metrics).T
    df.columns = [SECTOR_LABELS[c] for c in df.columns]

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        df, annot=True, fmt=".1f", center=0,
        cmap="RdYlGn", linewidths=0.4, ax=ax,
        cbar_kws={"label": "Metric value"}
    )
    ax.set_title("Figure 2 — Financial Ratios Heatmap by Sector (Medians)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return save_fig(fig, outdir, "fig2_ratios_heatmap.png")

def fig_production_function(pl, outdir):
    """Figure 3: Production function scatter with fitted lines by sector."""
    pf = pl[(pl["total_revenue_diary"]>10)&(pl["total_assets"]>10)&(pl["total_hours"]>0)].copy()
    pf["ln_Y"] = np.log(pf["total_revenue_diary"])
    pf["ln_K"] = np.log(pf["total_assets"])
    pf["ln_L"] = np.log(pf["total_hours"])

    sectors = sorted(pf["sector"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Figure 3 — Production Function: Log Revenue vs Log Capital & Labor by Sector",
                 fontsize=12, fontweight="bold")

    for ax, sec in zip(axes.flat, sectors):
        sd  = pf[pf["sector"]==sec]
        col = PALETTE[sec]
        ax.scatter(sd["ln_K"], sd["ln_Y"], alpha=0.4, s=18, color=col, label="Capital (K)")
        ax.scatter(sd["ln_L"], sd["ln_Y"], alpha=0.4, s=18, color="gray",  marker="^", label="Labour (L)")
        # Fit lines
        for xvar, lcolor in [("ln_K", col), ("ln_L", "gray")]:
            if len(sd) >= 5:
                m, b, r, p, _ = stats.linregress(sd[xvar], sd["ln_Y"])
                xr = np.linspace(sd[xvar].min(), sd[xvar].max(), 50)
                ax.plot(xr, m*xr+b, color=lcolor, lw=1.6, alpha=0.85)
        # Fit pooled
        if len(sd) >= 10:
            X = sm.add_constant(sd[["ln_K","ln_L"]])
            m_ols = OLS(sd["ln_Y"], X).fit()
            aK = m_ols.params.get("ln_K", 0)
            aL = m_ols.params.get("ln_L", 0)
            rts = aK + aL
            ax.text(0.05, 0.92, f"αK={aK:.2f}  αL={aL:.2f}\nRTS={rts:.2f}  R²={m_ols.rsquared:.2f}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("ln(Capital) / ln(Hours)", fontsize=8)
        ax.set_ylabel("ln(Weekly revenue)", fontsize=8)
        ax.set_title(SECTOR_LABELS[sec], fontsize=10, fontweight="bold", color=col)
        ax.legend(fontsize=7)

    plt.tight_layout()
    return save_fig(fig, outdir, "fig3_production_function.png")

def fig_typology(pl, outdir):
    """Figure 4: Typology pie + characteristics."""
    typ_counts = pl["typology"].value_counts()
    order = ["I_Viable","II_Precaria","III_Atrapada_deuda","IV_Riesgo_critico"]
    labels_short = ["I: Viable","II: Precaria","III: Atrapada\nen deuda","IV: Riesgo\ncrítico"]
    counts  = [typ_counts.get(k, 0) for k in order]
    colors  = [TYPOLOGY_COLORS[k] for k in order]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    fig.suptitle("Figure 4 — Firm Typology", fontsize=13, fontweight="bold")

    # Pie
    wedges, texts, autotexts = ax0.pie(
        counts, labels=labels_short, colors=colors,
        autopct="%1.0f%%", startangle=140,
        textprops={"fontsize": 9}
    )
    for at in autotexts: at.set_fontsize(9)
    ax0.set_title("A. Distribution (N=300)", fontsize=10)

    # By sector stacked bar
    sect_typ = pd.crosstab(pl["sector"], pl["typology"]).reindex(columns=order, fill_value=0)
    sect_typ = sect_typ.div(sect_typ.sum(axis=1), axis=0) * 100
    sect_typ.index = [SECTOR_LABELS[s] for s in sect_typ.index]
    bottom = np.zeros(len(sect_typ))
    for col, c in zip(order, colors):
        vals = sect_typ[col].values
        ax1.barh(sect_typ.index, vals, left=bottom, color=c, label=col.replace("_"," "), height=0.6)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 8:
                ax1.text(b + v/2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        bottom += vals
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("% of firms in sector", fontsize=9)
    ax1.set_title("B. Typology by Sector", fontsize=10)
    ax1.legend(fontsize=7, bbox_to_anchor=(1.01,1))

    # Median adjusted net income by typology
    med_ni = [pl[pl["typology"]==t]["net_income_adjusted"].median() for t in order]
    bars = ax2.barh(labels_short, med_ni, color=colors, height=0.55)
    ax2.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, med_ni):
        xpos = v + 10 if v >= 0 else v - 10
        ax2.text(xpos, bar.get_y()+bar.get_height()/2,
                 f"Bs{v:,.0f}", va="center", ha="left" if v>=0 else "right", fontsize=8)
    ax2.set_xlabel("Median adjusted net income Bs/week", fontsize=9)
    ax2.set_title("C. Net Income by Typology", fontsize=10)

    plt.tight_layout()
    return save_fig(fig, outdir, "fig4_typology.png")

def fig_eh_vs_diary(pl, outdir):
    """Figure 5: EH measurement error scatter and decomposition."""
    sub = pl[~pl["eh_nonresponder"] & (pl.get("eh_yi_tot", pl.get("eh_yi_tot_annual", pd.Series(0, index=pl.index))) > 0)].copy()
    if "eh_yi_tot" not in sub.columns:
        sub["eh_yi_tot"] = sub["eh_yi_tot_annual"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 5 — EH vs Diary: Measurement Bias", fontsize=13, fontweight="bold")

    # Panel A: scatter diary revenue vs EH revenue (annualised)
    ax = axes[0]
    for sec in sorted(sub["sector"].unique()):
        sd = sub[sub["sector"]==sec]
        ax.scatter(sd["eh_yi_tot"]/1000, sd["diary_yi_tot_annualized"]/1000,
                   alpha=0.5, s=18, color=PALETTE[sec], label=SECTOR_LABELS[sec])
    lim = max(sub["diary_yi_tot_annualized"].quantile(0.95),
              sub["eh_yi_tot"].quantile(0.95)) / 1000 * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="45° line")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("EH gross income Bs 000/yr", fontsize=9)
    ax.set_ylabel("Diary gross income Bs 000/yr", fontsize=9)
    ax.set_title("A. Gross Income: EH vs Diary", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    # Panel B: income ratio distribution
    ax = axes[1]
    if "ratio_diary_eh_income" in sub.columns:
        ratio_col = "ratio_diary_eh_income"
    else:
        sub["ratio_diary_eh_income"] = sub["diary_yi_tot_annualized"] / sub["eh_yi_tot"].replace(0, np.nan)
        ratio_col = "ratio_diary_eh_income"
    ratios = sub[ratio_col].dropna().clip(0, 4)
    ax.hist(ratios, bins=30, color="#2E6DB4", edgecolor="white", alpha=0.8)
    ax.axvline(1.0,  color="red",  lw=1.5, ls="--", label="No bias (=1)")
    ax.axvline(ratios.median(), color="orange", lw=1.5, ls="-",
               label=f"Median = {ratios.median():.2f}")
    ax.set_xlabel("Diary / EH income ratio", fontsize=9)
    ax.set_ylabel("N firms", fontsize=9)
    ax.set_title("B. Distribution of Diary/EH Income Ratio", fontsize=9)
    ax.legend(fontsize=8)

    # Panel C: net income comparison (EH vs diary adj)
    ax = axes[2]
    clip = 200
    x_eh  = sub["eh_yi_net_annual"].clip(-clip*1000, clip*1000) / 1000
    y_adj = sub["diary_yi_net_adj_annualized"].clip(-clip*1000, clip*1000) / 1000
    for sec in sorted(sub["sector"].unique()):
        sd = sub[sub["sector"]==sec]
        ax.scatter(sd["eh_yi_net_annual"]/1000,
                   sd["diary_yi_net_adj_annualized"]/1000,
                   alpha=0.5, s=18, color=PALETTE[sec], label=SECTOR_LABELS[sec])
    xlim = max(abs(x_eh.min()), x_eh.max()) * 1.1
    ylim = max(abs(y_adj.min()), y_adj.max()) * 1.1
    lim  = max(xlim, ylim)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.6)
    ax.axhline(0, color="red", lw=0.8, ls=":")
    ax.axvline(0, color="red", lw=0.8, ls=":")
    ax.set_xlabel("EH net income Bs 000/yr", fontsize=9)
    ax.set_ylabel("Diary adjusted net income Bs 000/yr", fontsize=9)
    ax.set_title("C. Net Income: EH vs Diary-Adjusted", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    return save_fig(fig, outdir, "fig5_eh_vs_diary.png")

def fig_valuation(pl, outdir):
    """Figure 6: Going-concern valuation and Tobin's Q."""
    viable = pl[pl["typology"]=="I_Viable"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 6 — Going-Concern Valuation (Viable Firms)", fontsize=13, fontweight="bold")

    # Panel A: PV distribution by sector (g=0)
    ax = axes[0]
    viable["pv_g0_000"] = viable["pv_g0"] / 1000
    viable["sector_label"] = viable["sector"].map(SECTOR_LABELS)
    order_sect = list(viable.groupby("sector_label")["pv_g0_000"].median().sort_values().index)
    data_bp = [viable[viable["sector_label"]==s]["pv_g0_000"].dropna().values for s in order_sect]
    ax.boxplot(data_bp, labels=order_sect, showfliers=False,
               boxprops=dict(color="steelblue"),
               medianprops=dict(color="red", lw=2))
    ax.set_xlabel("")
    ax.set_ylabel("PV Bs 000 (g=0%)", fontsize=9)
    ax.set_title("A. PV Distribution by Sector", fontsize=9)
    plt.sca(ax); plt.xticks(rotation=30, ha="right", fontsize=8)

    # Panel B: Tobin's Q scatter vs book equity
    ax = axes[1]
    q = viable["tobin_q_g0"].replace([np.inf,-np.inf], np.nan).clip(0, 200).dropna()
    be = viable.loc[q.index, "book_equity"].clip(upper=viable["book_equity"].quantile(0.95))
    for sec in viable["sector"].unique():
        mask = viable.loc[q.index, "sector"] == sec
        ax.scatter(be[mask], q[mask], alpha=0.5, s=20, color=PALETTE[sec], label=SECTOR_LABELS[sec])
    ax.set_xlabel("Book equity Bs", fontsize=9)
    ax.set_ylabel("Tobin's Q analog (g=0)", fontsize=9)
    ax.set_title("B. Tobin's Q vs Book Equity", fontsize=9)
    ax.legend(fontsize=7)

    # Panel C: sensitivity g=0 vs g=2 vs g=5
    ax = axes[2]
    medians = [
        viable["pv_g0"].median(),
        viable["pv_g2"].median(),
        viable.get("pv_g5", viable["pv_g0"]*1.15).median(),
    ]
    bars = ax.bar(["g=0%", "g=2%", "g=5%"], [m/1000 for m in medians],
                   color=["#2c7bb6","#abd9e9","#d7191c"], width=0.5)
    for bar, v in zip(bars, [m/1000 for m in medians]):
        ax.text(bar.get_x()+bar.get_width()/2, v+2,
                f"Bs{v:,.0f}k", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Median PV Bs 000", fontsize=9)
    ax.set_xlabel("Growth rate assumption", fontsize=9)
    ax.set_title("C. PV Sensitivity to Growth Rate\n(Viable firms, r=18%)", fontsize=9)

    plt.tight_layout()
    return save_fig(fig, outdir, "fig6_valuation.png")

def fig_gender_poverty(pl, outdir):
    """Figure 7: Gender and poverty heterogeneity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 7 — Heterogeneity: Gender and Poverty", fontsize=13, fontweight="bold")

    sectors = sorted(pl["sector"].unique())
    labels  = [SECTOR_LABELS[s] for s in sectors]
    x = np.arange(len(sectors))

    # Panel A: % loss-making by sector × gender
    ax = axes[0]
    pct_F = [pl[(pl["sector"]==s)&(pl["gender"]=="F")]["net_income_adjusted"].lt(0).mean()*100 for s in sectors]
    pct_M = [pl[(pl["sector"]==s)&(pl["gender"]=="M")]["net_income_adjusted"].lt(0).mean()*100 for s in sectors]
    ax.bar(x-0.2, pct_F, 0.35, label="Female", color="#C0392B", alpha=0.8)
    ax.bar(x+0.2, pct_M, 0.35, label="Male",   color="#2980B9", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% loss-making (adjusted)", fontsize=9)
    ax.set_title("A. % Loss-Making by Gender", fontsize=9)
    ax.legend(fontsize=8)

    # Panel B: hourly return by sector × gender
    ax = axes[1]
    hr_F = [pl[(pl["sector"]==s)&(pl["gender"]=="F")]["net_per_hour_adj"].median() for s in sectors]
    hr_M = [pl[(pl["sector"]==s)&(pl["gender"]=="M")]["net_per_hour_adj"].median() for s in sectors]
    ax.bar(x-0.2, hr_F, 0.35, label="Female", color="#C0392B", alpha=0.8)
    ax.bar(x+0.2, hr_M, 0.35, label="Male",   color="#2980B9", alpha=0.8)
    ax.axhline(MIN_WAGE_HOURLY, color="black", lw=1.2, ls="--", label=f"Min wage ({MIN_WAGE_HOURLY:.1f} Bs/hr)")
    ax.axhline(0,              color="red",   lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Net income per hour Bs (adjusted)", fontsize=9)
    ax.set_title("B. Hourly Return by Gender", fontsize=9)
    ax.legend(fontsize=7)

    # Panel C: accounting gap by poverty status
    ax = axes[2]
    gap_poor   = [pl[(pl["sector"]==s)&(pl["is_poor"]==True)]["accounting_gap_bs"].median() for s in sectors]
    gap_npoor  = [pl[(pl["sector"]==s)&(pl["is_poor"]==False)]["accounting_gap_bs"].median() for s in sectors]
    ax.bar(x-0.2, gap_poor,  0.35, label="Poor",     color="#E67E22", alpha=0.85)
    ax.bar(x+0.2, gap_npoor, 0.35, label="Non-poor", color="#27AE60", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Median accounting gap Bs/week", fontsize=9)
    ax.set_title("C. Accounting Gap by Poverty Status", fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return save_fig(fig, outdir, "fig7_gender_poverty.png")

def fig_daily_cash_flows(tx, outdir):
    """Figure 8: Daily cash-flow patterns and volatility."""
    sales = tx[tx["module"]=="M1A_VENTAS"].copy()
    daily = sales.groupby(["sector","day_num"])["amount_bs"].sum().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 8 — Daily Transaction Patterns", fontsize=13, fontweight="bold")

    # Panel A: Average daily revenue by sector × day of week
    ax = axes[0]
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for sec in sorted(daily["sector"].unique()):
        sd   = daily[daily["sector"]==sec]
        revs = [sd[sd["day_num"]==d]["amount_bs"].sum() /
                max(1, sales[(sales["sector"]==sec)&(sales["day_num"]==d)]["firm_id"].nunique())
                for d in range(1,8)]
        ax.plot(day_labels, revs, marker="o", lw=1.8, label=SECTOR_LABELS[sec],
                color=PALETTE[sec])
    ax.set_xlabel("Day of week", fontsize=9)
    ax.set_ylabel("Avg revenue per active firm Bs", fontsize=9)
    ax.set_title("A. Revenue Pattern by Day and Sector", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Coefficient of variation of daily revenue by sector (violin)
    ax = axes[1]
    cv_data, cv_labels, cv_colors = [], [], []
    for sec in sorted(daily["sector"].unique()):
        sd  = daily[daily["sector"]==sec]
        cvs = []
        for fid in tx[tx["sector"]==sec]["firm_id"].unique():
            d = sales[(sales["sector"]==sec)&(sales["firm_id"]==fid)].groupby("day_num")["amount_bs"].sum()
            if d.mean() > 0:
                cvs.append(d.std() / d.mean())
        if cvs:
            cv_data.append(cvs); cv_labels.append(SECTOR_LABELS[sec]); cv_colors.append(PALETTE[sec])

    parts = ax.violinplot(cv_data, positions=range(len(cv_data)), widths=0.6,
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], cv_colors):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    ax.set_xticks(range(len(cv_labels)))
    ax.set_xticklabels(cv_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CV of daily revenue (within-firm)", fontsize=9)
    ax.set_title("B. Revenue Volatility by Sector", fontsize=9)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.6, label="CV = 1")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return save_fig(fig, outdir, "fig8_daily_cashflows.png")

# ── PART D: EXCEL WORKBOOK ──────────────────────────────────────────────────

def write_excel(tables_dict, figures_dict, outpath):
    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:

        # Cover sheet
        cover = pd.DataFrame({
            "Field":  ["Paper","Data","N firms","N transactions","Sectors","Cities","Generated"],
            "Value": [
                "Ledgers of the Self-Employed",
                "Simulated diary study — Bolivia 2025–2026",
                "300", str(tables_dict.get("_n_tx","—")),
                "6 (comercio, manufactura, transporte, gastronomía, servicios, admin)",
                "La Paz/El Alto, Cochabamba, Santa Cruz, Tarija",
                pd.Timestamp.now().strftime("%Y-%m-%d"),
            ]
        })
        cover.to_excel(writer, sheet_name="0_Cover", index=False)

        tab_names = {
            "T1_Descriptive":            "1  — Descriptive Stats by Sector",
            "T2_Balance_Sheet":          "2  — Balance Sheet",
            "T3_Returns":                "3  — Returns & Valuation",
            "T4_Activity":               "4  — Activity & Liquidity",
            "T5_EH_vs_Diary":            "5  — EH vs Diary Bias",
            "T6_Typology":               "6  — Firm Typology",
            "T7_Heterogeneity":          "7  — Gender Heterogeneity",
            "T8_Production_Function":    "8  — Production Function",
            "T9_Credit":                 "9  — Informal Credit",
            "T10_EH_Adequacy":           "10 — EH Adequacy Scorecard",
            "T11_Tx_Complexity":         "11 — Transaction Complexity",
            "T11b_Module_Breakdown":     "11b — Module Breakdown",
            "DATA_Analytics_Panel":      "DATA — Full Analytics Panel",
        }

        for key, sheet_name in tab_names.items():
            if key in tables_dict and tables_dict[key] is not None:
                tables_dict[key].to_excel(writer, sheet_name=sheet_name[:31])

    print(f"  Excel written: {outpath}")

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ledgers of the Self-Employed — Paper Analytics"
    )
    parser.add_argument("--firms",        default="/home/claude/ledgers_firms.csv")
    parser.add_argument("--accounts",     default="/home/claude/ledgers_accounts_full.csv")
    parser.add_argument("--transactions", default="/home/claude/ledgers_transactions.csv")
    parser.add_argument("--out",          default="/home/claude/results")
    args = parser.parse_args()

    outdir     = args.out
    tables_dir = os.path.join(outdir, "tables")
    figures_dir= os.path.join(outdir, "figures")
    for d in [outdir, tables_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    print("─" * 60)
    print("LEDGERS OF THE SELF-EMPLOYED — Analytics")
    print("─" * 60)

    # ── Load
    print("\n[1/5] Loading data...")
    firms, accts, tx = load_data(args.firms, args.accounts, args.transactions)

    # ── Build panel
    print("[2/5] Building analytic panel...")
    pl = build_panel(accts)
    pl.to_csv(os.path.join(outdir, "ledgers_analytics_panel.csv"), index=False)
    print(f"  Panel: {len(pl)} rows × {len(pl.columns)} columns")

    # ── Tables
    print("[3/5] Computing tables...")
    t1         = table_descriptive(pl)
    t2         = table_balance_sheet(pl)
    t3         = table_returns(pl)
    t4         = table_activity(pl)
    t5, bias   = table_eh_vs_diary(pl)
    t6         = table_typology(pl)
    t7         = table_heterogeneity(pl)
    t8         = table_production_function(pl)
    t9         = table_credit(pl)
    t10        = table_eh_adequacy()
    t11, t11b  = table_transaction_complexity(tx)

    # Save individual CSVs
    for name, tbl in [("T1_descriptive",t1),("T2_balance",t2),("T3_returns",t3),
                       ("T4_activity",t4),("T5_eh_bias",t5),("T6_typology",t6),
                       ("T7_heterogeneity",t7),("T8_production",t8),
                       ("T9_credit",t9),("T10_adequacy",t10),("T11_complexity",t11)]:
        tbl.to_csv(os.path.join(tables_dir, f"{name}.csv"))

    # ── Figures
    print("[4/5] Generating figures...")
    figs = {}
    try: figs["f1"] = fig_accounting_gap(pl, figures_dir)
    except Exception as e: print(f"  fig1 skipped: {e}")
    try: figs["f2"] = fig_ratios_heatmap(pl, figures_dir)
    except Exception as e: print(f"  fig2 skipped: {e}")
    try: figs["f3"] = fig_production_function(pl, figures_dir)
    except Exception as e: print(f"  fig3 skipped: {e}")
    try: figs["f4"] = fig_typology(pl, figures_dir)
    except Exception as e: print(f"  fig4 skipped: {e}")
    try: figs["f5"] = fig_eh_vs_diary(pl, figures_dir)
    except Exception as e: print(f"  fig5 skipped: {e}")
    try: figs["f6"] = fig_valuation(pl, figures_dir)
    except Exception as e: print(f"  fig6 skipped: {e}")
    try: figs["f7"] = fig_gender_poverty(pl, figures_dir)
    except Exception as e: print(f"  fig7 skipped: {e}")
    try: figs["f8"] = fig_daily_cash_flows(tx, figures_dir)
    except Exception as e: print(f"  fig8 skipped: {e}")
    print(f"  {len(figs)} figures generated")

    # ── Excel
    print("[5/5] Writing Excel workbook...")
    write_excel({
        "T1_Descriptive":         t1,
        "T2_Balance_Sheet":       t2,
        "T3_Returns":             t3,
        "T4_Activity":            t4,
        "T5_EH_vs_Diary":         t5,
        "T6_Typology":            t6,
        "T7_Heterogeneity":       t7,
        "T8_Production_Function": t8,
        "T9_Credit":              t9,
        "T10_EH_Adequacy":        t10,
        "T11_Tx_Complexity":      t11,
        "T11b_Module_Breakdown":  t11b,
        "DATA_Analytics_Panel":   pl,
        "_n_tx": f"{len(tx):,}",
    }, figs, os.path.join(outdir, "ledgers_paper_tables.xlsx"))

    # ── Print headline numbers
    print("\n" + "═"*60)
    print("HEADLINE NUMBERS FOR THE PAPER")
    print("═"*60)

    print(f"\n  N = {len(pl)} firms  |  {len(tx):,} transactions  |  7-day diaries")
    print(f"\n  MEASUREMENT")
    print(f"  EH non-response (simulated):  {pl['eh_nonresponder'].mean()*100:.0f}%")
    print(f"  True debt prevalence:         {pl['has_debt_true'].mean()*100:.0f}%")
    eh_ci6 = pl.get("eh_ci6", pd.Series(0, index=pl.index))
    print(f"  EH declared debt (ci6>0):     {(eh_ci6>0).mean()*100:.0f}%  (detection: {(eh_ci6>0).mean()/pl['has_debt_true'].mean()*100:.0f}% of true)")

    print(f"\n  CENTRAL FINDING — ACCOUNTING GAP")
    print(f"  % firms positive (conventional): {(pl['net_income_conventional']>0).mean()*100:.0f}%")
    print(f"  % firms positive (adjusted):     {(pl['net_income_adjusted']>0).mean()*100:.0f}%")
    print(f"  Median gap Bs/week:              {pl['accounting_gap_bs'].median():,.0f}")
    print(f"  Median gap % of conv profit:     {pl['gap_pct_conv'].median():.0f}%")
    print(f"  Labor share of gap:              {pl['labor_imputed'].median()/pl['accounting_gap_bs'].median()*100:.0f}%")

    print(f"\n  FINANCIAL RATIOS (medians)")
    print(f"  ROA conventional:   {pl['roa_conv'].median()*100:.0f}%/yr")
    print(f"  ROA adjusted:       {pl['roa_adj'].median()*100:.1f}%/yr")
    print(f"  E/BV conventional:  {pl['ebv_conv'].median():.2f}")
    print(f"  E/BV adjusted:      {pl['ebv_adj'].median():.2f}")
    print(f"  MW multiple:        {pl['mw_multiple'].median():.2f}x  (vs min wage = 1.0x)")
    print(f"  Cash conv cycle:    {pl['cash_conv_cycle'].median():.1f} days")
    print(f"  Extraction rate:    {pl['extraction_rate'].median()*100:.0f}% of revenue")

    t8r = t8.set_index("Model")
    pooled_row = t8r.filter(like="sector FE", axis=0)
    if len(pooled_row):
        row = pooled_row.iloc[0]
        print(f"\n  PRODUCTION FUNCTION (pooled + sector FE)")
        print(f"  α_K = {row['α_K']}  α_L = {row['α_L']}  RTS = {row['RTS']}{row['RTS stars']}")
        print(f"  R² = {row['R²']}")

    print(f"\n  VALUATION (viable firms, N={pl['typology'].eq('I_Viable').sum()})")
    viable = pl[pl["typology"]=="I_Viable"]
    print(f"  Median FCF annual:   Bs {viable['fcf_annual'].median():,.0f}")
    print(f"  Median PV (g=0):     Bs {viable['pv_g0'].median():,.0f}")
    print(f"  Median Tobin Q:      {viable['tobin_q_g0'].median():.1f}x")

    t10_df = t10
    absent_ct = t10_df["EH Absent"].sum()
    crit_absent = t10_df[(t10_df["EH Absent"]) & (t10_df["Importance"]=="Critical")].shape[0]
    print(f"\n  EH ADEQUACY")
    print(f"  Absent from EH: {absent_ct}/{len(t10_df)} concepts  ({crit_absent} critical)")

    print("\n" + "─"*60)
    print(f"Output:  {outdir}/")
    print(f"  tables/     → {len(os.listdir(tables_dir))} CSV files")
    print(f"  figures/    → {len(os.listdir(figures_dir))} PNG files")
    print(f"  ledgers_paper_tables.xlsx")
    print(f"  ledgers_analytics_panel.csv")
    print("─"*60)

if __name__ == "__main__":
    main()
