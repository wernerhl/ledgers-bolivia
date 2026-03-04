# Ledgers of the Self-Employed
## Accounting for the Invisible Firm — Bolivia 2025

[![Dashboard](https://img.shields.io/badge/Dashboard-Live-red)](https://wernerhl.github.io/ledgers-bolivia)
[![Paper](https://img.shields.io/badge/Paper-Working%20Draft-blue)](https://github.com/wernerhl/ledgers-bolivia/raw/main/paper/ledgers_paper.pdf)

> *"Standard household surveys report positive profits for virtually every self-employed worker in the developing world. Using a seven-day financial diary with full double-entry accounting administered to 300 informal firms across four Bolivian cities, we show that 57 percent are loss-making once owner labour, depreciation, and informal debt are correctly accounted for."*
> 
> — Hernani-Limarino (2026)

---

## 🔍 Interactive Dashboard

**[→ Open Dashboard](https://wernerhl.github.io/ledgers-bolivia)**

The dashboard covers 8 analytical modules:

| § | Module | What you can explore |
|---|--------|---------------------|
| 1 | Headline statistics | Key aggregates, filterable by sector/city/gender/typology |
| 2 | The Profit Reversal | Distribution shift, sector comparisons, waterfall decomposition |
| 3 | Firm Map | Every firm in profit × debt-burden space, color-coded by typology |
| 4 | Persistence Mechanisms | Why loss-making firms survive: cash flow, hourly returns, household draws |
| 5 | Survey vs Diary | EH vs diary income, bias direction by sector |
| 6 | Financial Ratios | ROA, Tobin's Q, effective annual interest rates by lender type |
| 7 | Tax Simulator | Slide the IVA rate and see how many firms are pushed into loss |
| 8 | Firm Explorer | Sortable table of all 300 firms |

---

## 📊 Data

All data are **simulated** to match the statistical properties of a seven-day financial diary study. Real data collection is ongoing. The simulation preserves:
- Sector and city stratification
- True debt prevalence (46%) vs EH-declared (3%)
- Labour imputation structure (owner hours + family hours × shadow wage)
- Household-firm boundary flows (Module 7)
- Dual-currency exposure (88% of firms, 7.1% of sales in USD)

### Files

| File | Description | Rows | Cols |
|------|-------------|------|------|
| `data/ledgers_dashboard_data.csv` | Firm-level panel (key variables) | 300 | 56 |
| `data/ledgers_analytics.csv` | Full analytics panel (all variables) | 300 | 142 |
| `data/ledgers_transactions.csv` | Transaction-level records | 14,902 | 18 |
| `data/ledgers_firms.csv` | Firm metadata | 300 | 15 |

### Key Variables

| Variable | Description |
|----------|-------------|
| `net_income_conventional` | Survey-implied weekly profit (no labour/depreciation adjustment) |
| `net_income_adjusted` | Diary-adjusted profit after A1–A4 entries |
| `accounting_gap_bs` | Gap = conventional − adjusted (always ≥ 0) |
| `labor_imputed` | Owner + family labour × shadow wage (Bs 13.64/hr) |
| `cash_flow_operating` | Operating cash flow (always positive) |
| `typology` | I\_Viable / II\_Precaria / III\_Atrapada\_deuda / IV\_Riesgo\_critico |
| `tobin_q_g0` | Going-concern PV / book assets (viable firms, g=0) |
| `effective_annual_rate` | True EAR on debt (IFD ~25%, prestamista ~201%) |
| `mw_multiple` | Hourly adjusted return / minimum wage (< 1 = below min. wage) |
| `household_draws` | Weekly household extraction from firm cash |

---

## 📁 Repository Structure

```
ledgers-bolivia/
├── index.html              # Interactive dashboard (GitHub Pages)
├── data/
│   ├── ledgers_dashboard_data.csv
│   ├── ledgers_analytics.csv
│   ├── ledgers_transactions.csv
│   └── ledgers_firms.csv
├── code/
│   ├── simulate_diary.py   # Data simulation engine (300 firms × 7 days)
│   ├── ledgers_analytics.py # Full analytics pipeline
│   └── paper_illustrations.py # Publication figures
├── paper/
│   └── ledgers_paper.pdf   # Working paper
└── README.md
```

---

## 🛠️ Run Locally

```bash
git clone https://github.com/wernerhl/ledgers-bolivia
cd ledgers-bolivia
python -m http.server 8000
# open http://localhost:8000
```

To reproduce the analytics:
```bash
pip install pandas numpy matplotlib scipy
python code/simulate_diary.py      # generates raw data
python code/ledgers_analytics.py   # computes all firm-level metrics
python code/paper_illustrations.py # generates paper figures
```

---

## 📐 Methodology

### The Diary Protocol
Seven-day financial diary with nine modules administered daily by trained enumerators. All monetary amounts in bolivianos (Bs); USD transactions additionally recorded with P2P exchange rate.

### Double-Entry Accounting
Every transaction coded with debit and credit account from a structured chart of accounts. Four adjustment entries (A1–A4) applied ex post:

| Entry | Item | Basis |
|-------|------|-------|
| A1 | Depreciation | Asset value ÷ (52 × useful life years) |
| A2 | Owner labour | Hours × min(min. wage, WTA elicited) |
| A3 | Family labour | Hours × 0.85 × shadow wage |
| A4 | Own-consumption | Replacement cost of household draws from inventory |

### Typology
| Type | Condition | N | % |
|------|-----------|---|---|
| I — Viable | π_adj > 0, interest burden < 10% | 126 | 42% |
| II — Precaria | π_adj ≤ 0, interest burden < 10% | 147 | 49% |
| III — Debt-trapped | π_adj > 0, interest burden ≥ 10% | 3 | 1% |
| IV — Critical risk | π_adj ≤ 0, interest burden ≥ 10% | 24 | 8% |

---

## 📖 Citation

```bibtex
@techreport{HernaniLimarino2026,
  author  = {Hernani-Limarino, Werner},
  title   = {Ledgers of the Self-Employed: Accounting for the Invisible Firm},
  year    = {2026},
  institution = {IA Analytics},
  type    = {Working Paper},
  note    = {Available at: https://github.com/wernerhl/ledgers-bolivia}
}
```

---

## 📬 Contact

Werner Hernani-Limarino · [wernerhl@gmail.com](mailto:wernerhl@gmail.com)

*The opinions expressed in this paper do not reflect the position of any affiliated institution. All errors are the author's.*
