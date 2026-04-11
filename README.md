# Bolivia Informality Research
## Financial Diary Data & the Cash Flow Trap

[![Dashboard](https://img.shields.io/badge/Dashboard-Live-red)](https://wernerhl.github.io/ledgers-bolivia)
[![Dataverse](https://img.shields.io/badge/Harvard-Dataverse-blue)](https://doi.org/10.7910/DVN/KZMHGU)

> *"43 percent of informal firms are loss-making after imputing owner labor at shadow wage—yet 97% have positive cash flow. The 'cash flow veil' explains persistence: owners observe cash, not economic profit."*

---

## 📊 Key Findings

| Statistic | Value |
|-----------|-------|
| Firms | 500 |
| Transactions | 25,338 |
| Loss-making (π_adj ≤ 0) | 42.8% |
| Type C (trapped) share of loss-makers | 70.1% |
| ALMP mistargeting ratio | 2.96:1 |
| Median accounting gap | Bs 1,002/week |

---

## 📄 Papers

### Ledgers of the Self-Employed: Accounting for the Invisible Firm
*Measurement Companion*

Using a seven-day double-entry financial diary, we construct complete income statements, balance sheets, and going-concern valuations for 500 informal firms. The median accounting gap—Bs 1,002/week—is driven 88% by imputed labor.

- [Paper (EN)](https://wernerhl.github.io/ledgers-bolivia/docs/ledgers_v2.pdf) · [Paper (ES)](https://wernerhl.github.io/ledgers-bolivia/docs/ledgers_v2_es.pdf)
- [Slides (EN)](https://wernerhl.github.io/ledgers-bolivia/docs/presentation_en.pdf) · [Slides (ES)](https://wernerhl.github.io/ledgers-bolivia/docs/presentation_es.pdf)

### Trapped by Cash: A Theory of Voluntary Informality Among the Economically Non-Viable
*Theory + Empirics*

Why do loss-making informal firms persist? Standard models assume rationing. This paper proposes a different mechanism: the cash flow veil. Owners observe cash flow, not economic profit. Among loss-makers, 70% would decline a formal job offer because CF ≥ w*.

- [Paper](https://wernerhl.github.io/ledgers-bolivia/docs/trapped_by_cash.pdf)
- [Slides (60 min)](https://wernerhl.github.io/ledgers-bolivia/docs/trapped_slides.pdf)

---

## 🔍 Interactive Dashboard

**[→ Open Dashboard](https://wernerhl.github.io/ledgers-bolivia)**

Explore both firm typologies:

| Ledgers: Viability × Debt | Trapped: Viability × Cash Flow |
|---------------------------|-------------------------------|
| I — Viable (55.6%) | A — Viable (56.8%) |
| II — Precaria (34.4%) | B — Rationed (0.4%) |
| III — Debt-trapped (1.6%) | C — Trapped (30.0%) |
| IV — Critical (8.4%) | D — ALMP Target (12.8%) |

---

## 📊 Data & Replication

| Source | Link |
|--------|------|
| Harvard Dataverse | [doi.org/10.7910/DVN/KZMHGU](https://doi.org/10.7910/DVN/KZMHGU) |
| GitHub | [github.com/wernerhl/ledgers-bolivia](https://github.com/wernerhl/ledgers-bolivia) |

### Files

| File | Description | Rows |
|------|-------------|------|
| `data/ledgers_firms.csv` | Firm-level data | 500 |
| `data/ledgers_transactions.csv` | Transaction-level records | 25,338 |
| `data/ledgers_analytics.csv` | Full analytics panel | 500 |

### Key Variables

| Variable | Description |
|----------|-------------|
| `net_income_conventional` | Cash flow (π_conv) |
| `net_income_adjusted` | Economic profit (π_adj) |
| `accounting_gap_bs` | Gap = π_conv − π_adj |
| `labor_imputed` | Owner + family labor × shadow wage |
| `typology_ledgers` | I/II/III/IV (viability × debt) |
| `typology_trapped` | A/B/C/D (viability × cash flow) |

---

## 📁 Repository Structure

```
ledgers-bolivia/
├── index.html              # Interactive dashboard
├── docs/
│   ├── ledgers_v2.pdf      # Ledgers paper (EN)
│   ├── ledgers_v2_es.pdf   # Ledgers paper (ES)
│   ├── presentation_en.pdf # Ledgers slides (EN)
│   ├── presentation_es.pdf # Ledgers slides (ES)
│   ├── trapped_by_cash.pdf # Trapped paper
│   └── trapped_slides.pdf  # Trapped slides
├── data/
│   ├── ledgers_firms.csv
│   ├── ledgers_transactions.csv
│   └── ledgers_analytics.csv
├── code/
│   ├── ledgers_analytics.py
│   └── paper_illustrations.py
└── README.md
```

---

## 📐 Methodology

### The Diary Protocol
Seven-day double-entry financial diary administered to 500 informal owner-operated firms across six sectors (Commerce, Transport, Manufacturing, Food Service, Construction, Professional Services) and four cities (La Paz/El Alto, Cochabamba, Santa Cruz, Tarija). September–October 2024.

### Accounting Gap Decomposition

| Component | Share of Gap |
|-----------|-------------|
| Imputed owner labor | 61% |
| Imputed family labor | 31% |
| Depreciation | 12% |
| **Total** | 100% |

Shadow wage: Bs 13.64/hour (Mincer regression on formal sector)

---

## 📖 Citation

```bibtex
@techreport{HernaniLimarino2026Ledgers,
  author  = {Hernani-Limarino, Werner},
  title   = {Ledgers of the Self-Employed: Accounting for the Invisible Firm},
  year    = {2026},
  type    = {Working Paper},
  note    = {Data: https://doi.org/10.7910/DVN/KZMHGU}
}

@techreport{HernaniLimarino2026Trapped,
  author  = {Hernani-Limarino, Werner},
  title   = {Trapped by Cash: A Theory of Voluntary Informality Among the Economically Non-Viable},
  year    = {2026},
  type    = {Working Paper},
  note    = {Data: https://doi.org/10.7910/DVN/KZMHGU}
}
```

---

## 📬 Contact

Werner Hernani-Limarino · [wernerhl@gmail.com](mailto:wernerhl@gmail.com) · [wernerhl.github.io](https://wernerhl.github.io)

*The opinions expressed do not reflect the position of affiliated institutions. All errors are the author's.*
