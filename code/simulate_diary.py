"""
LEDGERS OF THE SELF-EMPLOYED — Bolivia Diary Study
Simulation of 300 informal firms × 7 days of diary data

Design principles:
  1. Sector-specific cost economics grounded in Bolivia field knowledge
  2. Realistic daily income variance (NOT smooth — lumpy, seasonal, weather-dependent)
  3. Dual-currency transactions (Bs vs USD at P2P rate)
  4. Informal credit with realistic interest rates (prestamistas: 5-15%/month)
  5. Anchored EH baseline replicating the heaping/recall bias observed in the data
  6. Family labor embedded structurally (not as explicit cost)
  7. Inventory dynamics with perishable and non-perishable goods
  8. Household-firm boundary crossings (retiros, aportes)
  9. Debt under-reporting in EH baseline (6% declared vs ~45% true prevalence)

Output files:
  - firms.csv         : 300-row firm-level summary
  - transactions.csv  : daily transaction ledger (~7,000+ rows)
  - weekly_accounts.csv: weekly accounting statements per firm
  - eh_baseline.csv   : simulated EH questionnaire responses (with biases)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

rng = np.random.default_rng(seed=42)

# ============================================================
# CONSTANTS
# ============================================================

P2P_RATE = 9.5          # Bs per USD (2025 P2P market rate)
MIN_WAGE_MONTHLY = 2362  # Bs/month (2024)
MIN_WAGE_HOURLY = MIN_WAGE_MONTHLY / (4.33 * 8 * 5)  # ~13.6 Bs/hr (8hrs x 5days x 4.33 weeks)

SECTORS = {
    'comercio':     {'label': 'Comercio y reparaciones',          'n': 90,  'pct_female': 0.65, 'pct_poor': 0.28},
    'manufactura':  {'label': 'Industria manufacturera',          'n': 55,  'pct_female': 0.44, 'pct_poor': 0.38},
    'transporte':   {'label': 'Transporte y almacenamiento',      'n': 55,  'pct_female': 0.02, 'pct_poor': 0.22},
    'gastronomia':  {'label': 'Alojamiento y gastronomía',        'n': 55,  'pct_female': 0.87, 'pct_poor': 0.32},
    'servicios':    {'label': 'Otras actividades de servicios',   'n': 30,  'pct_female': 0.49, 'pct_poor': 0.25},
    'admin':        {'label': 'Servicios administrativos',        'n': 15,  'pct_female': 0.69, 'pct_poor': 0.18},
}

CITIES = {
    'La Paz':     {'n_share': 0.35, 'altitude_factor': 1.0},
    'Cochabamba': {'n_share': 0.28, 'altitude_factor': 0.95},
    'Santa Cruz': {'n_share': 0.27, 'altitude_factor': 0.90},
    'Tarija':     {'n_share': 0.10, 'altitude_factor': 0.92},
}

# ============================================================
# SECTOR PARAMETERS
# For each sector: (weekly_revenue_mean, weekly_revenue_cv,
#                   cogs_share, rent_share, transport_share,
#                   utilities_share, wages_share,
#                   has_inventory, perishable, input_currency)
# All shares are fractions of gross revenue
# ============================================================

SECTOR_PARAMS = {
    'comercio': {
        'rev_weekly_mean': 2200,   # Bs/week gross
        'rev_weekly_cv': 0.55,     # high variance — feria days vs. slow days
        'cogs_share': 0.55,        # main cost: merchandise
        'rent_share': 0.06,
        'transport_share': 0.04,
        'utilities_share': 0.01,
        'wages_share': 0.02,       # mostly family labor (unpaid)
        'guildfees_share': 0.01,
        'has_inventory': True,
        'perishable_frac': 0.30,   # 30% of inventory is perishable
        'usd_input_frac': 0.25,    # 25% of inputs priced in USD
        'debt_true_prev': 0.45,
        'informal_interest_monthly': 0.08,
        'owner_hours_weekly': 54,
        'family_workers_mean': 1.4,
        'family_hours_each': 30,
        'days_active': 6,           # 6 days/week (Sunday rest)
        'rev_lumpiness': 0.35,      # fraction of weekly rev on best day
        'inventory_weeks': 1.5,     # weeks of stock held
    },
    'manufactura': {
        'rev_weekly_mean': 1900,
        'rev_weekly_cv': 0.65,     # lumpy: orders arrive irregularly
        'cogs_share': 0.45,        # raw materials
        'rent_share': 0.05,
        'transport_share': 0.03,
        'utilities_share': 0.03,
        'wages_share': 0.04,
        'guildfees_share': 0.01,
        'has_inventory': True,
        'perishable_frac': 0.05,
        'usd_input_frac': 0.40,    # many raw materials imported
        'debt_true_prev': 0.50,
        'informal_interest_monthly': 0.07,
        'owner_hours_weekly': 52,
        'family_workers_mean': 1.2,
        'family_hours_each': 20,
        'days_active': 6,
        'rev_lumpiness': 0.45,     # irregular orders → lumpier
        'inventory_weeks': 2.0,
    },
    'transporte': {
        'rev_weekly_mean': 1500,
        'rev_weekly_cv': 0.35,     # more predictable (routes, fares fixed)
        'cogs_share': 0.28,        # fuel + maintenance
        'rent_share': 0.02,        # terminal fee
        'transport_share': 0.00,
        'utilities_share': 0.01,
        'wages_share': 0.03,       # co-driver sometimes
        'guildfees_share': 0.03,   # sindicato transportistas
        'has_inventory': False,
        'perishable_frac': 0.0,
        'usd_input_frac': 0.60,    # fuel & parts often USD-priced
        'debt_true_prev': 0.60,    # vehicle loans very common
        'informal_interest_monthly': 0.06,
        'owner_hours_weekly': 60,  # long days
        'family_workers_mean': 0.2,
        'family_hours_each': 5,
        'days_active': 6,
        'rev_lumpiness': 0.20,     # daily routes → less lumpy
        'inventory_weeks': 0,
    },
    'gastronomia': {
        'rev_weekly_mean': 1700,
        'rev_weekly_cv': 0.45,
        'cogs_share': 0.42,        # ingredients (highly perishable)
        'rent_share': 0.07,
        'transport_share': 0.03,
        'utilities_share': 0.04,   # gas, water, electricity
        'wages_share': 0.03,
        'guildfees_share': 0.01,
        'has_inventory': True,
        'perishable_frac': 0.90,   # almost all perishable
        'usd_input_frac': 0.15,
        'debt_true_prev': 0.40,
        'informal_interest_monthly': 0.08,
        'owner_hours_weekly': 65,  # starts at 4am
        'family_workers_mean': 1.8,
        'family_hours_each': 35,
        'days_active': 6,
        'rev_lumpiness': 0.22,
        'inventory_weeks': 0.3,    # buy daily
    },
    'servicios': {
        'rev_weekly_mean': 1100,
        'rev_weekly_cv': 0.70,     # very lumpy — client-by-client
        'cogs_share': 0.12,        # mostly labor
        'rent_share': 0.08,
        'transport_share': 0.05,
        'utilities_share': 0.02,
        'wages_share': 0.02,
        'guildfees_share': 0.01,
        'has_inventory': False,
        'perishable_frac': 0.0,
        'usd_input_frac': 0.10,
        'debt_true_prev': 0.30,
        'informal_interest_monthly': 0.09,
        'owner_hours_weekly': 48,
        'family_workers_mean': 0.5,
        'family_hours_each': 10,
        'days_active': 5,
        'rev_lumpiness': 0.40,
        'inventory_weeks': 0,
    },
    'admin': {
        'rev_weekly_mean': 700,
        'rev_weekly_cv': 0.80,     # most unpredictable
        'cogs_share': 0.05,
        'rent_share': 0.10,
        'transport_share': 0.08,
        'utilities_share': 0.03,
        'wages_share': 0.00,
        'guildfees_share': 0.01,
        'has_inventory': False,
        'perishable_frac': 0.0,
        'usd_input_frac': 0.05,
        'debt_true_prev': 0.25,
        'informal_interest_monthly': 0.10,
        'owner_hours_weekly': 45,
        'family_workers_mean': 0.3,
        'family_hours_each': 8,
        'days_active': 5,
        'rev_lumpiness': 0.50,
        'inventory_weeks': 0,
    },
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def lognormal_params(mean, cv):
    """Convert mean and CV to lognormal mu, sigma."""
    sigma2 = np.log(1 + cv**2)
    mu = np.log(mean) - 0.5 * sigma2
    return mu, np.sqrt(sigma2)

def round_to_anchor(x, anchors=[50, 100, 200, 500, 1000]):
    """Simulate EH recall: round to nearest anchor."""
    diffs = [abs(x - a * round(x/a)) for a in anchors]
    best_anchor = anchors[np.argmin(diffs)]
    return best_anchor * round(x / best_anchor)

def weekly_to_eh_annual(weekly_mean, weekly_cv, n_weeks=52):
    """
    Simulate EH anchoring bias:
    - Respondent thinks of a 'typical' week
    - Rounds to a socially salient number
    - Multiply by 52
    - Under-reports by ~20% on average (social desirability)
    """
    typical_week = weekly_mean * rng.uniform(0.75, 0.95)  # underestimate
    # Round to nearest 50 or 100
    if typical_week < 500:
        anchor = 50
    elif typical_week < 2000:
        anchor = 100
    else:
        anchor = 500
    anchored = anchor * round(typical_week / anchor)
    return anchored * 52

def generate_daily_revenue(sector, p, n_days=7):
    """
    Generate 7-day revenue sequence for a firm.
    Captures: lumpiness, day-of-week effects, occasional zero days.
    """
    sp = SECTOR_PARAMS[sector]
    weekly_target = p['rev_weekly']
    days_active = sp['days_active']
    lump = sp['rev_lumpiness']
    
    # Day weights: lumpiness on market days (typically Sat/Sun for commerce)
    if sector == 'comercio':
        day_weights = np.array([0.12, 0.13, 0.13, 0.14, 0.16, lump, 0.0])  # Sun = market day, Mon rest
        day_weights = day_weights / day_weights.sum()
    elif sector == 'transporte':
        day_weights = np.array([0.18, 0.17, 0.16, 0.17, 0.18, 0.14, 0.0])
        day_weights = day_weights / day_weights.sum()
    elif sector == 'manufactura':
        # Orders come Monday and Thursday typically
        day_weights = np.array([lump*0.6, 0.12, 0.12, lump*0.4, 0.16, 0.12, 0.0])
        day_weights = day_weights / day_weights.sum()
    elif sector == 'gastronomia':
        day_weights = np.array([0.13, 0.12, 0.13, 0.14, 0.17, lump, 0.0])
        day_weights = day_weights / day_weights.sum()
    else:
        base = (1 - lump) / (days_active - 1)
        day_weights = np.array([base]*5 + [lump, 0.0])
        day_weights = day_weights / day_weights.sum()
    
    # Generate with noise around expected daily amounts
    expected_daily = weekly_target * day_weights
    revenues = np.zeros(7)
    for d in range(7):
        if day_weights[d] > 0:
            # Add multiplicative noise
            noise = rng.lognormal(0, 0.30)
            revenues[d] = max(0, expected_daily[d] * noise)
            # Occasional very bad day (0-20% of normal)
            if rng.random() < 0.08:
                revenues[d] *= rng.uniform(0, 0.2)
            # Occasional windfall (140-250% of normal)
            if rng.random() < 0.04:
                revenues[d] *= rng.uniform(1.4, 2.5)
    
    return revenues

# ============================================================
# GENERATE 300 FIRMS
# ============================================================

def generate_firms():
    firms = []
    firm_id = 1
    
    # Assign cities proportionally
    city_list = []
    for city, cfg in CITIES.items():
        n = round(300 * cfg['n_share'])
        city_list.extend([city] * n)
    city_list = city_list[:300]
    rng.shuffle(city_list)
    
    city_idx = 0
    for sector, scfg in SECTORS.items():
        sp = SECTOR_PARAMS[sector]
        n_firms = scfg['n']
        
        for i in range(n_firms):
            city = city_list[city_idx % len(city_list)]
            city_idx += 1
            
            # Demographics
            is_female = rng.random() < scfg['pct_female']
            is_poor = rng.random() < scfg['pct_poor']
            age = int(rng.normal(41, 12))
            age = max(18, min(70, age))
            
            # Firm age and capital base
            years_op = int(rng.exponential(6)) + 1
            years_op = min(years_op, 35)
            
            # Weekly revenue (true)
            mu, sig = lognormal_params(sp['rev_weekly_mean'], sp['rev_weekly_cv'])
            # Female premium in food, male in transport
            gender_adj = 1.0
            if sector == 'gastronomia' and is_female:
                gender_adj = 1.05
            elif sector == 'transporte' and not is_female:
                gender_adj = 1.08
            
            rev_weekly_true = rng.lognormal(mu, sig) * gender_adj
            
            # Asset base
            tool_value = rev_weekly_true * rng.uniform(8, 40)
            if sector == 'transporte':
                tool_value = rng.lognormal(*lognormal_params(45000, 0.8))
            elif sector == 'manufactura':
                tool_value = rng.lognormal(*lognormal_params(8000, 0.6))
            
            # Inventory initial (if applicable)
            inventory_init = 0
            if sp['has_inventory']:
                inventory_init = rev_weekly_true * sp['cogs_share'] * sp['inventory_weeks']
                inventory_init *= rng.uniform(0.7, 1.3)
            
            # Receivables
            receivables_init = 0
            if sector in ['manufactura', 'comercio']:
                receivables_init = rev_weekly_true * rng.uniform(0, 0.4)
            
            # Payables (supplier credit)
            payables_init = 0
            if sp['has_inventory']:
                payables_init = inventory_init * rng.uniform(0, 0.35)
            
            # DEBT — true prevalence much higher than EH captures
            has_debt_true = rng.random() < sp['debt_true_prev']
            debt_amount = 0
            debt_type = 'ninguno'
            monthly_interest_rate = 0
            monthly_installment = 0
            
            if has_debt_true:
                mu_d, sig_d = lognormal_params(8000, 1.0)
                debt_amount = rng.lognormal(mu_d, sig_d)
                if rng.random() < 0.30:
                    debt_type = 'IFD/cooperativa'
                    monthly_interest_rate = rng.uniform(0.015, 0.025)
                elif rng.random() < 0.50:
                    debt_type = 'prestamista_informal'
                    monthly_interest_rate = rng.uniform(0.05, 0.15)
                elif rng.random() < 0.70:
                    debt_type = 'familiar'
                    monthly_interest_rate = 0.0
                else:
                    debt_type = 'proveedor'
                    monthly_interest_rate = rng.uniform(0.02, 0.05)
                
                weekly_interest = debt_amount * monthly_interest_rate / 4.33
                monthly_installment = debt_amount * (monthly_interest_rate / (1 - (1 + monthly_interest_rate)**(-12))) if monthly_interest_rate > 0 else debt_amount / 12
                weekly_installment = monthly_installment / 4.33
            else:
                weekly_installment = 0
                weekly_interest = 0
            
            # Labor
            owner_hours = sp['owner_hours_weekly'] * rng.uniform(0.85, 1.15)
            n_family = max(0, int(rng.poisson(sp['family_workers_mean'])))
            family_hours_each = sp['family_hours_each'] * rng.uniform(0.7, 1.3) if n_family > 0 else 0
            total_family_hours = n_family * family_hours_each
            
            # Shadow wage for owner and family
            shadow_wage = MIN_WAGE_HOURLY * rng.uniform(0.8, 1.4)
            imputed_labor_cost_weekly = (owner_hours + total_family_hours) * shadow_wage / 52 * 7
            
            # Depreciation (straight line, 5-year life for most tools)
            depreciation_weekly = tool_value / (5 * 52)
            
            # Own-consumption
            own_consumption_weekly = 0
            if sector in ['gastronomia', 'comercio']:
                own_consumption_weekly = rev_weekly_true * sp['cogs_share'] * rng.uniform(0.02, 0.08)
            
            # Cash on hand at start
            cash_initial = rev_weekly_true * rng.uniform(0.3, 1.5)
            
            # Secondary activity
            has_secondary = rng.random() < 0.07
            secondary_hours = int(rng.uniform(5, 20)) if has_secondary else 0
            secondary_income_weekly = rev_weekly_true * rng.uniform(0.1, 0.3) if has_secondary else 0
            
            # EH BASELINE (with all the biases we documented)
            # 1. Non-response: 41.8% report zero
            eh_nonresponder = rng.random() < 0.418
            
            if eh_nonresponder:
                eh_yi_tot_annual = 0
                eh_yi_net = 0
                eh_ci_tot = 0
                eh_ci = {f'ci{i}': 0 for i in range(1,9)}
            else:
                # 2. Anchored weekly recall with downward bias
                eh_yi_tot_annual = weekly_to_eh_annual(rev_weekly_true, sp['rev_weekly_cv'])
                
                # 3. Cost recall: partial, with category-specific omissions
                true_annual_cogs = rev_weekly_true * sp['cogs_share'] * 52
                true_annual_rent = rev_weekly_true * sp['rent_share'] * 52
                true_annual_transport = rev_weekly_true * sp['transport_share'] * 52
                true_annual_utilities = rev_weekly_true * sp['utilities_share'] * 52
                true_annual_wages = rev_weekly_true * sp['wages_share'] * 52
                true_annual_debt_payment = weekly_installment * 52 if has_debt_true else 0
                true_annual_taxes = rev_weekly_true * 0.008 * 52
                true_annual_guild = rev_weekly_true * sp['guildfees_share'] * 52
                
                # ci1: raw materials — recalled at 70-95% of true
                eh_ci1 = true_annual_cogs * rng.uniform(0.70, 0.95) if true_annual_cogs > 0 else 0
                eh_ci1 = round_to_anchor(eh_ci1)
                
                # ci2: services — recalled at 50-85% (more abstract)
                eh_ci2 = true_annual_transport * rng.uniform(0.50, 0.85) if true_annual_transport > 50 else 0
                eh_ci2 = round_to_anchor(eh_ci2) if eh_ci2 > 0 else 0
                
                # ci3: wages — recall only if actually paid external worker
                if true_annual_wages > 500 and rng.random() < 0.70:
                    eh_ci3 = true_annual_wages * rng.uniform(0.80, 1.0)
                    eh_ci3 = round_to_anchor(eh_ci3)
                else:
                    eh_ci3 = 0  # family labor NEVER recorded here
                
                # ci4: rent — good recall if explicit contract
                if true_annual_rent > 200:
                    recall_prob = 0.75 if true_annual_rent > 1000 else 0.50
                    eh_ci4 = true_annual_rent * rng.uniform(0.85, 1.0) * (rng.random() < recall_prob)
                    eh_ci4 = round_to_anchor(eh_ci4)
                else:
                    eh_ci4 = 0
                
                # ci5: utilities — partial recall (home-business blur)
                if true_annual_utilities > 200:
                    eh_ci5 = true_annual_utilities * rng.uniform(0.30, 0.70)
                    eh_ci5 = round_to_anchor(eh_ci5)
                else:
                    eh_ci5 = 0
                
                # ci6: loan payments — massive under-reporting
                if has_debt_true and debt_type != 'familiar':
                    # Only 6% declare in EH; model: 13% declare conditional on having debt
                    if rng.random() < 0.13:
                        eh_ci6 = true_annual_debt_payment * rng.uniform(0.7, 1.0)
                        eh_ci6 = round_to_anchor(eh_ci6)
                    else:
                        eh_ci6 = 0
                else:
                    eh_ci6 = 0
                
                # ci7: taxes — only formal-ish declare
                if true_annual_taxes > 200 and rng.random() < 0.40:
                    eh_ci7 = true_annual_taxes * rng.uniform(0.5, 1.0)
                    eh_ci7 = round_to_anchor(eh_ci7)
                else:
                    eh_ci7 = 0
                
                # ci8: guild dues
                if rev_weekly_true > 800 and rng.random() < 0.55:
                    eh_ci8 = true_annual_guild * rng.uniform(0.7, 1.0)
                    eh_ci8 = round_to_anchor(eh_ci8)
                else:
                    eh_ci8 = 0
                
                eh_ci = {'ci1': eh_ci1, 'ci2': eh_ci2, 'ci3': eh_ci3,
                         'ci4': eh_ci4, 'ci5': eh_ci5, 'ci6': eh_ci6,
                         'ci7': eh_ci7, 'ci8': eh_ci8}
                eh_ci_tot = sum(eh_ci.values())
                
                # yi_net: respondent reports a "remembered" take-home
                # Tends to be between (yi_tot - ci_tot) and yi_tot
                eh_yi_net = max(0, eh_yi_tot_annual - eh_ci_tot) * rng.uniform(0.85, 1.10)
                eh_yi_net = round_to_anchor(eh_yi_net)
            
            firm = {
                'firm_id': f'F{firm_id:03d}',
                'sector': sector,
                'sector_label': scfg['label'],
                'city': city,
                'gender': 'M' if not is_female else 'F',
                'age': age,
                'is_poor': is_poor,
                'years_operating': years_op,
                'has_fixed_location': rng.random() < 0.65,
                'has_secondary_activity': has_secondary,
                # True economic parameters
                'rev_weekly_true': round(rev_weekly_true, 2),
                'cogs_share_true': sp['cogs_share'],
                'owner_hours_weekly': round(owner_hours, 1),
                'n_family_workers': n_family,
                'family_hours_weekly': round(total_family_hours, 1),
                'shadow_wage_per_hour': round(shadow_wage, 2),
                'imputed_labor_weekly': round(imputed_labor_cost_weekly, 2),
                'depreciation_weekly': round(depreciation_weekly, 2),
                'own_consumption_weekly': round(own_consumption_weekly, 2),
                # Assets/liabilities
                'tool_value': round(tool_value, 2),
                'inventory_init': round(inventory_init, 2),
                'receivables_init': round(receivables_init, 2),
                'payables_init': round(payables_init, 2),
                'cash_initial': round(cash_initial, 2),
                # Debt (true)
                'has_debt_true': has_debt_true,
                'debt_type': debt_type,
                'debt_amount': round(debt_amount, 2),
                'monthly_interest_rate': round(monthly_interest_rate, 4),
                'weekly_installment': round(weekly_installment, 2),
                'weekly_interest_true': round(weekly_interest if has_debt_true else 0, 2),
                # EH baseline
                'eh_nonresponder': eh_nonresponder,
                'eh_yi_tot': round(eh_yi_tot_annual, 0),
                'eh_yi_net': round(eh_yi_net if not eh_nonresponder else 0, 0),
                'eh_ci_tot': round(eh_ci_tot if not eh_nonresponder else 0, 0),
                'eh_ci1': round(eh_ci.get('ci1', 0) if not eh_nonresponder else 0, 0),
                'eh_ci2': round(eh_ci.get('ci2', 0) if not eh_nonresponder else 0, 0),
                'eh_ci3': round(eh_ci.get('ci3', 0) if not eh_nonresponder else 0, 0),
                'eh_ci4': round(eh_ci.get('ci4', 0) if not eh_nonresponder else 0, 0),
                'eh_ci5': round(eh_ci.get('ci5', 0) if not eh_nonresponder else 0, 0),
                'eh_ci6': round(eh_ci.get('ci6', 0) if not eh_nonresponder else 0, 0),
                'eh_ci7': round(eh_ci.get('ci7', 0) if not eh_nonresponder else 0, 0),
                'eh_ci8': round(eh_ci.get('ci8', 0) if not eh_nonresponder else 0, 0),
            }
            firms.append(firm)
            firm_id += 1
    
    return pd.DataFrame(firms)

# ============================================================
# GENERATE DAILY TRANSACTIONS
# ============================================================

DAYS_OF_WEEK = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

def generate_transactions(firms_df):
    transactions = []
    tx_id = 1
    
    for _, firm in firms_df.iterrows():
        sector = firm['sector']
        sp = SECTOR_PARAMS[sector]
        
        daily_revenues = generate_daily_revenue(sector, {'rev_weekly': firm['rev_weekly_true']})
        
        # Track daily state
        cash = firm['cash_initial']
        inventory = firm['inventory_init']
        receivables = firm['receivables_init']
        payables = firm['payables_init']
        
        for day_num, (day_name, day_revenue) in enumerate(zip(DAYS_OF_WEEK, daily_revenues)):
            is_rest_day = (day_revenue == 0) or (day_name == 'Domingo' and sp['days_active'] < 7)
            
            if is_rest_day:
                # Still record the day as active observation with zeros
                transactions.append({
                    'tx_id': f'TX{tx_id:05d}',
                    'firm_id': firm['firm_id'],
                    'sector': sector,
                    'city': firm['city'],
                    'day_num': day_num + 1,
                    'day_name': day_name,
                    'module': 'DIA_INACTIVO',
                    'category': 'descanso',
                    'description': 'Día de descanso / feria cerrada',
                    'amount_bs': 0,
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': None,
                    'is_credit': False,
                    'counterpart': None,
                    'account_db': None,
                    'account_cr': None,
                    'is_imputed': False,
                    'cash_end': round(cash, 2),
                    'inventory_end': round(inventory, 2),
                })
                tx_id += 1
                continue
            
            day_txs = []
            
            # --- MODULE 1A: SALES ---
            # Split day revenue into 2-8 individual transactions
            n_sales = int(rng.integers(2, 9))
            if day_revenue > 0 and n_sales > 0:
                # Allocate revenue across transactions
                alloc = rng.dirichlet(np.ones(n_sales))
                for j, frac in enumerate(alloc):
                    sale_amt = day_revenue * frac
                    if sale_amt < 5:
                        continue
                    # USD pricing: some inputs priced in USD
                    in_usd = rng.random() < (0.05 if sector != 'transporte' else 0.15)
                    if in_usd:
                        usd_amt = sale_amt / P2P_RATE
                        currency = 'USD'
                    else:
                        usd_amt = 0
                        currency = 'Bs'
                    
                    # Credit sales: common in manufactura and comercio
                    is_credit_sale = False
                    if sector in ['manufactura'] and rng.random() < 0.15:
                        is_credit_sale = True
                    elif sector == 'comercio' and rng.random() < 0.08:
                        is_credit_sale = True
                    
                    if is_credit_sale:
                        receivables += sale_amt
                        account_db, account_cr = '1110', '4020'
                    else:
                        cash += sale_amt
                        account_db, account_cr = '1010', '4010'
                    
                    day_txs.append({
                        'module': 'M1A_VENTAS',
                        'category': 'venta_contado' if not is_credit_sale else 'venta_credito',
                        'description': f'Venta {sector} día {day_num+1} tx{j+1}',
                        'amount_bs': round(sale_amt, 2),
                        'currency': currency,
                        'usd_amount': round(usd_amt, 2),
                        'payment_type': 'credito' if is_credit_sale else 'efectivo',
                        'is_credit': is_credit_sale,
                        'counterpart': 'cliente',
                        'account_db': account_db,
                        'account_cr': account_cr,
                        'is_imputed': False,
                    })
            
            # --- MODULE 1B: COLLECTIONS ON RECEIVABLES ---
            if receivables > 0 and rng.random() < 0.30:
                collection = receivables * rng.uniform(0.20, 0.60)
                receivables -= collection
                cash += collection
                day_txs.append({
                    'module': 'M1B_COBROS',
                    'category': 'cobro_cxc',
                    'description': 'Cobro cuenta por cobrar cliente',
                    'amount_bs': round(collection, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': 'cliente',
                    'account_db': '1010',
                    'account_cr': '1110',
                    'is_imputed': False,
                })
            
            # --- MODULE 1C: OWN CONSUMPTION ---
            if firm['own_consumption_weekly'] > 0 and rng.random() < 0.60:
                oc_amt = firm['own_consumption_weekly'] / sp['days_active'] * rng.uniform(0.5, 1.5)
                inventory -= oc_amt * 0.3  # rough: in-kind from stock
                day_txs.append({
                    'module': 'M1C_AUTOCONSUMO',
                    'category': 'autoconsumo',
                    'description': 'Consumo del hogar desde mercadería/producción',
                    'amount_bs': round(oc_amt, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'en_especie',
                    'is_credit': False,
                    'counterpart': 'hogar',
                    'account_db': '4040',
                    'account_cr': '5010',
                    'is_imputed': True,
                })
            
            # --- MODULE 2: INPUT PURCHASES ---
            if sp['has_inventory'] or sector == 'gastronomia':
                # Replenish inventory
                purchase_prob = 0.7 if sp['perishable_frac'] > 0.5 else 0.4
                if rng.random() < purchase_prob:
                    purchase_amt = day_revenue * sp['cogs_share'] * rng.uniform(0.8, 1.4)
                    # USD-priced inputs
                    in_usd = rng.random() < sp['usd_input_frac']
                    if in_usd:
                        # Pay at P2P rate
                        usd_amt = purchase_amt / P2P_RATE
                        currency = 'USD'
                    else:
                        usd_amt = 0
                        currency = 'Bs'
                    
                    # Credit purchase from supplier
                    is_credit_purchase = rng.random() < 0.30 and payables < purchase_amt * 5
                    if is_credit_purchase:
                        payables += purchase_amt
                        account_db, account_cr = '1210', '2010'
                    else:
                        cash -= purchase_amt
                        account_db, account_cr = '1210', '1010'
                    
                    inventory += purchase_amt * 0.85  # 15% immediate use
                    
                    day_txs.append({
                        'module': 'M2_COMPRAS',
                        'category': 'compra_insumos',
                        'description': f'Compra insumos/mercadería {sector}',
                        'amount_bs': round(purchase_amt, 2),
                        'currency': currency,
                        'usd_amount': round(usd_amt, 2),
                        'payment_type': 'credito' if is_credit_purchase else 'efectivo',
                        'is_credit': is_credit_purchase,
                        'counterpart': 'proveedor',
                        'account_db': account_db,
                        'account_cr': account_cr,
                        'is_imputed': False,
                    })
            
            # --- MODULE 3: OPERATING EXPENSES ---
            # Transport cost (daily or 2-3x/week)
            if sp['transport_share'] > 0 and rng.random() < 0.5:
                transport_daily = firm['rev_weekly_true'] * sp['transport_share'] / sp['days_active']
                transport_amt = transport_daily * rng.uniform(0.7, 1.5)
                cash -= transport_amt
                day_txs.append({
                    'module': 'M3_GASTOS_OP',
                    'category': 'gasto_transporte',
                    'description': 'Pasaje / flete / combustible',
                    'amount_bs': round(transport_amt, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': 'proveedor_servicio',
                    'account_db': '6010',
                    'account_cr': '1010',
                    'is_imputed': False,
                })
            
            # Rent (paid once a week typically on Monday or Friday)
            if sp['rent_share'] > 0 and day_num == 0 and rng.random() < 0.65:
                rent_weekly = firm['rev_weekly_true'] * sp['rent_share']
                rent_amt = rent_weekly * rng.uniform(0.9, 1.1)
                cash -= rent_amt
                day_txs.append({
                    'module': 'M3_GASTOS_OP',
                    'category': 'gasto_alquiler',
                    'description': 'Alquiler puesto / local / feria',
                    'amount_bs': round(rent_amt, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': 'arrendador',
                    'account_db': '6020',
                    'account_cr': '1010',
                    'is_imputed': False,
                })
            
            # Utilities (sporadic)
            if sp['utilities_share'] > 0 and rng.random() < 0.15:
                util_weekly = firm['rev_weekly_true'] * sp['utilities_share']
                cash -= util_weekly
                day_txs.append({
                    'module': 'M3_GASTOS_OP',
                    'category': 'gasto_servicios',
                    'description': 'Gas / agua / luz / internet',
                    'amount_bs': round(util_weekly, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': 'servicio_basico',
                    'account_db': '6030',
                    'account_cr': '1010',
                    'is_imputed': False,
                })
            
            # Guild dues (monthly, paid 1 day out of ~26)
            if sp['guildfees_share'] > 0 and rng.random() < 0.08:
                guild_monthly = firm['rev_weekly_true'] * sp['guildfees_share'] * 4.33
                cash -= guild_monthly
                day_txs.append({
                    'module': 'M3_GASTOS_OP',
                    'category': 'gasto_gremio',
                    'description': 'Cuota gremio / sindicato / asociación',
                    'amount_bs': round(guild_monthly, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': 'gremio',
                    'account_db': '6060',
                    'account_cr': '1010',
                    'is_imputed': False,
                })
            
            # --- MODULE 4: FINANCING ---
            # Debt payment (weekly installment)
            if firm['has_debt_true'] and firm['weekly_installment'] > 0 and day_num == 4:  # Friday
                installment = firm['weekly_installment']
                interest_part = firm['weekly_interest_true']
                capital_part = max(0, installment - interest_part)
                
                cash -= installment
                day_txs.append({
                    'module': 'M4B_PAGO_DEUDA',
                    'category': 'pago_deuda',
                    'description': f'Cuota {firm["debt_type"]}: capital + interés',
                    'amount_bs': round(installment, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': firm['debt_type'],
                    'account_db': '2110_capital+6100_interes',
                    'account_cr': '1010',
                    'is_imputed': False,
                    # Extra fields for debt decomposition
                    'capital_paid': round(capital_part, 2),
                    'interest_paid': round(interest_part, 2),
                })
            
            # Occasional new loan (10% of active firms on any given week)
            if firm['has_debt_true'] and rng.random() < 0.03:
                new_loan = rng.uniform(200, 1500)
                cash += new_loan
                day_txs.append({
                    'module': 'M4A_PRESTAMO',
                    'category': 'prestamo_recibido',
                    'description': 'Nuevo préstamo informal recibido',
                    'amount_bs': round(new_loan, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': True,
                    'counterpart': 'prestamista',
                    'account_db': '1010',
                    'account_cr': '2120',
                    'is_imputed': False,
                })
            
            # --- MODULE 5: IMPUTED LABOR (end of day adjustment) ---
            # Owner labor
            owner_daily_hrs = firm['owner_hours_weekly'] / sp['days_active']
            owner_daily_imputed = owner_daily_hrs * firm['shadow_wage_per_hour']
            day_txs.append({
                'module': 'M5_TRABAJO_IMPUTADO',
                'category': 'mano_obra_imputada_titular',
                'description': f'Trabajo titular: {owner_daily_hrs:.1f} hrs × Bs{firm["shadow_wage_per_hour"]:.1f}/hr',
                'amount_bs': round(owner_daily_imputed, 2),
                'currency': 'Bs',
                'usd_amount': 0,
                'payment_type': 'imputado',
                'is_credit': False,
                'counterpart': 'titular',
                'account_db': '6050',
                'account_cr': '3030',
                'is_imputed': True,
            })
            
            # Family labor (if any)
            if firm['n_family_workers'] > 0 and firm['family_hours_weekly'] > 0:
                fam_daily_hrs = firm['family_hours_weekly'] / sp['days_active']
                fam_daily_imputed = fam_daily_hrs * firm['shadow_wage_per_hour'] * 0.85
                day_txs.append({
                    'module': 'M5_TRABAJO_IMPUTADO',
                    'category': 'mano_obra_imputada_familiar',
                    'description': f'Trabajo familiar no remunerado: {fam_daily_hrs:.1f} hrs',
                    'amount_bs': round(fam_daily_imputed, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'imputado',
                    'is_credit': False,
                    'counterpart': 'familiar',
                    'account_db': '6050',
                    'account_cr': '3030',
                    'is_imputed': True,
                })
            
            # --- MODULE 7: HOUSEHOLD TRANSFERS ---
            # Daily draw for household expenses
            if rng.random() < 0.70:
                draw = day_revenue * rng.uniform(0.05, 0.25)
                draw = min(draw, cash * 0.6)  # can't draw more than 60% of cash
                cash -= draw
                day_txs.append({
                    'module': 'M7_TRANSFERENCIAS',
                    'category': 'retiro_titular',
                    'description': 'Retiro para gastos del hogar',
                    'amount_bs': round(max(0, draw), 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': 'hogar',
                    'account_db': '3030',
                    'account_cr': '1010',
                    'is_imputed': False,
                })
            
            # Occasional household injection into business
            if rng.random() < 0.05:
                injection = rng.uniform(100, 500)
                cash += injection
                day_txs.append({
                    'module': 'M7_TRANSFERENCIAS',
                    'category': 'aporte_hogar',
                    'description': 'Inyección de efectivo del hogar al negocio',
                    'amount_bs': round(injection, 2),
                    'currency': 'Bs',
                    'usd_amount': 0,
                    'payment_type': 'efectivo',
                    'is_credit': False,
                    'counterpart': 'hogar',
                    'account_db': '1010',
                    'account_cr': '3040',
                    'is_imputed': False,
                })
            
            # Add all day transactions with firm/day metadata
            for tx in day_txs:
                record = {
                    'tx_id': f'TX{tx_id:05d}',
                    'firm_id': firm['firm_id'],
                    'sector': sector,
                    'city': firm['city'],
                    'gender': firm['gender'],
                    'is_poor': firm['is_poor'],
                    'day_num': day_num + 1,
                    'day_name': day_name,
                }
                record.update(tx)
                # Ensure consistent columns
                for col in ['capital_paid', 'interest_paid']:
                    if col not in record:
                        record[col] = 0
                record['cash_end'] = round(max(0, cash), 2)
                record['inventory_end'] = round(max(0, inventory), 2)
                transactions.append(record)
                tx_id += 1
    
    return pd.DataFrame(transactions)

# ============================================================
# WEEKLY ACCOUNTING STATEMENTS
# ============================================================

def compute_weekly_accounts(firms_df, tx_df):
    records = []
    
    for _, firm in firms_df.iterrows():
        fid = firm['firm_id']
        sector = firm['sector']
        sp = SECTOR_PARAMS[sector]
        
        ftx = tx_df[tx_df['firm_id'] == fid].copy()
        
        # Revenue
        sales_cash = ftx[ftx['category']=='venta_contado']['amount_bs'].sum()
        sales_credit = ftx[ftx['category']=='venta_credito']['amount_bs'].sum()
        collections = ftx[ftx['category']=='cobro_cxc']['amount_bs'].sum()
        autoconsumo = ftx[ftx['category']=='autoconsumo']['amount_bs'].sum()
        total_revenue = sales_cash + sales_credit + autoconsumo
        
        # COGS
        purchases = ftx[ftx['category']=='compra_insumos']['amount_bs'].sum()
        cogs = purchases * 0.85  # simplification: COGS = purchases less end inventory change
        gross_profit = total_revenue - cogs
        
        # Operating expenses (cash, non-imputed)
        transport_exp = ftx[ftx['category']=='gasto_transporte']['amount_bs'].sum()
        rent_exp = ftx[ftx['category']=='gasto_alquiler']['amount_bs'].sum()
        utilities_exp = ftx[ftx['category']=='gasto_servicios']['amount_bs'].sum()
        guild_exp = ftx[ftx['category']=='gasto_gremio']['amount_bs'].sum()
        total_op_exp_conventional = transport_exp + rent_exp + utilities_exp + guild_exp
        
        # Imputed costs (diary adds these)
        labor_imputed = ftx[ftx['category'].isin(['mano_obra_imputada_titular','mano_obra_imputada_familiar'])]['amount_bs'].sum()
        depreciation_imputed = firm['depreciation_weekly']
        
        total_op_exp_adjusted = total_op_exp_conventional + labor_imputed + depreciation_imputed
        
        # Interest
        interest_paid = ftx[ftx['category']=='pago_deuda']['interest_paid'].sum()
        
        # Net income
        net_income_conventional = gross_profit - total_op_exp_conventional - interest_paid
        net_income_adjusted = gross_profit - total_op_exp_adjusted - interest_paid
        
        # Cash flows
        cash_receipts = ftx[ftx['category'].isin(['venta_contado','cobro_cxc'])]['amount_bs'].sum()
        cash_payments_op = ftx[ftx['category'].isin(['compra_insumos','gasto_transporte','gasto_alquiler','gasto_servicios','gasto_gremio'])].query('payment_type=="efectivo"')['amount_bs'].sum()
        cash_flow_op = cash_receipts - cash_payments_op - interest_paid
        
        loans_received = ftx[ftx['category']=='prestamo_recibido']['amount_bs'].sum()
        debt_payments = ftx[ftx['category']=='pago_deuda']['amount_bs'].sum()
        household_draws = ftx[ftx['category']=='retiro_titular']['amount_bs'].sum()
        household_injections = ftx[ftx['category']=='aporte_hogar']['amount_bs'].sum()
        
        # EH comparison
        eh_yi_tot_weekly = firm['eh_yi_tot'] / 52 if not firm['eh_nonresponder'] else 0
        eh_yi_net_weekly = firm['eh_yi_net'] / 52 if not firm['eh_nonresponder'] else 0
        
        # Accounting gap
        gap_net_income = net_income_conventional - net_income_adjusted
        gap_pct = gap_net_income / net_income_conventional * 100 if net_income_conventional > 0 else 0
        
        # Financial ratios
        gross_margin = gross_profit / total_revenue * 100 if total_revenue > 0 else 0
        net_margin_conv = net_income_conventional / total_revenue * 100 if total_revenue > 0 else 0
        net_margin_adj = net_income_adjusted / total_revenue * 100 if total_revenue > 0 else 0
        interest_burden = interest_paid / total_revenue * 100 if total_revenue > 0 else 0
        labor_imputed_burden = labor_imputed / gross_profit * 100 if gross_profit > 0 else 0
        
        # ROA (annualized)
        total_assets = firm['cash_initial'] + firm['inventory_init'] + firm['receivables_init'] + firm['tool_value']
        roa = (net_income_adjusted * 52) / total_assets * 100 if total_assets > 0 else 0
        
        # Hourly return
        total_hours = (firm['owner_hours_weekly'] + firm['family_hours_weekly'])
        hourly_return_adj = net_income_adjusted / total_hours if total_hours > 0 else 0
        
        # Typology
        if net_income_adjusted > 0 and interest_burden < 10:
            typology = 'I_Viable'
        elif net_income_adjusted <= 0 and interest_burden < 10:
            typology = 'II_Precaria'
        elif net_income_adjusted > 0 and interest_burden >= 10:
            typology = 'III_Atrapada_deuda'
        else:
            typology = 'IV_Riesgo_critico'
        
        records.append({
            'firm_id': fid,
            'sector': sector,
            'city': firm['city'],
            'gender': firm['gender'],
            'is_poor': firm['is_poor'],
            'has_debt_true': firm['has_debt_true'],
            'debt_type': firm['debt_type'],
            # Revenue
            'total_revenue_diary': round(total_revenue, 2),
            'sales_cash': round(sales_cash, 2),
            'sales_credit': round(sales_credit, 2),
            'autoconsumo_imputed': round(autoconsumo, 2),
            # Costs
            'cogs': round(cogs, 2),
            'gross_profit': round(gross_profit, 2),
            'transport_exp': round(transport_exp, 2),
            'rent_exp': round(rent_exp, 2),
            'utilities_exp': round(utilities_exp, 2),
            'guild_exp': round(guild_exp, 2),
            'total_op_exp_conventional': round(total_op_exp_conventional, 2),
            'labor_imputed': round(labor_imputed, 2),
            'depreciation_imputed': round(depreciation_imputed, 2),
            'total_op_exp_adjusted': round(total_op_exp_adjusted, 2),
            'interest_paid': round(interest_paid, 2),
            # Net income
            'net_income_conventional': round(net_income_conventional, 2),
            'net_income_adjusted': round(net_income_adjusted, 2),
            'accounting_gap_bs': round(gap_net_income, 2),
            'accounting_gap_pct': round(gap_pct, 1),
            # Cash flows
            'cash_flow_operating': round(cash_flow_op, 2),
            'loans_received': round(loans_received, 2),
            'debt_payments_total': round(debt_payments, 2),
            'household_draws': round(household_draws, 2),
            'household_injections': round(household_injections, 2),
            # EH comparison
            'eh_nonresponder': firm['eh_nonresponder'],
            'eh_yi_tot_annual': firm['eh_yi_tot'],
            'eh_yi_net_annual': firm['eh_yi_net'],
            'diary_yi_tot_annualized': round(total_revenue * 52, 2),
            'diary_yi_net_conv_annualized': round(net_income_conventional * 52, 2),
            'diary_yi_net_adj_annualized': round(net_income_adjusted * 52, 2),
            'ratio_diary_eh_income': round((total_revenue * 52) / firm['eh_yi_tot'], 3) if firm['eh_yi_tot'] > 0 else None,
            # Ratios
            'gross_margin_pct': round(gross_margin, 1),
            'net_margin_conventional_pct': round(net_margin_conv, 1),
            'net_margin_adjusted_pct': round(net_margin_adj, 1),
            'interest_burden_pct': round(interest_burden, 1),
            'labor_imputed_pct_gross_profit': round(labor_imputed_burden, 1),
            'roa_adjusted_annualized': round(roa, 1),
            'hourly_return_adjusted_bs': round(hourly_return_adj, 2),
            'hourly_vs_minwage_ratio': round(hourly_return_adj / MIN_WAGE_HOURLY, 3),
            # Typology
            'typology': typology,
        })
    
    return pd.DataFrame(records)

# ============================================================
# MAIN
# ============================================================

print("Generating 300 firms...")
firms_df = generate_firms()
print(f"  Firms generated: {len(firms_df)}")

print("Generating daily transactions...")
tx_df = generate_transactions(firms_df)
print(f"  Transactions generated: {len(tx_df)}")

print("Computing weekly accounting statements...")
accounts_df = compute_weekly_accounts(firms_df, tx_df)
print(f"  Accounts computed: {len(accounts_df)}")

# Save
firms_df.to_csv('/home/claude/firms.csv', index=False)
tx_df.to_csv('/home/claude/transactions.csv', index=False)
accounts_df.to_csv('/home/claude/weekly_accounts.csv', index=False)

print("\n=== VALIDATION SUMMARY ===")
print(f"\nFirms by sector:")
print(firms_df['sector'].value_counts().to_string())

print(f"\nEH non-response rate (simulated): {firms_df['eh_nonresponder'].mean()*100:.1f}%")
print(f"True debt prevalence: {firms_df['has_debt_true'].mean()*100:.1f}%")
print(f"EH debt declaration rate: {(firms_df['eh_ci6']>0).mean()*100:.1f}%")

print(f"\n--- WEEKLY ACCOUNTING (among non-zero revenue) ---")
active = accounts_df[accounts_df['total_revenue_diary'] > 0]
print(f"Median weekly revenue (diary): Bs {active['total_revenue_diary'].median():,.0f}")
print(f"Median net income conventional: Bs {active['net_income_conventional'].median():,.0f}")
print(f"Median net income adjusted: Bs {active['net_income_adjusted'].median():,.0f}")
print(f"Median accounting gap: Bs {active['accounting_gap_bs'].median():,.0f}")
print(f"Median gap as % of conventional profit: {active['accounting_gap_pct'].median():.1f}%")
print(f"% firms with negative adjusted income: {(active['net_income_adjusted']<0).mean()*100:.1f}%")
print(f"% firms with positive conventional income: {(active['net_income_conventional']>0).mean()*100:.1f}%")

print(f"\n--- TYPOLOGY DISTRIBUTION ---")
print(accounts_df['typology'].value_counts().to_string())

print(f"\n--- EH vs DIARY (non-zero responders) ---")
resp = accounts_df[~accounts_df['eh_nonresponder'] & (accounts_df['total_revenue_diary']>0)]
print(f"N firms (EH responders with diary income): {len(resp)}")
print(f"Median ratio diary/EH income: {resp['ratio_diary_eh_income'].median():.2f}")
print(f"EH median yi_net annual: Bs {resp['eh_yi_net_annual'].median():,.0f}")
print(f"Diary median net income adj (annual): Bs {resp['diary_yi_net_adj_annualized'].median():,.0f}")

print(f"\n--- HOURLY RETURN ---")
print(f"Median hourly return (adjusted): Bs {active['hourly_return_adjusted_bs'].median():.2f}")
print(f"vs min wage Bs {MIN_WAGE_HOURLY:.2f}/hr")
print(f"Median ratio hourly/min wage: {active['hourly_vs_minwage_ratio'].median():.2f}x")

print(f"\n--- TOTAL TRANSACTIONS ---")
print(tx_df['module'].value_counts().to_string())

print("\nFiles saved: firms.csv, transactions.csv, weekly_accounts.csv")
