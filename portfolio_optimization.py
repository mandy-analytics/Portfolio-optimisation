# ============================================================
#  Portfolio Optimization Model
#  Markowitz Mean-Variance + Efficient Frontier + VaR/CVaR
#  Author: Actuarial Science Student Project
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
})

# ── 1. CONFIGURATION ─────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "XOM", "BRK-B", "PG"]
START   = "2019-01-01"
END     = "2024-01-01"
RF      = 0.05          # Risk-free rate (5% annual)
N_SIM   = 10_000        # Monte Carlo portfolios
CONF    = 0.95          # VaR/CVaR confidence level

print("=" * 60)
print("  PORTFOLIO OPTIMIZATION MODEL")
print("  Markowitz Mean-Variance Framework")
print("=" * 60)

# ── 2. DATA GENERATION (realistic synthetic prices) ──────────
print(f"\n[1/6] Generating realistic synthetic price data for: {TICKERS}")
np.random.seed(0)
dates = pd.bdate_range(start=START, end=END)
n_days = len(dates)

# Realistic annual return/vol parameters per asset
params = {
    "AAPL":  (0.28, 0.30), "MSFT":  (0.24, 0.26),
    "GOOGL": (0.20, 0.28), "JPM":   (0.14, 0.24),
    "JNJ":   (0.08, 0.15), "XOM":   (0.10, 0.22),
    "BRK-B": (0.13, 0.18), "PG":    (0.09, 0.14),
}
# Correlation structure (rough market correlation)
corr_seed = np.array([
    [1.00, 0.75, 0.70, 0.45, 0.25, 0.30, 0.40, 0.30],
    [0.75, 1.00, 0.72, 0.48, 0.27, 0.28, 0.42, 0.32],
    [0.70, 0.72, 1.00, 0.42, 0.22, 0.25, 0.38, 0.28],
    [0.45, 0.48, 0.42, 1.00, 0.35, 0.40, 0.55, 0.38],
    [0.25, 0.27, 0.22, 0.35, 1.00, 0.30, 0.40, 0.55],
    [0.30, 0.28, 0.25, 0.40, 0.30, 1.00, 0.38, 0.32],
    [0.40, 0.42, 0.38, 0.55, 0.40, 0.38, 1.00, 0.45],
    [0.30, 0.32, 0.28, 0.38, 0.55, 0.32, 0.45, 1.00],
])
daily_vols  = np.array([params[t][1]/np.sqrt(252) for t in TICKERS])
daily_rets  = np.array([params[t][0]/252          for t in TICKERS])
cov_seed    = np.outer(daily_vols, daily_vols) * corr_seed
L           = np.linalg.cholesky(cov_seed)
shocks      = np.random.randn(n_days, len(TICKERS))
daily_r     = daily_rets + shocks @ L.T
prices      = 100 * np.exp(np.cumsum(daily_r, axis=0))
raw         = pd.DataFrame(prices, index=dates, columns=TICKERS)
print(f"      Data range: {raw.index[0].date()} → {raw.index[-1].date()}  ({len(raw)} trading days)")

# ── 3. RETURNS & STATISTICS ──────────────────────────────────
print("[2/6] Computing returns and statistics...")
returns = raw.pct_change().dropna()

annual_ret   = returns.mean() * 252
annual_vol   = returns.std()  * np.sqrt(252)
cov_matrix   = returns.cov() * 252
corr_matrix  = returns.corr()

print("\n  Asset Summary:")
print(f"  {'Ticker':<8} {'Ann.Return':>10} {'Ann.Volatility':>15} {'Sharpe':>8}")
print("  " + "-" * 45)
for t in TICKERS:
    sharpe = (annual_ret[t] - RF) / annual_vol[t]
    print(f"  {t:<8} {annual_ret[t]:>9.1%} {annual_vol[t]:>14.1%} {sharpe:>8.2f}")

# ── 4. MONTE CARLO SIMULATION ────────────────────────────────
print(f"\n[3/6] Running Monte Carlo simulation ({N_SIM:,} portfolios)...")
n_assets = len(TICKERS)
mc_weights  = np.zeros((N_SIM, n_assets))
mc_returns  = np.zeros(N_SIM)
mc_vols     = np.zeros(N_SIM)
mc_sharpes  = np.zeros(N_SIM)

np.random.seed(42)
for i in range(N_SIM):
    w = np.random.dirichlet(np.ones(n_assets))   # random weights summing to 1
    mc_weights[i]  = w
    mc_returns[i]  = w @ annual_ret.values
    mc_vols[i]     = np.sqrt(w @ cov_matrix.values @ w)
    mc_sharpes[i]  = (mc_returns[i] - RF) / mc_vols[i]

# ── 5. SCIPY OPTIMISATION ────────────────────────────────────
print("[4/6] Running analytical optimisation...")

def neg_sharpe(w):
    r = w @ annual_ret.values
    v = np.sqrt(w @ cov_matrix.values @ w)
    return -(r - RF) / v

def portfolio_vol(w):
    return np.sqrt(w @ cov_matrix.values @ w)

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds      = [(0, 1)] * n_assets
w0          = np.ones(n_assets) / n_assets

# Max Sharpe
opt_sharpe = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
w_sharpe   = opt_sharpe.x
r_sharpe   = w_sharpe @ annual_ret.values
v_sharpe   = portfolio_vol(w_sharpe)
s_sharpe   = (r_sharpe - RF) / v_sharpe

# Min Volatility
opt_minvol = minimize(portfolio_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
w_minvol   = opt_minvol.x
r_minvol   = w_minvol @ annual_ret.values
v_minvol   = portfolio_vol(w_minvol)

# Efficient Frontier curve
target_rets = np.linspace(annual_ret.min(), annual_ret.max(), 60)
ef_vols, ef_rets = [], []
for tr in target_rets:
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tr=tr: w @ annual_ret.values - tr}]
    res = minimize(portfolio_vol, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if res.success:
        ef_vols.append(res.fun)
        ef_rets.append(tr)

# ── 6. VAR & CVAR (Actuarial Risk Measures) ──────────────────
print("[5/6] Computing VaR and CVaR (actuarial risk measures)...")

def compute_var_cvar(weights, returns_df, conf=CONF):
    port_returns = returns_df.values @ weights
    var  = np.percentile(port_returns, (1 - conf) * 100)
    cvar = port_returns[port_returns <= var].mean()
    return var, cvar

var_sharpe,  cvar_sharpe  = compute_var_cvar(w_sharpe,  returns)
var_minvol,  cvar_minvol  = compute_var_cvar(w_minvol,  returns)
var_equal,   cvar_equal   = compute_var_cvar(w0,         returns)

print("\n  Risk Measures (daily, 95% confidence):")
print(f"  {'Portfolio':<20} {'VaR':>8} {'CVaR':>8}")
print("  " + "-" * 38)
print(f"  {'Max Sharpe':<20} {var_sharpe:>7.2%} {cvar_sharpe:>7.2%}")
print(f"  {'Min Volatility':<20} {var_minvol:>7.2%} {cvar_minvol:>7.2%}")
print(f"  {'Equal Weight':<20} {var_equal:>7.2%} {cvar_equal:>7.2%}")

# ── 7. PLOTTING ──────────────────────────────────────────────
print("\n[6/6] Generating charts...")
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Portfolio Optimization Analysis", fontsize=18,
             fontweight="bold", color="#58a6ff", y=0.98)

GOLD   = "#f0b429"
CYAN   = "#39d0d8"
GREEN  = "#3fb950"
RED    = "#f85149"
PURPLE = "#bc8cff"

# --- Plot 1: Efficient Frontier ---
ax1 = fig.add_subplot(2, 3, (1, 2))
sc = ax1.scatter(mc_vols, mc_returns, c=mc_sharpes, cmap="plasma",
                 alpha=0.4, s=6, zorder=1)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio")
ax1.plot(ef_vols, ef_rets, color=CYAN, lw=2.5, label="Efficient Frontier", zorder=2)
ax1.scatter(v_sharpe, r_sharpe, color=GOLD, s=200, marker="*",
            label=f"Max Sharpe ({s_sharpe:.2f})", zorder=5)
ax1.scatter(v_minvol, r_minvol, color=GREEN, s=160, marker="D",
            label="Min Volatility", zorder=5)
ax1.set_xlabel("Annual Volatility (Risk)")
ax1.set_ylabel("Annual Expected Return")
ax1.set_title("Efficient Frontier  ·  Monte Carlo Simulation", color="#58a6ff")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True)

# --- Plot 2: Optimal Weights Bar ---
ax2 = fig.add_subplot(2, 3, 3)
colors_bar = [GOLD if w > 0.15 else CYAN for w in w_sharpe]
bars = ax2.barh(TICKERS, w_sharpe * 100, color=colors_bar, edgecolor="#21262d", height=0.6)
for bar, val in zip(bars, w_sharpe):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{val:.1%}", va="center", fontsize=9, color="#c9d1d9")
ax2.set_xlabel("Weight (%)")
ax2.set_title("Max Sharpe Portfolio — Weights", color="#58a6ff")
ax2.set_xlim(0, max(w_sharpe * 100) + 8)
ax2.grid(True, axis="x")

# --- Plot 3: Correlation Heatmap ---
ax3 = fig.add_subplot(2, 3, 4)
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, ax=ax3, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, linecolor="#21262d",
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax3.set_title("Asset Correlation Matrix", color="#58a6ff")

# --- Plot 4: VaR/CVaR Distribution ---
ax4 = fig.add_subplot(2, 3, 5)
port_ret_series = returns.values @ w_sharpe
ax4.hist(port_ret_series, bins=80, color="#1f6feb", alpha=0.7, edgecolor="none", label="Daily Returns")
ax4.axvline(var_sharpe,  color=RED,    lw=2, linestyle="--", label=f"VaR  {var_sharpe:.2%}")
ax4.axvline(cvar_sharpe, color=PURPLE, lw=2, linestyle="--", label=f"CVaR {cvar_sharpe:.2%}")
ax4.fill_betweenx([0, ax4.get_ylim()[1] if ax4.get_ylim()[1] > 0 else 200],
                   port_ret_series.min(), var_sharpe,
                   color=RED, alpha=0.15, label="Tail Risk")
ax4.set_xlabel("Daily Return")
ax4.set_ylabel("Frequency")
ax4.set_title("Return Distribution — VaR & CVaR (Max Sharpe)", color="#58a6ff")
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax4.legend(fontsize=9)
ax4.grid(True)

# --- Plot 5: Cumulative Performance ---
ax5 = fig.add_subplot(2, 3, 6)
port_sharpe_cum = (1 + returns.values @ w_sharpe).cumprod()
port_minvol_cum = (1 + returns.values @ w_minvol).cumprod()
port_equal_cum  = (1 + returns.values @ w0).cumprod()
dates = returns.index
ax5.plot(dates, port_sharpe_cum,  color=GOLD,  lw=2, label="Max Sharpe")
ax5.plot(dates, port_minvol_cum,  color=GREEN, lw=2, label="Min Volatility")
ax5.plot(dates, port_equal_cum,   color=CYAN,  lw=2, label="Equal Weight", alpha=0.7)
ax5.axhline(1, color="#8b949e", lw=1, linestyle=":")
ax5.set_xlabel("Date")
ax5.set_ylabel("Growth of $1")
ax5.set_title("Cumulative Portfolio Performance", color="#58a6ff")
ax5.legend(fontsize=9)
ax5.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/mnt/user-data/outputs/portfolio_optimization.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("      Chart saved.")

# ── 8. SUMMARY REPORT ────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"\n  MAX SHARPE PORTFOLIO  (Sharpe = {s_sharpe:.2f})")
print(f"  Expected Return : {r_sharpe:.2%}")
print(f"  Volatility      : {v_sharpe:.2%}")
print(f"  Daily VaR (95%) : {var_sharpe:.2%}")
print(f"  Daily CVaR(95%) : {cvar_sharpe:.2%}")
print(f"\n  Weights:")
for t, w in sorted(zip(TICKERS, w_sharpe), key=lambda x: -x[1]):
    bar = "█" * int(w * 40)
    print(f"    {t:<8} {w:>6.1%}  {bar}")

print(f"\n  MIN VOLATILITY PORTFOLIO")
print(f"  Expected Return : {r_minvol:.2%}")
print(f"  Volatility      : {v_minvol:.2%}")
print(f"  Daily VaR (95%) : {var_minvol:.2%}")
print(f"  Daily CVaR(95%) : {cvar_minvol:.2%}")
print("\n" + "=" * 60)
print("  Done! See portfolio_optimization.png for all charts.")
print("=" * 60)
