import numpy as np
from scipy import integrate, optimize

print("="*70)
print("(1) OU tracking-filter lag variance: numeric vs closed form")
print("    sigma_x^2 = integral |H(w)|^2 S_theta(w) dw/2pi,  predicted = s2*phi/(phi+h)")
print("="*70)
def lag_var_numeric(phi, s2, h):
    # S_theta(w) = 2*phi*s2/(w^2+phi^2);  |H_lag(w)|^2 = w^2/(w^2+h^2)
    integ = lambda w: (w**2/(w**2+h**2)) * (2*phi*s2/(w**2+phi**2)) / (2*np.pi)
    val,_ = integrate.quad(integ, -np.inf, np.inf, limit=400)
    return val
for (phi,s2,h) in [(1.0,1.0,0.5),(2.0,3.0,4.0),(0.3,1.0,5.0),(5.0,2.0,0.2)]:
    num = lag_var_numeric(phi,s2,h); pred = s2*phi/(phi+h)
    print(f"  phi={phi:<4} s2={s2:<4} h=ga={h:<4}  numeric={num:.5f}  pred={pred:.5f}  ratio={num/pred:.4f}")

print()
print("="*70)
print("(2) Evolved-M anisotropy (slope of log m vs log a) by regime/objective")
print("    isotropic disaster Omega=s2*I, fixed A=diag(a_grid); a in [0.25..16]")
print("="*70)
a_grid = np.geomspace(0.25, 16.0, 9)

def fit_slope(a, m):
    m = np.maximum(m, 1e-12)
    return np.polyfit(np.log(a), np.log(m), 1)[0]

# --- (2a) LAG / arithmetic-mean coverage: optimize standing g, then M = a g^2 (MSB) ---
# per-mode objective f(g) = -1/2 log(1+g a) - 1/2 * s2 * a/(1+g a)
def lag_M(s2):
    ms=[]
    for a in a_grid:
        f = lambda g: -(-0.5*np.log(1+g*a) - 0.5*s2*a/(1+g*a))  # negate to minimize
        r = optimize.minimize_scalar(f, bounds=(1e-9, 50.0), method='bounded')
        g = r.x
        ms.append(a*g*g)         # m = a g^2 from GAG=M
    return np.array(ms)
for s2 in [0.5, 2.0, 8.0]:
    m = lag_M(s2); print(f"  LAG  s2={s2:<4}  slope(log m vs log a) = {fit_slope(a_grid,m):+.3f}  (predict ~ +1: M ~ A)")

# --- (2b) BET-HEDGE logdet benefit (l/2)log det M  minus mutation-load cost ---
# robustness to cost form: linear  c*tr(A M) = c*a*m   vs   sqrt cost  k*sqrt(a m) [= k*tr(A Ghat)]
def hedge_M(lam, cost='sqrt', k=1.0):
    ms=[]
    for a in a_grid:
        if cost=='sqrt':
            f = lambda m: -((lam/2)*np.log(m) - k*np.sqrt(a*m))
        else:
            f = lambda m: -((lam/2)*np.log(m) - k*a*m)
        r = optimize.minimize_scalar(f, bounds=(1e-9, 1e6), method='bounded')
        ms.append(r.x)
    return np.array(ms)
for cost in ['sqrt','linear']:
    m = hedge_M(1.0, cost=cost, k=0.5)
    print(f"  HEDGE logdet, cost={cost:<7} slope = {fit_slope(a_grid,m):+.3f}  (predict -1: M ~ A^-1)")

# --- (2c) light-tail coverage benefit -1/2 tr(M^-1 Omega) = -s2/(2m), to show it is COST-form dependent
def cover_M(s2, cost='linear', k=0.5):
    ms=[]
    for a in a_grid:
        if cost=='linear':
            f = lambda m: -(-0.5*s2/m - 0.5*a*m)        # note: 1/2 tr(A M) linear cost
        else:
            f = lambda m: -(-0.5*s2/m - k*np.sqrt(a*m))
        r = optimize.minimize_scalar(f, bounds=(1e-9, 1e6), method='bounded')
        ms.append(r.x)
    return np.array(ms)
for cost in ['linear','sqrt']:
    m = cover_M(2.0, cost=cost)
    print(f"  COVER tr(M^-1), cost={cost:<7} slope = {fit_slope(a_grid,m):+.3f}  (linear->-1/2; sqrt->-1/3: COST-DEPENDENT)")
