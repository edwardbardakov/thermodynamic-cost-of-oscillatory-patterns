"""
Supplemental Material for:
  "Irreducible thermodynamic cost of oscillatory patterns
   in the Stuart-Landau normal form"

Reproduces: Tables 1-2, Figure 1, Eq.(9) verification.
Requirements: numpy, scipy, matplotlib
Usage: python supplemental_code.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm as ndist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- Compatibility shim for np.trapezoid / np.trapz ---
try:
    _trapz = np.trapezoid          # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz              # NumPy < 2.0

# --- Core functions ---

def r2_oscillatory(mu, g, eps):
    """<r^2> for 2D Stuart-Landau. rho(r) ~ r exp(-V/eps).

    Subtracts V_min before exponentiating to avoid overflow
    at small eps (the constant cancels in the ratio), and
    integrates over a peak-centred window to maintain accuracy
    when the distribution is very narrow.
    """
    V = lambda r: -mu*r**2/2 + g*r**4/4
    V_min = -mu**2/(4*g)
    r_peak = np.sqrt(mu/g)
    width = np.sqrt(eps/mu)                      # characteristic width in r
    lo = max(0, r_peak - 10*width)
    hi = r_peak + 10*width
    w = lambda r: r*np.exp(-(V(r) - V_min)/eps)
    Z  = quad(w, lo, hi, limit=500)[0]
    I2 = quad(lambda r: r**2*w(r), lo, hi, limit=500)[0]
    return I2/Z

def x2_steady(mu, g, eps):
    """<x^2> for 1D Landau (positive well). rho(x) ~ exp(-V/eps).

    Same overflow/peak-centred treatment as above.
    """
    V = lambda x: -mu*x**2/2 + g*x**4/4
    V_min = -mu**2/(4*g)
    x_peak = np.sqrt(mu/g)
    width = np.sqrt(eps/(2*mu))
    lo = max(0, x_peak - 10*width)
    hi = x_peak + 10*width
    w = lambda x: np.exp(-(V(x) - V_min)/eps)
    Z  = quad(w, lo, hi, limit=500)[0]
    I2 = quad(lambda x: x**2*w(x), lo, hi, limit=500)[0]
    return I2/Z

def r2_closed(mu, g, eps):
    """Closed-form <r^2>, Eq.(9): truncated Gaussian mean."""
    c, sig = mu/g, np.sqrt(2*eps/g)
    return c + sig*ndist.pdf(-c/sig)/ndist.cdf(c/sig)

def ps_dist(s_arr, mu, g, eps):
    """Normalised p(s) for s=r^2."""
    lp = mu*s_arr/(2*eps) - g*s_arr**2/(4*eps)
    lp -= lp.max()
    ps = np.exp(lp)
    ps /= _trapz(ps, s_arr)
    return ps

g=1.0; mu0=0.5; A2=mu0/g

# --- Table 1 ---
print("TABLE 1: mu=0.5, g=1, A*^2=0.5")
print(f"{'eps':>8} {'<r2>/A*2 osc':>14} {'<x2>/A*2 std':>14} {'Eq(9)':>10}")
for e in [0.001, 0.01, 0.05, 0.1, 0.5]:
    print(f"{e:>8.3f} {r2_oscillatory(mu0,g,e)/A2:>14.4f} {x2_steady(mu0,g,e)/A2:>14.4f} {r2_closed(mu0,g,e)/A2:>10.4f}")

# --- Table 2 ---
print("\nTABLE 2: <r2>/A*2 (g=1)")
for mu in [0.1, 0.3, 0.5, 1.0, 2.0]:
    a2=mu/g; vals=[r2_oscillatory(mu,g,e)/a2 for e in [0.01,0.05,0.1,0.5]]
    print(f"  mu={mu:.1f}: "+"  ".join(f"{v:.3f}" for v in vals))

# --- Verification ---
mx=0
for mu in [0.1,0.3,0.5,1.0,2.0]:
    for e in [0.001,0.01,0.05,0.1,0.5,1.0]:
        mx=max(mx, abs(r2_oscillatory(mu,g,e)-r2_closed(mu,g,e))/r2_oscillatory(mu,g,e))
print(f"\nEq.(9) vs exact: max rel err = {mx:.2e} {'PASS' if mx<1e-10 else 'FAIL'}")

# --- Figure 1 ---
fig=plt.figure(figsize=(7.0,3.4))
gs=GridSpec(1,2,figure=fig,wspace=0.37,left=0.09,right=0.97,bottom=0.16,top=0.91)
ax1=fig.add_subplot(gs[0])
ea=np.logspace(-3,0.3,200)
osc=np.array([r2_oscillatory(mu0,g,e)/A2 for e in ea])
std=np.array([x2_steady(mu0,g,e)/A2 for e in ea])
cf=np.array([r2_closed(mu0,g,e)/A2 for e in ea])
ax1.fill_between(ea,0,1,color='#cb4335',alpha=0.05,zorder=0)
ax1.semilogx(ea,osc,'-',color='#2066a8',lw=2,label=r'Oscillatory (Hopf, 2D)')
ax1.semilogx(ea,cf,'--',color='#7fb3d8',lw=1.3,label=r'Closed form, Eq. (9)')
ax1.semilogx(ea,std,'-',color='#cb4335',lw=2,label=r'Steady (pitchfork, 1D)')
ax1.axhline(1,color='0.4',ls=':',lw=0.8,zorder=0)
ax1.text(1.6,1.04,r'$\frac{\langle\cdot\rangle}{A^{*2}}=1$',fontsize=8,color='0.4',va='bottom')
ax1.text(0.0025,0.6,'bound\nviolated',fontsize=7,color='#cb4335',ha='center',fontstyle='italic')
ax1.set(xlabel=r'Noise intensity $\varepsilon$',ylabel=r'$\langle\,\cdot\,\rangle\;/\;A^{*2}$',xlim=(1e-3,2),ylim=(0.4,2.5))
ax1.legend(fontsize=6.5,loc='upper left',framealpha=0.95)
ax1.set_title(r'$\mathbf{(a)}$ Mean-square amplitude ratio',fontsize=8.5,loc='left')
ax1.text(0.97,0.03,r'$\mu{=}0.5,\;g{=}1$',fontsize=7,transform=ax1.transAxes,ha='right',va='bottom',color='0.5')

ax2=fig.add_subplot(gs[1])
for e,c,lw in zip([0.01,0.05,0.2],['#1a5276','#2980b9','#85c1e9'],[2,1.7,1.4]):
    sig=np.sqrt(2*e/g); sa=np.linspace(max(0,A2-5*sig),max(A2+6*sig,1.2),500)
    ax2.plot(sa,ps_dist(sa,mu0,g,e),'-',color=c,lw=lw,label=rf'$\varepsilon={e}$')
ax2.axvline(A2,color='0.4',ls=':',lw=0.8,zorder=0)
ax2.text(A2+0.03,4.0,r'$A^{*2}$',fontsize=9.5,color='0.4')
ax2.set(xlabel=r'$s=r^2$',ylabel=r'$p(s)$',xlim=(0,1.25),ylim=(0,4.6))
ax2.legend(fontsize=7.5,loc='upper right',framealpha=0.95)
ax2.set_title(r'$\mathbf{(b)}$ Distribution of $s=r^2$',fontsize=8.5,loc='left')
ax2.text(0.97,0.89,'exact Gaussian\nin $s$',fontsize=7,transform=ax2.transAxes,ha='right',va='top',color='0.4',fontstyle='italic')
plt.savefig('fig1.pdf',dpi=300); plt.savefig('fig1.png',dpi=200)
print("Figure saved: fig1.pdf, fig1.png")
