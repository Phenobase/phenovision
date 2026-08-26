"""
drift_estimator_prototype.py
============================
De-risks the V3/V4 measurement machinery (sde_validation_plan.md) BEFORE building
an evolvable-M IBM. It runs on a Gaussian-closure surrogate where the true slow
drift b(M) = -G_M . grad L_eff(M) is KNOWN by construction, so it tests whether the
ENSEMBLE DRIFT ESTIMATOR can recover that drift from noisy slow dynamics -- it does
NOT test the biological theory (which is the IBM's job).

What it establishes (validated numbers, d=2, M parameterized by vech=[m00,m01,m11]):

  (1) Recovery: the ensemble estimator (R replicates from fixed M0, average
      (M(tau)-M0)/(tau*etaM)) recovers the drift DIRECTION to cos>0.999 with as
      few as R~200 replicates.

  (1b) Magnitude bias = the bias-variance tradeoff in the window. As the distance
      travelled during the window (etaM*tau*|b|) grows from 0.045 to 1.12, the
      magnitude error grows 5% -> 59% and direction degrades. LESSON: keep slow
      steps small and the window short relative to the scale over which the drift
      field varies; lengthen tau only enough to average the fast fluctuations.

  (2) The metric confound is real and the correction works. Comparing the measured
      drift to the bare gradient -grad L gives cos=0.70 (you would wrongly conclude
      the theory fails); comparing to -G_M grad L gives 0.999. Estimating G_M from
      as few as N=50 modifier breeding values already gives 0.99. LESSON: never
      compare drift to the bare gradient; estimate G_M and use -G_M grad L.

  (3) The eigenvalue-scaling vs eigenvector-rotation decomposition (rotate the drift
      into M's eigenbasis; diagonal = scaling, off-diagonal = rotation) cleanly
      separates and validates the two components (cos 0.999 / 1.000).

  (4) The window must exceed the fast correlation time (~1/fast_relax): tau=3 gives
      cos=0.985 (fast jitter not averaged), tau=30 gives 0.9995.

  (5) Curl detection (V4): the rotational fraction ||asym(K)||/||K|| of the metric-
      corrected field's Jacobian reads ~0.00 for a pure-gradient drift and jumps when
      a rotational component is injected -- so the V4 machinery reliably detects the
      presence/absence of curl (it saturates at large curl, so it is a detector, not
      a precise magnitude gauge).

Run: python3 drift_estimator_prototype.py
Pure numpy. Swap the surrogate increment() for the real evolvable-M IBM to run V3/V4.
"""

import numpy as np
rng = np.random.default_rng(0)

def vech_to_M(v): return np.array([[v[0], v[1]], [v[1], v[2]]])
def sqrtm_spd(X):
    w, Q = np.linalg.eigh(X); return Q @ np.diag(np.sqrt(np.maximum(w,1e-12))) @ Q.T

A = np.array([[3.0, 0.6], [0.6, 1.0]]); Ah = sqrtm_spd(A); Ahinv = np.linalg.inv(Ah)
Nstar = 8.0; kappa = 0.5 + 1.0/(4*Nstar); lam = 0.8
def L_eff(v):
    M = vech_to_M(v); Gh = Ahinv @ sqrtm_spd(Ah @ M @ Ah) @ Ahinv
    return kappa*np.trace(A @ Gh) - 0.5*lam*np.log(max(np.linalg.det(M),1e-12))
def grad_L(v, h=1e-5):
    g=np.zeros(3)
    for i in range(3):
        vp,vm=v.copy(),v.copy(); vp[i]+=h; vm[i]-=h; g[i]=(L_eff(vp)-L_eff(vm))/(2*h)
    return g

v0 = np.array([0.8, 0.15, 0.5]); gL = grad_L(v0)
# strongly anisotropic, misaligned modifier metric (dominant axis ~orthogonal to grad L)
u = np.array([-gL[1], gL[0], 0.3]); u/=np.linalg.norm(u)
w2 = np.array([1.0,1.0,1.0]); w2-= (w2@u)*u; w2/=np.linalg.norm(w2)
w3 = np.cross(u,w2)
G_M = 0.03*(8*np.outer(u,u) + 1*np.outer(w2,w2) + 1*np.outer(w3,w3))
def true_drift(v): return -G_M @ grad_L(v)
b_true = true_drift(v0)

fast_relax, fast_noise = 0.3, 0.4
fvar = fast_noise**2/(2*fast_relax-fast_relax**2)
def increment(v, etaM, fs, sel=0.05, ind=0.01, r=None):
    r = rng if r is None else r
    fs = (1-fast_relax)*fs + fast_noise*r.standard_normal(2)
    f = sel*np.array([fs[0]**2-fvar, fs[0]*fs[1], fs[1]**2-fvar])   # zero-mean over fast eq
    dM = etaM*true_drift(v) + etaM*f + ind*np.sqrt(etaM)*(G_M@r.standard_normal(3))
    return dM, fs
def estimate(v0, R, tau, etaM=0.05, seed=1):
    r=np.random.default_rng(seed); acc=np.zeros(3)
    for _ in range(R):
        v=v0.copy(); fs=r.standard_normal(2)
        for _ in range(tau):
            dM,fs=increment(v,etaM,fs,r=r); v=v+dM
        acc += (v-v0)/(tau*etaM)
    return acc/R
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

print("=== (1) recovery vs replicates R (small steps etaM=0.05, tau=30) ===")
print(f"{'R':>6} | {'cos(dir)':>9} {'mag err':>9}")
for R in [200,1000,5000]:
    bh=estimate(v0,R,30,seed=R)
    print(f"{R:>6} | {cos(bh,b_true):>9.4f} {abs(np.linalg.norm(bh)-np.linalg.norm(b_true))/np.linalg.norm(b_true):>9.3f}")

print("\n=== (1b) magnitude bias from distance travelled during window (etaM*tau) ===")
print(f"{'etaM':>6} {'tau':>5} {'travel':>8} | {'cos(dir)':>9} {'mag err':>9}")
for etaM,tau in [(0.02,30),(0.05,30),(0.2,30),(0.5,30)]:
    bh=estimate(v0,4000,tau,etaM=etaM,seed=11)
    travel=np.linalg.norm(b_true)*etaM*tau
    print(f"{etaM:>6} {tau:>5} {travel:>8.3f} | {cos(bh,b_true):>9.4f} "
          f"{abs(np.linalg.norm(bh)-np.linalg.norm(b_true))/np.linalg.norm(b_true):>9.3f}")

print("\n=== (2) metric confound + estimated G_M ===")
bh=estimate(v0,5000,30,seed=7)
print(f"  cos(drift, -grad L)        = {cos(bh,-gL):.3f}   <- NAIVE, no metric: the confound")
print(f"  cos(drift, -G_M grad L)    = {cos(bh,-G_M@gL):.3f}   <- true metric")
for N in [50,200,2000]:
    bv=rng.multivariate_normal(np.zeros(3),G_M,size=N); Gh=np.cov(bv.T)
    print(f"  cos(drift, -Ghat_M grad L) = {cos(bh,-Gh@gL):.3f}   <- ESTIMATED metric (N={N})")

print("\n=== (3) eigenvalue-scaling vs eigenvector-rotation decomposition ===")
M0=vech_to_M(v0); _,Q0=np.linalg.eigh(M0)
def split(vd):
    R=Q0.T@vech_to_M(vd)@Q0; return np.array([R[0,0],R[1,1]]), np.array([R[0,1]])
sh,rh=split(bh); st,rt=split(b_true)
print(f"  eigenvalue-scaling : cos={cos(sh,st):.3f}  (subspace norm {np.linalg.norm(sh):.2e})")
print(f"  eigenvector-rotate : cos={cos(rh,rt):.3f}  (subspace norm {np.linalg.norm(rh):.2e})")

print("\n=== (4) window tau: long enough to average fast fluctuations ===")
print(f"{'tau':>5} | {'cos(dir)':>9}")
for tau in [3,8,30,100]:
    print(f"{tau:>5} | {cos(estimate(v0,3000,tau,seed=tau),b_true):>9.4f}")

print("\n=== (5) curl detection (V4 machinery): rotational fraction of metric-corrected field ===")
def curl_fraction(extra=0.0):
    pts=np.array([v0+0.06*np.array([i,j,k]) for i in(-1,0,1) for j in(-1,0,1) for k in(-1,0,1)])
    J=np.array([[0,1,0],[-1,0,1],[0,-1,0]],float)
    fields=np.array([np.linalg.solve(G_M, true_drift(p)+extra*(J@(p-v0))) for p in pts])
    Z=np.hstack([pts-v0, np.ones((len(pts),1))]); Kc,*_=np.linalg.lstsq(Z,fields,rcond=None)
    K=Kc[:3].T; asym=0.5*(K-K.T)
    return np.linalg.norm(asym)/np.linalg.norm(K)     # in [0,1]
print(f"  pure gradient drift   curl-frac = {curl_fraction(0.0):.3f}   (~0 expected)")
print(f"  injected curl=0.3     curl-frac = {curl_fraction(0.3):.3f}")
print(f"  injected curl=1.0     curl-frac = {curl_fraction(1.0):.3f}   (grows monotonically)")
