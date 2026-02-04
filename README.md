# V-Learner: Cross-Fit Doubly Robust CATE for Uplift

This repository implements **V-Learner** for heterogeneous treatment effect (HTE) estimation.
The V-Learner estimates the conditional average treatment effect (CATE)

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\tau(x)%20:=%20\mathbb{E}[Y(1)-Y(0)\mid%20X=x]" />
</p>

from observational (or randomized) data <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}(X_i,%20T_i,%20Y_i)_{i=1}^n" />, where <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20T\in\{0,1\}" />.

The core idea is to convert CATE estimation into supervised regression by constructing a
**cross-fitted doubly robust (DR) pseudo-outcome** and regressing it on covariates.

## Is V-Learner "fundamentally different" than S / T / X learners?

Yes, in mechanism, not in goal. All meta-learners aim to learn <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\tau(x)" />, but they differ in what
supervised learning problem they reduce CATE estimation to.

Where **S/T/X** learners are primarily *outcome-modeling* strategies (fit outcome models, then
transform them into effects), the **V-Learner** is an *orthogonalized pseudo-outcome* strategy:

- It constructs a single per-unit training target <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau_{DR}(X_i)" /> that is already an estimate of the
  individual treatment effect, and then
- learns a CATE model by regressing that target on <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20X" />.

This makes the V-Learner closest in spirit to what is often called a DR-learner / AIPW pseudo-outcome
learner, with the extra "V" emphasis in this repo coming from variance stabilization and practical
uncertainty tooling.

## Why V-Learner?

Classic meta-learners estimate CATE indirectly:

- **S-Learner:** fit one outcome model <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\mu(X,T)" /> with <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20T" /> as a feature, then
  <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau(x)=\hat\mu(x,1)-\hat\mu(x,0)" />.
- **T-Learner:** fit separate outcome models per arm, <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\mu_0(x),\hat\mu_1(x)" />, then
  <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x)" />.
- **X-Learner:** impute individual effects using outcome models + propensity weighting, then fit effect models.

The **V-Learner** instead constructs a single training target using a doubly robust (DR) expression:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau_{\mathrm{DR}}(X)%20=%20\big(\hat\mu_1(X)%20-%20\hat\mu_0(X)\big)%20+%20\frac{T\,(Y-\hat\mu_1(X))}{\hat%20e(X)}%20-%20\frac{(1-T)\,(Y-\hat\mu_0(X))}{1-\hat%20e(X)}" />
</p>

Then it learns <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\tau(x)" /> by regressing <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau_{\mathrm{DR}}" /> on <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20X" />.

### What you get

- **Orthogonalization (DR / AIPW):** reduced sensitivity to nuisance-model error.
  Intuition: first-order nuisance errors cancel in the pseudo-outcome, so the second stage is less fragile.
- **Cross-fitting:** prevents leakage when creating pseudo-outcomes.
- **Overlap-aware stability (the "V" in practice):** optional variance-stabilizing weights
  <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20w(x)=\hat%20e(x)(1-\hat%20e(x))" />.
- **Uncertainty signal:** bootstrap ensemble standard deviation + optional conformal prediction intervals.

## Meta-learner mechanism differences (S vs T vs X vs V)

Below is a mechanistic comparison on what each learner fits and what supervision signal it uses.

| Learner | What you fit (high level) | Supervision target in 2nd stage | Key failure mode | When it tends to shine |
|---|---|---|---|---|
| **S** | Single outcome model <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\mu(X,T)" /> | None (effect is plug-in difference) | Underfits interactions; treatment effect can be "washed out" if <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20T" /> is weak signal | Strong regularization / limited data; simple effects |
| **T** | Two outcome models <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\mu_0(x),\hat\mu_1(x)" /> | None (difference of two predictors) | High variance if one arm is small; subtracting two noisy models | RCT-like balance, plenty of data per arm |
| **X** | Outcome models + imputation + effect models | Imputed ITEs (often weighted) | Can be sensitive to propensity / arm imbalance; multiple stages can compound error | Strong arm imbalance; good outcome models |
| **V (this repo)** | Propensity + both outcome models + one effect model | DR pseudo-outcome <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau_{DR}(X)" /> | Pseudo-outcomes can be high-variance when overlap is poor; needs clipping/regularization | Observational data; when robustness to nuisance error matters |

**Core Difference: V vs S/T/X:** the V-Learner's target is already orthogonalized via DR/AIPW, so you are not
learning effects by subtracting outcome models (T/S), nor by multi-step imputation (X). Instead you learn
"directly from a debiased signal".

## Setup and identification

Assume standard causal identification conditions:

1. **Consistency/SUTVA:** <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20Y%20=%20Y(T)" />
2. **Unconfoundedness:** <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}(Y(0),Y(1))%20\perp%20T%20\mid%20X" />
3. **Overlap:** <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}0%20%3C%20e(x)%20%3C%201" /> where <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20e(x)%20=%20\mathbb{P}(T=1\mid%20X=x)" />

Define nuisance functions:

- Propensity: <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20e(x)%20=%20\mathbb{P}(T=1\mid%20X=x)" />
- Outcome regressions: <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\mu_t(x)%20=%20\mathbb{E}[Y%20\mid%20X=x,%20T=t]" /> for <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20t\in\{0,1\}" />

## Algorithm

The DR + cross-fitting pipeline is as follow:

1. **Cross-fit nuisance models (K-fold):**
   - Fit a propensity model <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat%20e(x)=\mathbb{P}(T=1\mid%20X=x)" />.
   - Fit two outcome models <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\mu_0(x),%20\hat\mu_1(x)" />.
   - Predict <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat%20e_i,%20\hat\mu_{0,i},%20\hat\mu_{1,i}" /> *out-of-fold* for every sample.

2. **Construct the DR pseudo-outcome** per sample:

   <p align="center">
   <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau_{DR,i}%20=%20(\hat\mu_{1,i}-\hat\mu_{0,i})%20+%20\frac{T_i(Y_i-\hat\mu_{1,i})}{\hat%20e_i}%20-%20\frac{(1-T_i)(Y_i-\hat\mu_{0,i})}{1-\hat%20e_i}" />
   </p>

   Practical guards included here:
   - **Propensity clipping** (default `0.02–0.98`) to reduce extreme inverse-weight blowups.
   - **Outcome clipping** for binary outcomes to keep probabilities away from 0/1.
   - **Winsorization** of pseudo-outcomes (default 1st–99th quantile) to control heavy tails.

3. **Second-stage CATE regression:** fit a single model <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau(x)" /> with optional
   **variance-stabilizing weights**

   <p align="center">
   <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20w(x)%20=%20\hat%20e(x)(1-\hat%20e(x))" />
   </p>

   which downweights regions with poor overlap where the DR pseudo-outcome has intrinsically higher variance.

4. **Uncertainty tooling (practical, not asymptotic):**
   - **Bootstrap ensemble** of the second-stage model for an epistemic-uncertainty proxy
     (use `effect_uncertainty`).
   - **Leakage-free conformal prediction intervals:**
     split off a calibration set *after* pseudo-outcomes are computed, calibrate a residual quantile,
     and return <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau(x)\pm%20q" /> (use `effect_interval`).

## Conformal intervals

The conformal interval is calibrated on residuals of the *pseudo-outcome regression target*
<img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\hat\tau_{\mathrm{DR}}" /> on a held-out calibration split.

**Important:** this is a predictive interval for the *second-stage regression target*,
not a causal confidence interval for the structural CATE <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\tau(x)" />.
In practice it can be conservative / wide when pseudo-outcomes are high-variance
(e.g., binary <img src="https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}%20Y" />, limited overlap).

## API

```python
from v_learner import VLearner

v = VLearner(...)
v.fit(Y, T, X)

tau_hat = v.effect(X_new)
uncertainty = v.effect_uncertainty(X_new)
lo, hi = v.effect_interval(X_new, alpha=0.1)
```
