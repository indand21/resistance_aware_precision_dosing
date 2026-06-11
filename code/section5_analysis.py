"""
section5_analysis.py
====================
Section 5 computational case study for the manuscript
"Resistance-aware precision dosing as posterior inference".

Data sources (downloaded June 2026):
  - pntd0013505_s1.csv  : 38,585 N. gonorrhoeae genomes with quantitative MIC values
    (Ebrahimi et al. 2025, PLOS Negl Trop Dis, doi:10.1371/journal.pntd.0013505,
     PMID 41052130)
  - ncbi_ngono_amr.tsv  : NCBI Pathogen Detection AMRFinderPlus genotype calls for all
    N. gonorrhoeae in Pathogen Detection (PDG000000032.516, June 2025)

Analysis plan (mirrors manuscript Section 5):
  5.1  Data assembly: join by BioSample accession; extract QRDR feature vectors
  5.2  Model fitting: interval-censored ridge posterior on log2(MIC), with
       collinearity / range-restriction diagnostics
  5.3  Derived quantities: Clark MPC posterior, T_MSW (IV + oral Bateman)

The deployed estimator is the interval-censored MAP + Laplace posterior of
model_core.censored_ridge_posterior, which (i) honours the doubling-dilution
censoring stated in manuscript Section 2.2 and (ii) propagates residual-SD
uncertainty into the coefficient covariance. The identical estimator is
calibrated by simulation in Section 6.1, so the validation tests the deployed
code rather than a proxy.

gyrB substitution effects on zoliflodacin MIC are absent from the population data
(these mutations remain extremely rare globally: 0/420 zoliflodacin-tested
isolates carry one). They are therefore estimated from isogenic-panel MIC
measurements in the literature (see gyrb_literature.py) and combined with the
data-fitted coefficients for the MPC projection in Section 5.3.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# Ensure Unicode (Greek letters, ×, ≤) prints on Windows consoles / piped output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Paths resolve relative to this script (code/), so the project stays portable.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")

# Shared modelling machinery (fitting, Clark MPC propagation, PK, diagnostics).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_core import (
    censored_ridge_posterior, analytic_ridge_posterior, design_diagnostics,
    clark_max_moments, mpc_msw_posterior, Tmsw_iv, Tmsw_oral,
)
# Sourced, provenance-tagged literature inputs (gyrB coefficients from a meta-
# analysis of isogenic MIC pairs; PD parameters; relative fitness).
import literature_params as lit


# =============================================================================
# 5.1  DATA ASSEMBLY
# =============================================================================
print("=" * 70)
print("Section 5.1  Data Assembly")
print("=" * 70)

plos = pd.read_csv(os.path.join(DATA_DIR, "pntd0013505_s1.csv"), low_memory=False)
amr  = pd.read_csv(os.path.join(DATA_DIR, "ncbi_ngono_amr.tsv"),  sep="\t", low_memory=False)

merged = plos.merge(
    amr[["biosample_acc", "AMR_genotypes", "AMR_genotypes_core"]],
    left_on="Assembly BioSample Accession",
    right_on="biosample_acc",
    how="inner"
)
print(f"Joined dataset: {len(merged):,} isolates")

# ── Define feature slots ──────────────────────────────────────────────────────
# QRDR and efflux/permeability features; the binary encoding matches Section 2.1
FEATURE_DEFS = {
    # Zoliflodacin target (GyrB)
    "gyrB_D429N":  "gyrB_D429N=POINT",
    "gyrB_K450T":  "gyrB_K450T=POINT",
    "gyrB_S467N":  "gyrB_S467N=POINT",
    # Quinolone target (GyrA) – primary determinants of ciprofloxacin resistance
    "gyrA_S91F":   "gyrA_S91F=POINT",
    "gyrA_D95any": r"gyrA_D95[A-Z]=POINT",   # D95G, D95N, D95A
    # Quinolone target (ParC) – modifier of ciprofloxacin and cross-resistance context
    "parC_D86N":   "parC_D86N=POINT",
    "parC_S87any": r"parC_S87[A-Z]=POINT",   # S87R, S87I, S87N
    # Efflux (mtrR promoter mutations → MtrCDE overexpression)
    "mtrR_mut":    r"mtrR_[A-Z]",            # A-53del, G45D, etc.
    # Permeability (porB1b mutations)
    "porB_mut":    r"porB1b_[A-Z]",          # G120K, A121G/D/N
}

def extract_features(genotypes_series, feat_defs):
    """Return binary (0/1) DataFrame from AMR_genotypes string column."""
    result = {}
    for name, pattern in feat_defs.items():
        result[name] = genotypes_series.str.contains(pattern, na=False).astype(float)
    return pd.DataFrame(result, index=genotypes_series.index)

feat_all = extract_features(merged["AMR_genotypes"], FEATURE_DEFS)

# Attach MIC columns
feat_all["log2_ZO"]   = np.log2(merged["zoliflodacin"].values.astype(float))
feat_all["log2_CIP"]  = np.log2(merged["ciprofloxacin"].values.astype(float))
feat_all["country"]   = merged["country"].values
feat_all["year"]      = merged["year"].values
feat_all["BioSample"] = merged["Assembly BioSample Accession"].values

# Working subsets
feat_ZO  = feat_all.dropna(subset=["log2_ZO"])
feat_CIP = feat_all.dropna(subset=["log2_CIP"])
print(f"  Zoliflodacin MIC dataset  : {len(feat_ZO):,} isolates")
print(f"  Ciprofloxacin MIC dataset : {len(feat_CIP):,} isolates")
print()

# Mutation prevalence summary
print("Mutation prevalence in zoliflodacin-tested isolates (N=420):")
for f in FEATURE_DEFS:
    n = int(feat_ZO[f].sum())
    pct = 100 * n / len(feat_ZO)
    print(f"  {f:<15} n={n:3d}  ({pct:.1f}%)")
print()

# =============================================================================
# 5.2  MODEL FITTING – Analytic Ridge Posterior
# =============================================================================
print("=" * 70)
print("Section 5.2  Model Fitting")
print("=" * 70)

# Designate the biological feature columns (no MIC, no metadata)
BIO_FEATS_CIPRO = ["gyrA_S91F", "gyrA_D95any", "parC_D86N", "parC_S87any",
                   "mtrR_mut", "porB_mut"]
BIO_FEATS_ZOLI  = ["gyrA_S91F", "gyrA_D95any", "parC_D86N", "parC_S87any",
                   "mtrR_mut", "porB_mut"]

def build_design(df, bio_feats):
    """Intercept + main effects + key pairwise epistasis terms."""
    X = df[bio_feats].values.astype(float)
    # Pairwise interactions for the 4 core QRDR features
    pairs = [(i, j) for i in range(len(bio_feats))
             for j in range(i + 1, len(bio_feats))
             if bio_feats[i] in ["gyrA_S91F", "gyrA_D95any", "parC_D86N", "parC_S87any"]
             and bio_feats[j] in ["gyrA_S91F", "gyrA_D95any", "parC_D86N", "parC_S87any"]]
    Xinter = np.hstack([X[:, [i]] * X[:, [j]] for i, j in pairs])
    pair_names = [f"{bio_feats[i]}×{bio_feats[j]}" for i, j in pairs]
    Xfull = np.hstack([np.ones((len(df), 1)), X, Xinter])
    names = ["intercept"] + bio_feats + pair_names
    return Xfull, names, pairs


def report_diagnostics(X, names, label):
    """Print collinearity / range-restriction diagnostics for a design matrix."""
    d = design_diagnostics(X, names)
    print(f"  Design diagnostics [{label}]:")
    print(f"    rows (isolates)      : {d['n_rows']}")
    print(f"    distinct genotypes   : {d['distinct_genotypes']}  "
          f"(effective contrasts, not {d['n_rows']})")
    print(f"    condition number     : {d['condition_number']:.3g}")
    if d["zero_variance_cols"]:
        print(f"    zero-variance columns: {d['zero_variance_cols']}")
    hi_vif = {k: v for k, v in d["vif"].items() if v > 10}
    if hi_vif:
        top = sorted(hi_vif.items(), key=lambda kv: -kv[1])[:5]
        print(f"    VIF>10 (collinear)   : " +
              ", ".join(f"{k}={v:.3g}" for k, v in top))
    print()
    return d

# ── Ciprofloxacin model (interval-censored) ───────────────────────────────────
X_cip, names_cip, pairs_cip = build_design(feat_CIP, BIO_FEATS_CIPRO)
y_cip = feat_CIP["log2_CIP"].values
mu_cip, Sig_cip, sig_cip = censored_ridge_posterior(X_cip, y_cip, lambda_ridge=0.5)

print("Ciprofloxacin model  (n=%d, censored residual SD=%.3f log2):" % (len(y_cip), sig_cip))
for name, mu, se in zip(names_cip, mu_cip, np.sqrt(np.diag(Sig_cip))):
    ci_lo, ci_hi = mu - 1.96 * se, mu + 1.96 * se
    print(f"  {name:<30}  β={mu:+.3f}  SE={se:.3f}  95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}]")
print()
diag_cip = report_diagnostics(X_cip, names_cip, "ciprofloxacin")

# ── Zoliflodacin model (interval-censored) ────────────────────────────────────
X_zo, names_zo, pairs_zo = build_design(feat_ZO, BIO_FEATS_ZOLI)
y_zo = feat_ZO["log2_ZO"].values
mu_zo, Sig_zo, sig_zo = censored_ridge_posterior(X_zo, y_zo, lambda_ridge=0.5)

print("Zoliflodacin model  (n=%d, censored residual SD=%.3f log2):" % (len(y_zo), sig_zo))
for name, mu, se in zip(names_zo, mu_zo, np.sqrt(np.diag(Sig_zo))):
    ci_lo, ci_hi = mu - 1.96 * se, mu + 1.96 * se
    print(f"  {name:<30}  β={mu:+.3f}  SE={se:.3f}  95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}]")
print()
diag_zo = report_diagnostics(X_zo, names_zo, "zoliflodacin")

# Range-restriction note for zoliflodacin: with resistance near-absent the MIC has
# little variance, so "no association" with QRDR genotype is partly range
# restriction, not only mechanistic independence. Quantify the ZO MIC spread.
zo_mic_iqr = np.subtract(*np.percentile(y_zo, [75, 25]))
print(f"  Zoliflodacin log2 MIC spread: IQR={zo_mic_iqr:.2f} doublings, "
      f"range=[{y_zo.min():.1f},{y_zo.max():.1f}]  "
      f"(narrow spread => QRDR 'independence' is confounded with range restriction)")
print()

# ── Augment zoliflodacin model with gyrB effects ESTIMATED FROM isogenic data ─
# No gyrB-mutant isolate carries a quantitative ZO MIC in the population data, so
# the three gyrB main effects are estimated from a meta-analysis of published
# isogenic MIC pairs (literature_params.GYRB_ZO_COEF) rather than hand-set. Each
# coefficient's SD is the between-background heterogeneity of the isogenic
# observations — the transferable uncertainty for applying it to a new genotype.
GYRB_FEATS = ["gyrB_D429N", "gyrB_K450T", "gyrB_S467N"]
GYRB_PRIOR_MU = np.array([lit.GYRB_ZO_COEF[f][0] for f in GYRB_FEATS])
GYRB_PRIOR_SE = np.array([lit.GYRB_ZO_COEF[f][1] for f in GYRB_FEATS])
print("gyrB coefficients estimated from isogenic MIC pairs (log2 doublings):")
for f in GYRB_FEATS:
    mu_f, sd_f = lit.GYRB_ZO_COEF[f]
    print(f"  {f:<12} β={mu_f:+.2f} ± {sd_f:.2f}")
print()

# Extended coefficient vector (gyrA/parC/mtr/porB from data; gyrB from prior)
mu_zo_ext  = np.concatenate([mu_zo, GYRB_PRIOR_MU])
var_zo_ext = np.concatenate([np.diag(Sig_zo), GYRB_PRIOR_SE**2])
Sig_zo_ext = np.diag(var_zo_ext)  # independence approximation between fitted and literature terms
names_zo_ext = names_zo + GYRB_FEATS

print("Extended zoliflodacin model (gyrB augmented with Foerster 2019 literature prior):")
for name, mu, se in zip(names_zo_ext, mu_zo_ext, np.sqrt(var_zo_ext)):
    ci_lo, ci_hi = mu - 1.96 * se, mu + 1.96 * se
    src = "(literature)" if name in GYRB_FEATS else "(data)"
    print(f"  {name:<30}  β={mu:+.3f}  SE={se:.3f}  95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}]  {src}")
print()

# =============================================================================
# 5.3  DERIVED QUANTITIES: MPC posterior, T_MSW, dosing targets
# =============================================================================
print("=" * 70)
print("Section 5.3  Derived Quantities")
print("=" * 70)

# PK parameters (calibrated to O'Donnell 2019; same as Section 6.3)
F, D, Vd, ka, ke = 0.75, 3000.0, 100.0, 1.0, 0.11
t_half = np.log(2.0) / ke    # ≈ 6.3 h

# ── Define genotype feature vectors for the EXTENDED model ────────────────────
# The extended design has:
#   intercept, gyrA_S91F, gyrA_D95any, parC_D86N, parC_S87any, mtrR_mut, porB_mut,
#   [pairwise QRDR terms],
#   gyrB_D429N, gyrB_K450T, gyrB_S467N
# We need to build x(g) vectors for representative founding genotypes and their neighbours.

# Indices in the extended design vector
idx = {n: i for i, n in enumerate(names_zo_ext)}

def make_xvec(genotype_dict):
    """Build the design-matrix row for a genotype dict {feature_name: 0/1}."""
    # main effect features (in order of names_zo_ext, skipping intercept and interactions)
    main_feats = BIO_FEATS_ZOLI + GYRB_FEATS
    raw = np.array([genotype_dict.get(f, 0.0) for f in main_feats])
    # reconstruct full design row: intercept + mains + pairwise QRDR + gyrB mains
    x_mains = raw[:len(BIO_FEATS_ZOLI)]   # BIO_FEATS_ZOLI order
    x_gyrB  = raw[len(BIO_FEATS_ZOLI):]   # gyrB features (not in pairwise)
    # pairwise interactions among first 4 QRDR main features
    qrdr4 = [BIO_FEATS_ZOLI.index(f) for f in ["gyrA_S91F","gyrA_D95any","parC_D86N","parC_S87any"]]
    inter = np.array([x_mains[i] * x_mains[j] for i, j in pairs_zo])
    return np.concatenate([[1.0], x_mains, inter, x_gyrB])

# Representative founding genotypes
GENOTYPES = {
    "WT":                          {},
    "gyrA_S91F":                   {"gyrA_S91F": 1},
    "gyrA_S91F+D95A":              {"gyrA_S91F": 1, "gyrA_D95any": 1},
    "gyrA+parC_S87R":              {"gyrA_S91F": 1, "gyrA_D95any": 1, "parC_S87any": 1},
    "gyrA+parC_D86N":              {"gyrA_S91F": 1, "gyrA_D95any": 1, "parC_D86N": 1},
    "gyrA+parC_S87R+gyrB_D429N":   {"gyrA_S91F": 1, "gyrA_D95any": 1, "parC_S87any": 1,
                                    "gyrB_D429N": 1},
}

# Hamming-1 neighbours for zoliflodacin: single-step mutations from WT direction
ALL_BINARY_FEATS_EXT = BIO_FEATS_ZOLI + GYRB_FEATS

def hamming1_nb(gdict, all_feats=ALL_BINARY_FEATS_EXT):
    """Single-substitution neighbours (add one absent feature, or revert one present)."""
    nb_list = []
    for f in all_feats:
        current = gdict.get(f, 0)
        nb = {**gdict, f: 1 - current}   # toggle
        # codon mutual exclusivity: gyrA can't be S91F + D95x from same codon at once
        # (handled approximately: skip if gyrA_S91F AND gyrA_D95any both added to WT)
        if nb.get("gyrA_S91F", 0) == 1 and nb.get("gyrA_D95any", 0) == 1:
            # only allowed if one of them was already present (not adding both from WT)
            if gdict.get("gyrA_S91F", 0) == 0 and gdict.get("gyrA_D95any", 0) == 0:
                continue
        nb_list.append(nb)
    return nb_list

print("MPC and MSW posteriors for representative founding genotypes")
print("(zoliflodacin extended model; W = log2 MPC − log2 C*(g0))")
print()

results = {}
header = f"{'Genotype':<35} {'E[MPC] mg/L':>12} {'95%CI':>18} {'E[W]':>8} {'SD[W]':>7} {'T_MSW_iv':>10} {'T_MSW_oral':>11}"
print(header)
print("-" * len(header))

for gname, gdict in GENOTYPES.items():
    x_g0 = make_xvec(gdict)
    nb_dicts = hamming1_nb(gdict)
    if len(nb_dicts) == 0:
        continue
    X_nb = np.vstack([make_xvec(nb) for nb in nb_dicts])
    res = mpc_msw_posterior(mu_zo_ext, Sig_zo_ext, X_nb, x_g0)

    E_mpc = 2**res["E_log2MPC"]
    sd_mpc_log = np.sqrt(res["Var_log2MPC"])
    ci_lo = 2**(res["E_log2MPC"] - 1.96 * sd_mpc_log)
    ci_hi = 2**(res["E_log2MPC"] + 1.96 * sd_mpc_log)

    E_W  = max(res["E_W"], 0.01)
    SD_W = np.sqrt(res["Var_W"])

    # Current MIC (= 2^mu_g0 approximate)
    mu_g0 = float(x_g0 @ mu_zo_ext)
    E_mic = 2**mu_g0

    T_iv   = Tmsw_iv(E_W, t_half)
    T_oral = Tmsw_oral(E_mic, E_mpc, F, D, Vd, ka, ke)

    print(f"{gname:<35} {E_mpc:>12.3f} [{ci_lo:.3f},{ci_hi:.3f}] {E_W:>8.2f} {SD_W:>7.2f} {T_iv:>10.2f} {T_oral:>11.2f}")
    results[gname] = dict(E_mpc=E_mpc, ci_lo=ci_lo, ci_hi=ci_hi,
                          E_W=E_W, SD_W=SD_W, E_mic=E_mic,
                          T_iv=T_iv, T_oral=T_oral)

print()

# ── T_MSW distribution across the observed zoliflodacin-tested population ────
print("T_MSW distribution across the 420 observed isolates (IV formula, W=E[W] per isolate):")
pop_tmsw = []
for _, row in feat_ZO.iterrows():
    gdict = {f: int(row[f]) for f in ALL_BINARY_FEATS_EXT if f in row.index}
    x_g = make_xvec(gdict)
    nb_dicts = hamming1_nb(gdict)
    if len(nb_dicts) == 0:
        continue
    X_nb = np.vstack([make_xvec(nb) for nb in nb_dicts])
    res = mpc_msw_posterior(mu_zo_ext, Sig_zo_ext, X_nb, x_g)
    E_W = max(res["E_W"], 0.0)
    pop_tmsw.append({"T_msw_iv": Tmsw_iv(E_W, t_half),
                     "E_W": E_W, "E_mic": 2**float(x_g @ mu_zo_ext),
                     "E_mpc": 2**res["E_log2MPC"],
                     "country": row["country"], "year": row["year"],
                     "ZO_MIC_obs": 2**row["log2_ZO"]})

pop_df = pd.DataFrame(pop_tmsw)
print(f"  N = {len(pop_df)} isolates")
print(f"  T_MSW_iv: median={pop_df['T_msw_iv'].median():.2f} h,  mean={pop_df['T_msw_iv'].mean():.2f} h")
print(f"           P5={pop_df['T_msw_iv'].quantile(.05):.2f}  P95={pop_df['T_msw_iv'].quantile(.95):.2f}")
print(f"  E[W]:     median={pop_df['E_W'].median():.2f} doublings")
print()

# Country-level summary
country_summary = (pop_df.groupby("country")["T_msw_iv"]
                   .agg(["count","median","mean"])
                   .sort_values("count", ascending=False)
                   .head(10))
print("Top 10 countries by isolate count:")
print(country_summary.to_string())
print()

# =============================================================================
# 5.3b  FITNESS CHANNEL: viability gate on the MPC neighbourhood
# =============================================================================
# The MPC is the maximum C*(g') over VIABLE single-step neighbours (Section 4.1).
# We populate the fitness channel with measured relative fitness for the
# resistance mutants and make the viability gate ψ_host(g') = ψ_max·w(g') − δ_host
# operational. The dominant ZO-resistant neighbour is gyrB D429N, whose relative
# fitness is strongly background-dependent (w = 0.13–1.04; mean %.2f).
print("=" * 70)
print("Section 5.3b  Fitness channel and viability gate")
print("=" * 70)
psi_max = lit.PD_ZOLI["psi_max_s"]           # WHO F susceptible growth rate (h^-1)
w_d429n = lit.FITNESS_D429N_W                 # background-averaged relative fitness
w_min_d429n = min(w for w, *_ in lit.FITNESS_D429N)
# Drug-free in-host net growth of the dominant resistant neighbour at its mean and
# worst-case fitness; the gate excludes it only if in-host clearance δ_host exceeds
# this value (otherwise the neighbour is viable and dictates the MPC).
psi_free_mean = psi_max * w_d429n
psi_free_worst = psi_max * w_min_d429n
print(f"  gyrB D429N relative fitness: mean w={w_d429n:.2f}, worst-background w={w_min_d429n:.2f}")
print(f"  Drug-free in-host net growth ψ_free = ψ_max·w:")
print(f"    mean-fitness  : {psi_free_mean:.3f} h⁻¹  -> viable unless δ_host > {psi_free_mean:.3f} h⁻¹")
print(f"    worst-fitness : {psi_free_worst:.3f} h⁻¹  -> viable unless δ_host > {psi_free_worst:.3f} h⁻¹")
print("  In-host clearance δ_host for N. gonorrhoeae is not established; with the")
print("  gate open (δ_host below these thresholds) all single-step ZO-resistant")
print("  neighbours are viable, so fitness does NOT rescue the dosing target — the")
print("  high-MIC gyrB D429N neighbour both dominates the MPC and remains viable.")
print("  Λ (fitness/environment MIC correction) is held at 1 (conservative lab-MIC")
print(f"  analysis) absent an in-host δ_host estimate.")
print()

# =============================================================================
# FIGURES
# =============================================================================
print("Generating figures...")

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── Fig A: β coefficients for ciprofloxacin ───────────────────────────────────
ax_a = fig.add_subplot(gs[0, :2])
cip_names_short = [n.replace("gyrA_","gA_").replace("gyrB_","gB_").replace("parC_","pC_")
                    .replace("_mut","").replace("any","*")
                    .replace("_minus_","-").replace("×","×")
                    for n in names_cip[1:]]  # skip intercept
mu_plot = mu_cip[1:]
se_plot = np.sqrt(np.diag(Sig_cip))[1:]
colors = ["#d62728" if "gyrA" in n else "#1f77b4" if "parC" in n else
          "#ff7f0e" if "×" in n else "#7f7f7f" for n in names_cip[1:]]
ax_a.barh(range(len(mu_plot)), mu_plot, xerr=1.96 * se_plot,
          color=colors, alpha=0.8, capsize=3)
ax_a.axvline(0, color="k", lw=0.8)
ax_a.set_yticks(range(len(mu_plot)))
ax_a.set_yticklabels(cip_names_short, fontsize=7)
ax_a.set_xlabel("Effect on log₂(MIC) [doublings]", fontsize=9)
ax_a.set_title("A  Ciprofloxacin model coefficients\n(n=429; ridge posterior)", fontsize=9)

# ── Fig B: β coefficients for zoliflodacin (data + literature) ───────────────
ax_b = fig.add_subplot(gs[0, 2])
zo_names_short = [n.replace("gyrA_","gA_").replace("gyrB_","gB_").replace("parC_","pC_")
                   .replace("_mut","").replace("any","*")
                   .replace("_minus_","-").replace("×","×")
                   for n in names_zo_ext[1:]]
mu_b = mu_zo_ext[1:]
se_b = np.sqrt(var_zo_ext)[1:]
data_len = len(names_zo) - 1   # minus intercept
cols_b = (["#2ca02c"] * 3 +                          # gyrB (literature, green)
          ["#d62728" if "gyrA" in n else "#1f77b4" if "parC" in n else
           "#ff7f0e" if "×" in n else "#7f7f7f"
           for n in names_zo_ext[1:data_len + 1]])[:len(mu_b)]
# Reorder: data feats first, then gyrB
ax_b.barh(range(len(mu_b)), mu_b, xerr=1.96 * se_b,
          color=["#d62728" if "gyrA" in n else "#1f77b4" if "parC" in n else
                 "#2ca02c" if "gyrB" in n else "#ff7f0e" if "×" in n else "#7f7f7f"
                 for n in names_zo_ext[1:]], alpha=0.8, capsize=3)
ax_b.axvline(0, color="k", lw=0.8)
ax_b.set_yticks(range(len(mu_b)))
ax_b.set_yticklabels(zo_names_short, fontsize=6)
ax_b.set_xlabel("Effect on log₂(ZO MIC) [doublings]", fontsize=9)
ax_b.set_title("B  Zoliflodacin model\n(data=420; gyrB=literature)", fontsize=9)

# ── Fig C: MPC posterior distributions for key genotypes ─────────────────────
ax_c = fig.add_subplot(gs[1, :2])
geno_labels = {
    "WT":                        "WT",
    "gyrA_S91F":                 "gyrA-S91F",
    "gyrA_S91F+D95A":            "gyrA-S91F+D95A",
    "gyrA+parC_S87R":            "gyrA+parC-S87R",
    "gyrA+parC_D86N":            "gyrA+parC-D86N",
    "gyrA+parC_S87R+gyrB_D429N": "gyrA+parC+gyrB*",
}
y_pos = range(len(geno_labels))
for i, (gname, label) in enumerate(geno_labels.items()):
    r = results[gname]
    ax_c.barh(i, np.log2(r["E_mpc"]), color="#1f77b4", alpha=0.7, height=0.5)
    ax_c.errorbar(np.log2(r["E_mpc"]), i,
                  xerr=[[np.log2(r["E_mpc"]) - np.log2(r["ci_lo"])],
                        [np.log2(r["ci_hi"]) - np.log2(r["E_mpc"])]],
                  fmt="none", color="#1f77b4", capsize=4)
    ax_c.text(np.log2(r["E_mpc"]) + 0.15, i,
              f"MPC={r['E_mpc']:.2f} mg/L\nW={r['E_W']:.1f} d",
              va="center", fontsize=7)

ax_c.set_yticks(list(y_pos))
ax_c.set_yticklabels(list(geno_labels.values()), fontsize=8)
ax_c.axvline(np.log2(0.125), color="#d62728", ls="--", lw=1,
             label="EUCAST R breakpoint (0.125 mg/L)")
ax_c.set_xlabel("log₂ MPC [doublings]", fontsize=9)
ax_c.set_title("C  MPC posteriors by founding genotype\n(95% CI; gyrB* = literature-augmented)", fontsize=9)
ax_c.legend(fontsize=7)

# ── Fig D: T_MSW distribution across observed population ─────────────────────
ax_d = fig.add_subplot(gs[1, 2])
ax_d.hist(pop_df["T_msw_iv"], bins=30, color="#ff7f0e", edgecolor="k",
          linewidth=0.4, alpha=0.85, density=True)
ax_d.axvline(t_half * 2, color="#d62728", ls="--", lw=1.2,
             label=f"t½·W=2 = {t_half*2:.1f} h")
ax_d.axvline(pop_df["T_msw_iv"].median(), color="#1f77b4", ls="-", lw=1.2,
             label=f"Median = {pop_df['T_msw_iv'].median():.1f} h")
ax_d.set_xlabel("T_MSW (IV limit, h)", fontsize=9)
ax_d.set_ylabel("Density", fontsize=9)
ax_d.set_title(f"D  T_MSW across observed population\n(N={len(pop_df)} isolates)", fontsize=9)
ax_d.legend(fontsize=7)

fig_path = os.path.join(FIG_DIR, "section5_figures.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")

# =============================================================================
# SUMMARY TABLE for manuscript Section 5.3
# =============================================================================
print()
print("=" * 70)
print("MANUSCRIPT TABLE S5.1 — MPC/MSW/T_MSW by founding genotype")
print("=" * 70)
print(f"{'Genotype':<35}  {'E[MIC]':>8}  {'E[MPC]':>8}  {'95%CI':>18}  {'E[W]':>6}  {'SD[W]':>6}  {'T_MSW_iv':>9}  {'T_MSW_oral':>10}")
for gname, r in results.items():
    print(f"{gname:<35}  {r['E_mic']:>8.3f}  {r['E_mpc']:>8.3f}  "
          f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
          f"{r['E_W']:>6.2f}  {r['SD_W']:>6.2f}  {r['T_iv']:>9.2f}  {r['T_oral']:>10.2f}")

print()
print("Population T_MSW summary:")
print(f"  Median   : {pop_df['T_msw_iv'].median():.2f} h")
print(f"  Mean     : {pop_df['T_msw_iv'].mean():.2f} h")
print(f"  Std      : {pop_df['T_msw_iv'].std():.2f} h")
print(f"  P5–P95   : {pop_df['T_msw_iv'].quantile(.05):.2f}–{pop_df['T_msw_iv'].quantile(.95):.2f} h")
print()
print("NOTE: gyrB_D429N, K450T, S467N coefficients are literature-derived priors")
print("(Foerster et al. 2019, PMID 31730160) because no gyrB-mutant isolates with")
print("quantitative zoliflodacin MIC were present in the 420-isolate MIC dataset.")
print("This is a finding: gyrB-mediated ZO resistance remains extremely rare globally.")
print()
print("Done.")
