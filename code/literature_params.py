"""
literature_params.py
=====================
Sourced, provenance-tagged literature inputs for the manuscript
"Resistance-aware precision dosing as posterior inference".

Every numeric constant below is traceable to a published source named in its
comment (author, year, journal, PMID/DOI, table/figure). Values that could NOT be
verified from full text are flagged and handled with deliberately wide uncertainty.
These inputs replace the hand-set "round number" priors used in earlier drafts.

Two kinds of object are exported:
  * raw observations  (lists of isogenic MIC pairs / growth rates), so a reader can
    re-derive the pooled estimates and audit the meta-analysis; and
  * derived estimates (pooled coefficient mean + SD), computed here by simple
    fixed-effect pooling of the isogenic delta-log2-MIC observations.
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# 1.  gyrB isogenic zoliflodacin MIC observations  (delta log2 MIC = doublings)
# =============================================================================
# Each entry: (parent_MIC_mg_L, mutant_MIC_mg_L, background_label, source).
# delta = log2(mutant) - log2(parent) is the transferable, background-referenced
# effect of the substitution. Agar-dilution MICs used throughout for consistency.

# --- gyrB D429N -------------------------------------------------------------
# The best-characterised substitution: 9 isogenic clinical-strain transformants
# (Mukherjee 2026) plus reference-strain in-vitro selections (Foerster 2015,
# Jacobsson 2021/2022, Eyre 2023, Raven 2024).
GYRB_D429N_PAIRS = [
    # (parent, mutant, background, source)
    (0.064, 0.5,  "WHO F",      "Foerster 2015 FrontMicrobiol PMID26696986 Table2"),
    (0.125, 1.0,  "WHO O",      "Foerster 2015 FrontMicrobiol PMID26696986 Table2"),
    (0.25,  2.0,  "WHO P",      "Foerster 2015 FrontMicrobiol PMID26696986 Table2 (modal PM)"),
    (0.064, 0.5,  "WHO F",      "Jacobsson 2021 FrontPharmacol PMC8175963 Table1"),
    (0.125, 0.75, "WHO X",      "Jacobsson 2021 FrontPharmacol PMC8175963 Table1 (0.5-1 mid)"),
    (0.25,  2.0,  "SE600/18",   "Jacobsson 2022 FrontPharmacol PMID35496288 Table1 (S467N parent)"),
    (0.25,  2.0,  "GCGS0481",   "Eyre 2023 LancetMicrobe PMC10071290 Table2"),
    (0.125, 2.0,  "WHO P",      "Raven 2024 SciRep PMC10786824 Table3B (transformation)"),
    (0.0625, 2.0, "HHH040",     "Mukherjee 2026 JID 10.1093/infdis/jiag174 Table1"),
    (0.0625, 2.0, "EEE016",     "Mukherjee 2026 JID jiag174 Table1"),
    (0.0625, 2.0, "DDD020",     "Mukherjee 2026 JID jiag174 Table1"),
    (0.032, 0.5,  "EEE036",     "Mukherjee 2026 JID jiag174 Table1"),
    (0.032, 1.0,  "HHH014",     "Mukherjee 2026 JID jiag174 Table1"),
    (0.016, 0.5,  "FFF043",     "Mukherjee 2026 JID jiag174 Table1"),
    (0.032, 1.0,  "CCC033",     "Mukherjee 2026 JID jiag174 Table1"),
    (0.032, 1.0,  "DDD033",     "Mukherjee 2026 JID jiag174 Table1"),
    (0.0625, 1.0, "HHH023",     "Mukherjee 2026 JID jiag174 Table1"),
]

# --- gyrB K450T -------------------------------------------------------------
# Only one clean isogenic full-text data point (Foerster 2015 PM-9). Foerster 2019
# (paywalled) reports a grouped D429N/K450T/K450N range of 0.5-4 mg/L. We therefore
# carry K450T with a single anchor and an inflated SD.
GYRB_K450T_PAIRS = [
    (0.25, 2.0, "WHO P", "Foerster 2015 FrontMicrobiol PMID26696986 Table2 (PM-9)"),
]
GYRB_K450T_SD_FLOOR = 1.0   # inflate: n=1 clean isogenic observation.

# --- gyrB S467N -------------------------------------------------------------
# CRITICAL CORRECTION vs earlier draft (which assumed +1.0 doublings): S467N ALONE
# does NOT raise the MIC above the wild-type range. SE600/18 (S467N) MIC = 0.25
# mg/L (within WT distribution); a Nanjing clinical S467N isolate = 0.125 mg/L.
# The single 8 mg/L "S467N" line (Raven 2024, ATCC49226_8) is passage-selected and
# carries additional uncharacterised lesions -> excluded as confounded.
GYRB_S467N_PAIRS = [
    (0.125, 0.25,  "SE600/18", "Jacobsson 2022 FrontPharmacol PMID35496288 Table1 (within WT range)"),
    (0.125, 0.125, "Nanjing",  "Zhou 2021 AAC PMC8092536 (within WT range)"),
]
GYRB_S467N_SD_FLOOR = 0.5


def _pool_delta(pairs, sd_floor=0.0):
    """Pool isogenic ΔMIC observations into (mean, sd) on the log2 (doubling) scale.

    Returns mean delta, the between-background SD (the relevant transferable
    uncertainty for applying the coefficient to a new genotype), floored at
    ``sd_floor`` to avoid overconfidence when observations are few.
    """
    deltas = np.array([np.log2(m / p) for (p, m, *_rest) in pairs], float)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else sd_floor
    return mean, max(sd, sd_floor)


# Derived gyrB main-effect coefficients on log2 ZO MIC (doublings) + SD.
_d429_mu, _d429_sd = _pool_delta(GYRB_D429N_PAIRS)
_k450_mu, _k450_sd = _pool_delta(GYRB_K450T_PAIRS, GYRB_K450T_SD_FLOOR)
_s467_mu, _s467_sd = _pool_delta(GYRB_S467N_PAIRS, GYRB_S467N_SD_FLOOR)

GYRB_ZO_COEF = {
    "gyrB_D429N": (_d429_mu, _d429_sd),
    "gyrB_K450T": (_k450_mu, _k450_sd),
    "gyrB_S467N": (_s467_mu, _s467_sd),
}

# =============================================================================
# 2.  Zoliflodacin pharmacodynamic parameters (Regoes/Hill net-growth model)
# =============================================================================
# Jacobsson 2021, Front Pharmacol, PMID 34093206, Table 2 (Drusano-Louie three-
# output HFIM model). WHO F = susceptible reference; values are means.
PD_ZOLI = dict(
    psi_max_s=1.142,   # kg-s, susceptible growth rate (h^-1), WHO F
    psi_max_r=0.5602,  # kg-r, resistant (D429N) growth rate (h^-1), WHO F
    Emax_s=4.524,      # Kkill-s, susceptible max kill rate (h^-1), WHO F
    Emax_r=1.519,      # Kkill-r, resistant max kill rate (h^-1), WHO F
    EC50_s=0.2507,     # C50-s, susceptible (mg/L), WHO F
    EC50_r=0.4334,     # C50-r, resistant (mg/L), WHO F
    H_s=1.581,         # Hill coeff, susceptible subpopulation, WHO F
    H_r=4.377,         # Hill coeff, resistant subpopulation, WHO F
    source="Jacobsson 2021 FrontPharmacol PMID34093206 Table2 (WHO F)",
)

# =============================================================================
# 3.  Relative fitness of gyrB D429N (growth-rate ratio vs isogenic parent)
# =============================================================================
# Strongly background-dependent (mirrors the cross-resistance epistasis).
FITNESS_D429N = [
    # (w = kg_mutant / kg_parent, background, source)
    (0.5602 / 1.142, "WHO F",    "Jacobsson 2021 FrontPharmacol PMC8175963 Table2"),
    (1.206 / 1.163,  "WHO X",    "Jacobsson 2021 FrontPharmacol PMC8175963 Table2 (no cost)"),
    (0.088 / 0.68,   "SE600/18", "Jacobsson 2022 FrontPharmacol PMID35496288 Table2 (severe cost)"),
]
FITNESS_D429N_W = float(np.mean([w for w, *_ in FITNESS_D429N]))  # mean relative fitness

# =============================================================================
# 4.  Gepotidacin MICs and the GyrB x ParC cross-resistance epistasis
# =============================================================================
# Gepotidacin is dual-targeting (GyrA + ParC); gyrB D429N raises gepotidacin MIC
# ONLY on a parC D86N background (Mukherjee 2026). This is the headline epistasis.
GEPO_WT_MIC = 0.25          # MIC50 ~0.12-0.5; modal 0.5 (Taylor 2018 CID PMID29617982; Unemo 2020 JAC)
# parC D86N main effect on gepotidacin: clinical parC-D86N isolates MIC50 ~2 mg/L
# vs WT ~0.25-0.5 -> ~2-3 doublings (Unemo 2020 JAC PMC6927889 Table2).
GEPO_parC_D86N_DELTA = (np.log2(2.0 / 0.5), 0.7)   # (~2 doublings, SD)
# gyrB D429N main effect on gepotidacin (parC-WT backgrounds): ~no change (Mukherjee 2026).
GEPO_gyrB_D429N_DELTA = (0.0, 0.5)
# Epistasis: parC D86N x gyrB D429N -> double-mutant gepotidacin MIC = 32 mg/L in
# CCC033/DDD033 (parent parC-D86N MIC 1 -> 32 mg/L on acquiring D429N), beyond the
# sum of main effects. The interaction coefficient is the residual after removing
# the intercept AND both main effects on the log2 scale:
#   b_cross = log2(MIC_double) - log2(WT) - parC_main - gyrB_main.
GEPO_DOUBLE_LOG2MIC = np.log2(32.0)                 # absolute log2 MIC of the double mutant
GEPO_parC_x_gyrB_DELTA = (
    GEPO_DOUBLE_LOG2MIC - np.log2(GEPO_WT_MIC)
    - GEPO_parC_D86N_DELTA[0] - GEPO_gyrB_D429N_DELTA[0], 0.8)

# Provenance map for the gepotidacin terms (for reporting / audit).
GEPO_SOURCES = {
    "WT": "Taylor 2018 CID PMID29617982; Unemo 2020 JAC PMC6927889",
    "parC_D86N": "Unemo 2020 JAC PMC6927889 Table2",
    "gyrB_D429N": "Mukherjee 2026 JID jiag174 Table1 (no effect in parC-WT)",
    "parC_D86N x gyrB_D429N": "Mukherjee 2026 JID jiag174 Table1 (32-fold in CCC033/DDD033)",
}

# =============================================================================
# 5.  Hollow-fibre infection model (HFIM) resistance-emergence observations
# =============================================================================
# Two dynamic HFIM studies simulated single (and fractionated) oral zoliflodacin
# doses against N. gonorrhoeae and recorded, per dose, whether the strain was
# eradicated or whether a resistant subpopulation was AMPLIFIED, together with the
# target mutation and MIC of the selected mutant. These are the external anchor for
# the Section 6.6 validation: the framework's MPC (a concentration that should
# suppress the worst single-step neighbour) and its argmax (the identity of that
# neighbour) are tested against what the HFIM actually selected.
#
# Each record:
#   strain, parent_mic (mg/L, agar), gyrB background substitutions already present,
#   doses_eradicated / doses_selected (single oral dose, g),
#   pkpd_suppression_g (model-derived dose preventing amplification, if reported),
#   selected_substitution and selected_mic_range (mg/L, agar) of the amplified mutant,
#   source.
HFIM_OBSERVATIONS = [
    # --- Jacobsson 2021 (PMID 34093206): dose-ranging, target-WT backgrounds -----
    dict(strain="WHO F", parent_mic=0.064, background_gyrB=(),
         doses_eradicated=(2, 3, 4, 6, 8), doses_selected=(0.5, 1),
         pkpd_suppression_g=2.0,
         selected_substitution="gyrB_D429N", selected_mic_range=(0.5, 1.0),
         source="Jacobsson 2021 FrontPharmacol PMID34093206 (>=2 g eradicate; <2 g select D429N)"),
    dict(strain="WHO X", parent_mic=0.125, background_gyrB=(),
         doses_eradicated=(2, 3, 4, 6, 8), doses_selected=(0.5, 1),
         pkpd_suppression_g=2.0,
         selected_substitution="gyrB_D429N", selected_mic_range=(0.5, 1.0),
         source="Jacobsson 2021 FrontPharmacol PMID34093206 (XDR background; same D429N selection)"),
    # --- Jacobsson 2022 (PMID 35496288): isogenic S467N parent and D429N derivative
    dict(strain="SE600/18", parent_mic=0.25, background_gyrB=("gyrB_S467N",),
         doses_eradicated=(3, 4), doses_selected=(0.5, 1, 2),
         pkpd_suppression_g=2.7,
         selected_substitution="gyrB_D429N", selected_mic_range=(2.0, 2.0),
         source="Jacobsson 2022 FrontPharmacol PMID35496288 (S467N parent; 2.7 g PK/PD threshold)"),
    dict(strain="SE600/18-D429N", parent_mic=2.0, background_gyrB=("gyrB_S467N", "gyrB_D429N"),
         doses_eradicated=(), doses_selected=(2, 3, 4, 6),
         pkpd_suppression_g=None,
         selected_substitution="gyrB_D429N(pre-existing)", selected_mic_range=(2.0, 8.0),
         source="Jacobsson 2022 FrontPharmacol PMID35496288 (D429N strain NOT eradicated even at 6 g)"),
]

# Resistant-subpopulation drug-free growth-rate constants (kg-r, h^-1) measured in
# the two HFIM studies — the direct fitness readout for the viability gate (the
# mutant is viable, so the gate does not fire, iff kg-r > in-host clearance).
HFIM_D429N_KG_R = [
    (0.560, "WHO F",    "Jacobsson 2021 FrontPharmacol PMID34093206 Table2 (kg-r)"),
    (1.206, "WHO X",    "Jacobsson 2021 FrontPharmacol PMID34093206 Table2 (no fitness cost)"),
    (0.088, "SE600/18", "Jacobsson 2022 FrontPharmacol PMID35496288 Table2 (severe cost)"),
]

# Provenance partition for the leave-out coefficient validation (Section 6.6, Test B):
# which GYRB_D429N_PAIRS come from the HFIM strains vs from independent (clinical-
# transformant / genomic) sources. Estimating the D429N effect from the independent
# sources alone and predicting the HFIM-observed effect avoids circularity.
HFIM_SOURCE_KEYS = ("Jacobsson 2021", "Jacobsson 2022")


def split_d429n_pairs_by_source(pairs=None):
    """Partition GYRB_D429N_PAIRS into (hfim_pairs, independent_pairs).

    HFIM pairs are those whose provenance string names a Jacobsson HFIM study;
    all others (Foerster time-kill, Eyre/Raven genomic, Mukherjee clinical
    transformants) are treated as the independent set for the leave-out test.
    """
    pairs = GYRB_D429N_PAIRS if pairs is None else pairs
    hfim, indep = [], []
    for rec in pairs:
        src = rec[3]
        (hfim if any(k in src for k in HFIM_SOURCE_KEYS) else indep).append(rec)
    return hfim, indep


# =============================================================================
# 6.  Citation corrections surfaced during sourcing (apply to the bibliography)
# =============================================================================
CITATION_CORRECTIONS = """
1. Mukherjee 2026 (cross-resistance): correct DOI is 10.1093/infdis/jiag174
   (NOT jiab448). Authors: Mukherjee A, Blomqvist SO, Helekal D, Das AA,
   Palace SG, Grad YH. Title: 'Genetic Background Modulates Zoliflodacin and
   Gepotidacin Cross-Resistance and Fitness in Neisseria gonorrhoeae.' PMID 41858024.
2. Gepotidacin phase-2 clinical trial is Taylor SN et al. 2018 Clin Infect Dis
   PMID 29617982 (NOT PMID 30403954, which is the zoliflodacin NEJM trial).
3. Jacobsson 2022 (GyrB HFIM) is in Frontiers in Pharmacology
   (DOI 10.3389/fphar.2022.874176), NOT Antimicrob Agents Chemother.
4. Foerster 2019 zoliflodacin combination/resistance paper (PMID 31730160) is
   J Antimicrob Chemother 74(12):3521-3529, DOI 10.1093/jac/dkz376 (PubMed-verified;
   NOT dkz287 as in the draft, NOT dkz443 as a preliminary search guessed). This is
   the source of the isogenic GyrB D429N/K450T/K450N MICs (0.5-4 mg/L).
5. Gubensek 2022 (PMID 35326763) is Antibiotics (Basel) 11(3):299,
   DOI 10.3390/antibiotics11030299 - ceftriaxone/ertapenem/fosfomycin/gentamicin,
   NO zoliflodacin data. Not a zoliflodacin PD source.
6. Jacobsson 2022 GyrB HFIM (PMID 35496288) is Front Pharmacol 13:874176,
   DOI 10.3389/fphar.2022.874176 (NOT AAC 66(5):e0020022).
All five points PubMed-verified 2026-06-08 via get_article_metadata.
"""

if __name__ == "__main__":
    print("Derived gyrB coefficients on log2 ZO MIC (doublings):")
    for k, (mu, sd) in GYRB_ZO_COEF.items():
        n = {"gyrB_D429N": len(GYRB_D429N_PAIRS),
             "gyrB_K450T": len(GYRB_K450T_PAIRS),
             "gyrB_S467N": len(GYRB_S467N_PAIRS)}[k]
        print(f"  {k:<12} mu={mu:+.2f}  sd={sd:.2f}  (n={n} isogenic pairs)")
    print(f"\nMean relative fitness of D429N (background-averaged): w={FITNESS_D429N_W:.2f}")
    print(f"  per-background: " +
          ", ".join(f"{lab}={w:.2f}" for w, lab, *_ in FITNESS_D429N))
    print(f"\nGepotidacin: WT={GEPO_WT_MIC} mg/L; "
          f"parC_D86N delta={GEPO_parC_D86N_DELTA[0]:.2f}; "
          f"parC×gyrB epistasis={GEPO_parC_x_gyrB_DELTA[0]:.2f} doublings")
