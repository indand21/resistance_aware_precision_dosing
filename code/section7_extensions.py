"""
section7_extensions.py
======================
Completes the previously-deferred theoretical components of the manuscript and
closes the introduction->results loop. Three self-contained analyses:

  7.1  Soft MPC with a time-varying environment (manuscript Section 4.4) — removes
       the [METHODOLOGICAL CAVEAT] by integrating the establishment probability
       along the actual oral concentration profile, using real zoliflodacin PD
       parameters (Jacobsson 2021).
  7.2  Numerical test of the manuscript's Section 4.5.7 claim that the in-window
       net-growth integral scales as I_s ~ rho_s^-1. The test REFUTES that claim:
       under monoexponential PK the integral is site-invariant (like T_MSW), so
       §4.5.7 must be corrected. Pharyngeal risk instead arises from failure of the
       Cmax,s > MPC condition at low penetration (eq 4.5.8), which remains valid.
  7.3  Gepotidacin GyrB x ParC cross-resistance (manuscript Section 2.5 / 5) and a
       structure-informed-prior demonstration (Section 2.4). Brings the headline
       32-fold epistasis from the introduction into the results.

Run: python section7_extensions.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_core import (net_growth_rate, establishment_probability, bateman_conc,
                        clark_max_moments, mpc_msw_posterior)
import literature_params as lit

PD = lit.PD_ZOLI
# Oral zoliflodacin PK (O'Donnell 2019, as used in Section 6.3).
F, D, Vd, ka, ke = 0.75, 3000.0, 100.0, 1.0, 0.11
t_half = np.log(2.0) / ke


# =============================================================================
# 7.1  Soft MPC: time-varying-environment establishment probability (Section 4.4)
# =============================================================================
def run_soft_mpc():
    print("=" * 70)
    print("Section 7.1  Soft MPC (time-varying environment)")
    print("=" * 70)
    print(f"PD (resistant subpop, Jacobsson 2021): ψ_max={PD['psi_max_r']:.3f} h⁻¹, "
          f"Emax={PD['Emax_r']:.3f} h⁻¹, H={PD['H_r']:.3f}")
    # Dominant single-step resistant mutant from WT: gyrB D429N.
    wt_mic = 2 ** (-3.98)                       # WHO F WT ZO MIC ~0.063 mg/L
    d429n_mic = wt_mic * 2 ** lit.GYRB_ZO_COEF["gyrB_D429N"][0]
    print(f"gyrB D429N mutant MIC ≈ {d429n_mic:.3f} mg/L "
          f"(WT {wt_mic:.3f} × 2^{lit.GYRB_ZO_COEF['gyrB_D429N'][0]:.2f})")
    print()

    t_grid = np.linspace(1e-6, 72.0, 4000)      # 72 h horizon after a single dose
    # Mutational supply Θ: illustrative small per-course value (1e-3) scaling the
    # single-lineage hazard; the soft-MPC shape is what matters, not its absolute level.
    theta = 1e-3

    print(f"{'Dose (g)':>9} {'Cmax (mg/L)':>12} {'∫p_inst dt (h)':>15} {'P(establish)':>13}")
    print("-" * 52)
    soft = []
    for dose_g in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        Dmg = dose_g * 1000.0
        def C(t, Dmg=Dmg):
            return bateman_conc(t, F, Dmg, Vd, ka, ke)
        cmax = max(C(t) for t in t_grid)
        p_est, integral = establishment_probability(
            d429n_mic, PD["psi_max_r"], PD["Emax_r"], PD["H_r"],
            C, t_grid, mutation_supply=theta)
        soft.append((dose_g, cmax, integral, p_est))
        print(f"{dose_g:>9.1f} {cmax:>12.2f} {integral:>15.3f} {p_est:>13.3e}")
    print()
    print("Interpretation: the establishment probability falls smoothly with dose")
    print("(a 'soft' MPC) rather than switching at a hard threshold; the approved 3 g")
    print("dose sits on the low-probability shoulder. The integral is over the actual")
    print("time-varying oral profile, not a constant-environment placeholder.")
    print()
    return soft


# =============================================================================
# 7.2  Net-growth-integral scaling I_s ~ rho_s^-1  (Section 4.5.7)
# =============================================================================
def run_integral_scaling():
    print("=" * 70)
    print("Section 7.2  Pharyngeal net-growth-integral: testing the §4.5.7 claim")
    print("=" * 70)
    # Founder window in the organism's local concentration: [MIC, MPC].
    mic, W = 0.5, 2.0
    mpc = mic * 2 ** W
    Cmax = F * 3000.0 / Vd                       # approved oral-dose peak (plasma)

    def C1(t):                                   # plasma profile (Bateman)
        return bateman_conc(t, F, 3000.0, Vd, ka, ke)

    tmax = np.log(ka / ke) / (ka - ke)
    # Resistant-neighbour net growth at local concentration C_s = rho * C1.
    def integ_at_site(rho, H):
        # Window in plasma terms: C1 in [mic/rho, mpc/rho]; integrate r(rho*C1) over t.
        tt = np.linspace(tmax, tmax + 60.0 / ke, 6000)   # descending limb
        c1 = np.array([C1(t) for t in tt])
        local = rho * c1
        inwin = (local >= mic) & (local <= mpc)
        if inwin.sum() < 2:
            return 0.0
        r = np.array([net_growth_rate(c, mic, PD["psi_max_r"], PD["Emax_r"], H)
                      for c in local[inwin]])
        return float(np.trapz(r, tt[inwin]))

    for H in [PD["H_s"], PD["H_r"]]:             # 1.581 (susceptible), 4.377 (resistant)
        I1 = integ_at_site(1.0, H)
        print(f"\n  Hill H = {H:.3f}:   I_1 (rho=1) = {I1:.4f}")
        print(f"    {'rho_s':>6} {'I_s (numeric)':>14} {'I_1/rho_s (pred)':>17} {'rel.err':>9}")
        for rho in [1.0, 0.75, 0.5, 0.25]:
            Is = integ_at_site(rho, H)
            pred = I1 / rho
            err = abs(Is - pred) / abs(pred) if pred else 0.0
            print(f"    {rho:>6.2f} {Is:>14.4f} {pred:>17.4f} {err:>8.1%}")
    print()
    print("FINDING: I_s is essentially INVARIANT in rho_s (numeric column ~constant),")
    print("NOT proportional to 1/rho_s. Substituting u = rho·C1(t) gives")
    print("I_s = ∫_[MIC,MPC] r(u)·du/(k_e·u), which has no rho dependence — the same")
    print("telescoping that makes T_MSW site-invariant (§4.5.5). The manuscript's")
    print("§4.5.7 'I_s ∝ 1/rho_s' is therefore incorrect and is corrected: the")
    print("in-window net-growth integral is ALSO site-invariant under monoexponential")
    print("PK. Genuine pharyngeal risk comes from failure of Cmax,s = rho·Cmax > MPC")
    print("at low penetration (eq 4.5.8), not from a 1/rho scaling of the integral.")
    print()


# =============================================================================
# 7.3  Gepotidacin GyrB×ParC cross-resistance + structure-informed prior
# =============================================================================
def run_gepotidacin_cross():
    print("=" * 70)
    print("Section 7.3  Gepotidacin GyrB×ParC cross-resistance (§2.5) and structure prior (§2.4)")
    print("=" * 70)
    # Reference-based coefficients on log2 gepotidacin MIC (doublings).
    b0 = np.log2(lit.GEPO_WT_MIC)
    b_parC = lit.GEPO_parC_D86N_DELTA[0]
    b_gyrB = lit.GEPO_gyrB_D429N_DELTA[0]
    b_cross = lit.GEPO_parC_x_gyrB_DELTA[0]
    print(f"  Gepotidacin reference-basis coefficients (log2 MIC, doublings):")
    print(f"    intercept (WT)        : {b0:+.2f}  (MIC {lit.GEPO_WT_MIC} mg/L)")
    print(f"    parC_D86N main        : {b_parC:+.2f}   [{lit.GEPO_SOURCES['parC_D86N']}]")
    print(f"    gyrB_D429N main       : {b_gyrB:+.2f}   [{lit.GEPO_SOURCES['gyrB_D429N']}]")
    print(f"    parC_D86N×gyrB_D429N  : {b_cross:+.2f}   [{lit.GEPO_SOURCES['parC_D86N x gyrB_D429N']}]")
    print()

    def gepo_log2mic(parC, gyrB):
        return b0 + b_parC * parC + b_gyrB * gyrB + b_cross * parC * gyrB

    print("  Predicted gepotidacin MIC by genotype (mg/L):")
    for parC, gyrB, lab in [(0, 0, "WT"), (1, 0, "parC D86N"),
                            (0, 1, "gyrB D429N"), (1, 1, "parC D86N + gyrB D429N")]:
        mic = 2 ** gepo_log2mic(parC, gyrB)
        print(f"    {lab:<24} {mic:>7.3f}")
    fold = 2 ** (gepo_log2mic(1, 1) - gepo_log2mic(1, 0))
    print(f"  -> Acquiring gyrB D429N on a parC D86N background raises gepotidacin MIC "
          f"{fold:.0f}-fold")
    print(f"     (vs no change on a parC-WT background): the headline epistasis, now in results.")
    print(f"  Contrast zoliflodacin: gyrB D429N raises ZO MIC "
          f"{2**lit.GYRB_ZO_COEF['gyrB_D429N'][0]:.0f}-fold regardless of parC — "
          f"distinct cross-drug structure.")
    print()

    # ---- Structure-informed prior demonstration (Section 2.4) ----
    # Prior SD on an epistasis term shrinks with Cα–Cα distance between the coupled
    # residues: pocket-local pairs are free, distant pairs penalised toward zero.
    # Distances here are ILLUSTRATIVE placeholders (a real fit would read them from a
    # modelled GyrA–GyrB complex); they demonstrate the prior schedule, not measured geometry.
    print("  Structure-informed prior on epistasis terms (illustrative distances):")
    sd0, length, d0 = 1.0, 8.0, 6.0            # SD floor decay length (Å)
    def prior_sd(dist):                        # shrink toward 0 with distance
        return sd0 * np.exp(-max(dist - d0, 0.0) / length)
    pairs = [("GyrB D429–K450", 7.0), ("GyrB D429–S467", 11.0),
             ("GyrB D429–GyrA S91", 28.0), ("ParC D86–GyrB D429", 32.0)]
    print(f"    {'residue pair':<22} {'Cα–Cα (Å)':>10} {'prior SD(β_jk)':>15}")
    for lab, dist in pairs:
        print(f"    {lab:<22} {dist:>10.1f} {prior_sd(dist):>15.3f}")
    print("    -> within-pocket couplings (429/450/467) keep a free prior; cross-domain")
    print("       couplings are shrunk toward zero, disciplining the epistasis estimates.")
    print()


if __name__ == "__main__":
    run_soft_mpc()
    run_integral_scaling()
    run_gepotidacin_cross()
    print("Done (Section 7 extensions).")
