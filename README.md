# Resistance-aware precision dosing

Analysis and validation code for **"Genotype-informed mutant prevention concentration and
selection window for zoliflodacin in *Neisseria gonorrhoeae*: a modeling and
external-validation study."**

The work reframes the mutant prevention concentration (MPC) and the mutant-selection
window (MSW) as predictions of a fitted genotype→(MIC, fitness) model with propagated
uncertainty, couples the map to a Hill pharmacodynamic function and a pharmacokinetic
model in closed form, and validates the predictions against two independent hollow-fiber
infection-model studies. It is instantiated on the topoisomerase-resistance landscape of
*Neisseria gonorrhoeae* and the spiropyrimidinetrione **zoliflodacin**.

## Layout

```
.
├── code/                       Analysis & validation scripts
│   ├── model_core.py            shared machinery: interval-censored estimator (MAP+Laplace),
│   │                            Clark MPC propagation, PK, soft-MPC integral, design diagnostics
│   ├── literature_params.py     provenance-tagged sourced inputs, isogenic gyrB meta-analysis,
│   │                            and the hollow-fiber resistance-emergence observations
│   ├── section5_analysis.py     empirical case study (censored fit, diagnostics, MPC/MSW/T_MSW)
│   │                            → figures/section5_figures.png        [needs data/]
│   ├── section6_validation.py   estimator calibration, Clark-vs-Monte-Carlo, oral-PK sensitivity
│   ├── section64_robustness.py  epistasis-basis (reference vs Walsh–Hadamard) robustness
│   │                            → figures/section64_figures.png       [needs data/]
│   ├── section7_extensions.py   soft-MPC, pharyngeal scaling check, gepotidacin cross-resistance,
│   │                            structure-informed prior
│   ├── section6_hfim_validation.py  external validation vs hollow-fiber infection-model studies
│   │                            → figures/section66_hfim_validation.png
│   └── graphical_abstract.py    graphical-abstract figure → figures/graphical_abstract.png
├── data/                       Input datasets — NOT redistributed here; see data/README.md
└── figures/                    Generated figures (regenerable from code/)
    ├── section5_figures.png
    ├── section64_figures.png
    ├── section66_hfim_validation.png
    └── graphical_abstract.png
```

## Requirements

Python 3 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt        # numpy, pandas, scipy, matplotlib
```

## Reproducing the analysis

The two raw datasets are **not bundled** in this repository (see `data/README.md` for the
public sources). Scripts that consume the population data expect the files under `data/`;
the validation and extension scripts run without them.

```bash
python code/section6_validation.py        # estimator calibration + Clark-vs-MC (synthetic ground truth; no data needed)
python code/section7_extensions.py        # soft-MPC, pharyngeal scaling, gepotidacin cross-resistance
python code/section6_hfim_validation.py   # external HFIM validation → figures/section66_hfim_validation.png
python code/literature_params.py          # prints the pooled gyrB coefficients + provenance
python code/section5_analysis.py          # empirical results + figures/section5_figures.png   [needs data/]
python code/section64_robustness.py       # basis robustness + figures/section64_figures.png   [needs data/]
```

Scripts resolve `data/` and `figures/` relative to their own location, so the repository
can be moved or renamed without editing any paths.

## Headline results

- Wild-type MSW = 4.16 log₂ doublings; MPC = 1.17 mg/L (95% CI 0.36–3.78). The window is
  set by the most resistant single-step neighbor, *gyrB* D429N (≈4 doublings), whose effect
  is estimated from a meta-analysis of 17 published isogenic MIC pairs (no *gyrB* mutant
  carries a quantitative zoliflodacin MIC in the 420-isolate sample).
- Median population T_MSW = 26.8 h (IV-bolus limit). The approved 3 g dose (C_max ≈ 17.1 mg/L)
  clears the MPC of all circulating genotypes, but a *gyrB*-D429N-carrying genotype's MPC
  posterior (E 7.4 mg/L) has an upper credible bound exceeding C_max — the margin is not
  assured if D429N emerges.
- Design diagnostics: only ~18 distinct genotype profiles underlie the 420 isolates
  (condition number ≈ 2.2×10⁶); intervals are wide and honest, not spuriously tight.
- Internal validation: deployed interval-censored estimator calibrated at 91.3% (50-rep,
  target 95%); Clark vs Monte Carlo agree to within 0.005 log₂ doublings (std-ratio 0.997–1.012).
- External validation: against two hollow-fiber infection-model studies it never used for
  fitting, the framework reproduces the selection threshold, the decisive *gyrB* D429N
  coefficient (leave-out test), and the identity of the selected mutant, and shows that a
  peak-above-MPC rule is necessary but not sufficient.

## Data sources

- **MIC dataset** — Ebrahimi E, et al. (2025), *PLoS Negl Trop Dis* 19(10):e0013505,
  doi:10.1371/journal.pntd.0013505 (PMID 41052130); article supplementary material.
- **Genotype calls** — NCBI Pathogen Detection / AMRFinderPlus, *N. gonorrhoeae*
  (release PDG000000032.516, June 2025).
- **Pharmacokinetics** — O'Donnell J, et al. (2019), *Antimicrob Agents Chemother*
  63(6):e02604-18 (PMID 30373802).

See `data/README.md` for download instructions.

## Citation

If you use this code, please cite it via `CITATION.cff` (GitHub's "Cite this repository"
button). The associated manuscript is listed there as the preferred citation; update its
journal/DOI once published.

## License

Released under the MIT License — see [LICENSE](LICENSE).
