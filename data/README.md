# Data sources

The analysis uses two publicly available datasets, which are **not redistributed in this
repository**. Download them from the original sources and place them in this `data/` folder
(with the exact filenames below) to reproduce the data-dependent scripts
(`section5_analysis.py`, `section64_robustness.py`).

## 1. MIC dataset → `data/pntd0013505_s1.csv`

Ebrahimi E, Hadi Z, Farsioo S, Hasani B, Badmasti F, Beig M, Sholeh M.
*Global genomic and antimicrobial resistance profiling of* Neisseria gonorrhoeae*:
insights from whole genome sequencing and minimum inhibitory concentration analysis.*
**PLoS Negl Trop Dis** 2025;19(10):e0013505. doi:10.1371/journal.pntd.0013505 (PMID 41052130).

Available as the article's supplementary material (Supplementary file S1). Open access
(CC BY).

## 2. Genotype calls → `data/ncbi_ngono_amr.tsv`

NCBI Pathogen Detection / AMRFinderPlus point-mutation calls for *Neisseria gonorrhoeae*
(release PDG000000032.516, June 2025). <https://www.ncbi.nlm.nih.gov/pathogens/>

Public domain (U.S. Government work).

---

Scripts resolve `data/` relative to the repository root, so no path edits are needed once
the files are in place.
