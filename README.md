# megalith-hazard-model

Discrete-time hazard model testing the association between ancestry components and the emergence of megalithic monuments.

## Requirements

Python 3 with numpy, pandas, scipy and matplotlib. Tested on macOS 26 with Python 3.14, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2 (`requirements.txt`). No special hardware.

## Installation

```
git clone https://github.com/ras-nielsen/megalith-hazard-model.git
cd megalith-hazard-model
pip install -r requirements.txt
```

Install time: about 1 minute.

## Reproducing the results in the Supplementary Information

```
python model.py megalith_clean.csv 5000 .
python make_figures.py
```

Output: `results_A.csv` (marginal baseline), `results_B.csv` (time-controlled baseline), `fig_strength.png` (Figure S1) and `fig_panels.png` (Figure S2). Run time: about 20 seconds.

## Usage

```
python model.py <input.csv> [n_permutations] [output_dir]
```

The input table has one row per sample and ancestry source, with columns `sample_id, source_pop, p, se, res_norm, comb_idx, role, country, ageAverage, megalith_emerge` (see `megalith_clean.csv`). `megalith_emerge` is the region's emergence date (years BP), empty for regions without megaliths.

## License

PolyForm Noncommercial License 1.0.0 (`LICENSE`).
