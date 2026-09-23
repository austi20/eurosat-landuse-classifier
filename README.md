# Satellite Land Use Classifier (EuroSAT)

[![tests](https://github.com/austi20/eurosat-landuse-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/austi20/eurosat-landuse-classifier/actions/workflows/tests.yml)

Land use classification on Sentinel-2 satellite imagery: 27,000 labeled 64x64 image
patches across 10 land use classes, in PyTorch. A small CNN trained from scratch is
compared against an ImageNet pretrained ResNet18 fine tuned on the same split, so the
gap between them measures what transfer learning actually bought.

Overhead imagery is a real alternative data source in quantitative finance. Funds
estimate economic activity from satellite views of parking lots, storage tanks and
construction sites. This project is the computer vision half of that pipeline: given a
patch of ground, name what is on it, and be specific about where that fails.

**Status:** the baseline is recorded. The models and the single test score come next,
and the test split gets scored exactly once.

## Questions

1. How much does an ImageNet pretrained ResNet18 beat a CNN trained from scratch on
   64x64 RGB patches, and how much does either beat guessing the majority class?
2. Which land use classes get confused with each other, and is the confusion a real
   visual ambiguity at 64x64 resolution or a modeling failure?
3. How close does a short fine tune on a CPU get to the 98.57% overall accuracy
   reported in the EuroSAT paper, and where does the gap come from?

## Results

| Model | Val accuracy | Test accuracy |
|---|---|---|
| Majority class (AnnualCrop) | 11.11% | 11.11% |

**Majority class baseline: 11.11%.** Recorded before any model was trained. Six classes
tie for largest, with 2,100 training images each, so "the majority class" is really a
six way tie. The code takes the first one, AnnualCrop. Any of the six gives the same
11.11%, because the stratified split puts exactly 450 of each into val and into test.

## Data

**EuroSAT**: 27,000 labeled, georeferenced 64x64 patches from the ESA **Sentinel-2**
satellite, in 10 land use classes. MIT licensed. Sentinel-2 data is free and open under
EU law.

- Dataset and citation: https://github.com/phelber/EuroSAT
- Access used here: `torchvision.datasets.EuroSAT(root=..., download=True)`, the RGB
  variant built into torchvision (a 94 MB zip).
  https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html

The download is not committed. `data/` is gitignored and the dataset is fetched on
first run.

### It is not balanced

EuroSAT is often described as balanced. It is close, but not exactly. The largest
classes have 3,000 images and Pasture has 2,000, a 1.5x spread. Counts as printed by
`src/split.py`:

| Class | Images | Share | Train | Val | Test |
|---|---|---|---|---|---|
| AnnualCrop | 3,000 | 11.11% | 2,100 | 450 | 450 |
| Forest | 3,000 | 11.11% | 2,100 | 450 | 450 |
| HerbaceousVegetation | 3,000 | 11.11% | 2,100 | 450 | 450 |
| Highway | 2,500 | 9.26% | 1,750 | 375 | 375 |
| Industrial | 2,500 | 9.26% | 1,750 | 375 | 375 |
| Pasture | 2,000 | 7.41% | 1,400 | 300 | 300 |
| PermanentCrop | 2,500 | 9.26% | 1,750 | 375 | 375 |
| Residential | 3,000 | 11.11% | 2,100 | 450 | 450 |
| River | 2,500 | 9.26% | 1,750 | 375 | 375 |
| SeaLake | 3,000 | 11.11% | 2,100 | 450 | 450 |
| **Total** | **27,000** | | **18,900** | **4,050** | **4,050** |

The imbalance is mild enough that plain accuracy is still a fair headline, but it is why
the baseline is 11.11% and not 10%.

### The split

Stratified 70/15/15 into train, val and test, seed 42, using two calls to scikit-learn's
`train_test_split`. Every image's split assignment is committed in
[`splits/eurosat_split_seed42.csv`](splits/eurosat_split_seed42.csv), keyed by its path
inside the dataset, so the split can be audited or reused without rerunning anything.

## Method

The evaluation protocol was fixed before any modeling:

1. **Majority class baseline**, written down first.
2. **Stratified 70/15/15 split** with a fixed seed, saved to disk.
3. **Model A:** small CNN from scratch, 3 convolutional blocks.
4. **Model B:** ResNet18 pretrained on ImageNet, head replaced with 10 outputs,
   normalized with ImageNet statistics, input resized 64 -> 224.
5. **Model selection on validation only.** The test split is scored once, at the end,
   with the model already chosen.
6. **Confusion matrix** committed as a figure and read out in this README by class name.

## Repo layout

```
src/         data loading, split, baseline and training code
splits/      committed split assignments
results/     committed metrics as JSON
tests/       pytest suite
figures/     committed figures, including the confusion matrix
data/        EuroSAT download (gitignored)
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

On Linux, the default PyPI `torch` wheel is the CUDA build and pulls several
gigabytes of NVIDIA runtime packages with it. On Windows the default is already
CPU only. To force CPU wheels on any platform, install torch first from the
PyTorch CPU index:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then, from the repo root:

```bash
python src/split.py        # downloads EuroSAT, prints class counts, writes the split
python src/baseline.py     # majority class baseline -> results/baseline.json
python -m pytest
```

## Caveats

These are known before any results exist, and they do not go away if the accuracy is
good:

- **European imagery only.** Every patch is over Europe. Nothing here licenses a claim
  that the model transfers to other geographies, biomes or agricultural practices.
- **64x64 is coarse.** Several classes are nearly identical at that resolution. Narrow
  linear features and green texture classes are the ones to watch.
- **RGB only.** Sentinel-2 carries 13 spectral bands. The torchvision RGB variant throws
  away the near infrared and shortwave infrared bands, which are exactly the bands that
  separate vegetation types most cleanly. A multispectral model would likely do better.
- **A single test score is a point estimate.** One number on 15% of 27,000 patches, with
  no confidence interval and no repeated seeds.

## License

MIT. See [LICENSE](LICENSE).
