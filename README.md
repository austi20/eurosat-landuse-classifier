# Satellite Land-Use Classifier (EuroSAT)

[![tests](https://github.com/austi20/eurosat-landuse-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/austi20/eurosat-landuse-classifier/actions/workflows/tests.yml)

Land-use classification on Sentinel-2 satellite imagery: 27,000 labeled 64x64 image
patches across 10 land-use classes, in PyTorch. A small CNN trained from scratch is
compared against an ImageNet-pretrained ResNet18 fine-tuned on the same split, so the
gap between them measures what transfer learning actually bought.

Overhead imagery is a real alternative-data source in quantitative finance - funds
estimate economic activity from satellite views of parking lots, storage tanks and
construction sites. This project is the computer-vision half of that pipeline: given a
patch of ground, name what is on it, and be specific about where that fails.

**Status:** scaffolded. No results yet. Numbers go here once the test split has been
scored, and it gets scored exactly once.

## Questions

1. How much does an ImageNet-pretrained ResNet18 beat a from-scratch CNN on 64x64
   multispectral-derived RGB patches, and how much does either beat guessing the
   majority class?
2. Which land-use classes get confused with each other, and is the confusion a real
   visual ambiguity at 64x64 resolution or a modeling failure?
3. How close does a two-session fine-tune get to the 98.57% overall accuracy reported
   in the EuroSAT paper, and where does the gap come from?

## Data

**EuroSAT** - 27,000 labeled, geo-referenced 64x64 patches from the ESA **Sentinel-2**
satellite, in 10 land-use classes. MIT licensed; Sentinel-2 data is free and open under
EU law.

- Dataset and citation: https://github.com/phelber/EuroSAT
- Access used here: `torchvision.datasets.EuroSAT(root=..., download=True)`, the RGB
  variant built into torchvision.
  https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html

The download is not committed. `data/` is gitignored and the dataset is fetched on
first run.

Per-class counts are printed by the loader and will be recorded here rather than
assumed - the dataset is commonly described as balanced, and that claim gets checked.

## Method

Planned, in order, with the evaluation protocol fixed before any modeling:

1. **Majority-class baseline** on the test split, written down first.
2. **Stratified 70/15/15 split** into train/val/test with a fixed seed. Index lists are
   saved to disk so the split is reproducible and auditable.
3. **Model A:** small CNN from scratch, 3 convolutional blocks.
4. **Model B:** ResNet18 pretrained on ImageNet, head replaced with 10 outputs,
   normalized with ImageNet statistics, input resized 64 -> 224.
5. **Model selection on validation only.** The test split is scored once, at the end,
   with the model already chosen.
6. **Confusion matrix** committed as a figure and read out in this README by class name.

## Results

Pending. This section will carry the majority-class baseline, the scratch-CNN number,
the fine-tuned ResNet18 number, and a single test-set score for the selected model.

## Repo layout

```
src/         training, evaluation and data-loading code
tests/       pytest suite
notebooks/   exploratory work
figures/     committed figures, including the confusion matrix
data/        EuroSAT download (gitignored)
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA-enabled `torch` build from PyPI. For a CPU-only
machine, install torch first from the CPU index:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Caveats

These are known before any results exist, and they do not go away if the accuracy is
good:

- **European imagery only.** Every patch is over Europe. Nothing here licenses a claim
  that the model transfers to other geographies, biomes or agricultural practices.
- **64x64 is coarse.** Several classes are near-identical at that resolution. Narrow
  linear features and green-texture classes are the ones to watch.
- **RGB only.** Sentinel-2 carries 13 spectral bands. The torchvision RGB variant throws
  away the near-infrared and short-wave-infrared bands, which are exactly the bands that
  separate vegetation types most cleanly. A multispectral model would likely do better.
- **A single test score is a point estimate.** One number on 15% of 27,000 patches, with
  no confidence interval and no repeated seeds.

## License

MIT. See [LICENSE](LICENSE).
