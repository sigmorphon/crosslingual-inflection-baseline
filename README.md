# crosslingual-inflection-baseline

## Environment Installation
Machine with GPU (Tested on Linux):
```bash
conda env create -f environment.yml
```

Manually Installation (Recommend for machine without GPU):
```
python>=3.6
pytorch=1.0
numpy
tqdm
```

## Baseline Performance

Available for all baselines [here](https://docs.google.com/spreadsheets/d/1vvSuy2LBarS20zK8lg_YCTauntDsmoxfqSaSrAQsJrM/edit?usp=sharing).

## Train from Scratch

```bash
# 0-soft
sh scripts/run-task1-tag.sh soft adyghe--kabardian
# 0-hard
sh scripts/run-task1-tag.sh hard adyghe--kabardian
# 0-mono
sh scripts/run-task1-monotag.sh hmm adyghe--kabardian
# 1-mono
sh scripts/run-task1-monotag.sh hmmfull adyghe--kabardian
```

## Decode trained Model

```bash
python src/sigmorphon19-task1-decode.py \
    --in_file sample/task1/adyghe--kabardian/kabardian-dev \
    --out_file decode/task1/adyghe--kabardian-dev-out \
    --lang kabardian \
    --model sample/task1/model/adyghe--kabardian.1-mono.pth
```