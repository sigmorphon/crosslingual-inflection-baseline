# crosslingual-inflection-baseline

## Environment Installation
Machine with GPU:
```bash
conda env create -f environment.yml
```

Manually Installation:
```
python>=3.6
pytorch=1.0
numpy
tqdm
```

## Training from scratch

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

## Decoding with Pretrained model

```bash
python src/sigmorphon19-task1-decode.py \
    --file sample/adyghe--kabardian/kabardian-dev \
    --lang kabardian \
    --model sample/model/adyghe--kabardian.1-mono.pth
```