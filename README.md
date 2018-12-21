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

## Training from Scratch

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

## Decoding with Pretrained Model

```bash
python src/sigmorphon19-task1-decode.py \
    --in_file sample/adyghe--kabardian/kabardian-dev \
    --out_file decode/adyghe--kabardian-dev-out \
    --lang kabardian \
    --model sample/model/adyghe--kabardian.1-mono.pth
```

## Pretrained Models

Link: https://www.dropbox.com/sh/lx010wra0jagfhw/AACFQPWlMCD2xIzQ5ChgHVWha

Size of models:
```
2.9G	sigmorphon19-task1-model/0-hard
2.9G	sigmorphon19-task1-model/0-mono
2.9G	sigmorphon19-task1-model/0-soft
2.9G	sigmorphon19-task1-model/1-mono
```

## Baseline Performance

Link: https://docs.google.com/spreadsheets/d/1yqCf38Qub5TDEH_F9PzxHuF3adymhKZZHnEVUW94yKY/