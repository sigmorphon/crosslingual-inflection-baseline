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

## Baseline Performance (Task 1 & 2)

Link: https://docs.google.com/spreadsheets/d/1yqCf38Qub5TDEH_F9PzxHuF3adymhKZZHnEVUW94yKY/

Task 1 decoded files:
- `0-soft`: https://www.dropbox.com/s/ruom2g921t4kzzr/BASELINE-0SOFT-00-1.tar.gz
- `0-hard`: https://www.dropbox.com/s/g0b3z5hh9dkp3nz/BASELINE-0HARD-00-1.tar.gz
- `0-mono`: https://www.dropbox.com/s/dg2fawbahqqzqdl/BASELINE-0MONO-00-1.tar.gz
- `1-mono`: https://www.dropbox.com/s/n9asg9fo8e5yklx/BASELINE-1MONO-00-1.tar.gz

Task 2 decoded files: https://www.dropbox.com/s/2kqkhsu0kil6rzu/BASELINE-DEV-00-2.tar.gz?dl=0

## Task 1

### Training from Scratch

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

### Decoding with Pretrained Model

```bash
python src/sigmorphon19-task1-decode.py \
    --in_file sample/task1/adyghe--kabardian/kabardian-dev \
    --out_file decode/task1/adyghe--kabardian-dev-out \
    --lang kabardian \
    --model sample/task1/model/adyghe--kabardian.1-mono.pth
```

### Pretrained Models

Link: https://www.dropbox.com/sh/lx010wra0jagfhw/AACFQPWlMCD2xIzQ5ChgHVWha

Size of models:
```
2.9G	sigmorphon2019/public/task1/model/0-hard
2.9G	sigmorphon2019/public/task1/model/0-mono
2.9G	sigmorphon2019/public/task1/model/0-soft
2.9G	sigmorphon2019/public/task1/model/1-mono
```

## Task 2

### Training from Scratch

We train the model with the [jackknifing training data](https://www.dropbox.com/s/swf9cq22uxgr5wv/task2_jackknife_training_data_public.tar.gz) and at dev time, we decode the lemma with [predicted tag](https://www.dropbox.com/s/qt6nqa3gn96rbl3/baseline_predictions_public.tar.gz).


```bash
# 0-mono
sh scripts/run-task2.sh af_afribooms
```

### Decoding with Pretrained Model

```bash
python src/sigmorphon19-task2-decode.py \
    --in_file sample/task2/af_afribooms-um-dev.conllu.baseline.pred \
    --out_file decode/task2/af_afribooms-um-dev.conllu.output \
    --model sample/task2/model/af_afribooms.pth
```

### Pretrained Models

Link: https://www.dropbox.com/sh/p4vu5imyn69wyyp/AAA-3bQeGJmnCex78xx7T0ZPa

Size of models:
```
3.4G	sigmorphon2019/public/task2/lemmatizer-model
```