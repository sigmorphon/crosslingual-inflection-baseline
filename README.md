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

### Baseline Performance

Available for all baselines [here](https://docs.google.com/spreadsheets/d/1R1dtj2YFhPaOv4-VE1TpcCJ5_WzKO6rZ8ObMxJsM020/edit?usp=sharing).

Task 1 decoded files:
- `0-soft`: https://www.dropbox.com/s/ruom2g921t4kzzr/BASELINE-0SOFT-00-1.tar.gz
- `0-hard`: https://www.dropbox.com/s/g0b3z5hh9dkp3nz/BASELINE-0HARD-00-1.tar.gz
- `0-mono`: https://www.dropbox.com/s/dg2fawbahqqzqdl/BASELINE-0MONO-00-1.tar.gz
- `1-mono`: https://www.dropbox.com/s/n9asg9fo8e5yklx/BASELINE-1MONO-00-1.tar.gz

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
