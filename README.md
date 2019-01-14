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

Available for all baselines [here](https://docs.google.com/spreadsheets/d/1R1dtj2YFhPaOv4-VE1TpcCJ5_WzKO6rZ8ObMxJsM020/edit?usp=sharing).

## Training from Scratch

We train the model with the [jackknifing training data](https://www.dropbox.com/s/swf9cq22uxgr5wv/task2_jackknife_training_data_public.tar.gz) and at dev time, we decode the lemma with [predicted tag](https://www.dropbox.com/s/qt6nqa3gn96rbl3/baseline_predictions_public.tar.gz).

```bash
# 0-mono
sh scripts/run-task2.sh af_afribooms
```

## Decoding with Pretrained Model

```bash
python src/sigmorphon19-task2-decode.py \
    --in_file sample/task2/af_afribooms-um-dev.conllu.baseline.pred \
    --out_file decode/task2/af_afribooms-um-dev.conllu.output \
    --model sample/task2/model/af_afribooms.pth
```

## Pretrained Models

Link: https://www.dropbox.com/sh/p4vu5imyn69wyyp/AAA-3bQeGJmnCex78xx7T0ZPa

Size of models:
```
3.4G	sigmorphon2019/public/task2/lemmatizer-model
```

## Decoded Files

Task 2 decoded files: https://www.dropbox.com/s/2kqkhsu0kil6rzu/BASELINE-DEV-00-2.tar.gz