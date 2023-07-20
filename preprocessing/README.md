# Simplification model

## Requirements

```
pip install -r requirements.txt
```
from the top-level directory.

## Training the model on ASSET

Download the [ASSET dataset](https://github.com/facebookresearch/asset)
for training.  If the data is found in `asset/dataset`, preprocess
their examples into a JSONL format in an output directory `asset-data` with
```
python asset-dataset.py asset/dataset asset-data
```

Then train a sequence to sequence model on `asset-data` with:

```
python run_generate2.py --predict_with_generate --do_train --do_predict --train_file asset-data/train.json --validation_file asset-data/test.json --test_file asset-data/test.json --model_name_or_path t5-base --source_prefix "hypothesis: " --output_dir asset-30epoch-output --overwrite_output_dir --per_gpu_train_batch_size=8 --max_source_length=256 --max_target_length=256 --per_gpu_eval_batch_size=8 --num_train_epochs=30
```

## Applying the model to preprocess reviews

Edit `simpchewing.py` to change `p2h_model_name` to be the path to your
trained model.

Run
```
python rewrite_simp_source.py ../Copycat-abstractive-opinion-summarizer/gold_summs/test.csv test-simp-rewritten.csv
```
to preprocess the test set of [Copycat Amazon reviews](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/gold_summs).
Then drop in `test-simp-rewritten.csv` as
a replacement for `test.csv` when running any opinion summarization system.

