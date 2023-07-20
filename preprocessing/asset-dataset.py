import sys
import os
import json
import tqdm

if(len(sys.argv) != 3):
  raise ValueError("Usage")

in_dir = sys.argv[1]
out_dir = sys.argv[2]


train_orig_file = os.path.join(in_dir, "asset.valid.orig")
train_files = []
for i in range(10):
  train_files.append(os.path.join(in_dir, "asset.valid.simp." + str(i)))

test_orig_file = os.path.join(in_dir, "asset.test.orig")
test_files = []
for i in range(10):
  test_files.append(os.path.join(in_dir, "asset.test.simp." + str(i)))


out_file = os.path.join(out_dir, "train.json")
if(os.path.exists(out_file)):
  raise ValueError(out_file + " already exists")

in_fp = open(train_orig_file, "r")
out_fp = open(out_file, "w")
ref_fp = []
for i in range(10):
  ref_fp.append(open(train_files[i], "r"))

n = 0
for line in tqdm.tqdm(in_fp.readlines()):
  p = line.rstrip()
  for i in range(10):
    h = ref_fp[i].readline().rstrip()
    struct = {"index": "asset:" + str(n) + ":" + str(i), "premise": p,
              "hypothesis": h, "label": "entailment"}
    out_fp.write(json.dumps(struct) + "\n")
  n = n + 1

in_fp.close()
for i in range(10):
  ref_fp[i].close()
ref_fp = []

out_file = os.path.join(out_dir, "test.json")
if(os.path.exists(out_file)):
  raise ValueError(out_file + " already exists")

in_fp = open(test_orig_file, "r")
out_fp = open(out_file, "w")
ref_fp = []
for i in range(10):
  ref_fp.append(open(test_files[i], "r"))

n = 0
for line in tqdm.tqdm(in_fp.readlines()):
  p = line.rstrip()
  for i in range(10):
    h = ref_fp[i].readline().rstrip()
    struct = {"index": "asset:" + str(n) + ":" + str(i), "premise": p,
              "hypothesis": h, "label": "entailment"}
    out_fp.write(json.dumps(struct) + "\n")
  n = n + 1

in_fp.close()
for i in range(10):
  ref_fp[i].close()
ref_fp = []



