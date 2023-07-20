import json
import tqdm
import sys
import os
import nltk.tokenize

if(len(sys.argv) != 3):
  raise ValueError("Usage")

in_file = sys.argv[1]
out_file = sys.argv[2]

if(os.path.exists(out_file)):
  raise ValueError("Output already exists")

# document.text / claim

in_fp = open(in_file, "r")
out_fp = open(out_file, "w")

hdr = in_fp.readline()

for line in tqdm.tqdm(in_fp.readlines()):
  cat, prod_id, rev1, rev2, rev3, rev4, rev5, rev6, rev7, rev8, summ1, summ2, summ3 = line.rstrip().split("\t")
  for k, original in enumerate([rev1, rev2, rev3, rev4, rev5, rev6, rev7, rev8]):
    for j, summ in enumerate([summ1]):
      sentences = []
      for i, sentence in enumerate(nltk.tokenize.sent_tokenize(summ)):
        generated = sentence
        sentences.append(sentence)

      idx = prod_id + ":" + str(k) + ":" + str(j)
      answer = {"index": idx, "review": original, "summary": sentences}
      out_fp.write(json.dumps(answer) + "\n")

out_fp.close()


