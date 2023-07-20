import json
import tqdm
import sys
import os
import nltk.tokenize
import pandas
from simpchewing import SimpChewing

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
out_fp.write(hdr)

mouth = SimpChewing()

for line in tqdm.tqdm(in_fp.readlines()):
  cat, prod_id, rev1, rev2, rev3, rev4, rev5, rev6, rev7, rev8, summ1, summ2, summ3 = line.rstrip().split("\t")

  rewrote = []
  for k, original in enumerate([rev1, rev2, rev3, rev4, rev5, rev6, rev7, rev8]):

    rewritten = []
    for j, sentence in enumerate(nltk.tokenize.sent_tokenize(original)):
      bites = mouth.generate_possible_hypotheses([sentence])
      rewritten.extend(bites[0])
    rewrote.append(" ".join(rewritten))

  out_fp.write("\t".join([cat, prod_id, rewrote[0], rewrote[1], rewrote[2], rewrote[3], rewrote[4], rewrote[5], rewrote[6], rewrote[7], summ1, summ2, summ3]) + "\n")

in_fp.close()
out_fp.close()

