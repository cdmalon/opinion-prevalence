import json
import tqdm
import sys
import os
import nltk.tokenize
from summac.model_summac import SummaCZS

if(len(sys.argv) != 5):
  raise ValueError("Usage")

in_file = sys.argv[1]
product_file = sys.argv[2]
summary_file = sys.argv[3]
out_file = sys.argv[4]

if(os.path.exists(out_file)):
  raise ValueError("Output already exists")

model = SummaCZS(granularity="document", model_name="mnli", bins="percentile", use_con=False, device="cuda")

products = {}
product_fp = open(product_file, "r")
for line in product_fp.readlines():
  tok = line.rstrip().split(" ")
  products[tok[0]] = " ".join(tok[1:])

in_fp = open(in_file, "r")
summ_fp = open(summary_file, "r")
out_fp = open(out_file, "w")

hdr = in_fp.readline()

threshold = .04

tot_prevalence = 0
nsummaries = 0
for line in tqdm.tqdm(in_fp.readlines()):
  nsummaries = nsummaries + 1

  cat, prod_id, rev1, rev2, rev3, rev4, rev5, rev6, rev7, rev8, summ1, summ2, summ3 = line.rstrip().split("\t")

  trivial = "I bought " + products[prod_id] + "."

  summ = summ_fp.readline().rstrip()
  nsent = 0
  prevalence = 0
  sents = nltk.tokenize.sent_tokenize(summ)
  output = ""
  for i, sentence in enumerate(sents):
    nsent = nsent + 1
    generated = sentence

    implied = 0
    tot = 0

    if model.score([trivial], [generated])["scores"][0] > threshold:
      output = output + " " + generated + " (T)"
      continue

    redundant = False
    for j in range(i):
      if model.score([sents[j]], [generated])["scores"][0] > threshold:
        redundant = True

    if redundant:
      output = output + " " + generated + " (R)"
      continue

    for k, original in enumerate([rev1, rev2, rev3, rev4, rev5, rev6, rev7, rev8]):
      tot = tot + 1
      score = model.score([original], [generated])["scores"][0]
      if score > threshold:
        implied = implied + 1

    prevalence = prevalence + (implied/tot)
    output = output + " " + generated + " (" + str(implied) + ")"

  prevalence = prevalence / nsent

  out_fp.write(("%.4f" % prevalence) + " " + output + "\n")
  tot_prevalence = tot_prevalence + prevalence

out_fp.close()

tot_prevalence = tot_prevalence / nsummaries
print("Total prevalence: " + ("%.4f" % tot_prevalence))

