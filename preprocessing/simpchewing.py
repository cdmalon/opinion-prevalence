import numpy as np
import os
import sys
import tqdm
import re
import nltk.tokenize
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from transformers.utils import check_min_version

check_min_version("4.8.1")

class SimpChewing:
  def __init__(self):
    p2h_model_name = "/zdata/users/malon/p2h/asset-30epoch-output"
    ent_model_name = "roberta-large-mnli"
    self.ent_idx = 2
    self.con_idx = 0

    set_seed(42)

    p2h_config = AutoConfig.from_pretrained(p2h_model_name)
    self.p2h_tokenizer = AutoTokenizer.from_pretrained(
        p2h_model_name,
        cache_dir=None,
        use_fast=True,
        revision="main",
        use_auth_token=False,
    )
    self.p2h_model = AutoModelForSeq2SeqLM.from_pretrained(
        p2h_model_name,
        from_tf=False,
        config=p2h_config,
        cache_dir=None,
        revision="main",
        use_auth_token=False,
    )

    self.p2h_model.resize_token_embeddings(len(self.p2h_tokenizer))

    # source_prefix?
    self.p2h_prefix = "hypothesis: "
    self.num_sequences = 10

    self.p2h_max_target_length = 64
    self.p2h_max_source_length = 64

    self.ent_tokenizer = AutoTokenizer.from_pretrained(ent_model_name)
    self.ent_model = AutoModelForSequenceClassification.from_pretrained(ent_model_name)
    self.ent_max_seq_length = 500

    self.device = "cuda:0"
    self.p2h_model.to(self.device)
    self.ent_model.to(self.device)

  def generate_possible_hypotheses(self, premises):
    inputs = [self.p2h_prefix + x for x in premises]
    model_inputs = self.p2h_tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, max_length=self.p2h_max_source_length, return_tensors="pt")

    self.p2h_model.eval()
    input_ids = model_inputs["input_ids"].to(self.device)
    attention_mask = model_inputs["attention_mask"].to(self.device)
    # need "labels"?

    with torch.no_grad():
      outputs = self.p2h_model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        num_return_sequences=self.num_sequences,
                                        do_sample=True, top_p=.9, top_k=50,
                                        max_length=self.p2h_max_target_length)
    predictions = self.p2h_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    answer = []

    # Take the first prediction with the greater number of sentences
    # Return each sentence as a separate bite
    for i in range(len(premises)):
      answer.append([])
      nbest = 0

      for j in range(self.num_sequences):
        all_sent = nltk.tokenize.sent_tokenize(predictions[self.num_sequences*i+j])
        if(len(all_sent) > nbest):
          nbest = len(all_sent)
          answer[i] = []
          for k in range(len(all_sent)):
            answer[i].append(re.sub(r" *$", "", all_sent[k]))
            if(re.search(r"[.,?!]$", answer[i][k]) == None):
              answer[i][k] = answer[i][k] + "."

    return answer

  def generate_supported_hypotheses(self, premises):
    possible = self.generate_possible_hypotheses(premises)
    answer = []
    for i, p in enumerate(premises):
      batch = []
      for j in range(self.num_sequences):
        h = possible[self.num_sequences*i+j]
        batch.append((p,h))

      batch_inputs = self.ent_tokenizer.batch_encode_plus(batch, padding=True, truncation=True, max_length=self.ent_max_seq_length, truncation_strategy="only_first", return_tensors="pt")
      input_ids = batch_inputs["input_ids"].to(self.device)
      attention_mask = batch_inputs["attention_mask"].to(self.device)

      with torch.no_grad():
        outputs = self.ent_model(input_ids=input_ids, attention_mask=attention_mask)
      # No labels
      logits = outputs[0].detach().cpu().numpy()
      preds = np.argmax(logits, axis=1)

      accepted = []
      for j in range(self.num_sequences):
        h = possible[self.num_sequences*i+j]
        if(preds[j] == self.ent_idx):
          accepted.append(h)

      if(len(accepted) < 2):
        accepted = [p]

      answer.append(accepted)

    return answer

  def generate_nontrivial_supported_hypotheses(self, premises, trivial=None):
    supported = self.generate_supported_hypotheses(premises)
    if trivial == None:
      return supported

    answer = []
    for i, p in enumerate(premises):
      batch = []
      for j in range(len(supported[i])):
        h = supported[i][j]
        batch.append((trivial,h))

      batch_inputs = self.ent_tokenizer.batch_encode_plus(batch, padding=True, truncation=True, max_length=self.ent_max_seq_length, truncation_strategy="only_first", return_tensors="pt")
      input_ids = batch_inputs["input_ids"].to(self.device)
      attention_mask = batch_inputs["attention_mask"].to(self.device)

      with torch.no_grad():
        outputs = self.ent_model(input_ids=input_ids, attention_mask=attention_mask)
      # No labels
      logits = outputs[0].detach().cpu().numpy()
      preds = np.argmax(logits, axis=1)

      accepted = []
      for j in range(len(supported[i])):
        h = supported[i][j]
        if(preds[j] != self.ent_idx):
          accepted.append(h)

      if(len(accepted) < 2):
        accepted = [p]

      answer.append(accepted)

    return answer

  def generate_nontrivial_nonredundant_supported_hypotheses(self, premises, trivial=None):
    hypotheses = self.generate_nontrivial_supported_hypotheses(premises, trivial)

    answer = []
    for i, p in enumerate(premises):
      accepted = []
      for j, h in enumerate(hypotheses[i]):
        if j == 0:
          accepted = [h]
        else:
          batch = []
          for k, other in enumerate(accepted):
            batch.append((other, h))

          batch_inputs = self.ent_tokenizer.batch_encode_plus(batch, padding=True, truncation=True, max_length=self.ent_max_seq_length, truncation_strategy="only_first", return_tensors="pt")
          input_ids = batch_inputs["input_ids"].to(self.device)
          attention_mask = batch_inputs["attention_mask"].to(self.device)
    
          with torch.no_grad():
            outputs = self.ent_model(input_ids=input_ids, attention_mask=attention_mask)
          # No labels
          logits = outputs[0].detach().cpu().numpy()
          preds = np.argmax(logits, axis=1)
   
          ok = True
          # for k in range(len(accepted)):
          for k, other in enumerate(accepted):
            print(other + " / " + h + " = " + str(preds[k]))
            if(preds[k] == self.ent_idx):
              ok = False
   
          if ok:
            accepted.append(h)

      answer.append(accepted)

    return answer

