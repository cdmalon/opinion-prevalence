# ReviewNLI Data

## Associating judgments with NLI pairs

NLI pairs are constructed from the reviews and summaries
distributed with [Copycat](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/gold_summs).  Referring to the `dev.csv` and
`test.csv` files provided in their `gold_summs` directory, execute

```
python preprocess.py gold_summs/dev.csv dev-questions.jsonl
python preprocess.py gold_summs/test.csv test-questions.jsonl
```

to create JSONL files `dev-questions.jsonl` and `test-questions.jsonl`
where each line contains a review and a list
of sentences from a summary.  The index field in each example
consists of the Amazon ASIN code for the product, the review
number (from 0 to 7), and the summary number (always 0),
separated by a colon.

Each example consists of multiple NLI pairs to be classified,
where the premise is the entire review and the hypothesis is one
of the sentences in the "summary" array.  Human judgments are
available in the provided JSONL files in the "answers" array,
which is in one-to-one correspondence with the "summary" array of the
associated example.  A value of 1 means the summary statement is
mostly supported by the review and 0 means that it is not.
Be sure to match the "index" field of the examples you are comparing,
as `dev-questions.jsonl` and `test-questions.jsonl` may be
written in a different order from the judgment files.

The `*workers.jsonl` files have individual worker judgments for
each NLI pair.  For anonymity, we have hashed the Turk worker ID's.
All experiments in our paper are based on the results of majority
voting among three workers, which are recorded in `*majority.jsonl`.

For details about dataset construction, see our paper.

