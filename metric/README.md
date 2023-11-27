# Opinion prevalence metric

This script computes opinion prevalence of generated summaries
against input sets of reviews.  Reference summaries are not required.

A file of product names is used to define trivial statements for each
review, but if these are not known, you can make a file where each
product name is "a product".

## Requirements

Tested with Python 3.7:
```sh
pip install -r requirements.txt
```

## Running the script

The script takes four arguments, consisting of the input review set file,
the file with product names, the file with the generated summaries,
and the desired output filename (which should not yet exist).

The review set file is in the CSV format of the Amazon review dataset
distributed with [Copycat](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/gold_summs).  The script is easily modified
to handle other formats.

Generated summaries are expected one per line, corresponding to the
products in the review set file.

Entries in the product name file can be in any order.
On each line, the product ID (ASIN) used in the review set file is expected,
followed by a space, followed by the product name, such as:
```
B006NNLX2C Norpro Silicone Steamer with Insert, Green 
```

Referring to the `test.csv` file provided in Copycat's `gold_summs` directory,
you can compute the prevalence of the greedy summaries distributed in this
repository by executing
```sh
python verbose-prevalence-mnli-doc-text.py /path/to/gold_summs/test.csv ../data/product_names.txt ../greedy-outputs/test-greedy-summaries.txt greedy-prevalence.txt
```

In the output file `greedy-prevalence.txt`, the prevalence of each individual
summary is written, followed by the summary text with an annotation after each
sentence in parentheses.  The annotation is either a number, indicating the
number of reviews that supported the sentence, or 'T', indicating the sentence
was judged to be trivial, or an 'R', indicating the sentence was judged
to be redundant.

The overall (average) prevalence is reported on stdout.
Expected result: `Total prevalence: 0.4744`.


## Object-Oriented wrapper

Alternatively, if you want to use the metric within your own code, you can use the OO version:

```python
from prevalence_metric import PrevalenceMetric

metric = PrevalenceMetric()

prevalence, redundant_fraction, trivial_fraction = metric.get_prevalence(
    reviews, # a list of lists of reviews
    generated_summaries, # a list of summaries
    product_names, # a list of product names
    trivial_template = "I bought {:}.",
    trivial_default = "a product",
    batch_size = 32,
)
```