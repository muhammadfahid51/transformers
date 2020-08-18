"""Testing language model on mlm task"""

from transformers import pipeline
import argparse

parser = argparse.ArgumentParser(description="Test on MLM task")

parser.add_argument('-t', '--text', type=str, required=True)
args = parser.parse_args()


fill_mask = pipeline("fill-mask", model="Urdu_Roberta/", tokenizer="Urdu_Roberta/")

results = fill_mask(args.text)

for result in results:
    print(result)

from urduhack.datasets.text import ImdbUrduReviews