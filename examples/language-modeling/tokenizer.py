"""Pre-train Urdu Roberta tokenizer"""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("/home/ikram/workplace/datasets/sentences/").glob("*.txt")]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=52000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(".", "u_roberta")