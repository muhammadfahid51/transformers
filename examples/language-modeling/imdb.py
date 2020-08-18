"""Train sentiment model"""

import pandas as pd

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

# df_train = pd.read_csv("urdu_imdb_review_train.csv")
# df_test = pd.read_csv("urdu_imdb_review_test.csv")


tokenizer = RobertaTokenizer.from_pretrained("roberta-urdu-small")
model = TFRobertaForSequenceClassification.from_pretrained("roberta-urdu-small", from_pt=True)

print(model.summary())





