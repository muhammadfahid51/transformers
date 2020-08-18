"""Train sentiment model"""

import numpy as np
import pandas as pd
import tensorflow as tf

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

df_train = pd.read_csv("imdb_urdu_reviews_train.csv")
df_test = pd.read_csv("imdb_urdu_reviews_test.csv")


tokenizer = RobertaTokenizer.from_pretrained("tf-roberta-urdu-small")
# model = TFRobertaForSequenceClassification.from_pretrained("tf-roberta-urdu-small", num_labels=2)

# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.keras.losses.binary_crossentropy(),
#               metrics=tf.keras.metrics.Accuracy())


input_ids = []
attention_masks = []

for text in df_train["review"].values:
    encoded = tokenizer.encode_plus(text, max_length=512, padding="max_length")
    input_ids.append(encoded["input_ids"])
    attention_masks.append(encoded["attention_mask"])

label_to_int = {"positive": 1, "negative": 0}
train_labels = [label_to_int[label] for label in df_train["sentiment"].values]
train_labels = tf.keras.utils.to_categorical(train_labels)
print(train_labels)

input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)

print(input_ids.shape, attention_masks.shape)









