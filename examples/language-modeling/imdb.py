"""Train sentiment model"""

import pandas as pd
import tensorflow as tf

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

df_train = pd.read_csv("imdb_urdu_reviews_train.csv")
df_test = pd.read_csv("imdb_urdu_reviews_test.csv")


tokenizer = RobertaTokenizer.from_pretrained("tf-roberta-urdu-small")
model = TFRobertaForSequenceClassification.from_pretrained("tf-roberta-urdu-small", num_labels=2)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss="binary_crossentropy",
              metrics=["accuracy"])


train_input_ids = []
train_attention_masks = []
test_input_ids = []
test_attention_masks = []

for text in df_train["review"].values:
    encoded = tokenizer.encode_plus(text)
    train_input_ids.append(encoded["input_ids"])
    train_attention_masks.append(encoded["attention_mask"])

for text in df_test["review"].values:
    encoded = tokenizer.encode_plus(text)
    test_input_ids.append(encoded["input_ids"])
    test_attention_masks.append(encoded["attention_mask"])

label_to_int = {"positive": 1, "negative": 0}
train_labels = [label_to_int[label] for label in df_train["sentiment"].values]
test_labels = [label_to_int[label] for label in df_test["sentiment"].values]
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

train_input_ids = tf.keras.preprocessing.sequence.pad_sequences(train_input_ids, maxlen=256, padding="post", value=1)
train_attention_masks = tf.keras.preprocessing.sequence.pad_sequences(train_attention_masks, maxlen=256, padding="post", value=0)

test_input_ids = tf.keras.preprocessing.sequence.pad_sequences(test_input_ids, maxlen=256, padding="post", value=1)
test_attention_masks = tf.keras.preprocessing.sequence.pad_sequences(test_attention_masks, maxlen=256, padding="post", value=0)

print(train_input_ids.shape, train_attention_masks.shape, test_input_ids.shape, test_attention_masks.shape)

model.fit(x=[train_input_ids, train_attention_masks], y=train_labels, epochs=5, batch_size=32,
          validation_data=([test_input_ids, test_attention_masks], test_labels))

model.save("tf-sentiment-model.h5")








