import tensorflow_hub as hub
import  tensorflow as tf
import tensorflow_text as text
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
Import the dataset (Dataset is taken from kaggle)
import pandas as pd

df = pd.read_csv("spam.csv")
df.head(5)
df.groupby('Category').describe()
df['Category'].value_counts()


df_spam = df[df['Category']=='spam']
df_spam.shape
df_ham = df[df['Category']=='ham']
df_ham.shape
df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape
df_balanced = pd.concat([df_ham_downsampled, df_spam])
df_balanced.shape
df_balanced['Category'].value_counts()
df_balanced['spam']=df_balanced['Category'].apply(lambda x: 1 if x=='spam' else 0)
df_balanced.sample(5)
X_train, X_test, y_train, y_test = train_test_split(df_balanced['Message'],df_balanced['spam'], stratify=df_balanced['spam'])
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
bert_preprocess_model = hub.KerasLayer(preprocess_url)
text_test = ['nice movie indeed', 'I love python programming']
text_preprocessed = bert_preprocess_model(text_test)
tk = text_preprocessed.keys()
# dict_keys(['input_mask', 'input_type_ids', 'input_word_ids'])
bert_model = hub.KerasLayer(encoder_url)
bert_results = bert_model(text_preprocessed)
rk = bert_results.keys()
# dict_keys(['default', 'encoder_outputs', 'pooled_output', 'sequence_output'])
e = []
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([e[0]], [e[1]])
# Bert layers
bert_preprocess=bert_preprocess_model
bert_encoder=bert_model

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.summary()
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)
model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)
y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()
import numpy as np
y_predicted = np.where(y_predicted > 0.5, 1, 0)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predicted)
from matplotlib import pyplot as plt
import seaborn as sn
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
reviews = [
    'Enter a chance to win $5000, hurry up, offer valid until march 31, 2021',
    'You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p pÂ£3.99',
    'it to 80488. Your 500 free text messages are valid until 31 December 2005.',
    'Hey Sam, Are you coming for a cricket game tomorrow',
    "Why don't you wait 'til at least wednesday to see if you get your ."
]
model.predict(reviews)