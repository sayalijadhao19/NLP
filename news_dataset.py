import pandas as pd
import nltk
from langdetect import detect
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

# Load the dataset
df = pd.read_csv("NewsCategorizer.csv")

# Language detection
languages = []
for headline in df['headline']:
    languages.append(detect(headline))
df['language'] = languages

# Word count
word_counts = []
for headline in df['headline']:
    word_counts.append(len(word_tokenize(headline)))
df['word_count'] = word_counts

# Sentence count
sentence_counts = []
for headline in df['headline']:
    sentence_counts.append(len(sent_tokenize(headline)))
df['sentence_count'] = sentence_counts

# Word-level tokenization
tokens_list = []
for headline in df['headline']:
    tokens_list.append(word_tokenize(headline))
df['word_tokens'] = tokens_list

# Sentence-level tokenization
sent_tokens_list = []
for headline in df['headline']:
    sent_tokens_list.append(sent_tokenize(headline))
df['sentence_tokens'] = sent_tokens_list

# Display the results
print("Language detection:")
print(df[['headline', 'language']].head())

print("\nWord count:")
print(df[['headline', 'word_count']].head())

print("\nSentence count:")
print(df[['headline', 'sentence_count']].head())

print("\nWord-level tokenization:")
print(df[['headline', 'word_tokens']].head())

print("\nSentence-level tokenization:")
print(df[['headline', 'sentence_tokens']].head())
