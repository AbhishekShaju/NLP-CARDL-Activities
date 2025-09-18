# feature_extraction.py

# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample dataset (replace with your own dataset like movie reviews or tweets)
documents = [
    "I love this movie, it is fantastic and full of surprises!",
    "This film was terrible, I hated every moment.",
    "Amazing storyline and brilliant acting. Truly loved it!",
    "Not my taste, the movie was boring and predictable.",
    "One of the best movies I have ever seen!"
]

print("===== Original Dataset =====")
for i, doc in enumerate(documents, 1):
    print(f"Doc {i}: {doc}")
print("\n")

# ---------------------------
# a. Bag-of-Words (BoW)
# ---------------------------
print("===== Bag-of-Words (BoW) Representation =====")
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)

print("Feature Names:", bow_vectorizer.get_feature_names_out())
print("BoW Matrix Shape:", bow_matrix.shape)
print("BoW Matrix (as array):")
print(bow_matrix.toarray())
print("\n")

# ---------------------------
# b. Term Frequency-Inverse Document Frequency (TF-IDF)
# ---------------------------
print("===== TF-IDF Representation =====")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("Feature Names:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("TF-IDF Matrix (as array):")
print(tfidf_matrix.toarray())
