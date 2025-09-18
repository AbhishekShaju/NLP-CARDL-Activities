import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models

# --- Download NLTK resources (first run only) ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Example corpus ---
corpus = [
    "The economy is working better than ever.",
    "Sports events are gaining popularity worldwide.",
    "The new movie has received great reviews from critics.",
    "Stock markets are experiencing a significant drop today.",
    "Football is the most popular sport in the world.",
    "Academics are focusing more on artificial intelligence research.",
    "Machine learning and deep learning are transforming technology.",
    "Basketball championships are watched by millions of fans.",
    "The government announced new economic policies.",
    "AI applications are used in healthcare and finance."
]

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

processed_corpus = [preprocess(doc) for doc in corpus]

# --- Dictionary & Corpus ---
dictionary = corpora.Dictionary(processed_corpus)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_corpus]

# --- LDA Model ---
lda_model = gensim.models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42,
    passes=10,
    iterations=100,
    alpha='auto',
    per_word_topics=True
)

print("\nTopics discovered:")
for idx, topic in lda_model.show_topics(num_topics=3, num_words=10, formatted=False):
    print(f"\nTopic {idx}: " + ", ".join([word for word, _ in topic]))

# --- Visualization ---
vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.save_html(vis, "lda_vis.html")
print("\n✅ Interactive visualization saved as lda_vis.html (open it in a browser)")
