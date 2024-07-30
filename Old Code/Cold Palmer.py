import os
import pandas as pd
import numpy as np
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.decomposition import PCA
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Converts text to lowercase, tokenizes, removes non-alphabetical tokens and stopwords
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Load and preprocess texts from a folder
def load_texts_from_folder(folder_path):
    texts = []
    file_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts.append(preprocess(file.read()))
                    file_paths.append(file_path)
            except Exception as e:
                print(f"An error occurred while reading {file_path}: {e}")
    return texts, file_paths

# Perform LDA on the texts
def perform_lda(texts, num_topics=4):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, alpha='auto', eta='auto', random_state=42)
    return lda_model, dictionary, corpus

# Basic Sentiment Analysis (Simplified)
def basic_sentiment_analysis(text):
    positive_words = ['good', 'happy', 'joy', 'positive', 'fortunate', 'correct', 'superior']
    negative_words = ['bad', 'sad', 'pain', 'negative', 'unfortunate', 'wrong', 'inferior']
    
    text = text.lower()
    sentiment_score = sum([1 for word in text.split() if word in positive_words]) - \
                      sum([1 for word in text.split() if word in negative_words])
    
    return sentiment_score

# Plot LDA topic distributions in 2D space with PCA
def plot_lda_2d(lda_model, corpus, file_paths, sentiments):
    topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    
    topic_matrix = []
    for distribution in topic_distributions:
        topic_matrix.append([topic[1] for topic in distribution])
    
    topic_matrix = np.array(topic_matrix)

    # PCA analysis
    pca = PCA(n_components=2)
    topic_matrix_2d = pca.fit_transform(topic_matrix)

    print("Position vectors of the coordinates:")
    for i, vec in enumerate(topic_matrix_2d):
        print(f"Document {file_paths[i]}: {vec}")

    # Create DataFrame for PCA plot
    df_pca = pd.DataFrame({
        'x': topic_matrix_2d[:, 0],
        'y': topic_matrix_2d[:, 1],
        'sentiment': sentiments,
        'file_path': file_paths
    })

    # Plot PCA with sentiment
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca['x'], df_pca['y'], c=df_pca['sentiment'], cmap='RdYlGn')
    plt.colorbar(label='Sentiment Score')
    plt.title('PCA of Topic Distribution with Sentiment')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

# WordCloud for Topics
def plot_wordcloud(lda, dictionary, n_top_words=10):
    topic_words = []
    for t in range(lda.num_topics):
        words = lda.show_topic(t, topn=n_top_words)
        topic_words.append(" ".join([word[0] for word in words]))
        wc = WordCloud(width=800, height=400, max_words=10).generate(topic_words[t])
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {t}')
        plt.show()

# Main function
def main():
    folder_name = input("Enter the folder name containing the text files (default is 'texts'): ")
    if not folder_name:
        folder_name = 'texts'
    folder_path = os.path.join('.', folder_name)
    
    # Load and preprocess texts
    texts, file_paths = load_texts_from_folder(folder_path)
    if not texts:
        print("No texts loaded. Exiting.")
        return
    
    # Perform LDA
    num_topics = 4
    lda_model, dictionary, corpus = perform_lda(texts, num_topics=num_topics)
    
    # Print the topics
    for idx, topic in lda_model.print_topics(num_topics):
        print(f'Topic: {idx} \nWords: {topic}')
        
    # Optional: Save topics to a CSV
    topics_data = {'Topic': [], 'Words': []}
    for idx, topic in lda_model.print_topics(num_topics):
        topics_data['Topic'].append(idx)
        topics_data['Words'].append(topic)
    
    topics_df = pd.DataFrame(topics_data)
    topics_df.to_csv('lda_topics.csv', index=False)

    # Basic Sentiment Analysis
    texts_raw = [open(file, 'r', encoding='utf-8').read() for file in file_paths]
    sentiments = [basic_sentiment_analysis(text) for text in texts_raw]

    # Plot 2D visualization with PCA and sentiment
    plot_lda_2d(lda_model, corpus, file_paths, sentiments)

    # Plot wordclouds for each topic
    plot_wordcloud(lda_model, dictionary)

if __name__ == "__main__":
    main()
