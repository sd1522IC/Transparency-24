import os
import pandas as pd
import numpy as np
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from bertopic import BERTopic
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

# Sliding window chunks to handle long texts
def sliding_window_chunks(text, max_length=512, stride=256):
    """
    Splits the input text into overlapping chunks using a sliding window approach.

    Parameters:
    - text (str): The input text to be split.
    - max_length (int): The maximum length of tokens per chunk (must not exceed the model's limit).
    - stride (int): The number of tokens to skip between chunks.

    Returns:
    - List of text chunks.
    """
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    tokens = sentiment_pipeline.tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')['input_ids'][0]
    total_tokens = tokens.size(0)
    chunks = []
    for i in range(0, total_tokens, stride):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = sentiment_pipeline.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if i + max_length >= total_tokens:
            break
    return chunks

# Advanced Sentiment Analysis using the sliding window approach
def advanced_sentiment_analysis(texts, file_paths):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    all_sentiments = []
    all_files = []
    all_chunks = []
    for text, file_path in zip(texts, file_paths):
        chunks = sliding_window_chunks(text)
        chunk_sentiments = [sentiment_pipeline(chunk)[0]['label'] for chunk in chunks]
        all_sentiments.extend(chunk_sentiments)
        all_files.extend([file_path] * len(chunk_sentiments))
        all_chunks.extend([f"Chunk {i+1}" for i in range(len(chunk_sentiments))])
    
    # Convert 'POSITIVE'/'NEGATIVE' to numerical values
    sentiment_scores = [1 if s == 'POSITIVE' else -1 for s in all_sentiments]
    
    # Debugging: Print out sentiment distribution
    print("Sentiment scores:", sentiment_scores)
    print("Files:", all_files)
    print("Chunks:", all_chunks)

    return sentiment_scores, all_files, all_chunks

# Perform BERTopic for dynamic topic modeling
def perform_bertopic(texts, raw_texts):
    vectorizer_model = TfidfVectorizer(stop_words='english')
    topic_model = BERTopic(vectorizer_model=vectorizer_model, language="english")
    topics, probs = topic_model.fit_transform(raw_texts)
    return topic_model, topics, probs

# Perform LDA and PCA for topic modeling and dimensionality reduction
def perform_lda_pca(texts):
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(texts)
    # Convert the dictionary to a bag of words corpus.
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model on the corpus.
    lda_model = models.LdaModel(corpus, num_topics=5, random_state=42)
    lda_topics = lda_model.show_topics(num_words=10, formatted=False)

    # Extract topic distributions for each document.
    topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    topic_matrix = np.array([[topic[1] for topic in dist] for dist in topic_distributions])

    # Perform PCA on the topic distributions.
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(topic_matrix)

    return lda_model, pca_result, topic_matrix, lda_topics

# Plot LDA topics with PCA
def plot_lda_pca(pca_result, topic_matrix, file_paths):
    df_pca = pd.DataFrame({
        'x': pca_result[:, 0],
        'y': pca_result[:, 1],
        'file_path': file_paths
    })

    fig_pca = px.scatter(df_pca, x='x', y='y', hover_data=['file_path'])
    fig_pca.update_layout(
        title='PCA of LDA Topic Distribution',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2'
    )
    fig_pca.show()

# Plot BERTopic distributions in 2D space with t-SNE
def plot_bertopic_2d(topic_model, topics, probs, raw_texts, sentiments, files, chunks):
    embeddings = topic_model.embedding_model.embedding_model.encode(raw_texts)
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(raw_texts)-1))
    tsne_embeddings = tsne_model.fit_transform(embeddings)

    df_tsne = pd.DataFrame(tsne_embeddings, columns=['x', 'y'])
    df_tsne['topic'] = topics
    df_tsne['sentiment'] = sentiments  # Assuming normalized sentiment scores
    df_tsne['file'] = files
    df_tsne['chunk'] = chunks

    fig_tsne = px.scatter(df_tsne, x='x', y='y', color='sentiment', hover_data=['file', 'chunk'],
                          color_continuous_scale='RdYlGn', labels={'sentiment': 'Sentiment Score'})
    fig_tsne.update_layout(
        title='t-SNE of BERTopic Distribution with Sentiment',
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2'
    )
    fig_tsne.show()

# WordCloud for BERTopic topics
def plot_bertopic_wordcloud(topic_model):
    topic_freq = topic_model.get_topic_freq()
    for topic in topic_freq['Topic']:
        words = topic_model.get_topic(topic)
        wc = WordCloud(width=800, height=400, max_words=10).generate(' '.join([word[0] for word in words]))
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic}')
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

    # Load raw texts for advanced sentiment analysis
    raw_texts = [open(file, 'r', encoding='utf-8').read() for file in file_paths]

    # Perform advanced sentiment analysis
    sentiments, files, chunks = advanced_sentiment_analysis(raw_texts, file_paths)

    # Perform BERTopic
    topic_model, topics, probs = perform_bertopic(texts, raw_texts)
    
    # Perform LDA and PCA
    lda_model, pca_result, topic_matrix, lda_topics = perform_lda_pca(texts)
    
    # Plot 2D visualization with t-SNE and sentiment for BERTopic
    plot_bertopic_2d(topic_model, topics, probs, raw_texts, sentiments, files, chunks)

    # Plot PCA results for LDA topics
    plot_lda_pca(pca_result, topic_matrix, file_paths)

    # Plot wordclouds for each BERTopic topic
    plot_bertopic_wordcloud(topic_model)

if __name__ == "__main__":
    main()
