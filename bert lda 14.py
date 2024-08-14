import os
import pandas as pd
import numpy as np
import torch
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
import torch.nn.functional as F

nltk.download('punkt')
nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return tokens

    def load_texts_from_folder(self, folder_path):
        texts = []
        file_paths = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        texts.append(self.preprocess(file.read()))
                        file_paths.append(file_path)
                except Exception as e:
                    print(f"An error occurred while reading {file_path}: {e}")
        return texts, file_paths

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    def sliding_window_chunks(self, text, max_length=512, stride=256):
        tokens = self.sentiment_pipeline.tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')['input_ids'][0]
        total_tokens = tokens.size(0)
        chunks = []
        for i in range(0, total_tokens, stride):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = self.sentiment_pipeline.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if i + max_length >= total_tokens:
                break
        return chunks

    def softmax_with_temperature(self, logits, temperature=1.0):
        logits = logits / temperature
        return F.softmax(logits, dim=-1)

    def advanced_sentiment_analysis(self, texts, file_paths, temperature=1.0):
        all_sentiments = []
        all_files = []
        all_chunks = []
        for text, file_path in zip(texts, file_paths):
            chunks = self.sliding_window_chunks(text)
            chunk_sentiments = []
            for chunk in chunks:
                logits = self.sentiment_pipeline(chunk, return_all_scores=True)[0]
                scores = self.softmax_with_temperature(torch.tensor([score['score'] for score in logits]), temperature)
                # Use the continuous score for POSITIVE class (index 1)
                chunk_sentiments.append(scores[1].item())
            all_sentiments.extend(chunk_sentiments)
            all_files.extend([file_path] * len(chunk_sentiments))
            all_chunks.extend([f"Chunk {i+1}" for i in range(len(chunk_sentiments))])
        
        return all_sentiments, all_files, all_chunks

class TopicModeler:
    def __init__(self):
        self.vectorizer_model = TfidfVectorizer(stop_words='english')
        self.topic_model = BERTopic(vectorizer_model=self.vectorizer_model, language="english")

    def perform_bertopic(self, texts, raw_texts):
        topics, probs = self.topic_model.fit_transform(raw_texts)
        return topics, probs

    def perform_lda_pca(self, texts):
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = models.LdaModel(corpus, num_topics=5, random_state=42)
        topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        topic_matrix = np.array([[topic[1] for topic in dist] for dist in topic_distributions])
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(topic_matrix)

        # Attempt to name topics
        pca_labels = self.get_pca_axis_labels(lda_model)
        
        return lda_model, pca_result, topic_matrix, pca_labels

    def get_pca_axis_labels(self, lda_model):
        topics = lda_model.show_topics(num_words=1, formatted=False)
        pca_labels = {"x_label": "", "y_label": ""}
        if len(topics) > 0:
            pca_labels["x_label"] = topics[0][1][0][0]  # Top word for PCA1 (assume topic 0)
        if len(topics) > 1:
            pca_labels["y_label"] = topics[1][1][0][0]  # Top word for PCA2 (assume topic 1)
        return pca_labels

    def plot_lda_pca(self, pca_result, file_paths, pca_labels):
        df_pca = pd.DataFrame({
            'x': pca_result[:, 0],
            'y': pca_result[:, 1],
            'file_path': file_paths
        })

        fig_pca = px.scatter(df_pca, x='x', y='y', hover_data=['file_path'])
        fig_pca.update_layout(
            title='PCA of LDA Topic Distribution',
            xaxis_title=f'PCA1: {pca_labels["x_label"]}',
            yaxis_title=f'PCA2: {pca_labels["y_label"]}'
        )

        fig_pca.show()

    def plot_bertopic_2d(self, topics, probs, raw_texts, sentiments, files, chunks):
        embeddings = self.topic_model.embedding_model.embedding_model.encode(raw_texts)
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(raw_texts)-1))
        tsne_embeddings = tsne_model.fit_transform(embeddings)

        df_tsne = pd.DataFrame(tsne_embeddings, columns=['x', 'y'])
        df_tsne['topic'] = topics
        df_tsne['sentiment'] = sentiments
        df_tsne['file'] = files
        df_tsne['chunk'] = chunks

        fig_tsne = px.scatter(df_tsne, x='x', y='y', color='sentiment', hover_data=['file', 'chunk', 'sentiment'],
                              color_continuous_scale='RdYlGn', labels={'sentiment': 'Sentiment Score'})
        fig_tsne.update_layout(
            title='t-SNE of BERTopic Distribution with Sentiment',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2'
        )
        fig_tsne.show()

    def plot_bertopic_wordcloud(self):
        topic_freq = self.topic_model.get_topic_freq()
        for topic in topic_freq['Topic']:
            words = self.topic_model.get_topic(topic)
            wc = WordCloud(width=800, height=400, max_words=10).generate(' '.join([word[0] for word in words]))
            plt.figure()
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic}')
            plt.show()

class TextAnalysisPipeline:
    def __init__(self):
        self.folder_name = ''
        self.temperature = 1.0
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()

    def get_user_input(self):
        # Prompt the user for the folder name
        self.folder_name = input("Enter the folder name containing the text files (default is 'texts'): ")
        if not self.folder_name:
            self.folder_name = 'texts'

        # Prompt the user for the temperature value
        temp_input = input("Enter the temperature for softmax (default is 1.0): ")
        try:
            self.temperature = float(temp_input) if temp_input else 1.0
        except ValueError:
            print("Invalid input for temperature. Using default value 1.0.")
            self.temperature = 1.0

    def run(self):
        self.get_user_input()  # Get user inputs

        folder_path = os.path.join('.', self.folder_name)
        texts, file_paths = self.text_processor.load_texts_from_folder(folder_path)
        if not texts:
            print("No texts loaded. Exiting.")
            return

        raw_texts = [open(file, 'r', encoding='utf-8').read() for file in file_paths]
        sentiments, files, chunks = self.sentiment_analyzer.advanced_sentiment_analysis(raw_texts, file_paths, self.temperature)
        topics, probs = self.topic_modeler.perform_bertopic(texts, raw_texts)
        lda_model, pca_result, topic_matrix, pca_labels = self.topic_modeler.perform_lda_pca(texts)

        self.topic_modeler.plot_bertopic_2d(topics, probs, raw_texts, sentiments, files, chunks)
        self.topic_modeler.plot_lda_pca(pca_result, file_paths, pca_labels)
        self.topic_modeler.plot_bertopic_wordcloud()

if __name__ == "__main__":
    pipeline = TextAnalysisPipeline()
    pipeline.run()
