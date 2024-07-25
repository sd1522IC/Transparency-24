import pandas as pd
import numpy as np
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def load_texts_from_files(file_paths):
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(preprocess(file.read()))
    return texts

def perform_lda(texts, num_topics=3):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, alpha='auto', eta='auto', random_state=42)
    return lda_model, dictionary, corpus

def plot_lda_3d(lda_model, corpus, file_paths):
    # Get topic distributions for each document
    topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    
    # Convert topic distributions to a 2D array
    topic_matrix = []
    for distribution in topic_distributions:
        topic_matrix.append([topic[1] for topic in distribution])
    
    topic_matrix = np.array(topic_matrix)  # Convert to numpy array

    # Reduce dimensions to 3 using PCA
    pca = PCA(n_components=3)
    topic_matrix_3d = pca.fit_transform(topic_matrix)

    # Print the position vectors
    print("Position vectors of the coordinates:")
    for i, vec in enumerate(topic_matrix_3d):
        print(f"Document {file_paths[i]}: {vec}")

    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['r', 'g', 'b']
    labels = ['Trans Rights', 'Abortion', 'Veganism']
    
    for i, vec in enumerate(topic_matrix_3d):
        ax.scatter(vec[0], vec[1], vec[2], color=colors[i % len(colors)], label=file_paths[i])

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    # Set title and axis labels
    ax.set_title('3D Dirichlet Distribution of LDA Topic Vectors', fontname='Times New Roman', fontsize=12)
    ax.set_xlabel('PCA Component 1', fontname='Times New Roman', fontsize=10)
    ax.set_ylabel('PCA Component 2', fontname='Times New Roman', fontsize=10)
    #ax.set_zlabel('PCA Component 3', fontname='Times New Roman', fontsize=10)

    plt.show()

def main():
    # List of file paths
    file_paths = ['woke_Guardian_April_27_24.txt', 'woke_GBNews_July_24.txt', 'woke_ft_may_13_2024.txt']
    
    # Load and preprocess texts
    texts = load_texts_from_files(file_paths)
    
    # Perform LDA
    lda_model, dictionary, corpus = perform_lda(texts, num_topics=3)
    
    # Print the topics
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
        
    # Optional: Save topics to a CSV
    topics_data = {'Topic': [], 'Words': []}
    for idx, topic in lda_model.print_topics(-1):
        topics_data['Topic'].append(idx)
        topics_data['Words'].append(topic)
    
    topics_df = pd.DataFrame(topics_data)
    topics_df.to_csv('lda_topics.csv', index=False)
    
    # Plot 3D visualization
    plot_lda_3d(lda_model, corpus, file_paths)

if __name__ == "__main__":
    main()
