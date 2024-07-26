import os
import pandas as pd
import numpy as np
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.decomposition import PCA
import plotly.express as px

nltk.download('punkt')
nltk.download('stopwords')

# Converts text to lowercase.
# Tokenizes the text into words.
# Removes non-alphabetical tokens and stopwords (punctuation).
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# load_texts_from_folder(folder_path): Reads and preprocesses text files from a specified folder.
# Lists all files in the folder.
# Opens each .txt file, preprocesses the text, and stores it in a list.
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

# perform_lda(texts, num_topics=3): Performs Latent Dirichlet Allocation (LDA) on the preprocessed texts.
# Creates a dictionary and corpus from the texts.
# Fits an LDA model to the corpus.
def perform_lda(texts, num_topics=4):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, alpha='auto', eta='auto', random_state=42)
    return lda_model, dictionary, corpus

# plot_lda_2d(lda_model, corpus, file_paths): Plots the LDA topic distributions in 2D space.
# Gets topic distributions for each document.
# Converts the topic distributions to a matrix.
# Uses PCA to reduce the matrix to 2 dimensions.
# Prints the 2D coordinates of each document.
# Plots the documents in a 2D scatter plot with interactive hover functionality.
def plot_lda_2d(lda_model, corpus, file_paths):
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

    # Define colors based on filename prefixes
    color_map = {
        'G_': 'red',      # Red for G_
        'FT_': 'green',   # Green for FT_
        'GB_': 'blue',    # Blue for GB_
        'Areseblog_': 'magenta' # Magenta for Areseblog_
    }
    
    # Assign default color for files without a specific prefix
    default_color = 'black' # Black
    
    colors = []
    labels = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        color = default_color
        label = 'Other'
        for prefix, col in color_map.items():
            if file_name.startswith(prefix):
                color = col
                label = prefix
                break
        colors.append(color)
        labels.append(label)

    # Create DataFrame for PCA plot
    df_pca = pd.DataFrame({
        'x': topic_matrix_2d[:, 0],
        'y': topic_matrix_2d[:, 1],
        'color': colors,
        'label': labels,
        'file_path': file_paths
    })

    # Create PCA plot
    fig_pca = px.scatter(df_pca, x='x', y='y', color='label', hover_data=['file_path'],
                         color_discrete_map=color_map, labels={'label': 'Document Type'})
    
    fig_pca.update_layout(
        title='2D Dirichlet Distribution of LDA Topic Vectors with PCA',
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        font=dict(
            family='Times New Roman',
            size=12
        ),
        legend_title_text='Document Type',
        legend=dict(
            x=1.05,
            y=1,
            traceorder='normal',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        )
    )

    fig_pca.show()

    # Create DataFrame for LDA plot without PCA
    df_lda = pd.DataFrame(topic_matrix, columns=[f'Topic {i+1}' for i in range(topic_matrix.shape[1])])
    df_lda['file_path'] = file_paths
    df_lda['color'] = colors
    df_lda['label'] = labels

    # Create LDA plot without PCA
    fig_lda = px.scatter_matrix(df_lda, dimensions=df_lda.columns[:-3], color='label', hover_data=['file_path'],
                                color_discrete_map=color_map, labels={'label': 'Document Type'})

    fig_lda.update_layout(
        title='LDA Topic Distribution without PCA',
        font=dict(
            family='Times New Roman',
            size=12
        ),
        legend_title_text='Document Type',
        legend=dict(
            x=1.05,
            y=1,
            traceorder='normal',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        )
    )

    fig_lda.show()

# main(): The main function to execute the script.
# Sets the folder path containing text files.
# Loads and preprocesses the texts from the folder.
# Checks if any texts were loaded; if not, it exits.
# Performs LDA on the texts.
# Prints the topics found by LDA.
# Optionally saves the topics to a CSV file.
# Calls the function to plot the 2D visualization of the topic distributions.
def main():
    folder_path = 'texts'  # Folder containing the text files
    
    # Load and preprocess texts
    texts, file_paths = load_texts_from_folder(folder_path)
    
    if not texts:
        print("No texts loaded. Exiting.")
        return
    
    # Perform LDA with the specified number of topics
    num_topics = 4  # Change this value to set the number of topics
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
    
    # Plot 2D visualization with and without PCA
    plot_lda_2d(lda_model, corpus, file_paths)
    print('The magnitude of the vector element is the proportion of the document predicted to talk about this topic. The direction is ')

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()
