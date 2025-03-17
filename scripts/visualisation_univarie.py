import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Fonction pour générer un nuage de mots
def plot_wordcloud(df, column):
    """
    Affiche un nuage de mots pour une colonne textuelle.
    """
    text = " ".join(df[column].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Wordcloud de {column}')
    plt.show()

# Fonction pour afficher la densité d'une colonne numérique
def plot_density(df, column):
    """
    Affiche la densité (distribution) d'une colonne numérique.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[column].dropna(), fill=True, color='skyblue')
    plt.title(f'Distribution de {column}')
    plt.xlabel(column)
    plt.ylabel('Densité')
    plt.show()

# Fonction pour afficher les valeurs uniques d'une colonne
def plot_unique_values(df, column):
    """
    Affiche le nombre de valeurs uniques dans une colonne.
    """
    unique_values = df[column].nunique()
    print(f"Nombre de valeurs uniques dans {column} : {unique_values}")
    print(f"Valeurs uniques : {df[column].unique()}")

# Fonction pour afficher un barplot ou un histogramme
def plot_bar_or_histogram(df, column):
    """
    Affiche un barplot ou un histogramme selon le type de la colonne.
    """
    plt.figure(figsize=(10, 6))
    if df[column].dtype == 'object' or df[column].dtype == 'category':
        sns.countplot(y=df[column], order=df[column].value_counts().index, palette='viridis')
    else:
        sns.histplot(df[column], kde=True, color='skyblue')
    
    plt.title(f'Graphique de {column}')
    plt.xlabel('Fréquence' if df[column].dtype == 'object' else column)
    plt.ylabel(column if df[column].dtype == 'object' else 'Fréquence')
    plt.show()

# Fonction pour afficher un violin plot
def plot_violin(df, column):
    """
    Affiche un violin plot pour une colonne numérique.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=df[column], inner="quartile", color="skyblue")
    plt.title(f'Violin Plot de {column}')
    plt.ylabel(column)
    plt.show()
