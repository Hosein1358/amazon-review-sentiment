#import sys 
#print(sys.argv)

import argparse   # for reading arguments in a regular way
parser = argparse.ArgumentParser()
parser.add_argument("train_data")
parser.add_argument("test_data")
parser.add_argument("model", choices=["lr","svm","rf","gb","nb","lstm"], default = "lr", help = "choose a model")
parser.add_argument("output_name", default = "results.csv")
args = parser.parse_args()

args_dict = vars(args)
print(args_dict)

# import libraries
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords, wordnet
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

# Downloading necessary NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')


"""

1- Loading data set & Reading data

"""

# Load train and test datasets
column_names = ['rate', 'title', 'comment']
# Rating: either 1: negative or 2: positive.
train_df = pd.read_csv(args_dict["train_data"], header=None, names=column_names)
test_df = pd.read_csv(args_dict["test_data"], header=None, names=column_names)

# Reduce size to 10% of original dataset
train_df = train_df.sample(frac=0.1, random_state=1)
test_df = test_df.sample(frac=0.1, random_state=1)

"""

2- Defining the Cleaning Functions

"""
"""

2-1 Normalization

"""

import time
start_time = time.time()
print("Normalization")

def normalize_case_folding(train_df, test_df):
 # Normalize Case Folding - Converting to Lowercase
    train_df['title'] = train_df['title'].str.lower()
    train_df['comment'] = train_df['comment'].str.lower()
    test_df['title'] = test_df['title'].str.lower()
    test_df['comment'] = test_df['comment'].str.lower()
    
    return train_df, test_df

normalize_case_folding(train_df, test_df)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")


"""

2-2 Removing Punctuation

"""

print("Removing Punctuation")
start_time = time.time()

def remove_punctuation(train_df, test_df):
    train_df['title'] = train_df['title'].str.replace('[^\w\s]', '', regex=True)
    train_df['comment'] = train_df['comment'].str.replace('[^\w\s]', '', regex=True)
    test_df['title'] = test_df['title'].str.replace('[^\w\s]', '', regex=True)
    test_df['comment'] = test_df['comment'].str.replace('[^\w\s]', '', regex=True)
    
    return train_df, test_df

remove_punctuation(train_df, test_df)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")



"""

2-3 Removing Numbers

"""


print("Removing Numbers")
start_time = time.time()

def remove_numbers(train_df, test_df):
    train_df['title'] = train_df['title'].str.replace('\d', '', regex=True)
    train_df['comment'] = train_df['comment'].str.replace('\d', '', regex=True)
    test_df['title'] = test_df['title'].str.replace('\d', '', regex=True)
    test_df['comment'] = test_df['comment'].str.replace('\d', '', regex=True)
    
    return train_df, test_df

remove_numbers(train_df, test_df)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")


"""

2-4 Removing Missing & NAN Values

"""
print("Removing Missing & NAN Values")
start_time = time.time()

def remove_missNan(train_df, test_df):
    train_df['title'] = train_df['title'].fillna('').str.lower()
    train_df['comment'] = train_df['comment'].fillna('').str.lower()
    test_df['title'] = test_df['title'].fillna('').str.lower()
    test_df['comment'] = test_df['comment'].fillna('').str.lower()
    
    return train_df, test_df

remove_missNan(train_df, test_df)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")



"""

2-5 Removing URLs

"""

print("Removing URLs")
start_time = time.time()

def remove_urls(train_df, test_df):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    train_df['title'] = train_df['title'].apply(lambda x: url_pattern.sub(r'', x))
    train_df['comment'] = train_df['comment'].apply(lambda x: url_pattern.sub(r'', x))
    test_df['title'] = test_df['title'].apply(lambda x: url_pattern.sub(r'', x))
    test_df['comment'] = test_df['comment'].apply(lambda x: url_pattern.sub(r'', x))
    
    return train_df, test_df

remove_urls(train_df, test_df)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")

"""

2-6 Removing Hashtags

"""

def remove_hashtags(train_df, test_df):
    hashtag_pattern = re.compile(r'#\w+')
    
    train_df['title'] = train_df['title'].astype(str).apply(lambda x: hashtag_pattern.sub(r'', x))
    train_df['comment'] = train_df['comment'].astype(str).apply(lambda x: hashtag_pattern.sub(r'', x))
    test_df['title'] = test_df['title'].astype(str).apply(lambda x: hashtag_pattern.sub(r'', x))
    test_df['comment'] = test_df['comment'].astype(str).apply(lambda x: hashtag_pattern.sub(r'', x))
    
    return train_df, test_df

remove_hashtags(train_df, test_df)


"""

2-7 Removing Mentions

"""

def remove_mentions(train_df, test_df):
    mention_pattern = re.compile(r'@\w+')
    
    train_df['title'] = train_df['title'].apply(lambda x: mention_pattern.sub(r'', x))
    train_df['comment'] = train_df['comment'].apply(lambda x: mention_pattern.sub(r'', x))
    test_df['title'] = test_df['title'].apply(lambda x: mention_pattern.sub(r'', x))
    test_df['comment'] = test_df['comment'].apply(lambda x: mention_pattern.sub(r'', x))
    
    return train_df, test_df

remove_mentions(train_df, test_df)

"""

2-8 Removing Noise

"""
start_time = time.time()

def remove_noise(train_df, test_df):
    # Define patterns for noise removal
    noise_patterns = [
        r'http\S+',        # Remove URLs
        r'@\w+',           # Remove mentions (handles starting with '@')
        r'[^\w\s]',        # Remove punctuation and special characters
        r'\d'              # Remove digits
    ]
    
    # Combine patterns into a single regular expression
    noise_pattern = re.compile('|'.join(noise_patterns))
    
    # Apply noise removal to 'title' and 'comment' columns in train_df
    train_df['title'] = train_df['title'].apply(lambda x: noise_pattern.sub(r'', x))
    train_df['comment'] = train_df['comment'].apply(lambda x: noise_pattern.sub(r'', x))
    
    # Apply noise removal to 'title' and 'comment' columns in test_df
    test_df['title'] = test_df['title'].apply(lambda x: noise_pattern.sub(r'', x))
    test_df['comment'] = test_df['comment'].apply(lambda x: noise_pattern.sub(r'', x))
    
    return train_df, test_df

remove_noise(train_df, test_df)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")




"""

2-9 Lemmatization

"""





start_time = time.time()
"""
def lemmatize_text(train_df, test_df):
    # Function to lemmatize a single sentence using TextBlob
    def lemmatize_sentence(text):
        blob = TextBlob(text)
        lemmatized_words = [word.lemmatize() for word in blob.words]
        return ' '.join(lemmatized_words)
    
    # Apply lemmatization function to 'title' and 'comment' columns in train_df
    train_df['title'] = train_df['title'].apply(lemmatize_sentence)
    train_df['comment'] = train_df['comment'].apply(lemmatize_sentence)
    
    # Apply lemmatization function to 'title' and 'comment' columns in test_df
    test_df['title'] = test_df['title'].apply(lemmatize_sentence)
    test_df['comment'] = test_df['comment'].apply(lemmatize_sentence)
    
    return train_df, test_df

lemmatize_text(train_df, test_df)
"""
end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")




"""

2-10 Removing Stopwords

"""






start_time = time.time()

def remove_stopwords(train_df, test_df):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['wa'])
    
    # Function to remove stopwords from text
    def remove_stopwords_sentence(text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    # Apply remove_stopwords_sentence function to 'title' and 'comment' columns in train_df
    train_df['title'] = train_df['title'].apply(remove_stopwords_sentence)
    train_df['comment'] = train_df['comment'].apply(remove_stopwords_sentence)
    
    # Apply remove_stopwords_sentence function to 'title' and 'comment' columns in test_df
    test_df['title'] = test_df['title'].apply(remove_stopwords_sentence)
    test_df['comment'] = test_df['comment'].apply(remove_stopwords_sentence)
    
    return train_df, test_df

remove_stopwords(train_df, test_df)

# Display the first few rows of the dataframe to verify the changes
print(train_df.head())
print(test_df.head())

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")




"""

2-1 Normalization

"""










start_time = time.time()

def combine_text_and_count_frequency(train_df, test_df):
    """
    Combine all text from 'title' and 'comment' columns into a single list of words,
    count the frequency of each word, and store the word frequencies in a DataFrame.
    
    Parameters:
    train_df (pd.DataFrame): The training dataset.
    test_df (pd.DataFrame): The testing dataset.
    
    Returns:
    pd.DataFrame: A DataFrame with word frequencies sorted by frequency in descending order.
    """
    all_words = []

    def collect_words(df):
        for column in ['title', 'comment']:
            for text in df[column]:
                if pd.isna(text):
                    continue  # Skip NaN values
                words = text.split()  # Split text into words
                all_words.extend(words)  # Add to the list of all words

    collect_words(train_df)
    collect_words(test_df)

    # Count the frequency of each word
    word_freq = Counter(all_words)

    # Store word frequencies in a DataFrame for better visualization
    word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)

    return word_freq_df

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")











start_time = time.time()

stop_clean = combine_text_and_count_frequency(train_df, test_df)
# Print the resulting DataFrame
print(stop_clean)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time: {elapsed_time:.2f} seconds")




"""

2-1 Normalization

"""





# After all preprocessing steps, get word frequencies
word_freq_df = combine_text_and_count_frequency(train_df, test_df)

# Print the resulting DataFrame
print(word_freq_df)










# Specify the file path where you want to save the CSV file
file_path = f'{args_dict["output_name"]}_word_frequencies.csv'

# Save the DataFrame to a CSV file
word_freq_df.to_csv(file_path, index=False)





#drops = word_freq_df[word_freq_df['frequency'] <= 2]
word_freq_df = word_freq_df[word_freq_df['frequency'] > 2]
#len(word_freq_df)






word_vis = word_freq_df[word_freq_df['frequency'] > 30000]
plt.figure(figsize=(18, 10))
word_vis.plot.bar(x="word", y="frequency", ax=plt.gca())
plt.savefig(f"{args_dict["output_name"]}_word_frequency_graph.png")





# Convert the DataFrame to a dictionary
word_freq_dict = dict(zip(word_vis['word'], word_vis['frequency']))

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

# Display the word cloud
plt.figure(figsize=(12, 8))  # Width: 12 inches, Height: 8 inches
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.savefig(f"{args_dict["output_name"]}_word_frequency_pic.png")



if args_dict["model"] == "lr":
    start_time = time.time()

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature for training data
    train_df['combined_text'] = train_df['title'].astype(str) + " " + train_df['comment'].astype(str)
    
    # Input: Fit and transform on training data
    X_train = vectorizer.fit_transform(train_df['combined_text'])
    
    # Target
    y_train = train_df['rate']
    
    # Train a Logistic Regression model
    tf_model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
    tf_model.fit(X_train, y_train)
    
    # Evaluate the model on the training set
    y_train_pred = tf_model.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")

    # Test Model
    start_time = time.time()
    
    # Combine 'comment' and 'title' into a single feature for test data
    test_df['combined_text'] = test_df['title'].astype(str) + " " + test_df['comment'].astype(str)
    
    # Input: Transform the test data using the already fitted vectorizer
    X_test = vectorizer.transform(test_df['combined_text'])
    
    # Target
    y_test = test_df['rate']
    
    # Prediction of the model on the test set
    y_test_pred = tf_model.predict(X_test)
    
    # Print validation metrics
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], annot_kws={"size": 16})
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    

    
    
if args_dict["model"] == "svm":   
    # Train Model
    start_time = time.time()
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature for training data
    train_df['combined_text'] = train_df['title'].astype(str) + " " + train_df['comment'].astype(str)
    
    # Input: Fit and transform on training data
    X_train = vectorizer.fit_transform(train_df['combined_text'])
    
    # Target
    y_train = train_df['rate']
    
    # Train an SVM model
    svm_model = SVC(max_iter=1000)  # Increase max_iter if needed
    svm_model.fit(X_train, y_train)
    
    # Evaluate the model on the training set
    y_train_pred = svm_model.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Test Model
    start_time = time.time()
    
    # Combine 'comment' and 'title' into a single feature for test data
    test_df['combined_text'] = test_df['title'].astype(str) + " " + test_df['comment'].astype(str)
    
    # Input: Transform the test data using the already fitted vectorizer
    X_test = vectorizer.transform(test_df['combined_text'])
    
    # Target
    y_test = test_df['rate']
    
    # Prediction of the model on the test set
    y_test_pred = svm_model.predict(X_test)
    
    # Print validation metrics
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train), annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
        
    
    
    
    
    
if args_dict["model"] == "rf":   
    # Train Model
    start_time = time.time()
    
    # Vectorize the text using TF-IDF
    vectorizerRF = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature
    train_df['combined_text'] = train_df['title'].astype(str) + " " + train_df['comment'].astype(str)
    
    # Input
    X_train = vectorizerRF.fit_transform(train_df['combined_text'])
    
    # Target
    y_train = train_df['rate']
    
    # Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model on the training set
    y_train_pred = rf_model.predict(X_train)
    print("Validation Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Test Model
    start_time = time.time()
    
    # Vectorize the text using TF-IDF
    #vectorizer = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature
    test_df['combined_text'] = test_df['title'].astype(str) + " " + test_df['comment'].astype(str)
    
    # Input
    X_test = vectorizerRF.transform(test_df['combined_text'])
    
    # Target
    y_test = test_df['rate']
    
    
    # Prediction of the model on the test set
    y_test_pred = rf_model.predict(X_test)
    print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train), annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
        
        
    
    
    
if args_dict["model"] == "nb":       
    # Train Model
    start_time = time.time()
    
    # Vectorize the text using TF-IDF
    vectorizerNB = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature
    train_df['combined_text'] = train_df['title'].astype(str) + " " + train_df['comment'].astype(str)
    
    # Input
    X_train = vectorizerNB.fit_transform(train_df['combined_text'])
    
    # Target
    y_train = train_df['rate']
    
    # Train a Naive Bayes classifier
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    # Evaluate the model on the training set
    y_train_pred = nb_model.predict(X_train)
    print("Validation Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Test Model
    start_time = time.time()
    
    # Vectorize the text using TF-IDF
    # vectorizer = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature
    test_df['combined_text'] = test_df['title'].astype(str) + " " + test_df['comment'].astype(str)
    
    # Input
    X_test = vectorizerNB.transform(test_df['combined_text'])
    
    # Target
    y_test = test_df['rate']
    
    
    # Prediction of the model on the test set
    y_test_pred = nb_model.predict(X_test)
    print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train), annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    
    
    
    
if args_dict["model"] == "gb":     
    # Train Model
    start_time = time.time()
    
    # Vectorize the text using TF-IDF
    vectorizerGB = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature
    train_df['combined_text'] = train_df['title'].astype(str) + " " + train_df['comment'].astype(str)
    
    # Input
    X_train = vectorizerGB.fit_transform(train_df['combined_text'])
    
    # Target
    y_train = train_df['rate']
    
    # Train a Gradient Boosting classifier
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Evaluate the model on the training set
    y_train_pred = gb_model.predict(X_train)
    print("Validation Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Test Model
    start_time = time.time()
    
    # Vectorize the text using TF-IDF
    # vectorizer = TfidfVectorizer(max_features=2000)
    
    # Combine 'comment' and 'title' into a single feature
    test_df['combined_text'] = test_df['title'].astype(str) + " " + test_df['comment'].astype(str)
    
    # Input
    X_test = vectorizerGB.transform(test_df['combined_text'])
    
    # Target
    y_test = test_df['rate']
    
    
    # Prediction of the model on the test set
    y_test_pred = gb_model.predict(X_test)
    print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train), annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    
    
    
    
if args_dict["model"] == "lstm":    
    # Train Model
    start_time = time.time()
    
    # Combine 'comment' and 'title' into a single feature
    train_df['combined_text'] = train_df['title'].astype(str) + " " + train_df['comment'].astype(str)
    
    # Input
    X_train = train_df['combined_text']
    
    # Tokenize the text
    tokenizerLSTM = Tokenizer(num_words=2000)
    tokenizerLSTM.fit_on_texts(X_train)
    X_train = tokenizerLSTM.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=200)
    
    # Target
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['rate'])
    
    # Build LSTM Model
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=2000, output_dim=32, input_length=200))
    lstm_model.add(SpatialDropout1D(0.2))
    lstm_model.add(LSTM(50, dropout=0.1, recurrent_dropout=0.1))
    lstm_model.add(Dense(1, activation='sigmoid'))
    
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    lstm_model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2, verbose=2)
    
    # Evaluate the model on the training set
    y_train_pred_prob = lstm_model.predict(X_train)
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)
    
    print("Validation Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Classification Report:\n", classification_report(y_train, y_train_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    # Test Model
    start_time = time.time()
    
    # Combine 'comment' and 'title' into a single feature
    test_df['combined_text'] = test_df['title'].astype(str) + " " + test_df['comment'].astype(str)
    
    # Input
    X_test = test_df['combined_text']
    
    # Use the same tokenizer that was fitted on the training data
    X_test = tokenizerLSTM.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=200)
    
    # Target
    y_test = label_encoder.transform(test_df['rate'])  # Use the fitted label encoder
    
    # Prediction of the model on the test set
    y_test_pred_prob = lstm_model.predict(X_test)
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions
    
    print("Validation Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds")
    
    
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train), annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
plt.savefig(f"{args_dict["output_name"]}_confusion_mat_{args_dict["model"]}")














