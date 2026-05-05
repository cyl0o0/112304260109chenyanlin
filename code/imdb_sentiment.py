import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load data
print("Loading data...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Text preprocessing function
def preprocess_text(text):
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Handle special cases (preserve negations and punctuation with emotional value)
    # Replace contractions to full form for better processing
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "wasn't": "was not",
        "aren't": "are not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "cannot",
        "couldn't": "could not",
        "shouldn't": "should not",
        "mightn't": "might not",
        "mustn't": "must not",
        "needn't": "need not",
        "daren't": "dare not",
        "shan't": "shall not",
        "ain't": "is not",
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "we'd": "we would",
        "they'd": "they would",
        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will"
    }
    
    for contraction, full in contractions.items():
        text = text.replace(contraction, full)
    
    # 4. Keep important punctuation (exclamation marks, question marks for sentiment)
    # Remove other punctuation but preserve words
    text = re.sub(r'[^a-zA-Z0-9\s!?]', ' ', text)
    
    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 6. Handle stopwords - remove common stopwords but keep negations
    stop_words = set(stopwords.words('english'))
    # Remove negations from stopwords list
    negations = {'not', 'no', 'never', 'nor', 'none', 'nobody', 'nowhere', 'nothing'}
    stop_words = stop_words - negations
    
    words = [word for word in words if word not in stop_words]
    
    # Join back to text
    return ' '.join(words)

print("\nPreprocessing text data...")
train_df['clean_review'] = train_df['review'].apply(preprocess_text)
test_df['clean_review'] = test_df['review'].apply(preprocess_text)

# Create TF-IDF features
print("\nCreating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Fit on training data and transform both datasets
X_train_tfidf = tfidf.fit_transform(train_df['clean_review'])
X_test_tfidf = tfidf.transform(test_df['clean_review'])

y_train = train_df['sentiment'].values

print(f"Training features shape: {X_train_tfidf.shape}")
print(f"Test features shape: {X_test_tfidf.shape}")

# Train model
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42
)

model.fit(X_train_split, y_train_split)

# Evaluate on validation set
y_val_pred = model.predict(X_val_split)
print("\nValidation Results:")
print(f"Accuracy: {accuracy_score(y_val_split, y_val_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val_split, y_val_pred))

# Train on full training data
print("\nTraining on full dataset...")
model.fit(X_train_tfidf, y_train)

# Predict on test data
print("\nGenerating predictions...")
test_predictions = model.predict(X_test_tfidf)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_predictions
})

submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False, quoting=3)
print(f"\nSubmission file created: {submission_file}")
print("\nFirst 10 predictions:")
print(submission.head(10))
