# Common Data Science Library
import pandas as pd
import numpy as np

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize 
from sklearn.preprocessing import LabelEncoder

# nltk library
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Common python libraries
import re 
import string

""" Class containing preprocessing function for reddit comments """
class Preprocessor: 

    def __init__(self, normalizer):
        self.label_encoder = LabelEncoder()
        self.tf_idf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words=stopwords.words('english').append(["nt", "get", "like", "would","peopl", "one", "think", "time", "becaus"]), smooth_idf=True, norm="l2",lowercase=True, max_features=30000, use_idf=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode',  analyzer = "word")
        if normalizer == "stemmer": 
            self.normalizer = SnowballStemmer("english")
        elif normalizer == "lemmatizer":
            self.normalizer = WordNetLemmatizer()
        else:
            raise Exception("Normalizer must be \"stemmer\" or \"lemmatizer\".")        
            
    def preprocess_reddit_train(self):
        """ Stores a cleaned up version of the dataset on the current directory """
        # Read dataset
        df = pd.read_csv("data/reddit_train.csv")
        # Apply stemming function 
        df["cleaned"] = df["comments"].apply(self.clean_text)
        # Transform each subreddit into an unique integer
        df["label"] = self.label_encoder.fit_transform(df["subreddits"])
        # Save cleaned dataset
        df.to_csv("data/preprocessed_reddit_train_" + type(self.normalizer).__name__ + ".csv",index=False)       
        # TODO: Implement Regularization (i.e. PCA, SVD, L1, L2...?) 

    def clean_text(self, sentence): 
        # Put all words to lower case 
        sentence = sentence.lower()    
        # Tokenize words   
        word_tokens = word_tokenize(sentence)
        # Remove punctuation 
        word_tokens = [_ for _ in word_tokens if _ not in string.punctuation]
        # Remove non-alphabetical char 
        word_tokens = [re.sub(pattern="[^a-zA-Z0-9\s]", repl= "", string = _ ) for _ in word_tokens]
        # Remove empty strings 
        word_tokens = [_ for _ in word_tokens if _]
        # Stem words
        processed_sentence = self.normalize(" ".join(word_tokens))
        # TODO: Remove links?
        return processed_sentence.strip()
    
    def normalize(self, sentence):
        normalized_str = []
        word_tokens = word_tokenize(sentence)
        if type(self.normalizer).__name__ == "SnowballStemmer": 
            for i in word_tokens:
                normalized_str.append(self.normalizer.stem(i))
        elif type(self.normalizer).__name__ == "WordNetLemmatizer":
            for i in word_tokens:
                normalized_str.append(self.normalizer.lemmatize(i))        
        else:
            raise Exception("Normalizer must be \"stemmer\" or \"lemmatizer\".")

        return " ".join(normalized_str)
    
    def preprocess_reddit_test(self):
        """ Returns a cleaned up version of the test dataset """
        # Read dataset
        df = pd.read_csv("data/reddit_test.csv")
        # Apply stemming function 
        df["cleaned"] = df["comments"].apply(self.clean_text)
        # Store cleaned test set in 
        df.to_csv("data/preprocessed_reddit_test_" + type(self.normalizer).__name__  + ".csv", index=False)      