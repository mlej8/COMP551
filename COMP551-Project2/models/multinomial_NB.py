from sklearn.naive_bayes import MultinomialNB
from predict import classify

# Making final predictions with Multinomial NB
multi_NB = MultinomialNB(alpha=0.25)

# Calling classify function from predict module
classify(multi_NB)