import pandas as pd
import numpy as np
import nltk
import requests
from sklearn import naive_bayes, feature_extraction, model_selection, preprocessing
import matplotlib.pyplot as plt
import time

start_time = time.time()

#--------------------------------------------------------------------------------------------------
# Importing Corona_NLP_train csv file using pandas:
df = pd.read_csv('data/Corona_NLP_train.csv', encoding= 'latin-1')

print('--------------------------------------------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------
# Question 1.1:
print('Question 1.1')

# Calculating the possible sentiment a tweet may have:
poss_sentiment = df['Sentiment'].unique()
print('Here are the possible sentiments a tweet may have: {}.\n'.format(poss_sentiment))


# Calculating second most popular sentiment in the tweets:
second_most_popular = df['Sentiment'].value_counts(ascending = False).index[1]
print('The second most popular sentiment in the tweets is: {}.\n'.format(second_most_popular))


# Calculating date with greatest number of extremely positive tweets:
extrem_positive_df = df[df.loc[:,'Sentiment'] == 'Extremely Positive']

# Calculating the number of tweets on the date with the greatest number of extremely positive tweets:
needed_date_freq = extrem_positive_df['TweetAt'].value_counts(ascending = False).max()

# Getting date value with the greatest number of extremely positive tweets:
needed_date = extrem_positive_df['TweetAt'].value_counts(ascending = False).idxmax()

# Printing out date, along with the number of Extremely Positive tweets on the day:
print('The date with the greatest number of Extremely Positive tweets was {}, with a tweet count of {}.'.format(
    needed_date, needed_date_freq))


# Converting messages to lowercase, replacing non-alphabetical characters with whitespaces, and ensuring that
# words of a message are separated with a single whitespace:
df['OriginalTweet'] = df['OriginalTweet'].str.lower().replace(r'[^a-zA-Z]', ' ', regex = True).replace('/  +/g', ' ')

print('--------------------------------------------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------
# Question 1.2:
print('Question 1.2')

# Tokenizing each tweet in the OriginalTweet column using NLTK library:
df['OriginalTweet'] = df['OriginalTweet'].apply(nltk.word_tokenize)


total_word_df = pd.DataFrame() # Calculating the total number of words in the OriginalTweet column:
total_word_df['TweetLength'] = df['OriginalTweet'].apply(len)
total_words = total_word_df['TweetLength'].sum()
print('The total number of words in the OriginalTweet column of the dataset is {}.'.format(total_words))


distinct_word_df = pd.DataFrame() # Calculating the total number of distinct words in the OriginalTweet column:
distinct_word_df['DistinctWords'] = df['OriginalTweet'].explode().drop_duplicates()
distinct_words = len(distinct_word_df['DistinctWords'])
print('The total number of distinct words in the OriginalTweet column of the dataset is {}.'.format(distinct_words))


list_of_freq_words = df['OriginalTweet'].to_list() # Calculating the ten most frequent words in the corpus:
list_of_freq_words = np.hstack(list_of_freq_words)
list_of_freq_words = pd.Series(list_of_freq_words).value_counts().index[0:10]
print('The ten most frequent words in the corpus are shown in the following list: {}.\n'.format(
    list(list_of_freq_words)))


# Removing stopwords from corpus and words which are <=2 characters long:
stopwords = requests.get('https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt').content.decode('utf-8').split('\n')
df['OriginalTweet'] = df['OriginalTweet'].apply(
    lambda x: [tweet for tweet in x if len(tweet) > 2 and tweet not in stopwords])
print('Following the modifications to the tweets column:')

modified_tweet_df = pd.DataFrame() # Calculating the total number of words in the now modified OriginalTweet column:
modified_tweet_df['TweetLength'] = df['OriginalTweet'].apply(len)
modified_total_words = modified_tweet_df['TweetLength'].sum()
print('The total number of words in the modified OriginalTweet column of the dataset is {}.'.format(
    modified_total_words))


modified_distinct_df = pd.DataFrame() # Calculating the total number of distinct words in the OriginalTweet column:
modified_distinct_df['DistinctWords'] = df['OriginalTweet'].explode().drop_duplicates()
modified_distinct_words = len(modified_distinct_df['DistinctWords'])
print('The total number of distinct words in the modified OriginalTweet column of the dataset is {}.'.format(
    modified_distinct_words))

list_of_freq_words_distinct = df['OriginalTweet'].to_list() # Calculating the ten most frequent words in the corpus:
list_of_freq_words_distinct = np.hstack(list_of_freq_words_distinct)
list_of_freq_words_distinct = pd.Series(list_of_freq_words_distinct).value_counts().index[0:10]
print('The ten most frequent words in the modified corpus are shown in the following list: {}.'.format(
    list(list_of_freq_words_distinct)))

print('--------------------------------------------------------------------------------------------------')
# #--------------------------------------------------------------------------------------------------
# # # Question 1.3:
print('Question 1.3')
#
# Plotting histogram of word frequencies:
word_corpus = df['OriginalTweet']

frequency_words = {} #dictionary to store frequency of each word in the corpus

for tweet in word_corpus: # for loop to go through each tweet
    for distinct_word in tweet: #for loop to go through each word of each tweet
        # adding the word to the dictionary if it does not exist already:
        if distinct_word not in frequency_words:
            frequency_words[distinct_word] = 1/len(df['OriginalTweet'])
        #if the word already exists in the dictionary then add to its frequency value:
        else:
            frequency_words[distinct_word] += 1/len(df['OriginalTweet'])

#sorting dictionary in ascending order of frequency values
lists = sorted(frequency_words.values())

plt.plot(range(1, len(frequency_words) + 1), lists) #plotting word frequency histogram
plt.yscale('log') #setting the y-axis to log-scale -> this makes for easier viewing

#tweaking various parameters of the graph
plt.title('Word Frequencies Histogram')
plt.xlabel('Words')
plt.ylabel('Frequency')

plt.savefig('outputs/word_frequencies_histogram.jpg') # Saving plot
print('See report for Word Frequencies Histogram.')

print('--------------------------------------------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------
# # Question 1.4:
print('Question 1.4')

naivebayes_df = pd.read_csv('data/Corona_NLP_train.csv', encoding= 'latin-1') # Re-importing Corona_NLP_train csv file

corpus_array = np.array(naivebayes_df['OriginalTweet']) #isolating OriginalTweet column for X variable

# Using feature_extraction CountVectorizer function to build sparse representation of term-document matrix.
vectorizer = feature_extraction.text.CountVectorizer()
X_vectorizer = vectorizer.fit_transform(corpus_array) # Fitting tweets to vectorizer

y = naivebayes_df['Sentiment'] #isolating Sentiment column as target variable

clf = naive_bayes.MultinomialNB().fit(X_vectorizer, y) #fitting vectorizer to Multinomial Naive Bayes Classifier.

# Calculating error rate:
print('The error rate of the Multinomial Naive Bayes classifier is: {}'.format(1 - (clf.score(X_vectorizer, y))))
print('--------------------------------------------------------------------------------------------------')

end_time = time.time()
print('The script took {} seconds to run.'.format(end_time - start_time))
print('--------------------------------------------------------------------------------------------------')