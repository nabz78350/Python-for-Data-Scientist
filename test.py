# %%
# https://medium.com/swlh/how-to-scrape-tweets-by-location-in-python-using-snscrape-8c870fa6ec25
# https://medium.com/dataseries/how-to-scrape-millions-of-tweets-using-snscrape-195ee3594721
# https://larevueia.fr/nlp-avec-python-analyse-de-sentiments-sur-twitter/
# https://github.com/JosephAssaker/Twitter-Sentiment-Analysis-Classical-Approach-VS-Deep-Learning/blob/master/Twitter%20Sentiment%20Analysis%20-%20Classical%20Approach%20VS%20Deep%20Learning.ipynb


# %%
pip
install
git + https: // github.com / JustAnotherArchivist / snscrape.git

# %%
pip
install
snscrape

# %%
import snscrape.modules.twitter as sntwitter
import pandas as pd

# %%
tweets_list = []
start_date = '2021-10-01'
end_date = '2021-12-19'

# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
        f'Bitcoin since:{start_date} until:{end_date} min_faves:200 min_retweets:20').get_items()):
    if i > 50000:
        break
    tweets_list.append(
        [tweet.date, tweet.content, tweet.user.username, tweet.hashtags, tweet.likeCount, tweet.retweetCount,
         tweet.user.followersCount, tweet.user.verified])

# Creating a dataframe from the tweets list above
tweets_df = pd.DataFrame(tweets_list,
                         columns=['tweet_date', 'tweet_text', 'tweet_user', 'tweet_hashtags', 'tweet_likes',
                                  'tweet_retweets', 'tweet_user_followers', 'user_verified'])

# %%
tweets_df = pd.DataFrame(tweets_list,
                         columns=['tweet_date', 'tweet_text', 'tweet_user', 'tweet_hashtags', 'tweet_likes',
                                  'tweet_retweets', 'tweet_user_followers', 'user_verified'])
tweets_df['tweet_date'] = tweets_df['tweet_date'].dt.round('H')
tweets_df

# %%

tweets_df['tweet_date'] = tweets_df['tweet_date'].dt.tz_localize(None)
tweets_df.to_excel('tweets_2021_hourly_brut.xlsx')

# %%
tweets_df['tweet_date']

# %%
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# %%
tweets_df = pd.read_excel('tweets_2021_brut.xlsx')
tweets_df.drop('Unnamed: 0', axis=1, inplace=True)
tweets_df

# %%
# cleaning tweets corpus with regext
import re


def nlp_pipeline(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
    text = re.sub(r"(\s\-\s|-$)", "", text)
    text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
    text = re.sub(r"\&\S*\s", "", text)
    text = re.sub(r"\&", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\#", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\£", "", text)
    text = re.sub(r"\%", "", text)
    text = re.sub(r"\:", "", text)
    text = re.sub(r"\@", "", text)
    text = re.sub(r"\-", "", text)

    return text


# %%
import pandas as pd
from textblob import TextBlob


def text_processing(tweet):
    # Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)

    new_tweet = form_sentence(tweet)

    # Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess

    no_punc_tweet = no_user_alpha(new_tweet)

    # Normalizing the words in tweets
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word, 'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet

    return normalization(no_punc_tweet)


# %%
# KAGGLE Dataset with tweets and posititivity 0 if not 4 if positive

# Reading the dataset with no columns titles and with latin encoding
df_raw = pd.read_csv('training.csv', encoding="ISO-8859-1", header=None)

# As the data has no column titles, we will add our own
df_raw.columns = ["label", "time", "date", "query", "username", "text"]

# Show the first 5 rows of the dataframe.
# You can specify the number of rows to be shown as follows: df_raw.head(10)
df_raw.head()

# %%
# Ommiting every column except for the text and the label, as we won't need any of the other information
df = df_raw[['label', 'text']]
df_pos = df[df['label'] == 4]
df_neg = df[df['label'] == 0]
print(len(df_pos), len(df_neg))
df_pos = df_pos.iloc[:int(len(df_pos) / 4)]
df_neg = df_neg.iloc[:int(len(df_neg) / 4)]
print(len(df_pos), len(df_neg))
df = pd.concat([df_pos, df_neg])
df

# %%
from nltk.tokenize import TweetTokenizer

# The reduce_len parameter will allow a maximum of 3 consecutive repeating characters, while trimming the rest
# For example, it will tranform the word: 'Helloooooooooo' to: 'Hellooo'
tk = TweetTokenizer(reduce_len=True)

data = []

# Separating our features (text) and our labels into two lists to smoothen our work
X = df['text'].tolist()
Y = df['label'].tolist()

# Building our data list, that is a list of tuples, where each tuple is a pair of the tokenized text
# and its corresponding label
for x, y in zip(X, Y):
    if y == 4:
        data.append((tk.tokenize(x), 1))
    else:
        data.append((tk.tokenize(x), 0))

# Printing the CPU time and the first 5 elements of our 'data' list

data[:5]
data

# %%
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer


# Previewing the pos_tag() output

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        # First, we will convert the pos_tag output tags to a tag format that the WordNetLemmatizer can interpret
        # In general, if a tag starts with NN, the word is a noun and if it stars with VB, the word is a verb.
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


# Previewing the WordNetLemmatizer() output
print(lemmatize_sentence(data[0][0]))

# %%
import re, string

# Stopwords are frequently-used words (such as “the”, “a”, “an”, “in”) that do not hold any meaning useful to extract sentiment.
# If it's your first time ever using nltk, you can download nltk's stopwords using: nltk.download('stopwords')
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')


# A custom function defined in order to fine-tune the cleaning of the input text. This function is highly dependent on each usecase.
# Note: Only include misspelling or abbreviations of commonly used words.
#       Including many minimally present cases would negatively impact the performance.
def cleaned(token):
    if token == 'u':
        return 'you'
    if token == 'r':
        return 'are'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == 'secs':
        return 'seconds'
    if token == 'pls' or token == 'plz':
        return 'please'
    if token == '2morow':
        return 'tomorrow'
    if token == '2day':
        return 'today'
    if token == '4got' or token == '4gotten':
        return 'forget'
    return token


# This function will be our all-in-one noise removal function
def remove_noise(tweet_tokens):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        # Eliminating the token if it is a link
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        # Eliminating the token if it is a mention
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        cleaned_token = cleaned(token.lower())

        # Eliminating the token if its length is less than 3, if it is a punctuation or if it is a stopword
        if cleaned_token not in string.punctuation and len(cleaned_token) > 2 and cleaned_token not in STOP_WORDS:
            cleaned_tokens.append(cleaned_token)

    return cleaned_tokens


# Prevewing the remove_noise() output
print(remove_noise(data[0][0]))

# %%
from time import time

start_time = time()

cleaned_tokens_list = []

# Removing noise from all the data

for tokens, label in data:
    cleaned_tokens_list.append((remove_noise(tokens), label))

print('Removed Noise, CPU Time:', time() - start_time)
start_time = time()

# %%
# Transforming the data to fit the input structure of the Naive Bayesian classifier
# As the Naive Bayesian classifier accepts inputs in a dict-like structure,
# we have to define a function that transforms our data into the required input structure
final_data = []


def list_to_dict(cleaned_tokens):
    return dict([token, True] for token in cleaned_tokens)


for tokens, label in cleaned_tokens_list:
    final_data.append((list_to_dict(tokens), label))

print('Data Prepared for model, CPU Time:', time() - start_time)

# Previewing our final (tokenized, cleaned and lemmatized) data list
final_data[:5]

# %%
# As our data is currently ordered by label, we have to shuffle it before splitting it
# .Random(140) randomizes our data with seed = 140. This guarantees the same shuffling for every execution of our code
# Feel free to alter this value or even omit it to have different outputs for each code execution
import random

random.Random(140).shuffle(final_data)

# Here we decided to split our data as 90% train data and 10% test data
# Once again, feel free to alter this number and test the model accuracy
trim_index = int(len(final_data) * 0.9)

train_data = final_data[:trim_index]
test_data = final_data[trim_index:]

# %%
start_time = time()

from nltk import classify
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_data)

# Output the model accuracy on the train and test data
print('Accuracy on train data:', classify.accuracy(classifier, train_data))
print('Accuracy on test data:', classify.accuracy(classifier, test_data))

# Output the words that provide the most information about the sentiment of a tweet.
# These are words that are heavily present in one sentiment group and very rarely present in the other group.
print(classifier.show_most_informative_features(20))

print('\nCPU Time:', time() - start_time)

# %%
import pandas as pd
import numpy as np

tweets_df = pd.read_excel('tweets_2019_2020_2021_NLP_analysis.xlsx')
tweets_df['tweet_date'] = tweets_df['Unnamed: 0']

# %%
tweets_df.drop('Unnamed: 0', axis=1, inplace=True)
tweets_df

# %%
tweets_df['tweet_date'] = pd.to_datetime(tweets_df['tweet_date']).dt.date
tweets_df

# %%
pip
install
wordnet

# %%
import nltk

nltk.download('words')
from nltk.corpus import words
import pandas as pd

# %%
from langdetect import detect
import pandas as pd

# %%
test = tweets_df.copy()


def detect_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False


test = test[test['tweet_text'].apply(detect_en)]
test

# %%
tweets_df = test.copy()

# %%
import numpy as np

tweets_df['positivity'] = np.nan
for i in range(len(tweets_df)):
    tweet = tweets_df['tweet_text'][i]
    tweet = str(tweet)
    # print(tweet)
    custom_tokens = remove_noise(tk.tokenize(tweet))
    tweets_df['positivity'][i] = classifier.classify(dict([token, True] for token in custom_tokens))

# %%
tweets_df['user_verified'] = tweets_df['user_verified'].replace(True, 1.10)
tweets_df['user_verified'] = tweets_df['user_verified'].replace(False, 0.9)

# %%
tweets_df['user_verified'].mean()

# %%

# tweets_df['positivity']=tweets_df['positivity'].replace(-1,0)
# tweets_df.to_excel('tweets_2021_NLP_analysis.xlsx')
tweets_df['followers_score'] = tweets_df['positivity'] * tweets_df['tweet_user_followers']
tweets_df['like_score'] = tweets_df['positivity'] * tweets_df['tweet_likes']
# tweets_df.to_excel('tweets_2018_2019-2020_2021_NLP_analysis.xlsx')


tweets_df['score'] = tweets_df['positivity'] * tweets_df['tweet_likes'] * tweets_df['user_verified']

# %%
tweets_df['Date'] = tweets_df['tweet_date']
tweets_df.set_index('Date', inplace=True)
tweets_df.index.names = ['Date']
tweets_df.index = pd.to_datetime(tweets_df.index)

# %%
tweets_df

# %%
bitcoin_price = pd.read_csv('BTC-USD_2019_2020_2021.csv')
bitcoin_price.set_index('Date', inplace=True)
bitcoin_price.sort_values('Date', ascending=False, inplace=True)
bitcoin_price.index = pd.to_datetime(bitcoin_price.index).tz_localize('Etc/UCT')
bitcoin_price['pre_close'] = bitcoin_price['Close'].shift(-1)
bitcoin_price['next_open'] = bitcoin_price['Open'].shift(2)
bitcoin_price['diff'] = bitcoin_price['Close'].diff()
bitcoin_price

# %%
data = tweets_df.groupby(tweets_df.index)[['score', 'like_score', 'followers_score']].mean()
data.sort_index(ascending=False)
data.index = pd.to_datetime(data.index).tz_localize('Etc/UCT')

# %%
data

# %%
## final data for model

data = pd.merge(data, bitcoin_price, left_index=True, right_index=True)
data[['Close', 'Open', 'next_open']]

# %%
# data['score']=data['score'].expanding().mean()
data

# %%
X = data[['score', 'Close']].copy()
Y = pd.DataFrame({'Next_Open': data['next_open']})
Y.dropna(inplace=True)
Y['Next_Open'] = Y['Next_Open'].astype(int)
X.drop(X.tail(1).index, inplace=True)  # if shift >1

# %%
len(Y)

# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)  # 70% training and 30% test

# %%
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

result = y_test.copy()
result['pred'] = y_pred
result.sort_index(inplace=True)
# result.to_excel('result_decision_tree_2021.xlsx')
result

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)
result = y_test.copy()
result['pred'] = y_pred
result.sort_index(inplace=True)

lm.coef_

r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))

# %%
lm.coef_

# %%
r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

lm_2 = LinearRegression()
lm_2.fit(X_train, y_train)

y_pred = lm_2.predict(X_test)
result = y_test.copy()
result['pred'] = y_pred
result.sort_index(inplace=True)

lm_2.coef_

r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))

# %%
lm_2.coef_

# %%
import xgboost as xgb

# %%
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# %%
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))



