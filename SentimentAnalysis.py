#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r'C:\Users\rushi\OneDrive\Desktop\imp\project\SentimentAnalysis\vaccination_tweets.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


text_df = df.drop(['id', 'user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date', 'hashtags', 'source', 'retweets', 'favorites',
       'is_retweet'], axis = 1)
text_df.head()


# In[7]:


print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


# In[8]:


text_df.info()


# In[9]:


def data_processing(text):
    text = text.lower()  ##lower case all text
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)  ##remove URLs from text
    text = re.sub(r'\@w+|\#','',text)  ##remove hastags from text
    text = re.sub(r'[^\w\s]','',text)  ##remove punctuation from text
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]  ##remove stopwords from text
    return " ".join(filtered_text)


# In[10]:


import nltk
nltk.download('punkt')


# In[11]:


text_df.text = text_df['text'].apply(data_processing)


# In[12]:


text_df = text_df.drop_duplicates('text')  ##pre processing of data is complete


# In[13]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[14]:


text_df['text'] = text_df['text'].apply(lambda x: stemming(x))  ##stemming 


# In[15]:


text_df.head()


# In[16]:


text_df.info()  ##after removing duplicates and stemming, total entries is reduced


# In[17]:


def polarity(text):
    return TextBlob(text).sentiment.polarity


# In[18]:


text_df['polarity'] = text_df['text'].apply(polarity)  ##add polarity to text from -1 to 1


# In[19]:


text_df.head(10)  


# In[20]:


def sentiment(label):
    if label < 0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label > 0:
        return "Positive"


# In[21]:


text_df['sentiment'] = text_df['polarity'].apply(sentiment) ## adds extra column 'sentiment' according to polarity of statement


# In[22]:


text_df.head()


# In[23]:


fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = text_df)  ##visulaise positive, neutral and negative tweets


# In[24]:


fig = plt.figure(figsize=(7,7))
colors = ("blue", "gray", "green")
wp = {'linewidth':2, 'edgecolor':"white"}
tags = text_df['sentiment'].value_counts()
explode = (0.05, 0.05, 0.05)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')  


# In[25]:


pos_tweets = text_df[text_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
pos_tweets.head()  ##display only positive tweets


# In[26]:


text = ' '.join([word for word in pos_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show()  ##wordcloud of positive tweets


# In[27]:


neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending= False)
neg_tweets.head()  ##display only negative tweets


# In[28]:


text = ' '.join([word for word in neg_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative tweets', fontsize=19)
plt.show()  ##wordcloud of negative tweets


# In[29]:


neutral_tweets = text_df[text_df.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending= False)
neutral_tweets.head()  ##display only neutral tweets


# In[30]:


text = ' '.join([word for word in neutral_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()  ##wordcloud of neutral tweets


# In[31]:


vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])  ##vectorization of text data using bigram 


# In[32]:


feature_names = vect.get_feature_names_out()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features:\n {}".format(feature_names[:20]))


# In[33]:


X = text_df['text']
Y = text_df['sentiment']
X = vect.transform(X)  ##convert obtained vectors into X and Y for data training


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)##splitting into training and test data


# In[35]:


print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))


# In[36]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))  ##Training with Logistic Regression Model


# In[37]:


style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()   ##confusion matrix


# In[38]:


param_grid={'C':[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)


# In[39]:


print("Best parameters:", grid.best_params_)  ##best learning rate is found using GridSearchCV


# In[40]:


y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))  ##accuracy improved with hyperparameter tuning


# In[41]:


SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)


# In[42]:


svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("test accuracy: {:.2f}%".format(svc_acc*100))  ##training with Support Vector Classification Model


# In[43]:


grid = {
    'C':[0.01, 0.1, 1, 10],
    'kernel':["linear","poly","rbf","sigmoid"],
    'degree':[1,3,5,7],
    'gamma':[0.01,1]
}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)  ##Hyperparameter tuning


# In[44]:


print("Best parameter:", grid.best_params_)


# In[45]:


y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))  ##accuracy very sligthly improved 


# In[46]:


style.use('classic')
cm = confusion_matrix(y_test, svc_pred, labels=SVCmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=SVCmodel.classes_)
disp.plot()   ##confusion matrix


# In[47]:


mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb_pred = mnb.predict(x_test)
mnb_acc = accuracy_score(mnb_pred, y_test)
print("Test accuracy: {:.2f}%".format(mnb_acc*100))  ##training with Multinomial Naive Bayes Model,  low accuracy


# In[48]:


style.use('classic')
cm = confusion_matrix(y_test, mnb_pred, labels=mnb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=mnb.classes_)
disp.plot()   ##confusion matrix


# In[ ]:




