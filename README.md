# sentimentAnalysis

This project aims to train a machine learning model and analyse the sentiments of twitter users on the Pfizer COVID-19 vaccines. The dataset is obtained from Pfizer Vaccine data on Kaggle which has entries such as user ID, hashtags, text, tweet time, user location, number of retweets and likes, etc.

Data preprocessing is done by segregating only text data, then stemming to get the base form of the words. Polarity of tweets are found using TextBlob and the sentiments are displayed. Finally the tweets are vectorized and features are extracted.

Three models, viz., Logistic Regression, Support Vector Classification (SVC) and Multinomial Naive Bayes models are used to compare the model which produces the best accuracy. It is found that for the given dataset, SVC model gives the best accuracy of ~86% with hyperparameter tuning.

Confusion matrices and WordCloud are used to better visualise the results.
