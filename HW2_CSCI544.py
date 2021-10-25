#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
 


# In[2]:


#! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz


# ## Read Data

# In[3]:


#loading dataset
# data = pd.read_table('amazon_reviews_us_Kitchen_v1_00.tsv')

df = pd.read_csv('amazon_reviews_us_Kitchen_v1_00.tsv', sep='\t', error_bad_lines=False)


# ## Keep Reviews and Ratings

# In[7]:


data = df[["review_id","review_body", "star_rating"]]
# print(data.info())
data=data.dropna(subset=['review_body'])
# print(data.info())


# # Labelling Reviews:
# ## The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'

# In[8]:


discarded_count = len(data[data['star_rating'] == 3.0])
# print("Neutral Reviews: ", discarded_count)


# In[9]:


data.drop(data[data['star_rating'] == 3.0].index, inplace = True)
# print(data.info())

data['label'] = np.where(data['star_rating']>=4.0, 1, 0)
# data.head(10)


#  ## We select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews.
# 
# 

# In[10]:


pos_data = data[data['label']==1]
neg_data = data[data['label']==0]

# print("Positive Reviews: ", len(pos_data))
# print("Negative Reviews: ", len(neg_data))

pos = pos_data.sample(n=100000, random_state=1)
neg = neg_data.sample(n=100000, random_state=1)

data_both = pd.concat([pos, neg], ignore_index=True)
data_both = data_both.sample(frac=1).reset_index(drop=True)
# data_both.head(10)


# In[12]:


from statistics import mean

data_both['review_body'] = data_both['review_body'].astype(str)

avg_len1 = mean([len(s) for s in data_both["review_body"]])

#before cleaing and pre-processing
# print("BEFORE CLEANING")
# print(*[rev+", length: "+str(len(rev))+"\n" for rev in data_both["review_body"].head(3)])
# print("Average review length before processing: "+ str(avg_len1))


# # Data Cleaning

# In[13]:


#regexs for expressions to be removed
replace_url = re.compile(r'https?:\/\/.*[\r\n]*')
remove_href = re.compile(r'\<a href')
remove_amp = re.compile(r'&amp;') 
remove_sym = re.compile(r'[_"\-;%()|+&=*%.,~!?:#$\[\]/]')
remove_br = re.compile(r'<br />')
remove_space = re.compile(r' +')
replace_brackets = re.compile('[/(){}\[\]\|,;+_-]')
remove_numbers = re.compile("[\d]")


def cleaning_text(text):
    text = text.lower() # lowercase text
    text = replace_url.sub("", text) # delete URLs from text
    text = remove_numbers.sub("", text) # delete numbers tags from text
    text = remove_href.sub('', text) # delete HTML tags from text
    text = remove_amp.sub('', text) # delete "&" from text
    text = remove_sym.sub(' ', text)# replace "remove_sym" symbols by space in text
    text = remove_br.sub(' ', text) # delete HTML tags from text
    text = replace_brackets.sub(' ', text) # replace "replace_brackets" symbols by space in text
    text = remove_space.sub(' ', text)
    
    return text

#applying all the above cleaning to the dataset
data_both['review_body'] = data_both['review_body'].apply(cleaning_text)
# data_both.head(10)


# ## perform contractions on the reviews.

# In[14]:


def contractionfunction(s):
    
    s = re.sub(r"won\'t", "will not", s)
    s = re.sub(r"can\'t", "can not", s)
    s = re.sub(r"ain\'t", "are not", s)
    s = re.sub(r"\'cause","because",s)
    s = re.sub(r"let\'s", "let us",s)
    s = re.sub(r"o\'clock", "of the clock", s)
    s = re.sub(r"ma\'am", "madam",s)
    
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'m", " am", s)
    return s

data_both['review_body'] = data_both['review_body'].apply(contractionfunction)
# data_both.head(10)


# In[15]:


avg_len2 = mean([len(s) for s in data_both["review_body"]])

# #before cleaing and pre-processing
# print("AFTER CLEANING AND BEFORE PREPROCESSING")
# print(*[rev+", length: "+str(len(rev))+"\n" for rev in data_both["review_body"].head(3)])
# print("Average review length after cleaning and before processing: "+ str(avg_len2))


# # Pre-processing

# ## remove the stop words 

# In[16]:


from nltk.corpus import stopwords
 
def remove_stopwords(text):
    STOPWORDS = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

data_both['review_body'] = data_both['review_body'].apply(remove_stopwords)
# data_both.head(10)


# ## perform tokenization and lemmatization  

# In[17]:


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

#creating an tokenizer object
text_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

def tokenize_text(text):
    #tokenize text
    review_tokens = text_tokenizer.tokenize(text)
    
    return review_tokens
    
#tokenizing each review in the dataset
data_both['review_body'] = data_both['review_body'].apply(tokenize_text)
# data_both.head(10)


# In[18]:


#stemmer object
stemmer=PorterStemmer()

#Lemmatizer object 
lemmmatizer=WordNetLemmatizer()

def lemmatize_tokens(tokens):
    #stemming
    stemmed_tokens=[stemmer.stem(t) for t in tokens]

    #Stemming with lemmatization
    lemmatized_tokens = [lemmmatizer.lemmatize(st) for st in stemmed_tokens]
    lemmatize_review = ' '.join(lemmatized_tokens)
    
    return lemmatize_review
    
data_both['review_body'] = data_both['review_body'].apply(lemmatize_tokens)


# In[19]:


from statistics import mean

avg_len3 = mean([len(s) for s in data_both["review_body"]])

# #after cleaing and pre-processing
# print("AFTER PREPROCESSING")
# print(*[rev+", length: "+str(len(rev))+"\n" for rev in data_both["review_body"].head(3)])
# print("Average review lentgh after processing: "+ str(avg_len3))


# In[20]:


from sklearn.model_selection import train_test_split

#splitting the dataset into train and test sets
X = data_both.review_body
y = data_both.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Perceptron

# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Perceptron

#perceptron for reviews
perceptron = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', Perceptron(tol=1e-3, random_state=0)),
               ])
#training the model
perceptron.fit(X_train, y_train)

# print("---PERCEPTRON---")
#predicting the classes on train
y_pred_perceptron_train = perceptron.predict(X_train)

# #metrics
# print('Train Accuracy: %s' % accuracy_score(y_pred_perceptron_train, y_train))
# print('Train Precision: %s' % precision_score(y_pred_perceptron_train, y_train))
# print('Train Recall: %s' % recall_score(y_pred_perceptron_train, y_train))
# print('Train F1: %s \n' % f1_score(y_pred_perceptron_train, y_train))

#predicting the classes on test
y_pred_perceptron = perceptron.predict(X_test)

#metrics
# print('Test Accuracy: %s' % accuracy_score(y_pred_perceptron, y_test))
# print('Test Precision: %s' % precision_score(y_pred_perceptron, y_test))
# print('Test Recall: %s' % recall_score(y_pred_perceptron, y_test))
# print('Test F1: %s' % f1_score(y_pred_perceptron, y_test))


# # SVM

# In[23]:


#SVM classifier for reviews
from sklearn.linear_model import SGDClassifier

svm_classifier = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-5, max_iter=100, tol=None)),
               ])
#training the model
svm_classifier.fit(X_train, y_train)

# print("---SVM---")
#predicting the classes on train
y_pred_svm_train = svm_classifier.predict(X_train)

#metrics
# print('Train Accuracy: %s' % accuracy_score(y_pred_svm_train, y_train))
# print('Train Precision: %s' % precision_score(y_pred_svm_train, y_train))
# print('Train Recall: %s' % recall_score(y_pred_svm_train, y_train))
# print('Train F1: %s \n' % f1_score(y_pred_svm_train, y_train))

#predicting the classes on test
y_pred_svm = svm_classifier.predict(X_test)

#metrics
# print('Test Accuracy: %s' % accuracy_score(y_pred_svm, y_test))
# print('Test Precision: %s' % precision_score(y_pred_svm, y_test))
# print('Test Recall: %s' % recall_score(y_pred_svm, y_test))
# print('Test F1: %s' % f1_score(y_pred_svm, y_test))


# # Logistic Regression

# In[24]:


#logistic regression model for reviews
from sklearn.linear_model import LogisticRegression

log_reg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=5, C=1e6)),
               ])
#training the model
log_reg.fit(X_train, y_train)

# print("---LR---")

#predicting the classes on train
y_pred_lr_train = log_reg.predict(X_train)

#metrics
# print('Train Accuracy: %s' % accuracy_score(y_pred_lr_train, y_train))
# print('Train Precision: %s' % precision_score(y_pred_lr_train, y_train))
# print('Train Recall: %s' % recall_score(y_pred_lr_train, y_train))
# print('Train F1: %s \n' % f1_score(y_pred_lr_train, y_train))

#predicting the classes on test
y_pred_lr = log_reg.predict(X_test)

#metrics
# print('Test Accuracy: %s' % accuracy_score(y_pred_lr, y_test))
# print('Test Precision: %s' % precision_score(y_pred_lr, y_test))
# print('Test Recall: %s' % recall_score(y_pred_lr, y_test))
# print('Test F1: %s' % f1_score(y_pred_lr, y_test))


# # Naive Bayes

# In[25]:


#Multinomial Naive Bayes Classifier for reviews
from sklearn.naive_bayes import MultinomialNB

multi_naive_bayes = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
#training the classifier
multi_naive_bayes.fit(X_train, y_train)

# print("---NB---")

#predicting the classes on train
y_pred_nb_train = multi_naive_bayes.predict(X_train)

#metrics
# print('Train Accuracy: %s' % accuracy_score(y_pred_nb_train, y_train))
# print('Train Precision: %s' % precision_score(y_pred_nb_train, y_train))
# print('Train Recall: %s' % recall_score(y_pred_nb_train, y_train))
# print('Train F1: %s \n' % f1_score(y_pred_nb_train, y_train))

#predicting the classes on test
y_pred_nb = multi_naive_bayes.predict(X_test)

#metrics
# print('Test Accuracy: %s' % accuracy_score(y_pred_nb, y_test))
# print('Test Precision: %s' % precision_score(y_pred_nb, y_test))
# print('Test Recall: %s' % recall_score(y_pred_nb, y_test))
# print('Test F1: %s' % f1_score(y_pred_nb, y_test))


# In[37]:


#final output

print("\nNeutral Reviews:", discarded_count, "Positive Reviews:", len(pos_data), "Negative Reviews:", len(neg_data), "\n")

print(str(avg_len1)+","+ str(avg_len2))
print(str(avg_len2)+","+ str(avg_len3)+"\n")

print(str(accuracy_score(y_pred_perceptron_train, y_train))+","+str(precision_score(y_pred_perceptron_train, y_train))+","+ str(recall_score(y_pred_perceptron_train, y_train))+","+str(f1_score(y_pred_perceptron_train, y_train))+","+str(accuracy_score(y_pred_perceptron, y_test))+","+str(precision_score(y_pred_perceptron, y_test))+","+ str(recall_score(y_pred_perceptron, y_test))+","+str(f1_score(y_pred_perceptron, y_test))+"\n")
print(str(accuracy_score(y_pred_svm_train, y_train))+","+str(precision_score(y_pred_svm_train, y_train))+","+ str(recall_score(y_pred_svm_train, y_train))+","+str(f1_score(y_pred_svm_train, y_train))+","+str(accuracy_score(y_pred_svm, y_test))+","+str(precision_score(y_pred_svm, y_test))+","+ str(recall_score(y_pred_svm, y_test))+","+str(f1_score(y_pred_svm, y_test))+"\n")
print(str(accuracy_score(y_pred_lr_train, y_train))+","+str(precision_score(y_pred_lr_train, y_train))+","+ str(recall_score(y_pred_lr_train, y_train))+","+str(f1_score(y_pred_lr_train, y_train))+","+str(accuracy_score(y_pred_lr, y_test))+","+str(precision_score(y_pred_lr, y_test))+","+ str(recall_score(y_pred_lr, y_test))+","+str(f1_score(y_pred_lr, y_test))+"\n")
print(str(accuracy_score(y_pred_nb_train, y_train))+","+str(precision_score(y_pred_nb_train, y_train))+","+ str(recall_score(y_pred_nb_train, y_train))+","+str(f1_score(y_pred_nb_train, y_train))+","+str(accuracy_score(y_pred_nb, y_test))+","+str(precision_score(y_pred_nb, y_test))+","+ str(recall_score(y_pred_nb, y_test))+","+str(f1_score(y_pred_nb, y_test))+"\n")

