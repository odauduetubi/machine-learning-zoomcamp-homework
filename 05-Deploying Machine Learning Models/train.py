#!/usr/bin/env python
# coding: utf-8

# # Deploying Machine Learning Models
# In this module, we use the code from the previous module.
# 

# # Evaluation Metrics for Classification
# 
# We will use the same example as in the previous module. We basically want to answer the question "how good is it?"

# ## Churn prediction project
# 
# - Dataset:https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# - https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as  plt


# In[3]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[4]:


#get_ipython().system('wget $data -O data-week-5.csv')


# In[5]:


df = pd.read_csv('data-week-5.csv')


# We want to write our column names and entries consistent. We do that by writting them all in lower case and the spaces in _.

# In[6]:


df.columns


# In[7]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[8]:


df.totalcharges


# We obsserve that `totalcharges` is a str, lets try to convert it to a number.

# In[9]:


tc = pd.to_numeric(df.totalcharges, errors = 'coerce') #conversion to numbers, we also tell python to replace arrors by missing values with null


# In[10]:


df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')


# In[11]:


df.totalcharges = df.totalcharges.fillna(0)


# In[12]:


(df.churn == 'yes').astype(int).head()


# In[13]:


df.churn = (df.churn == 'yes').astype(int)


# In[14]:


df.churn


# ## Setting up the validation framework
# - Perform the train/validation/test split with Scikit-Learn.

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)  #random_state is to ensure that the result is reproducible.


# In[17]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)  #random_state is to ensure that the result is reproducible.


# In[18]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# Next, we get our churn variables. The code below takes the column with the target variable, churn, and saves it outside the dataframe.

# In[19]:


y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


# Deletes the churn columns from the dataframes to make sure we don't accidentally use the churn variable as a feature during training.

# In[20]:


del df_train['churn']
del df_val['churn']
del df_test['churn']


# ## EDA
# - Check missing values
# - Look at the target variables (churn)
# - Look at tthe numerical and categorical variiables

# In[21]:


#For this, we shall be using our full train dataset

df_full_train = df_full_train.reset_index(drop=True)


# In[22]:


#Let's look at the missing values
df_full_train.isnull().sum()


# In[23]:


#Next, we check the distribution of values in the targett variable (churn)
df_full_train.churn.value_counts()


# In[24]:


#Now to see the proportion of the above values we do

df_full_train.churn.value_counts(normalize = True)


# The churn rate in the full train dataset is $27\%$. Alternatively, we can simply calculate the mean and get the same value.

# In[25]:


global_churn_rate = df_full_train.churn.mean()
print(f"The churn rate is: {round(global_churn_rate, 2)}")


# Next, we create two lists from our dataset.

# In[26]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[27]:


df_full_train.columns


# In[28]:


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[29]:


#Now, we check how many unique values each variable has. We already know that we should have just a few for each column. 
df_full_train[categorical].nunique()


# ## Feature importance: Churn rate and risk ratio
# Feature importance analysis is a part of EDA that identifies which features affect our target variable.
# 
# - Churn rate
# - Risk ratio
# - Mutual information

# ### Churn rate

# Before now, we saw the global churn rate. In this section, we shall check the churn rate within each group (say gender).

# In[30]:


df_full_train.head()


# In[31]:


churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
print(f"The rate of churn among females is: {round(churn_female, 3)}")


# In[32]:


churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
print(f"The rate of churn among males is: {round(churn_male, 3)}")


# In[33]:


global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 3)


# The difference between the churn rate among males and females and the global churn rate is not significant and of no consequence. We can do the same analysis for the column `partner`.

# In[34]:


df_full_train.partner.value_counts()


# In[35]:


churn_with_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
print(f"The rate of churn with partner is: {round(churn_with_partner, 3)}")


# In[36]:


churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
print(f"The rate of churn without partner is: {round(churn_no_partner, 3)}")


# The difference between with or witthout partner is significant, compared to the global churn rate.

# #### Risk ratio

# In[37]:


churn_no_partner/global_churn_rate


# In[38]:


churn_with_partner/global_churn_rate


# In[39]:


df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group['diff'] = df_group['mean'] - global_churn_rate
df_group['risk'] = df_group['mean'] / global_churn_rate

df_group


# In[40]:


#This is used to display variables in a loop
from IPython.display import display


# In[41]:


for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate

    display(df_group)
    print()
    print()


# ## Feature importance: Mutual information
# Mutual information - concept from information theory, it tells us how much we can learn about one variable if we know the value of another variable.
# 
# - https://en.wikipedia.org/wiki/Mutual_information

# This is already implemented in Scikit-learn in the `mutual_info_score`.

# In[42]:


from sklearn.metrics import mutual_info_score


# In[43]:


mutual_info_score(df_full_train.churn, df_full_train.contract) #The oder in which we pass it doesn't matter


# This tells us about churn by  observing the variable contract and vice versa.

# In[44]:


mutual_info_score(df_full_train.churn, df_full_train.gender)


# In[45]:


mutual_info_score(df_full_train.churn, df_full_train.partner)


# Now, we apply this metric to all the categorical variables tthat we have.

# In[46]:


# We define a function with only one argument because of what comes next.
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# In[47]:


mi = df_full_train[categorical].apply(mutual_info_churn_score) #Applies the function mutual_iinfo_churn_score to each categorical column

#Sorts the values in descending order.
mi.sort_values(ascending = False)


# ## One-hot encoding
# - Use Scikit-Learn to encode categorical features.

# In the previous classes, we encoded it manually to get our feature matrix. However, we use scikit-learn to do it here.

# In[48]:


from sklearn.feature_extraction import DictVectorizer


# In[49]:


df_train[['gender', 'contract']].iloc[:100]


# In[50]:


dicts = df_train[['gender', 'contract']].iloc[:100].to_dict(orient= 'records') #Turns it row wise. Turns each row to a dictionary


# In[51]:


dv = DictVectorizer(sparse = False) #Create a new instance for this class. We will not use the sparse here, and hence we use False.This turns the output to a numpy array


# In[52]:


dv.fit(dicts)


# In[53]:


dv.get_feature_names_out()


# In[54]:


dv.transform(dicts) #produce a sparse matrix


# To learn more about sparse matrix, see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html

# Dictvectorizer is smart enough to leave a column intact that is numerical(Not categorical). 

# In[55]:


train_dicts = df_train[categorical + numerical].to_dict(orient= 'records')


# In[56]:


train_dicts[0] #Shows the first entry of the dictionary


# In[57]:


#Next, we fit our dict vectorizer

dv.fit(train_dicts)


# In[58]:


dv.get_feature_names_out() #we can learn the names of all the columns


# In[59]:


X_train = dv.transform(train_dicts)


# In[60]:


val_dicts = df_val[categorical + numerical].to_dict(orient= 'records')


# In[61]:


#We don't fit on the validatioon dataset, we just transform

X_val = dv.transform(val_dicts)


# Now, we are ready to train our model.

# In[62]:


from sklearn.feature_extraction import DictVectorizer


# In[63]:


dv = DictVectorizer(sparse = False)

train_dicts = df_train[categorical + numerical].to_dict(orient= 'records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient= 'records')
X_val = dv.transform(val_dicts)


# ## Training logistic regression with Scikit-learn
# 
# - Train a model with Scikit-learn
# - Apply it to the validation dataset
# - Calculate the accuracy

# We first import the model.

# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


#we train the model by calling the fit method
model = LogisticRegression()
model.fit(X_train, y_train)


# In[66]:


model.intercept_ #bias term


# In[67]:


model.intercept_[0]


# In[68]:


model.coef_ #these are the weights


# In[69]:


model.coef_[0].round(3)


# To use the model, we go as follows: 

# In[70]:


model.predict(X_train)


# These are Hard Predictions.

# In[71]:


model.predict_proba(X_train)


# These are Soft Predictions. The first element of the array is the probability of belonging to $0$ and the second element is the probability of belonging to $1$.

# In[72]:


y_pred = model.predict_proba(X_val)[: ,1]
y_pred


# In[73]:


churn_decision = (y_pred >= 0.5)
churn_decision


# In[74]:


df_val[churn_decision].customerid #selecting all the customers that we think are going to churn


# Now, we may decide to send them a promotional email of some discount.

# Now, we see how accurate our predictions are?

# In[75]:


(y_val == churn_decision).mean()


# ## Accuracy and dummy model
# - Evaluate the model on different thhresholds
# - Check the accuracy of dummy baselines

# In[76]:


len(y_val)


# To get the parts that we got correctly.

# In[77]:


(y_val == churn_decision).sum()


# In[78]:


1132/1409


# Next, we see whether or not the threshold of $0.5$ is good.

# In[79]:


thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    churn_decision = (y_pred >= t)
    score = (y_val == churn_decision).mean()

    print('%.2f %.3f' % (t, score))
    scores.append(score)


# In[80]:


scores


# In[81]:


plt.plot(thresholds, scores)


# $0.5$ is indeed the best threshold.

# We can actually do all of these in scikit-learn.

# In[82]:


from sklearn.metrics import accuracy_score


# In[83]:


accuracy_score(y_val, (y_pred >= 0.5))


# Now, let us tak the extreme cases i.e $t = 0$ and $t = 1.0$

# In[84]:


from collections import Counter # Helps us counts 


# In[85]:


Counter(y_pred >= 0.5)


# In[86]:


Counter(y_pred >= 1.0)


# In[87]:


Counter(y_pred >= 0)


# Let us call the model with $t = 1.0$ the dummy model, and we see that it has an accuuracy oof $73\%$ as opposed to our model ($t = 0.5$) with an accuracy of $80\%$.

# In[88]:


Counter(y_val)


# In[89]:


y_val.mean()


# $27\%$ of the users are churning.

# ## Confusion table
# 
# - Different types of errors and correct decisions
# - Arranging them in a table

# In[90]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[91]:


t = 0.5

predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[92]:


tp = (predict_positive & actual_positive).sum() #We use the sum() to see how many are there.
tn = (predict_negative & actual_negative).sum()


# In[93]:


fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[94]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]

])
confusion_matrix


# In[95]:


#Now, we normalise it
(confusion_matrix / confusion_matrix.sum()).round(2)


# ## Precision and Recall

# We calculate accuracy as follows:

# In[96]:


(tp + tn)/(tp+fp+tn+fn)


# Precision tells us fractions of positive predictions that turned out to be correct
# 
# $$ Precision = \frac{tp}{tp + fp}$$

# In[97]:


p = tp / (tp + fp)
p


# Recall is the fraction of correctly identified positive examples (i.e people who were going to churn).
# 
# $$ Recall = \frac{tp}{tp + fn} $$

# In[98]:


r = tp / (tp + fn)
r


# ## ROC Curves
# ROC stands for Reciever Operating Characteristics. This is a way of describing the perforrmance of a binary classification model.

# ### FPR and TPR
# Here, FPR is False Positive Rate, and TPR is True Positive Rrate.

# In[99]:


tpr = tp / (tp + fn)
tpr


# In[100]:


fpr = fp / (fp + tn)
fpr


# Instead of evaluating the above for just one threshold, ROC curvs looks at it for all possible thresholds. 

# In[101]:


scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum() 
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, fp, fn, tn))


# In[102]:


scores


# Next, we turn it to a dataframe.

# In[103]:


pd.DataFrame(scores)


# In[104]:


columns = ['thresholds', 'tp', 'fp', 'fn', 'tn']

df_scores = pd.DataFrame(scores, columns=columns)


# In[105]:


df_scores


# In[106]:


df_scores[::10]


# In[107]:


df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)


# In[108]:


df_scores[::10]


# In[109]:


plt.plot(df_scores.thresholds, df_scores['tpr'], label='TPR')
plt.plot(df_scores.thresholds, df_scores['fpr'], label='FPR')
plt.legend() #Add the legend


# ## Random model

# Now, we want to assign a random model to every customer.

# In[110]:


np.random.seed(1)
y_rand = np.random.uniform(0, 1, size = len(y_val))


# In[111]:


y_rand.round(3)


# In[112]:


((y_rand >= 0.5) == y_val).mean() #To check accuracy for our random model


# In[113]:


def tpr_fpr_dataframe(y_val, y_pred):

    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum() 
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['thresholds', 'tp', 'fp', 'fn', 'tn']

    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores


# In[114]:


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# In[115]:


df_rand[::10]


# In[116]:


plt.plot(df_rand.thresholds, df_rand['tpr'], label='TPR')
plt.plot(df_rand.thresholds, df_rand['fpr'], label='FPR')
plt.legend() #Add the legend


# ## Ideal model

# Next, we talk about a different benchmark. The ideal model (It is ideal because it gets the correct output for everyone). For that we need to first know the negative examples (i.e numbers of peope who are not churning).

# In[117]:


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

num_neg, num_pos


# Next we want to create a list of non-churning customers, then churning ones.

# In[118]:


y_ideal = np.repeat([0, 1], [num_neg, num_pos]) #We use a function that contains first 0 then 1, using the repeat function. The number of times it repeats is given by num_neg and num_pos respectively
y_ideal


# Next, we create our predictions i.e numbers between 0 and 1

# In[119]:


y_ideal_pred = np.linspace(0, 1, len(y_val))


# Now, tto check the accuracy, we go as follows.

# In[120]:


1 - y_val.mean()


# In[121]:


((y_ideal_pred >= 0.726) == y_ideal).mean()


# Although this model doesn't exist in reality, it helps us to benchmark our model to this one. Now, we do an exercise for the ideal model as we did for the random model.

# In[122]:


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)


# In[123]:


df_ideal[::10]


# Now, let us plot it.
# 

# In[124]:


plt.plot(df_ideal.thresholds, df_ideal['tpr'], label='TPR')
plt.plot(df_ideal.thresholds, df_ideal['fpr'], label='FPR')
plt.legend() #Add the legend


# ### Putting everything together

# In[125]:


#Actual model
plt.plot(df_scores.thresholds, df_scores['tpr'], label='TPR')
plt.plot(df_scores.thresholds, df_scores['fpr'], label='FPR')

#Random model
#plt.plot(df_rand.thresholds, df_rand['tpr'], label='TPR')
#plt.plot(df_rand.thresholds, df_rand['fpr'], label='FPR')

plt.plot(df_ideal.thresholds, df_ideal['tpr'], label='TPR', color = 'black')
plt.plot(df_ideal.thresholds, df_ideal['fpr'], label='FPR', color = 'black')
plt.legend() #Add the legend


# We want the TPR for the actual model to be as close to the TPR of the ideal model, from the above, we see that the blue curve is still far away from the upper black curve (Hence, we are making a mistake).
# 
# We notice that we have two thresholds $0.5$ which is good for the accuracy, and $0.726$ which is good for the ideal model. Next, we plot the FPR against the TPR.

# In[126]:


plt.figure(figsize=(5,5)) # To change the size of the graph

plt.plot(df_scores.fpr, df_scores.tpr, label = 'model')
plt.plot(df_rand.fpr, df_rand.tpr, label = 'random')
plt.plot(df_ideal.fpr, df_ideal.tpr, label = 'ideal')

#label the graph
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# Let us call the coordinate $(0, 1)$ the _NORTH STAR_ (ideally where we would like to get to).

# Next, we plot the model, and a straight line (mimiking the random model).

# In[127]:


plt.figure(figsize=(5,5)) # To change the size of the graph

plt.plot(df_scores.fpr, df_scores.tpr, label = 'model')
plt.plot([0,1], [0,1], label = 'random', linestyle = '--')
#plt.plot(df_ideal.fpr, df_ideal.tpr, label = 'ideal')

#label the graph
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# This is how the ROC curve look like. For this, we plot the TPR against the FPR, evaluated at a particular threshold. Our aim is for the curve to be as close to the _NORTHERN STAR_ and as far away from the random baseline.

# We can actually use Scikit-learn to plot the ROC curve.

# In[128]:


from sklearn.metrics import roc_curve


# In[129]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[130]:


plt.figure(figsize=(5,5)) # To change the size of the graph

plt.plot(fpr, tpr, label = 'model')
plt.plot([0,1], [0,1], label = 'random', linestyle = '--')
#plt.plot(df_ideal.fpr, df_ideal.tpr, label = 'ideal')

#label the graph
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# Observe that the above looks like what we initially have, However, it is more accurate because it uses more thresholds that when we solved it manually.

# ## ROC AUC
# - Area under the ROC curve
# - Intrepretatioon of AUC

# In[131]:


from sklearn.metrics import auc #works for any curve.


# In[132]:


auc(fpr, tpr) # when we calculated it using sklearn


# In[133]:


auc(df_scores.fpr, df_scores.tpr) #when we calculated it manually


# In[134]:


auc(df_ideal.fpr, df_ideal.tpr) #when we calculated it for the ideal case


# NB: We can do this in two lines
# ```
# fpr, tpr, thresholds = roc_curve(y_val, y_pred)
# auc(fpr, tpr) 
# 
# ```
# 
# We can however do this in sklearn

# In[135]:


from sklearn.metrics import roc_auc_score


# In[136]:


roc_auc_score(y_val, y_pred)


# Basically, AUC tells us the probability that randomly selected example has a higher score than a randomly selected negative example.

# In[137]:


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


# In[138]:


#Next we randomly select a negative example.

import random

n = 100000
success = 0

for i in range(n):

    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success/n


# We can implement the above with numpy.

# In[139]:


n = 50000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size = n) #here no need for minus 1 because it is not iinclusive in numpy
neg_ind = np.random.randint(0, len(neg), size = n)

pos_ind, neg_ind


# In[140]:


(pos[pos_ind] > neg[neg_ind]).mean()


# ## Cross-Validation
# - Evaluating the same model on different subsets of data.
# - Getting the aveerage prediction and the spread within predictions.

# In[141]:


def train(df_train, y_train, C = 1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


# In[142]:


dv, model = train(df_train, y_train, C=0.001)


# In[143]:


def predict(df, dv, model):

    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


# In[144]:


y_pred = predict(df_val, dv, model)


# Now, we use K-selection. 

# In[145]:


from sklearn.model_selection import KFold


# In[146]:


kfold = KFold(n_splits = 10, shuffle = True, random_state = 1)








#get_ipython().system('pip install tqdm')





from tqdm.auto import tqdm





n_splits = 5

C = 1.0
scores = []

kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)


for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[157]:


scores


# Now, we go on as before, extract the dictionary, train the feature variiable...

# In[152]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# This model still lives in our notebook. Our main aim here is to deploy it to the web services. Next, we save the model.
# 

# In[154]:


import pickle


# In[159]:


output_file = f'model_C={C}.bin'
output_file


# In[160]:


f_out = open(output_file, 'wb') #write bytes
pickle.dump((dv, model), f_out)
f_out.close()


# To make sure that we closed the file, we run the following

# In[161]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# Next, we load the model (In principle, we mimic starting afresh, this is done by restarting the kernel).

