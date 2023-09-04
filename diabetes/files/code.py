#!/usr/bin/env python
# coding: utf-8

# # Setup

# ### Basic

# In[1]:


import pandas as pd
import numpy as np


# ### Plots

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go


# ### data preprocessing

# In[70]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE


# ### Machine Learning

# In[4]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


# ### additional Packages

# In[5]:


from pandas_profiling import ProfileReport


# # Data + EDA

# In[6]:


df = pd.read_csv("../diabetes.csv")
df.head()


# In[7]:


df.describe().T


# In[8]:


def EDA(df) : 
    plt.figure(figsize = (15, 15))
    
    cnt = 1
    for col in df.columns : 
        plt.subplot(3, 3, cnt); cnt += 1;
        sns.histplot(data = df, x = col, hue = "Outcome")
    plt.show()   


# In[9]:


EDA(df)


# ## Pandas Profiling

# In[10]:


profile =  ProfileReport(df, title = "Diabetes Report")
profile.to_widgets()


# # Data Preprocessing

# In[11]:


COLS = df.columns.drop('Outcome')
df.head(), COLS


# In[12]:


df.isnull().sum()


# In[13]:


df.isna().sum()


# ## Correlation HITMAP

# In[14]:


plt.figure(figsize = (20,10))
sns.heatmap(df.corr(),annot=True , cmap ='YlGnBu' )

plt.title("Correlation between Features")


# ## Missing Values

# In[15]:


zero_null_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_null_fields] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

df.isna().sum()


# In[16]:


def handle_missing(data, column:str):
    data.loc[(data['Outcome'] == 0 ) & (data[column].isnull()), column] = df.groupby('Outcome')[column].median()[0]
    data.loc[(data['Outcome'] == 1 ) & (data[column].isnull()), column] = df.groupby('Outcome')[column].median()[1]
    
    return data

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for col in cols:
    handle_missing(df, col)
    
df.isna().sum()


# ## Outliers

# In[17]:


def handle_outlier(df, col) : 
    # IQR
    q1 = 0.25; q3 = 0.90
    
    Q1 = df[col].quantile(q1)
    Q3 = df[col].quantile(q3)
    iqr = Q3 - Q1
    
    up_b = Q3 + 1.5 * iqr
    low_b = Q1 - 1.5 * iqr
    
    df.loc[(df[col] < low_b), col] = low_b
    df.loc[(df[col] > up_b), col] = up_b


# In[18]:


for col in df.columns : 
    handle_outlier(df, col)


# ## Scaling Data

# In[19]:


rs = RobustScaler()
df[COLS] = rs.fit_transform(df[COLS])


# ## Drop Insulin (Bad :/)

# In[20]:


#df = df.drop(['Insulin'], axis = 1)
#df


# ## Pandas Profiling (after Preprocessing)

# In[21]:


profile =  ProfileReport(df, title = "Diabetes Report")
profile.to_widgets()


# # Machine Learning

# ## Train-Test Split

# In[22]:


x = df.drop(['Outcome'], axis=1)
y = df['Outcome']


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y, shuffle = True)


# ## Balancing

# In[24]:


oversample = SMOTE(random_state = 42, k_neighbors = 10)

x_smote, y_smote = oversample.fit_resample(x_train, y_train)
x_train, y_train = x_smote, y_smote


# In[25]:


y_smote.value_counts()


# ### Classifiers

# In[26]:


# models 
classifiers = [
    DecisionTreeClassifier(max_depth = 3, random_state = 42),
    AdaBoostClassifier(DecisionTreeClassifier(random_state = 42)),
    RandomForestClassifier(max_depth = 5, random_state = 42),
    GradientBoostingClassifier(random_state = 42),
    LogisticRegression(random_state = 42, solver='lbfgs', max_iter=10000),
    SVC(random_state = 42, probability = True),        
    KNeighborsClassifier(n_neighbors = 5, algorithm = "kd_tree"),
    GaussianNB(),
    MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes=(5, 2), random_state = 1),
    BaggingClassifier(SVC(random_state = 42, probability = True), max_samples = 0.5, max_features = 0.7),
    ExtraTreesClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2, random_state = 0),
    XGBClassifier(learning_rate= 0.01,max_depth = 3,n_estimators = 1000)
]


# In[63]:


def make_classification(classifiers, x_train, x_test, y_train, y_test) :
    
    # metrics
    acc, f1, AUC, recall, cross_val, prec = [ ], [ ], [ ], [ ], [ ], [ ]
    models = [ ]
    
    for classifier in classifiers : 
        clf = classifier
        clf.fit(x_train, y_train)
        
        y_pred = clf.predict(x_test)
        y_prob = clf.predict_proba(x_test)
        
        acc.append(((accuracy_score(y_test,y_pred))) * 100)
        cross_val.append(sum(cross_val_score(clf, x_train, y_train, cv = 10, scoring = "accuracy"))/10)
        f1.append(((f1_score(y_test,y_pred))) * 100)
        AUC.append(((roc_auc_score(y_test,y_prob[:, 1]))) * 100)
        recall.append(((recall_score(y_test,y_pred))) * 100)
        prec.append(((precision_score(y_test,y_pred))) * 100)
        models.append(clf.__class__.__name__)
        
    res = pd.DataFrame({
        "Accuracy" : acc,
        "Cross Val" : cross_val,
        "F1" : f1,
        "ROC" : AUC,
        "Recall" : recall,
        "Precision" : prec,
        "ML Models" : models,
    })
    
    res = (res.sort_values(by = ['ROC','F1'], ascending = False).reset_index(drop =  True))
    return res


# In[28]:


res = make_classification(classifiers, x_train, x_test, y_train, y_test)
res


# ### Ensemble

# In[29]:


gb_clf = GradientBoostingClassifier(random_state = 42)
rf_clf = RandomForestClassifier(max_depth = 5, random_state = 42)
nn_clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes=(5, 2), random_state = 1)
svm_clf = SVC(random_state = 42, probability = True)
svm_bg_clf = BaggingClassifier(SVC(random_state = 42, probability = True), max_samples = 0.5, max_features = 0.7)
xgb_clf = XGBClassifier(learning_rate= 0.01,max_depth = 3, n_estimators = 1000)


# In[30]:


voting_clf = VotingClassifier(
 estimators = [('gb', gb_clf), ('rf', rf_clf), ('svm_bagging', svm_bg_clf)],
 voting = 'soft')

voting_clf.fit(x_train, y_train)


# In[31]:


y_pred = voting_clf.predict(x_test)
y_prob = voting_clf.predict_proba(x_test)


# In[32]:


pd.Series({"Acc :" : accuracy_score(y_test, y_pred) * 100,"F1 : " : f1_score(y_test, y_pred) * 100,"AUC : " : roc_auc_score(y_test,y_prob[:, 1]) * 100,"Recall : " : ((recall_score(y_test,y_pred))) * 100})


# ## Cross Validation 

# In[33]:


def cross_val(classifiers, x_train, y_train) :
    cv_train, cv_test, diff, models =  [], [], [], []
    
    for classifier in classifiers : 
        clf = classifier
        #clf.fit(x_train, y_train)
        
        cv = cross_validate(clf, x_train, y_train, cv = 5, scoring = "accuracy", return_train_score = True)
        
        cv_train.append(cv['train_score'].mean() * 100)
        cv_test.append(cv['test_score'].mean() * 100)
        diff.append((cv['train_score'].mean() - cv['test_score'].mean()) * 100)
        models.append(clf.__class__.__name__)
        
        
    res = pd.DataFrame({
        "CV Train" : cv_train,
        "CV Test" : cv_test,
        "Diff" : diff,
        "ML Model" : models
    })
    
    res = (res.sort_values(by = ['CV Test', 'CV Train'], ascending = False).reset_index(drop =  True))
    
    return res


# In[34]:


nclf = classifiers
nclf.pop(8)


# In[35]:


nclf.append(
    VotingClassifier(
 estimators = [('gb', gb_clf), ('rf', rf_clf), ('svm_bagging', svm_bg_clf)],
 voting = 'soft')
)


# In[36]:


nclf


# In[37]:


cv_res = cross_val(nclf, x_train, y_train)
cv_res


# ## Fine Tuning

# ### 1) GradientBoostingClassifier

# In[38]:


gb_clf = GradientBoostingClassifier( 
    random_state = 42, 
    min_samples_split = 100,
    min_samples_leaf = 20,
    max_depth = 2,
    max_features = 3,
    learning_rate = 0.05,
    subsample = 0.65,
)


# In[39]:


cross_val([gb_clf], x_train, y_train)


# ### 2) ExtraTreesClassifier

# In[40]:


ex_clf = ExtraTreesClassifier(
    random_state = 42,
    max_depth = 6,
    min_samples_split = 20,
    max_features = "log2",
)


# In[41]:


cross_val([ex_clf], x_train, y_train)


# ### 3) SVC

# In[42]:


svm_clf = SVC(
    random_state = 42,
    probability = True,
    C = 1,
    kernel = "rbf",
    gamma = "scale",
)


# In[43]:


cross_val([svm_clf], x_train, y_train)


# ### 4) Ensemble

# In[44]:


voting_clf = VotingClassifier(
    estimators = [('gb', gb_clf), ('ex', ex_clf), ('svm', svm_clf)],
    voting = 'soft'
)


# In[45]:


cross_val([voting_clf], x_train, y_train)


# In[46]:


make_classification([ex_clf, svm_clf, gb_clf, voting_clf], x_train, x_test, y_train, y_test)


# In[47]:


y_pred_vt = voting_clf.predict(x_test)
y_pred_gb = gb_clf.predict(x_test)


# In[48]:


accuracy_score(y_test, y_pred_vt) * 100, accuracy_score(y_test, y_pred_gb) * 100, 


# In[49]:


cross_val([voting_clf, gb_clf], x_train, y_train)


# In[50]:


cross_val([voting_clf, gb_clf], x, y)


# # Using PCA (Just a Showcase)

# #### in this problem due to low number of demensions we do not need Demensionality Reduction Methods and Using PCA here is just for educational purposes

# In[51]:


from sklearn.decomposition import PCA


# In[52]:


pca = PCA()
pca.fit(x_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1


# In[53]:


d


# In[57]:


pca = PCA(n_components = 0.90)
x_redu = pca.fit_transform(x_train)


# In[58]:


pca_gb_clf = GradientBoostingClassifier( 
    random_state = 42, 
    min_samples_split = 100,
    min_samples_leaf = 20,
    max_depth = 2,
    max_features = 3,
    learning_rate = 0.05,
    subsample = 0.65,
)


# In[ ]:





# In[59]:


cross_val([pca_gb_clf], x_redu, y_train)


# # BEST MODEL

# ### as we saw in the previous parts of this notebook we had out 2 best models as "Ensemble" and "GradientBoosting" now its time to compare them in different metrics and declare a winner

# # Results

# In[64]:


make_classification([gb_clf, voting_clf], x_train, x_test, y_train, y_test)


# ## 1) Accuracy

# not the most important metric in this problem 

# as we can see the **GB Classifier** is slightly doing better 

# ## 2) Precision

# both have rather low precision but once again **GB Classifer** is doing better even much better ! **considering the 3.5% difference**

# ## 3) Recall

# in my eyes in this particular problem recall can be one of the most if not the most important metric

# the reasoning behind this idea is that it is really important to correctly detect the ones that have diabetes cause if not, they wont be diagnosed and as a result they will not be cured which may have some detrimental consequences

# as well as the other metrics **GB Classifier** is doing better my the smallest margin, but its still better !

# ## 4) F1

# well this one is obviously a win for **GB Classifier**

# ## 5) AUC

# roc curve is also a really useful metric for this problem and one way to measure it is using **AUC (Area Under Curve)**

# as we already had figured it out, **GB Classifier** has the upper hand

# ### Overfitting / Underfitting

# none of these two has underfitting by any means

# as for overfitting based of what we saw in the previous parts of this notebook we can say that none of these has overfitting problem as well however the diffrence between train/test in **GB Classifier** is somewhat smaller which makes it a slightly more stable model

# # Conclusion

# based on all models and parameter tuning that we did we can safely say that **"Gradient Boosting Classifer"** is our best model

# In[66]:


make_classification([gb_clf], x_train, x_test, y_train, y_test)


# In[68]:


cross_val([gb_clf], x, y)


# In[77]:


y_scores = cross_val_predict(gb_clf, x, y, cv = 3, method = "decision_function")


# In[79]:


precisions, recalls, thresholds = precision_recall_curve(y, y_scores)


# In[80]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds) : 
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label = "Recall")


# # Precision-Recall Curve

# In[81]:


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[82]:


fpr, tpr, thresholds = roc_curve(y, y_scores)


# In[83]:


def plot_roc_curve(fpr, tpr, label = None) :
    plt.plot(fpr, tpr, linewidth = 2,label = label)
    plt.plot([0, 1], [0, 1], 'k--')


# # ROC Curve

# In[84]:


plot_roc_curve(fpr, tpr, "ROC Curve")
plt.show()


# In[85]:


roc_auc_score(y, y_scores)

