# Cheat sheet examen ADA

## Artur Jesslen


# Importations
```python
# General packages
import pandas as pd
import numpy as np
from collections import Counter

# ML
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, SCORERS, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


# Stats
from statsmodels.stats import diagnostic, weightstats, proportion
import statsmodels.formula.api as smf
from scipy import stats

# Display figures
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# PySpark
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *

# NLP
import spacy
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.phrases import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


#  Create color generator (20 colors)
from itertools import cycle
colors = cycle([plt.get_cmap('tab20')(i/20) for i in range(20)])
default_color = next(colors)

```


# Utils
```python
# Drop rows with NA
df = df.dropna()

# Split train/test dataset
def split_set(data_to_split, ratio=0.8):
    mask = np.random.rand(len(data_to_split)) < ratio
    return (data_to_split[mask].reset_index(drop=True), data_to_split[~mask].reset_index(drop=True))

# Split trian/test dataset sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Convert Categorical (Male/Female) to numerical variables
X = pd.get_dummies(X)

# Normalize dataframe 
normalized_df = (df - df.mean()) / df.std()

# Min-Max normalization
normalized_df = (df - df.min()) / (df.max() - df.min())

# Normalization 
def normalize(df, minmax=False):
    df_norm = df.copy()
    df_num = df_norm.select_dtypes(include=[np.number])
    if minmax:
        df_num = (df_num - df_num.min()) / (df_num.max() - df_num.min())
    else:
        df_num = (df_num - df_num.mean()) / df_num.std()
    df_norm[df_num.columns] = df_num
    return df_norm

# Print sklearn metrics (for GriSearchCV for example)
print(sorted(SCORERS.keys()))

```

# Plots :
```python
## Pair plots 
sns.pairplot(df)


## barplot + density 
ax = sns.distplot(treated['re78'], hist=True, label='treated');
ax = sns.distplot(control['re78'], hist=True, label='control')
ax.set(title='Income distribution comparison in 1978, after matching',xlabel='Income 1978', ylabel='Income density')
plt.legend()
plt.show()


## boxplots
# --> OPTION 1
fig, ax = plt.subplots(figsize=(15,5))
ax.boxplot([pokemon.Attack, pokemon.Defense, pokemon['Sp. Atk'], pokemon['Sp. Def'], pokemon['Speed']])
plt.xticks([1, 2, 3, 4, 5], ['Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']);
# --> OPTION 2
lalonde_data.boxplot(by='treat', column='educ', figsize = [5, 5], grid=True)
# Plot educ column as function of treat (category) column

## Categorical bar plots 
plt.figure()
pokemon.Legendary.value_counts().plot(kind='bar')
#pokemon.Legendary.value_counts().sort_values().plot(kind='bar') # Sorteds
plt.xticks([0,1], ['Non legendary', 'Legendary']);


## Scatter plots
plt.scatter(pokemon.Attack, pokemon.Defense, marker='+')
plt.xlabel('Attack')
plt.ylabel('Defense');

## Plot 4*4 grid of subplots
fig, axes = plt.subplots(4,4, figsize=(15,15), sharey=True)
plt.subplots_adjust(hspace= 0.3)
for i in range(axes.size):
    row = df.loc[i]
    axes[i//4,i%4].hist(row.lengths.to_numpy(), bins=8, range=[0,200], color=next(colors))
    axes[i//4,i%4].set_title(row.Main_Genre)
    axes[i//4,i%4].set_xlabel('Length (minutes)')
    if not i%4:
        axes[i//4,i%4].set_ylabel('Number of movies')
```

# Groupby 
```python
# Group by name and compute mean and std
df = movies.groupby(by='name').agg(['mean', 'std'])
# Remove multi index
df.columns = df.columns.map('_'.join)


# Group by name and extract all lengths as series
df = movies.groupby(by='name').apply(lambda x: pd.Series({'lengths' : x.length}))
df = df.reset_index()
```

# Power laws
```python
# Histogram
array = plt.hist(pop_per_commune.population_Dec, bins=100, color=default_color, log=True)
plt.loglog(array[1][1:],array[0])

# Cumulative
array_cumulative = plt.hist(pop_per_commune.population_Dec,bins=100,log=True,cumulative=-1)
plt.loglog(array_cumulative[1][1:],array_cumulative[0])
```

# Series statistics
```python
df['name'].hist(bins = 50)
df['name'].describe()

```

# Tests
```python

#Ks-test
from statsmodels.stats import diagnostic

diagnostic.kstest_normal(df['name'].values, dist = 'norm') # Normal distribution ?
diagnostic.kstest_normal(df['name'].values, dist = 'exp') # Exponential distribution ? 
# --> return Tuple : (ksstat, pvalue)
# If p-value < 0.05, we can reject null hypothesis that distribution is normal/exp/ ...

#T-test
from scipy import stats
stats.ttest_ind(df.loc[df['State'] == 'New York']['IncomePerCap'], df.loc[df['State'] == 'California']['IncomePerCap'])
# --> return Ttest_indResult(statistic, pvalue)
# This is a two-sided test for the null hypothesis that the two independent samples have identical average (expected) values.
# if p is not less than 0.05 -> we cannot reject the null hypothesis that the 2 distribution are the same

# Binomial test ()
from statsmodels.stats import proportion

proportion.binom_test(count=28, nobs=100, prop=0.2, alternative='larger') #  alternative =[‘two-sided’, ‘smaller’, ‘larger’]

# Ttest ind (same means)
from statsmodels.stats import weightstats

weightstats.ttest_ind(grass_pokemon.Attack, rock_pokemon.Attack)
# This is a test for the null hypothesis that the two dustribution have the same mean
# if p is not less than 0.05 -> we cannot reject the null hypothesis that the 2 distribution have the same mean

# Given that we have fixed the significance threshold to 0.05 and that we get a p-value of ...., we reject the null hypothesis (as pvalue < significance threshold) that the two means are equals meaning that the means are different.


# Pearson correlation
from scipy import stats

stats.pearsonr(df['IncomePerCap'],df['Employed'])
# --> return Tuple : (corr, pvalue)
# If p-value < 0.05, correlation is significant

# Spearman correlation
from scipy import stats

stats.spearmanr(df['IncomePerCap'],df['Employed'])
# --> return SpearmanrResult(corr, pvalue)
# If p-value < 0.05, correlation is significant

```



# Linear regression
```python
import seaborn as sns
import statsmodels.formula.api as smf

# Linear regression plot
sns.lmplot(data=df, x='SelfEmployed',y='IncomePerCap', hue='State')


# Model of linear regression
mod = smf.ols(formula='time ~ C(high_blood_pressure) * C(DEATH_EVENT,  Treatment(reference=0)) + C(diabetes)',
              data=df)# C(DEATH_EVENT,  Treatment(reference=0)) implies that we are considering the population that did not die!
res = mod.fit()
print(res.summary())
```

# Logistic regression
```python
# FIRST --> STANDARDIZE DATA !!!!
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf

def prepare_data_for_logistic(data, dummies=True, categorical_columns=None, \
                            predicted_column=None, split=True, ratio=0.8, normalize=True):
    train_X = data
    test_X = None
    train_y = None
    test_y = None
    if predicted_column is not None:
        train_y = train_X[predicted_column]
        train_X = train_X.drop(columns=predicted_column)

    if dummies:
        train_X = pd.get_dummies(train_X, columns=categorical_columns)

    if split:
        mask = np.random.rand(len(train_X)) < ratio
        test_X = train_X[~mask].reset_index(drop=True)
        train_X = train_X[mask].reset_index(drop=True)
        if predicted_column is not None:
            test_y = train_y[~mask].reset_index(drop=True)
            train_y = train_y[mask].reset_index(drop=True)
    
    if normalize:
        means = train_X.mean()
        stddevs = train_X.std()

        train_X_std = pd.DataFrame()
        for c in train_X.columns:
            train_X_std[c] = (train_X[c]-means[c])/stddevs[c]
        if split:
            # Use the mean and stddev of the training set
            test_X_std = pd.DataFrame()
            for c in test_X.columns:
                test_X_std[c] = (test_X[c]-means[c])/stddevs[c]

    if predicted_column is None:
        if not split:
            return train_X
        return train_X, test_X
    if not split:
        return train_X, train_y
    return (train_X, train_y), (test_X, test_y)

# ----------------------------------------------------------------------------------------------------

logistic = LogisticRegression(solver='lbfgs', max_iter=10000)
logistic.fit(X_train, y_train)
y_pred = logistic.predict_proba(X_test)

# Show scores as function of threshold
scores = []
thresholds = np.arange(0,1,1/100)
for t in thresholds:
    confusion_matrix = compute_confusion_matrix(y_test.to_numpy(), y_pred, t)
    scores.append(compute_all_score(confusion_matrix, t))
scores = np.asarray(scores)
columns_score_name = ['Threshold', 'Accuracy', 'Precision P', 'Recall P', 'F1 score P', \
                                              'Precision N', 'Recall N', 'F1 score N']
threshold_score = pd.DataFrame(data=scores, columns=columns_score_name).set_index('Threshold')
threshold_score.plot()

# PRINT COEFFS (using smf)
tmp = []
for name, value in zip(X_train.columns, logistic.coef_[0]):
    tmp.append({"name": name, "value": value})
    
features_coef = pd.DataFrame(tmp).sort_values("value").set_index('name')
variables = features_coef.index # feature names
## quantifying uncertainty!
coefficients = features_coef.value
#sort them all by coefficients
# Plot coefficients
plt.subplots(figsize=(10,10))
plt.barh(features_coef.index, features_coef.value)


# -----------------------------
mod = smf.logit(formula='DEATH_EVENT ~  age + creatinine_phosphokinase + ejection_fraction + \
                        platelets + serum_creatinine + serum_sodium + \
                        C(diabetes) + C(high_blood_pressure) +\
                        C(sex) + C(anaemia) + C(smoking) + C(high_blood_pressure)', data=df)
res = mod.fit()
print(res.summary())
# PRINT COEFFS (using smf)
variables = res.params.index # feature names
## quantifying uncertainty!
coefficients = res.params.values
p_values = res.pvalues
standard_errors = res.bse.values
res.conf_int() #confidence intervals
#sort them all by coefficients
l1, l2, l3, l4 = zip(*sorted(zip(coefficients[1:], variables[1:], standard_errors[1:], p_values[1:]))) # (without intercept)
# Plot coefficients
plt.errorbar(l1, np.array(range(len(l1))), xerr= 2*np.array(l3), linewidth = 1,
             linestyle = 'none',marker = 'o',markersize= 3,
             markerfacecolor = 'black',markeredgecolor = 'black', capsize= 5)
plt.vlines(0,0, len(l1), linestyle = '--')
# Analysis:
#- Since all predictors are standardized, we can interpret in the following way:
    #- When all other predictors take mean values, an increase of age by 1 standard deviation, leads on average to an increase by 0.66 of log odds of death.
    #- When all other predictors take mean values, increase of ejection fraction by 1 standard deviation, leads on average to a  decrease by 0.83 of log odds of death.
### Interpreting log odds
#- Why log odds? remember that that's what logistic regression models.
    #- Notice that log odds are a bit difficult to interpret.
    #- If an event has probability p, it has odds 1/(1-p).
    #- This is a non-linear transformation over p. See the plot below!#
def p_to_log_odds(p):
    return np.log(p/(1-p))
def log_odds_to_p(odds):
    return np.exp(odds) / (1+ np.exp(odds))
### Interpretation of log transform of the dependent variable y
# In the first model, high_blood_pressure is associated with an additive coefficient of around -25. Thus, in the model, whenever a patient has high blood pressure we deduce -25 days out of the prediction. In the second model, high_blood_pressure is associated with an multiplicative coefficient of around -0.22. This means that, in the model, whenever a patient has high blood pressure we multiply his or her outcome by e^0.22≃0.80.
```


# Confusion matrix
```python
# Sklearn way
from sklearn.metrics import confusion_matrix
# When we have labels (Output from model.predict()):
conf_matrix = confusion_matrix(y_true, y_pred)

# When we have probabilities (Output from model.predict_proba()):
def confusion_matrix_from_proba(true_label, prediction_proba, decision_threshold=0.5):
    predict_label = (prediction_proba[:,1]>decision_threshold).astype(int)   
    return confusion_matrix(true_label, predict_label)
conf_matrix = confusion_matrix_from_proba(y_true, y_pred_proba,decision_threshold=0.5)

# -------------------------------

# By hand way
def compute_confusion_matrix(true_label, prediction_proba, decision_threshold=0.5): 
    
    predict_label = (prediction_proba[:,1]>decision_threshold).astype(int)   
                                                                                                                       
    TP = np.sum(np.logical_and(predict_label==1, true_label==1))
    TN = np.sum(np.logical_and(predict_label==0, true_label==0))
    FP = np.sum(np.logical_and(predict_label==1, true_label==0))
    FN = np.sum(np.logical_and(predict_label==0, true_label==1))
    
    confusion_matrix = np.asarray([[TP, FP],
                                    [FN, TN]])
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    [[TP, FP],[FN, TN]] = confusion_matrix
    label = np.asarray([[f'TP {TP}', f'FP {FP}'],
                        [f'FN {FN}', f'TN {TN}']])
    
    df_cm = pd.DataFrame(confusion_matrix, index=['Yes', 'No'], columns=['Positive', 'Negative']) 
    
    return sn.heatmap(df_cm, cmap='YlOrRd', annot=label, annot_kws={"size": 16}, cbar=False, fmt='')


def compute_all_score(confusion_matrix, t=0.5):
    [[TP, FP],[FN, TN]] = confusion_matrix.astype(float)
    
    accuracy =  (TP+TN)/np.sum(confusion_matrix)
    
    precision_positive = TP/(TP+FP) if (TP+FP) !=0 else np.nan
    precision_negative = TN/(TN+FN) if (TN+FN) !=0 else np.nan
    
    recall_positive = TP/(TP+FN) if (TP+FN) !=0 else np.nan
    recall_negative = TN/(TN+FP) if (TN+FP) !=0 else np.nan

    F1_score_positive = 2 *(precision_positive*recall_positive)/(precision_positive+recall_positive) if (precision_positive+recall_positive) !=0 else np.nan
    F1_score_negative = 2 *(precision_negative*recall_negative)/(precision_negative+recall_negative) if (precision_negative+recall_negative) !=0 else np.nan

    return {'threshold':t, 'accuracy':accuracy, 'precision+':precision_positive, 'recall+':recall_positive, 'f1+':F1_score_positive, 'precision-':precision_negative, 'recall-':recall_negative, 'f1-':F1_score_negative}
    #return [t, accuracy, precision_positive, recall_positive, F1_score_positive, precision_negative, recall_negative, F1_score_negative]

score = compute_all_score(confusion_matrix(y_test, y_pred))

print(f"Threshold:            {score['threshold']:.3f}\nAccuracy:             {score['accuracy']:.3f}\nPrecision (Positive): {score['precision+']:.3f}\nRecall (Positive):    {score['recall+']:.3f}\nF1 (Positive):        {score['f1+']:.3f}\nPrecision (Negative): {score['precision-']:.3f}\nRecall (Negative):    {score['recall-']:.3f}\nF1 (Negative):        {score['f1-']:.3f}\n")
```

# ROC and AUC curves
```python
logistic = LogisticRegression(solver='lbfgs')
precision = cross_val_score(logistic, X, y, cv=10, scoring="precision")
recall = cross_val_score(logistic, X, y, cv=10, scoring="recall")
# Predict the probabilities with a cross validationn
y_pred = cross_val_predict(logistic, X, y, cv=10, method="predict_proba")
# Compute the False Positive Rate and True Positive Rate
fpr, tpr, _ = roc_curve(y, y_pred[:, 1])
# Compute the area under the fpt-tpf curve
auc_score = auc(fpr, tpr)
```

# Random forest
```python
# With CV
from sklearn.ensemble import RandomForestClassifier

number_trees = [n for n in range(1, 21)]
precision_scores = []
recalls_scores = []


for nt in number_trees:
    clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=nt)
    clf.fit(X, y)
    precision = cross_val_score(clf, X, y, cv=10, scoring="precision")
    precision_scores.append(precision.mean())
    recall = cross_val_score(clf, X, y, cv=10, scoring="recall")
    recalls_scores.append(recall.mean())

fig, ax = plt.subplots(1, figsize=(6,4))

ax.plot(number_trees, precision_scores, label="Precision")
ax.plot(number_trees, recalls_scores, label="Recall")

ax.set_ylabel("Score value")
ax.set_xlabel("Number of trees")
ax.legend()

# Without CV
from sklearn.ensemble import RandomForestClassifier
n_estimators = [10, 25, 50, 100]
depth = [2, 4, 10]

(train_X,train_y), (test_X,test_y) = separate_test_train(pokemon_features, pokemon_labels)
scores=[]
for n in n_estimators:
    for d in depth:
        forest = RandomForestClassifier(n_estimators=n, max_depth=d)
        forest.fit(train_X, train_y)
        score = forest.score(test_X, test_y)
        print(f"Score: {score:.3f} with {n} estimators and a depth of {d}")
        scores.append(score)

```

# Clustering
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# ---------------------- KMEANS ----------------------
# Cluster the data with the current number of clusters
kmean = KMeans(n_clusters=k, random_state=42).fit(X)
# Plot the data by using the labels as color
ax.scatter(X[:,0], X[:,1], c=kmean.labels_, alpha=0.6)
ax.set_title(f"{n_clusters} clusters")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
# Plot the centroids
for c in kmean.cluster_centers_:
    ax.scatter(c[0], c[1], marker="+", color="red")

# Compute silhouette score
# Cluster the data and assigne the labels
labels = KMeans(n_clusters=k, random_state=10).fit_predict(X)
# Get the Silhouette score
score = silhouette_score(X, labels)

# Compute SSE
kmeans = KMeans(n_clusters=k, random_state=10).fit(X)

def plot_sse_silhouettes(df, start=2, end=11):
    scores = []
    for k in range(start, end):
        # Assign the labels to the clusters
        kmeans = KMeans(n_clusters=k, random_state=10).fit(X)
        labels = kmeans.predict(X)
        # Get the Silhouette score
        score = silhouette_score(X, labels)
        scores.append({"Number of clusters k": k, "sil": score, "sse": kmeans.inertia_})

    scores = pd.DataFrame(scores).set_index('Number of clusters k')
    # Plot the data
    fig, ax1 = plt.subplots(1,1)
    scores.sil.plot(ax=ax1, label='Silhouette score')
    ax2 = ax1.twinx()
    scores.sse.plot(ax=ax2, label='Sum of Squared Errors', color='green')
    ax1.set_ylabel('Silhouette score')
    ax2.set_ylabel('Sum of Squared Errors')
    ax1.legend(loc=3)
    ax2.legend(loc=0)
    plt.show()

plot_sse_silhouettes(X)

# ---------------------- DBSCAN ----------------------
# Create a list of eps
eps_list = np.linspace(0.05, 0.15, 16)

# Compute number of row and columns
fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharey=True, sharex=True)

for i in range(0, axs.size):
    eps = eps_list[i]
    labels = DBSCAN(eps=eps).fit_predict(X_moons)
    axes[i//4,i%4].scatter(X_moons[:,0], X_moons[:,1], c=labels, alpha=0.6)
    axes[i//4,i%4].set_title("eps = {:.3f}".format(eps))
    
plt.tight_layout()

```

# Dimensionality reduction
```python
X_norm = (X - X.mean()) / X.std()

# TSNE
X_reduced_tsne = TSNE(n_components=2, random_state=0).fit_transform(X_norm)

# Plot tSNE predicted vs real 
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax[0].scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=pred_labels, alpha=0.6)
ax[1].scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=labels, alpha=0.6)
plt.show()

# PCA
X_reduced_pca = PCA(n_components=2).fit(X_norm).transform(X_norm)


```

# Graphs
```python

# Undirected graph
familiarity_G = nx.Graph()
familiarity_G.add_nodes_from(recurrent_characters)

# Directed graph
gossip_G = nx.DiGraph()
gossip_G.add_nodes_from(recurrent_characters)

for _, value in character_by_scene.iteritems():
    if len(value) > 1:
        for character in value[1:]:
            if familiarity_G.has_edge(value[0], character):
                familiarity_G[value[0]][character]['weight'] += 1
            else:
                familiarity_G.add_edge(value[0], character, weight=1)


for _, row in lines_.iterrows():
    for mention in row['mentions']:
        if mention not in character_by_scene[row['Scene']]:
            if gossip_G.has_edge(row['Character'], mention):
                gossip_G[row['Character']][mention]['weight'] += 1
            else:
                gossip_G.add_edge(row['Character'], mention, weight=1)

# Easy solution
nx.draw_shell(familiarity_graph, with_labels=True, alpha = 0.6, node_size=2000, width=weights)
# Helper function for visualizing the graph
def visualize_graph(G, with_labels=True, k=None, alpha=1.0, node_shape='o'):
    #nx.draw_spring(G, with_labels=with_labels, alpha = alpha)
    pos = nx.spring_layout(G, k=k)
    if with_labels:
        lab = nx.draw_networkx_labels(G, pos, labels=dict([(n, n) for n in G.nodes()]))
    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='g', node_shape=node_shape)
    plt.axis('off')

```



# PySpark
```python
###### BEGINNING PYSPARK ######
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.executor.memory', '12g'),  # find
                                   ('spark.driver.memory','4g'), # your
                                   ('spark.driver.maxResultSize', '2G') # setup
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# create the context
sc = spark.sparkContext

scripts = sc.textFile("data/all_scripts.txt")

# FIX for Spark 2.x
locale = sc._jvm.java.util.Locale
locale.setDefault(locale.forLanguageTag("en-US"))

###### END BEGINNING PYSPARK ######

# Bag of words
def split_clean(x):
    a, b = x.split(':', 1)
    return a, b.strip()

char_line_rdd = scripts.filter(lambda x: not x.startswith('>')).map(split_clean).filter(lambda x: x[0] in rec_char).persist()
vocabulary = dict(char_line_rdd.flatMap(lambda x: x[1].split(' ')).distinct().zipWithIndex().collect())

def create_bow(x, vocabulary):
    counter = Counter(x.split(' '))
    data = []
    col_ind = []
    for k, v in count.items():
        data.append(v)
        col_ind.append(vocabulary[k])
    return csr_matrix((data, (np.zeros_like(data), col_ind)), shape=(1, len(vocabulary)))

voc_bc = sc.broadcast(vocabulary)
char_line_bow = char_line_rdd.map(lambda x: (x[0], create_bow(x[1], voc_bc.value))).persist()

agg_char_bow = char_line_bow.reduceByKey(lambda x, y: x+y)
agg_char_bow.collect()

#----------------------------------------------------------------------------
# Pokemons
pokemon_spark = spark.read.csv("pokemon.csv", header=True)
combats_spark = spark.read.csv("combats.csv", header=True)

counter_combats = combats_spark.groupby('Winner').count().cache()
pokemon_joined = counter_combats.join(pokemon_spark, counter_combats.Winner == pokemon_spark.pid)
top10_pokemon = pokemon_joined.sort(desc('count')).limit(10).toPandas()['Name'].values
top10_pokemon
```



# Bootstraping CIs
```python
num_records = len(y)
bootstrap_errors = []

for i in range(1000):
    # Compute training set
    train_indices = np.random.choice(range(num_records), num_records, replace=True)
    X_train, y_train = X.loc[train_indices], y.loc[train_indices] # Normalized data
    # Compute test set
    test_indices = np.setdiff1d(range(num_records), train_indices)
    X_test, y_test = X.loc[test_indices], y.loc[test_indices]  # Normalized data

    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    bootstrap_errors.append(mse)

bootstrap_errors_sorted = np.sort(bootstrap_errors)
print(f'95% CIs of the model: [{bootstrap_errors_sorted[25]:.3f}, {bootstrap_errors_sorted[975]:.3f}]')

mean_error = bootstrap_errors.mean()
deviation = bootstrap_errors_sorted - mean_error

xpos = [0, 1]
error_means = [mean_error, mean_error_2]
yerr = [[deviation[25], deviation[975]], [deviation_2[25], deviation_2[975]]]
fig, ax = plt.bar(xpos, error_means, yerr=yerr, align='center', alpha=0.5, ecolor='k', capsize=10)
plt.xticks(xpos, ('First Model', 'Second Model'))
plt.ylabel('Error')
plt.title('Comparison of the Two Models')

```


# Matching (Propensity scores)
```python
def get_similarity(propensity_score1, propensity_score2):
    '''Calculate similarity for instances with given propensity scores'''
    return 1-np.abs(propensity_score1-propensity_score2)


# Separate the treatment and control groups
treatment_df = lalonde_data[lalonde_data['treat'] == 1]
control_df = lalonde_data[lalonde_data['treat'] == 0]

# Create an empty undirected graph
G = nx.Graph()

# Loop through all the pairs of instances
for control_id, control_row in control_df.iterrows():
    for treatment_id, treatment_row in treatment_df.iterrows():

        # Calculate the similarity 
        similarity = get_similarity(control_row['Propensity_score'],
                                    treatment_row['Propensity_score'])

        # Add an edge between the two instances weighted by the similarity between them
        G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

# Generate and return the maximum weight matching on the generated graph
matching = nx.max_weight_matching(G)
```


# NLP
```python
# Exam 2019
from string import punctuation

EXCLUDE_CHARS = set(punctuation).union(set('’'))
def simple_tokeniser(text):
    return text.split()

with open("helpers/stopwords.txt") as f:
    stop_words = list(map(lambda x: x[:-1], f.readlines()))
tfidf = TfidfVectorizer(stop_words=stop_words, tokenizer=simple_tokeniser, min_df=2)
train_vectors = tfidf.fit_transform(train_set["Line"])
test_vectors = tfidf.transform(test_set["Line"])

words_for_chars = pd.concat([pd.Series(row["Character"], row['Line'].split(' '))
                             for _, row in lines.iterrows()]).reset_index()
words_for_chars.columns = ["Word", "Character"]

words_for_chars = words_for_chars.groupby("Word")["Character"].apply(set)
sheldon_words = words_for_chars[words_for_chars.apply(lambda x: ("Sheldon" in x) and (len(x) == 1))].index

def contains_sheldon_words(line):
    for word in sheldon_words:
        if word in line:
            return True
    return False
test_pred = test_set["Line"].apply(contains_sheldon_words)
test_true = test_set["Character"] == "Sheldon"

svd = TruncatedSVD(n_components=25)
train_svd = svd.fit_transform(train_vectors)
test_svd = svd.transform(test_vectors)

model = LogisticRegressionCV(cv=10)
train_labels = train_set["Character"] == "Sheldon"
model.fit(train_svd, train_labels)
test_pred = model.predict(test_svd)
train_pred = model.predict(train_svd)

# -------------------------------------

nlp = spacy.load('en')

### Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
vs = analyzer.polarity_scores(sentence)

print(sentence, '\n')
print('Negative sentiment:',vs['neg'])
print('Neutral sentiment:',vs['neu'])
print('Positive sentiment:',vs['pos'])
print('Compound sentiment:',vs['compound'])

nlp = spacy.load('en')
doc = nlp(text)
positive_sent = []
for sent in doc.sents:
    positive_sent.append(analyzer.polarity_scores(sent.text)['pos'])
plt.hist(positive_sent,bins=15)
plt.xlim([0,1])
plt.ylim([0,8000])
plt.xlabel('Positive sentiment')
plt.ylabel('Number of sentences')


### Document classification
def get_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

chunks = list()
size = 50 # how many sentences per chunk/page
# Convert to chunks of same size (text of n sentences)
chunks_of_sents = [x for x in get_chunks(sentences,size)] # this is a list of lists of sentences, which are a list of tokens
# regroup so to have a list of chunks which are strings
for c in chunks_of_sents:
    grouped_chunk = list()
    for s in c:
        grouped_chunk.extend(s)
    chunks.append(" ".join(grouped_chunk))

# initialize and specify minumum number of occurences to avoid untractable number of features
vectorizer = CountVectorizer(min_df = 2)

#create bag of words features
X = vectorizer.fit_transform(chunks)

# Alternative text representation: word emdeddings, pretrained on intergraded in Spacy (300-dimensional word vectors trained on Common Crawl with GloVe.)
X = nlp(sentence).vector


### Topic detection 
# use same chunks
STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

processed_docs = list()
for doc in nlp.pipe(chunks, n_threads=5, batch_size=10):

    # Process document using Spacy NLP pipeline.
    ents = doc.ents  # Named entities

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Remove common words from a stopword list and keep only words of length 3 or more.
    doc = [token for token in doc if token not in STOPWORDS and len(token) > 2]

    # Add named entities, but only if they are a compound of more than word.
    doc.extend([str(entity) for entity in ents if len(entity) > 1])

    processed_docs.append(doc)
docs = processed_docs
del processed_docs

# Add bigrams too
from gensim.models.phrases import Phrases

# Add bigrams to docs (only ones that appear 15 times or more).
bigram = Phrases(docs, min_count=15)

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)


# Create a dictionary representation of the documents, and filter out frequent and rare words.
dictionary = Dictionary(docs)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.5
min_wordcount = 5
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]
#MmCorpus.serialize("models/corpus.mm", corpus)

print('Number of unique tokens: %d' % len(dictionary))
print('Number of chunks: %d' % len(corpus))

# models
params = {'passes': 10, 'random_state': 1}
base_models = dict()
model = LdaMulticore(corpus=corpus, num_topics=4, id2word=dictionary, workers=6,
                passes=params['passes'], random_state=params['random_state'])
model.show_topics(num_words=5)
model.show_topic(1,20)
```