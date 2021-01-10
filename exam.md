## Cheat sheet examen ADA

### Artur Jesslen


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, auc, roc_curve
import seaborn as sns
%matplotlib inline
```

#### Utils
```python
# Drop rows with NA
df = df.dropna()

# Split train/test dataset
def split_set(data_to_split, ratio=0.8):
    mask = np.random.rand(len(data_to_split)) < ratio
    return (data_to_split[mask].reset_index(drop=True), data_to_split[~mask].reset_index(drop=True))

# Convert Categorical (Male/Female) to numerical variables
X = pd.get_dummies(X)

# Normalize dataframe 
normalized_df = (df - df.mean()) / df.std()

# Min-Max normalization
normalized_df = (df - df.min()) / (df.max() - df.min())
```

#### Create color generator (20 colors)
```python
from itertools import cycle
colors = cycle([plt.get_cmap('tab20')(i/20) for i in range(20)])
default_color = next(colors)
```

#### Plot 4*4 grid of subplots
```python
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

#### Plot : barplot + density 
```python
ax = sns.distplot(treated['re78'], hist=True, label='treated');
ax = sns.distplot(control['re78'], hist=True, label='control')
ax.set(title='Income distribution comparison in 1978, after matching',xlabel='Income 1978', ylabel='Income density')
plt.legend()
plt.show()
```

#### Groupby 
```python
# Group by name and compute mean and std
df = movies.groupby(by='name').agg(['mean', 'std'])
# Remove multi index
df.columns = df.columns.map('_'.join)


# Group by name and extract all lengths as series
df = movies.groupby(by='name').apply(lambda x: pd.Series({'lengths' : x.length}))
df = df.reset_index()
```

#### Power laws
```python
# Histogram
array = plt.hist(pop_per_commune.population_Dec, bins=100, color=default_color, log=True)
plt.loglog(array[1][1:],array[0])

# Cumulative
array_cumulative = plt.hist(pop_per_commune.population_Dec,bins=100,log=True,cumulative=-1)
plt.loglog(array_cumulative[1][1:],array_cumulative[0])
```

#### Series statistics
```python
df['name'].hist(bins = 50)
df['name'].describe()

```

#### Tests
```python
from statsmodels.stats import diagnostic
from statsmodels.stats import proportion
from scipy import stats

#K-test
diagnostic.kstest_normal(df['name'].values, dist = 'norm') # Normal distribution ?
diagnostic.kstest_normal(df['name'].values, dist = 'exp') # Exponential distribution ? 
# --> return Tuple : (ksstat, pvalue)
# If p-value < 0.05, we can reject null hypothesis that distribution is normal/exp/ ...

#T-test
stats.ttest_ind(df.loc[df['State'] == 'New York']['IncomePerCap'], df.loc[df['State'] == 'California']['IncomePerCap'])
# --> return Ttest_indResult(statistic, pvalue)
# This is a two-sided test for the null hypothesis that the two independent samples have identical average (expected) values.
# if p is not less than 0.05 -> we cannot reject the null hypothesis that the 2 distribution are the same

# Binomial test
proportion.binom_test(count=28, nobs=100, prop=0.2, alternative='larger') #  alternative =[‘two-sided’, ‘smaller’, ‘larger’]


# Pearson correlation 
stats.pearsonr(df['IncomePerCap'],df['Employed'])
# --> return Tuple : (corr, pvalue)
# If p-value < 0.05, correlation is significant

# Spearman correlation 
stats.spearmanr(df['IncomePerCap'],df['Employed'])
# --> return SpearmanrResult(corr, pvalue)
# If p-value < 0.05, correlation is significant

```



#### Linear regression
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

#### Confusion matrix
```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
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

#### Logistic regression
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
y_pred = res.predict_proba(X_test)

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
for name, value in zip(X_train.columns, res.coef_[0]):
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

#### ROC and AUC curves
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

#### Random forest
```python
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
```

#### Clustering
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

#### Dimensionality reduction
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

#### Create graph
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


#### Example
```python


```
#### Example
```python


```
#### Example
```python


```