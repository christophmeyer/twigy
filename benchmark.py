import numpy as np
import twigy
import pandas as pd
from sklearn import tree
from sklearn import ensemble
import time
from plotnine import ggplot, geom_point, geom_line, aes, geom_errorbar, position_dodge, labs


titanic_data = pd.read_csv("./test/testdata/titanic_data.csv")
X = titanic_data.drop(columns=['Survived']).to_numpy()
y = np.reshape(titanic_data[['Survived']].to_numpy(), (-1))

timing_data = {'implementation': [], 'threads': [], 'timing': []}
n_samples = 10
n_burn_in = 3

for n_threads in range(1, 5):
    for n in range(n_samples+n_burn_in):
        start_time = time.time()

        forest_clf = twigy.RandomForestClassifier(
            n_estimators=10000,
            impurity_measure=twigy.ImpurityMeasure.gini,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=5,
            n_threads=n_threads)

        forest_clf.build_forest(X, y)
        current_timing = (time.time() - start_time)
        if n >= n_burn_in:
            timing_data['implementation'].append('twigy 0.0.1')
            timing_data['threads'].append(n_threads)
            timing_data['timing'].append(current_timing)

for n_threads in range(1, 5):
    for n in range(n_samples+n_burn_in):
        start_time = time.time()

        sklearn_forest = ensemble.RandomForestClassifier(n_estimators=10000,
                                                         criterion="gini",
                                                         min_samples_leaf=1,
                                                         max_depth=5,
                                                         min_samples_split=2,
                                                         max_features=5,
                                                         n_jobs=n_threads)

        sklearn_forest.fit(X, y)
        current_timing = (time.time() - start_time)
        if n >= n_burn_in:
            timing_data['implementation'].append('scikit-learn 0.23.1')
            timing_data['threads'].append(n_threads)
            timing_data['timing'].append(current_timing)


df = pd.DataFrame(data=timing_data)
df = df.groupby(['implementation', 'threads']).agg(['mean', 'std']).reset_index()
df.columns = ['Implementation', 'threads', 'mean', 'std']
print(df)

df['error_min'] = df['mean'] - df['std']
df['error_max'] = df['mean'] + df['std']
p = (ggplot(df, aes(x='threads', y='mean', group='Implementation', color='Implementation')) +
     geom_line() +
     geom_point() +
     geom_errorbar(aes(ymin='error_min', ymax='error_max'), width=.2,
                   position=position_dodge(0.05)) +
     labs(x="Number of threads", y="timing [s]"))

p.save(filename='benchmark.png')
