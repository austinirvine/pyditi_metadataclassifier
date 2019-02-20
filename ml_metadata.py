import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('cancer_res_csv.csv')
# plt.hist(df['kurtosis_r_fc'])
# plt.show()
# sns.pairplot(df, hue='cancer')
# plt.show()

all_inputs = df[['skews_r_f','skews_r_fc','skews_g_f','skews_g_fc', 'skews_b_f','skews_b_fc','kurtosis_r_f','kurtosis_r_fc','kurtosis_g_f','kurtosis_g_fc', 'kurtosis_b_f','kurtosis_b_fc']].values
all_classes = df['cancer'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.8, random_state=1)

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs,train_classes)
res = dtc.score(test_inputs,test_classes)
print(res)