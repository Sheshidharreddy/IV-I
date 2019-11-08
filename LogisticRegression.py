@anvil.server.callable
def algorithm1():
    from sklearn.linear_model import LogisticRegression

    mdl = LogisticRegression(C = 1e20, solver = 'lbfgs', multi_class = 'auto')

    mdl.fit(X_train, Y_train)

    Y_pred1 = mdl.predict(X_test)
    
    return Y_pred1

Y_pred1 = algorithm1()
print(Y_pred1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

c_mat = confusion_matrix(Y_test,Y_pred1)
print(c_mat)

accuracy = accuracy_score(Y_test,Y_pred1) * 100 
print(accuracy, "%")
import seaborn as sns

hmap = sns.heatmap(c_mat, annot = True, fmt = 'd', cmap="YlGnBu")
print(hmap)

import anvil.mpl_util
@anvil.server.callable
def visual1():
    import seaborn as sns
    hmap = sns.heatmap(c_mat, annot = True, fmt = 'd', cmap="YlGnBu")
    return anvil.mpl_util.plot_image(), str(accuracy)+"%"
