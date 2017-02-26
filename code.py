from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
import IPython
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO
from IPython.display import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyPDF2 import PdfFileMerger
import os
import subprocess
"""
Data Engineering and Analysis
"""
#Load the dataset
AH_data = pd.read_csv("data.csv", encoding = "ISO-8859-1")
data_clean = AH_data.dropna()

"""
Modeling and Prediction
"""
#Split Predictors and targets

predictors = data_clean[['kids',	'say',	'things',	'president',	'diet',	'fitnessliving',	'wellparenting',	'tv',	'search',	'crime',	'east',	'digital',	'shows',	'kelly',	'wallace',	'november',	'chat',	'facebook',	'messenger',	'find',	'world',	'many',	'want',	'videos',	'must',	'watch',	'run',	'according',	'large',	'family',	'life',	'read',	'parents',	'twitter',	'school',	'interest',	'much',	'also',	'absolutely',	'ever',	'office',	'land',	'thing',	'go',	'could',	'told',	'america',	'march',	'presidential',	'campaign',	'end',	'million',	'order',	'get',	'money',	'first',	'take',	'time',	'might',	'american',	'times',	'way',	'election',	'children',	'inc',	'country',	'leader',	'free',	'high',	'thought',	'know',	'good',	'candidates',	'definitely',	'part',	'white',	'house',	'four',	'years',	'vice',	'top',	'young',	'really',	'call',	'public',	'service',	'show',	'beyond',	'vote',	'artist',	'model',	'someone',	'cancer',	'helping',	'animals',	'asked',	'make',	'better',	'place',	'latest',	'share',	'comments',	'health',	'hillary',	'clinton',	'female',	'even',	'actually',	'chance',	'lady',	'content',	'pay',	'card',	'save',	'enough',	'reverse',	'risk',	'paid',	'partner',	'cards',	'around',	'next',	'generation',	'big',	'network',	'system',	'rights',	'reserved',	'terms',	'mexican',	'meeting',	'trump',	'january',	'mexico',	'different',	'route',	'border',	'immigrants',	'trying',	'donald',	'wall',	'billion',	'signs',	'executive',	'actions',	'building',	'along',	'southern',	'nowstory',	'believe',	'fruitless',	'thursday',	'set',	'week',	'plan',	'tuesday',	'something',	'recently',	'wednesday',	'needed',	'tweet',	'trade',	'nafta',	'massive',	'@realdonaldtrump',	'jobs',	'companies',	'remarks',	'gathering',	'congressional',	'republicans',	'planned',	'together',	'unless',	'senate',	'gop',	'lawmakers',	'security',	'national',	'problem',	'illegal',	'immigration',	'see',	'need',	'statement',	'back',	'two',	'leaders',	'last',	'year',	'days',	'called',	'action',	'begin',	'process',	'announced',	'move',	'level',	'foreign',	'representatives',	'come',	'since',	'officials',	'including',	'staff',	'minister',	'government',	'team',	'car',	'department',	'homeland',	'work',	'help',	'united',	'states',	'forces',	'number',	'officers',	'visit',	'try',	'able',	'related',	'monday',	'migrants',	'home',	'city',	'conversation',	'made']]
# print(predictors)
targets = data_clean.SITE
# print(targets)

#Perform 5 fold cross validation
kf = KFold(n_splits=5)
# fold = 0
i=1
l=[] #list to store the accuracy values
for training, testing in kf.split(targets):
    pred_train = predictors.ix[training]
    tar_train = targets[training]
    pred_test = predictors.ix[testing]
    tar_test = targets[testing]

    #Build model on training data
    classifier=DecisionTreeClassifier()
    classifier=classifier.fit(pred_train,tar_train)
    predictions=classifier.predict(pred_test)
    print(sklearn.metrics.confusion_matrix(tar_test,predictions))

    #Displaying the decision tree
    out = StringIO()
    tree.export_graphviz(classifier, out_file=out)
    import pydotplus
    graph=pydotplus.graph_from_dot_data(out.getvalue())

    #Generate Graphs for the Decision Classifier
    millis = int(round(time.time() * 1000))  # Generate time system time in milliseconds
    Image(graph.write_pdf(str(i)+".pdf"))

    #Calculate accuracy
    l.append(sklearn.metrics.accuracy_score(tar_test, predictions)*100)
    print("Accuracy Score is "+str(l[i-1]))
    i+=1
    #  print(sklearn.metrics.accuracy_score(tar_test, predictions)*100)
    # l.append(sklearn.metrics.accuracy_score(tar_test, predictions)*100)
    # f1_score(y_test, y_pred, average="macro")

#Generating a list for accuracy
l=[int(x) for x in l]
a=0
for i in range(len(l)):
    a+=l[i]
print('The average accuracy after 5 Fold Cross Validation is '+str(a/len(l)))
objects=('Fold 1','Fold 2','Fold 3','Fold 4','Fold 5')
y_pos = np.arange(len(objects))
plt.bar(y_pos, l)
plt.show()

pdfs = ['1.pdf', '2.pdf', '3.pdf', '4.pdf','5.pdf']
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(open(pdf, 'rb'))
with open('DecisionTrees.pdf', 'wb') as fout:
    merger.write(fout)


# subprocess.call('python pdf.py ',shell=True)
sys.exit()
