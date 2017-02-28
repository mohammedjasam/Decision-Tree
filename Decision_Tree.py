### Scripted by Mohammed Jasam ###
### mnqnd@mst.edu


import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
import IPython
import matplotlib.pyplot as plt
import csv
from sklearn import tree
from io import StringIO
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

test=[]
train=[]

# Running the 5 Fold Cross Validation!
def getValues(count):
    if count==1:
        for i in range(100):
            if 0<=i<20:
                test.append(i)
            else:
                train.append(i)
    if count==2:
        for i in range(100):
            if 20<=i<40:
                test.append(i)
            else:
                train.append(i)
    if count==3:
        for i in range(100):
            if 40<=i<60:
                test.append(i)
            else:
                train.append(i)
    if count==4:
        for i in range(100):
            if 60<=i<80:
                test.append(i)
            else:
                train.append(i)
    if count==5:
        for i in range(100):
            if 80<=i<100:
                test.append(i)
            else:
                train.append(i)
    return test,train

# Loop runs 5 times to simulate the 5 Fold Cross Validation!
print('\n')
print("                       THE DECISION TREE ANALYSIS")
for x in range(5):
    test=[]
    train=[]
    # print(testing)
    test,train=getValues(x+1)
    pred_train = predictors.ix[train]
    tar_train = targets[train]
    pred_test = predictors.ix[test]
    tar_test = targets[test]

    #Build model on training data
    classifier=DecisionTreeClassifier()
    classifier=classifier.fit(pred_train,tar_train)
    predictions=classifier.predict(pred_test)
    #Displaying the decision tree
    out = StringIO()
    tree.export_graphviz(classifier, out_file=out)
    import pydotplus
    graph=pydotplus.graph_from_dot_data(out.getvalue())

    #Generate Graphs for the Decision Classifier
    millis = int(round(time.time() * 1000))  # Generate time system time in milliseconds
    Image(graph.write_pdf(str(i)+".pdf"))

    print("===========================================================")
    print('Iteration #'+str(x+1)+'\n')
    #Calculate accuracy
    l.append(sklearn.metrics.accuracy_score(tar_test, predictions)*100)
    print("Accuracy Score is "+str(l[i-1])+ '%')
    i+=1

    print(sklearn.metrics.confusion_matrix(tar_test,predictions))
    print(classification_report(tar_test,predictions))
#Generating a list for accuracy
l=[int(x) for x in l]
with open("DecisionTree_Accuracy.csv", "w") as fp_out:
    writer = csv.writer(fp_out, delimiter=",")
    writer.writerow(l)
a=0
for i in range(len(l)):
    a+=l[i]
# print(a)
print("===========================================================")
print("===========================================================")
print('The average accuracy after 5 Fold Cross Validation is: '+str(int(a/len(l)))+"%")
print("===========================================================")
print("===========================================================")


### VISUALIZING THE ACCURACY DATA ###
# data to plot
n_groups = 5

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects2 = plt.bar(index + bar_width, l, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Decision Tree')

plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Accuracy by Decision Tree')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5'))
plt.legend()

plt.tight_layout()
plt.show()

# Merging the Decision Tree graphs to clean the directory!
pdfs = ['1.pdf', '2.pdf', '3.pdf', '4.pdf','5.pdf']
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(open(pdf, 'rb'))
with open('DecisionTrees.pdf', 'wb') as fout:
    merger.write(fout)


sys.exit()
