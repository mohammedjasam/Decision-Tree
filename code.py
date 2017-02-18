from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
import IPython


"""
Data Engineering and Analysis
"""
#Load the dataset

AH_data = pd.read_csv("data.csv", encoding = "ISO-8859-1")

data_clean = AH_data.dropna()

print(data_clean.dtypes)
print(data_clean.describe())

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = data_clean[['kids',	'say',	'things',	'president',	'diet',	'fitnessliving',	'wellparenting',	'tv',	'search',	'crime',	'east',	'digital',	'shows',	'kelly',	'wallace',	'november',	'chat',	'facebook',	'messenger',	'find',	'world',	'many',	'want',	'videos',	'must',	'watch',	'run',	'according',	'large',	'family',	'life',	'read',	'parents',	'twitter',	'school',	'interest',	'much',	'also',	'absolutely',	'ever',	'office',	'land',	'thing',	'go',	'could',	'told',	'america',	'march',	'presidential',	'campaign',	'end',	'million',	'order',	'get',	'money',	'first',	'take',	'time',	'might',	'american',	'times',	'way',	'election',	'children',	'inc',	'country',	'leader',	'free',	'high',	'thought',	'know',	'good',	'candidates',	'definitely',	'part',	'white',	'house',	'four',	'years',	'vice',	'top',	'young',	'really',	'call',	'public',	'service',	'show',	'beyond',	'vote',	'artist',	'model',	'someone',	'cancer',	'helping',	'animals',	'asked',	'make',	'better',	'place',	'latest',	'share',	'comments',	'health',	'hillary',	'clinton',	'female',	'even',	'actually',	'chance',	'lady',	'content',	'pay',	'card',	'save',	'enough',	'reverse',	'risk',	'paid',	'partner',	'cards',	'around',	'next',	'generation',	'big',	'network',	'system',	'rights',	'reserved',	'terms',	'mexican',	'meeting',	'trump',	'january',	'mexico',	'different',	'route',	'border',	'immigrants',	'trying',	'donald',	'wall',	'billion',	'signs',	'executive',	'actions',	'building',	'along',	'southern',	'nowstory',	'believe',	'fruitless',	'thursday',	'set',	'week',	'plan',	'tuesday',	'something',	'recently',	'wednesday',	'needed',	'tweet',	'trade',	'nafta',	'massive',	'@realdonaldtrump',	'jobs',	'companies',	'remarks',	'gathering',	'congressional',	'republicans',	'planned',	'together',	'unless',	'senate',	'gop',	'lawmakers',	'security',	'national',	'problem',	'illegal',	'immigration',	'see',	'need',	'statement',	'back',	'two',	'leaders',	'last',	'year',	'days',	'called',	'action',	'begin',	'process',	'announced',	'move',	'level',	'foreign',	'representatives',	'come',	'since',	'officials',	'including',	'staff',	'minister',	'government',	'team',	'car',	'department',	'homeland',	'work',	'help',	'united',	'states',	'forces',	'number',	'officers',	'visit',	'try',	'able',	'related',	'monday',	'migrants',	'home',	'city',	'conversation',	'made']]

targets = data_clean.SITE

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

print(sklearn.metrics.confusion_matrix(tar_test,predictions))
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
#Image(graph.create_png())
Image(graph.write_pdf("graph.pdf"))
