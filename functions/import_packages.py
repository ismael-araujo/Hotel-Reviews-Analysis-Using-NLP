def import_packages():
    # Importing Packages
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
	import os

	# NLP Packages
	import nltk 
	from nltk.corpus import stopwords
	from textblob import TextBlob 
	from textblob import Word
	import re
	import string

	# WordCloud
	from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

	# Sklearn Packages
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction import text 
	from sklearn import metrics
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import mean_squared_error, precision_score, f1_score, confusion_matrix, accuracy_score
	from sklearn.model_selection import train_test_split
	from sklearn.utils import resample
	from sklearn.linear_model import LogisticRegression

	# ImbLearn Packages
	from imblearn.over_sampling import SMOTE

	# Pandas Settings
	pd.set_option('display.max_columns', 10000)
	pd.set_option('display.max_rows', 100)

	# Solve warnings
	import warnings
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	warnings.filterwarnings("ignore", category=FutureWarning)