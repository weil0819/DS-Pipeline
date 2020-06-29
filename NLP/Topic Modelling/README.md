
## Summary for Topic Modelling problem  

### Text Representation:
1. `tf-idf` vector
2. `token counts` vector

### Models: 
1. Logistic Regression
2. (Multinomial) Naive Bayes
3. Linear Support Vector Machine
4. Random Forest
5. Keras

### Libraries:
#### Math Libraries:
- `import pandas as pd`
- `import numpy as np`

#### Plot Libraries:
- `import seaborn as sns`
- `import matplotlib.pyplot as plt`
- `from IPython.display import display`
- `import warnings`
- `warnings.filterwarnings("ignore")`

#### Feature Extraction/Selection:
- `from sklearn.feature_selection import chi2`
- `from sklearn.feature_extraction.text import CountVectorizer`
- `from sklearn.feature_extraction.text import TfidfVectorizer`

#### Model Selection:
- `from sklearn.pipeline import Pipeline`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.model_selection import cross_val_score`
- `from sklearn.naive_bayes import MultinomialNB`
- `from sklearn.linear_model import LogisticRegression`
- `from sklearn.ensemble import RandomForestClassifier`
- `from sklearn.svm import LinearSVC`

#### Metric Libraries:
- `from sklearn.metrics import confusion_matrix`
- `from sklearn.metrics import classification_report`
- `from sklearn.metrics import accuracy_score`

#### NLP Libraries:
- `import nltk`
- `from nltk.corpus import stopwords`
- `import spacy`
- `from spacy.lang.en import English`

### Ref:
> Topic Modelling in Python with NLTK and Gensim  
https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21  


> When Topic Modeling is Part of the Text Pre-processing  
https://towardsdatascience.com/when-topic-modeling-is-part-of-the-text-pre-processing-294b58d35514  


