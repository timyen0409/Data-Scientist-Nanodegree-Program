import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from IPython import display
import warnings

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('CategorizedMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.iloc[:, 4:].columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    # nromalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    result = []
    for token in tokens:
        if token not in stop_words:
            result.append(lemmatizer.lemmatize(token))
            
    return result


def build_model():
     # text processing and model pipeline
    pipeline = Pipeline([("text_pipeline", 
                        Pipeline([("vect", CountVectorizer(tokenizer=tokenize)),
                                  ("tfidf", TfidfTransformer())])),
                                  ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=4)))])

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__criterion' : ('gini', 'entropy'),
        'clf__estimator__min_samples_leaf' : (1, 5)
        }

    # create gridsearch object and return as final model pipeline
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    
    res_lis = []
    avg_lis = []
    y_pred = model.predict(X_test)
    
    for i in range(y_pred.shape[1]):
        res_lis.append(classification_report(Y_test.iloc[:, i].values.tolist(), list(y_pred[:, i])))
        avg_lis.append(re.findall('\d*\.\d+|\d+', res_lis[i].split("avg / total", 1)[-1]))
        avg_lis[i].pop()
        array_avg = np.array(avg_lis).astype(np.float)
        print('--')
        print(category_names[i])
        print("Precision:{0:0.2f}  Recall:{1:0.2f}  f1-score:{2:0.2f}".format(array_avg[i][0], array_avg[i][1], array_avg[i][2]))     
        
    avg = np.mean(array_avg, axis = 0)
    print('Overall average')
    print("Precision:{0:0.2f}  Recall:{1:0.2f}  f1-score:{2:0.2f}".format(avg[0],avg[1],avg[2]))

    return


def save_model(model, model_filepath):
    
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)
    return

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
