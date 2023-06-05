import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV

from mlxtend.preprocessing import DenseTransformer

from sklearn.decomposition import PCA

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score, mean_squared_error, make_scorer, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score

from sklearn.naive_bayes import MultinomialNB, CategoricalNB, BernoulliNB, ComplementNB, GaussianNB

from pyfiglet import Figlet

from itertools import *

from PIL import Image

import time

import pickle

import warnings

warnings.filterwarnings('ignore')

def menu():
    print('Выберите действие: \n' + 
          '1 --> Загрузить файл\n' + 
          '2 --> Выбрать алгоритм\n' + 
          '3 --> Выход')
    
    user_input = input()
    match user_input:
        case '1':
            global using_data, data, data_test, file_path
            
            file_path = input('Введите абсолютный путь до файла: ')
            data = pd.read_csv(file_path, engine='python').sample(frac=1)
            data_test = pd.read_csv(file_path, skipfooter=1600, engine='python').sample(frac=1)
            using_data = pd.read_csv(file_path, engine='python').sample(frac=1)
            using_data = pd.concat([using_data, data_test]).drop_duplicates(keep=False)
            print(using_data.head())
            menu()
            
        case '2':
            try:
                using_data
            except:
                print('Вы не выбрали файл')
                return menu()
            global answer, number_of_algorithm, answers
            number_of_algorithm = int(input('Введите количество алгоритмов: '))
            if number_of_algorithm > 4:
                print('Количество алгоритмов не может быть больше 4, введите заново')
                return menu()
            
            if number_of_algorithm != 1:
                answer = algorithm_list(number_of_algorithm, file_path, using_data)
                
                save_or_not = input('Save results?\n' + 
                                    '1 --> Yes\n' + 
                                    '2 --> No\n')
                
                match save_or_not:
                    case '1':
                        answer.to_csv('answers.csv', index=False)
                    case '2':
                        pd.set_option('display.max_rows', None)
                        print(answer)
                        
            else:
                answers = pd.DataFrame(columns=['Command', 'Classifier Answer', 'True Answer'])
                model_chose = input('Введите название алгоритма: \n' + 
                                    '1 --> Байесовский классификатор \n' + 
                                    '2 --> Логистическая регрессия \n' + 
                                    '3 --> Бэггинг \n' + 
                                    '4 --> Дерево решений \n')

                match model_chose:
                    case '1':
                        # Naive Bayes
                        vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4)).fit(using_data['command_clear'])
                        df = vectorizer.transform(list(using_data['command_clear'])).toarray()
                        X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                                using_data['malicious'], 
                                                                test_size=0.2)
                        
                        loaded_model = pickle.load(open('bayes_model.sav', 'rb'))

                        for i in range(len(X_test)):
                            if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
                                # print(using_data['command_clear'].iloc[int(len(using_data) * 0.8) + i], 
                                #       Y_test.iloc[i], 
                                #       Y_test.iloc[i])
                                answers = answers.append({'Command': using_data['command_clear'].iloc[int(len(using_data) * 0.8) + i], 
                                                          'Classifier Answer': Y_test.iloc[i], 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                            else:
                                # print(using_data['command_clear'].iloc[int(len(using_data) * 0.8) + i], 
                                #       int(not Y_test.iloc[i]), 
                                #       Y_test.iloc[i])
                                answers = answers.append({'Command': using_data['command_clear'].iloc[int(len(using_data) * 0.8) + i], 
                                                          'Classifier Answer': int(not Y_test.iloc[i]), 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                        save_or_not = input('Save results?\n' + 
                                    '1 --> Yes\n' + 
                                    '2 --> No\n')
                
                        match save_or_not:
                            case '1':
                                answers.to_csv('bayes_answers.csv', index=False)
                            case '2':
                                pd.set_option('display.max_rows', None)
                                pd.set_option('display.max_rows', None)
                                print(answers)
                        
                    case '2':
                        # Logistic Regression
                        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1)).fit(using_data['command_clear'])
                        df = vectorizer.transform(list(using_data['command_clear'])).toarray()
                        X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                                            using_data['malicious'], 
                                                                            test_size=0.2)
                        loaded_model = pickle.load(open('log_res_model.sav', 'rb'))
                        
                        for i in range(len(X_test)):
                            if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
                                answers = answers.append({'Command': using_data['command_clear'].iloc[int(len(using_data) * 0.8) + i], 
                                                          'Classifier Answer': Y_test.iloc[i], 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                            else:
                                answers = answers.append({'Command': using_data['command_clear'].iloc[int(len(using_data) * 0.8) + i], 
                                                          'Classifier Answer': int(not Y_test.iloc[i]), 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                        save_or_not = input('Save results?\n' + 
                                    '1 --> Yes\n' + 
                                    '2 --> No\n')
                
                        match save_or_not:
                            case '1':
                                answers.to_csv('log_res_answers.csv', index=False)
                            case '2':
                                pd.set_option('display.max_rows', None)
                                print(answers)

                    case '3':
                        # Random Forest
                        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2)).fit(using_data['command_clear'])
                        df = vectorizer.transform(list(using_data['command_clear'])).toarray()
                        X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                                using_data['malicious'], 
                                                                test_size=0.2)
                        loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
                        for i in range(len(X_test)):
                            if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
                                # print(using_data['command_clear'].iloc[1280 + i], 
                                #       Y_test.iloc[i], 
                                #       Y_test.iloc[i])
                                answers = answers.append({'Command': using_data['command_clear'].iloc[1280 + i], 
                                                          'Classifier Answer': Y_test.iloc[i], 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                            else:
                                # print(using_data['command_clear'].iloc[1280 + i], 
                                #       int(not Y_test.iloc[i]), 
                                #       Y_test.iloc[i])
                                answers = answers.append({'Command': using_data['command_clear'].iloc[1280 + i], 
                                                          'Classifier Answer': int(not Y_test.iloc[i]), 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                        save_or_not = input('Save results?\n' + 
                                    '1 --> Yes\n' + 
                                    '2 --> No\n')
                
                        match save_or_not:
                            case '1':
                                answers.to_csv('random_forest_answers.csv', index=False)
                            case '2':
                                pd.set_option('display.max_rows', None)
                                print(answers)

                    case '4':
                        # Decision Tree
                        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2)).fit(using_data['command_clear'])
                        df = vectorizer.transform(list(using_data['command_clear'])).toarray()
                        X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                                using_data['malicious'], 
                                                                test_size=0.2)

                        loaded_model = pickle.load(open('decision_tree_model.sav', 'rb'))
                        for i in range(len(X_test)):
                            if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
                                # print(using_data['command_clear'].iloc[int(len(using_data) * 0.8) + i], 
                                #       Y_test.iloc[i], 
                                #       Y_test.iloc[i])
                                answers = answers.append({'Command': using_data['command_clear'].iloc[1280 + i], 
                                                          'Classifier Answer': Y_test.iloc[i], 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                            else:
                                # print(using_data['command_clear'].iloc[int(len(using_data) 0.8) + i], 
                                #       int(not Y_test.iloc[i]), 
                                #       Y_test.iloc[i])
                                answers = answers.append({'Command': using_data['command_clear'].iloc[1280 + i], 
                                                          'Classifier Answer': int(not Y_test.iloc[i]), 
                                                          'True Answer': Y_test.iloc[i]}, ignore_index=True)
                        save_or_not = input('Save results?\n' + 
                                    '1 --> Yes\n' + 
                                    '2 --> No\n')
                
                        match save_or_not:
                            case '1':
                                answers.to_csv('dec_tree_answers.csv', index=False)
                            case '2':
                                pd.set_option('display.max_rows', None)
                                print(answers)

                    case _:
                        print('Вы не выбрали алгоритм')

        case '3':
            print('Окончание работы')
            
        case _:
            print('Выберите заново')
            menu()


def algorithm_list(number_of_algorithm, file_path, using_data):
    algs = ['Command']
    for i in range(number_of_algorithm):
        model_chose = input('Введите название алгоритма: \n' + 
                            '1 --> Байесовский классификатор \n' + 
                            '2 --> Логистическая регрессия \n' + 
                            '3 --> Бэггинг \n' + 
                            '4 --> Дерево решений \n')
        
        match model_chose:
            case '1':
                if 'Naive Bayes' in algs:
                    print('Naive Bayes уже выбран, выберите алгоритмы заново')
                    return algorithm_list(number_of_algorithm, file_path, using_data)
                else:
                    algs.append('Naive Bayes')
            case '2':
                if 'Logistic Regression' in algs:
                    print('Logistic Regression уже выбрана, выберите алгоритмы заново')
                    return algorithm_list(number_of_algorithm, file_path, using_data)
                else:
                    algs.append('Logistic Regression')
            case '3':
                if 'Bagging' in algs:
                    print('Bagging уже выбран, выберите алгоритмы заново')
                    return algorithm_list(number_of_algorithm, file_path, using_data)
                else:
                    algs.append('Bagging')
            case '4':
                if 'Decision Tree' in algs:
                    print('Decision Tree уже выбран, выберите алгоритмы заново')
                    return algorithm_list(number_of_algorithm, file_path, using_data)
                else:
                    algs.append('Decision Tree')
    algs.append('True Answer')
    
    answers = pd.DataFrame(columns=algs)
    answers['Command'] = using_data['command_clear'].iloc[1280:]
    answers['True Answer'] = using_data['malicious'].iloc[1280:]

    for i in range(1, len(algs)):
        if algs[i] == 'Naive Bayes':
            naive_bayes_answers = []
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4)).fit(using_data['command_clear'])
            df = vectorizer.transform(list(using_data['command_clear'])).toarray()
            X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                    using_data['malicious'], 
                                                    test_size=0.2)

            loaded_model = pickle.load(open('bayes_model.sav', 'rb'))
            for i in range(len(X_test)):
                if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
#                     print(using_data['command_clear'].iloc[1280 + i], 
#                           Y_test.iloc[i], 
#                           Y_test.iloc[i])
                    naive_bayes_answers.append(Y_test.iloc[i])
                else:
#                     print(using_data['command_clear'].iloc[1280 + i], 
#                           int(not Y_test.iloc[i]), 
#                           Y_test.iloc[i])
                    naive_bayes_answers.append(int(not Y_test.iloc[i]))
            answers['Naive Bayes'] = naive_bayes_answers
        
        elif algs[i] == 'Bagging':
            bagging_answers = []
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2)).fit(using_data['command_clear'])
            df = vectorizer.transform(list(using_data['command_clear'])).toarray()
            X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                    using_data['malicious'], 
                                                    test_size=0.2)
            loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
            for i in range(len(X_test)):
                if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
#                     print(using_data['command_clear'].iloc[1280 + i], 
#                           Y_test.iloc[i], 
#                           Y_test.iloc[i])
                    bagging_answers.append(Y_test.iloc[i])
                else:
#                     print(using_data['command_clear'].iloc[1280 + i], 
#                           int(not Y_test.iloc[i]), 
#                           Y_test.iloc[i])
                    bagging_answers.append(int(not Y_test.iloc[i]))
            answers['Bagging'] = bagging_answers
            
        elif algs[i] == 'Logistic Regression':
            log_res_answers = []
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1)).fit(using_data['command_clear'])
            df = vectorizer.transform(list(using_data['command_clear'])).toarray()
            X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                                using_data['malicious'], 
                                                                test_size=0.2)
            loaded_model = pickle.load(open('log_res_model.sav', 'rb'))
            for i in range(len(X_test)):
                if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
#                     print(using_data['command_clear'].iloc[1280 + i], 
#                           Y_test.iloc[i], 
#                           Y_test.iloc[i])
                    log_res_answers.append(Y_test.iloc[i])
                else:
#                     print(using_data['command_clear'].iloc[1280 + i], 
#                           int(not Y_test.iloc[i]), 
#                           Y_test.iloc[i])
                    log_res_answers.append(int(not Y_test.iloc[i]))
            answers['Logistic Regression'] = log_res_answers
            
        elif algs[i] == 'Decision Tree':
            dec_tree_answers = []
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2)).fit(using_data['command_clear'])
            df = vectorizer.transform(list(using_data['command_clear'])).toarray()
            X_train, X_test, Y_train, Y_test = train_test_split(df,
                                                    using_data['malicious'], 
                                                    test_size=0.2)
            loaded_model = pickle.load(open('decision_tree_model.sav', 'rb'))
            
            for i in range(len(X_test)):
                if loaded_model.score(np.reshape(X_test[i], (-1, len(X_test[i]))), Y_test.iloc[i:i + 1]) == 1.0:
                    dec_tree_answers.append(Y_test.iloc[i])
                else:
                    dec_tree_answers.append(int(not Y_test.iloc[i]))
                    
            answers['Decision Tree'] = dec_tree_answers

    return answers

menu()