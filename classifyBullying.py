from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv
import pickle
import numpy as np
import constants
import pandas as pd
import os
from cleaningDataset import Cleaner

class Classifier:

    def __init__(self):
        self.modelStr = None
        self.model = None
        self.labelEncoder = None
        self.tfidf = None
        self.scale = None
        self.cleaner = Cleaner()
        self.prepare_train()

    def load_label_encoder(self):
        print("Loading Label Encoder")
        self.labelEncoder = pickle.load(open(f"{constants.MODELS_PATH}{constants.LBL_ENCODER_FILE}", 'rb'))

    def prepare_train(self, tst_sz=0.33):
        print("Preparing Training and Validation Data ...")
        df=pd.read_csv(constants.CSV_CLEANED)

        x=df['clean_data']
        y=df['cyberbullying_type']
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,test_size=tst_sz)

        if self.tfidf is None:
            path = f"{constants.MODELS_PATH}{constants.TFIDF_FILE}"
            if os.path.exists(path):
                print("Tfidf already exists")
                self.tfidf = pickle.load(open(path, 'rb'))
            else:
                self.tfidf=TfidfVectorizer(max_features=5000)
                self.x_train_tfidf=self.tfidf.fit(self.x_train)
                pickle.dump(self.tfidf, open(path, 'wb'))

        self.x_train_tfidf=self.tfidf.transform(self.x_train)
        self.x_test_tfidf=self.tfidf.transform(self.x_test)

        # print matriz train tfidf
        # print(x_train_tfidf)

        self.x_train_tfidf=self.x_train_tfidf.toarray()
        self.x_test_tfidf=self.x_test_tfidf.toarray()
        if self.scale is None:
            path = f"{constants.MODELS_PATH}{constants.SCALER_FILE}"
            if os.path.exists(path):
                print("Scaler already exists")
                self.scale = pickle.load(open(path, 'rb'))
            else:
                from sklearn.preprocessing import StandardScaler
                self.scale=StandardScaler()
                self.scaled_x_train=self.scale.fit(self.x_train_tfidf)
                pickle.dump(self.scale, open(path, 'wb'))

        self.scaled_x_train=self.scale.transform(self.x_train_tfidf)
        self.scaled_x_test=self.scale.transform(self.x_test_tfidf)

    def trainModel(self, modelStr):
        if modelStr == constants.LOGREG:
            print("Training with Logistc Regression")
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(solver = 'saga', verbose=True)
            param_grid = {'C': np.logspace(0, 10, 3)}

        elif modelStr == constants.GRAD_BOOST:
            print("Training with Gradient Boosting")
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(random_state = 42,n_iter_no_change=5, verbose=True)
            param_grid = {'learning_rate': [.1, .01, .5]}

        elif modelStr == constants.RND_FRST:
            print("Training with Random Forest")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state = 42, verbose=True)
            n_estimators = [64, 100, 128]
            bootstrap = [True, False] # Bootstrapping is true by default
            param_grid = {'n_estimators': n_estimators, 'bootstrap': bootstrap}

        elif modelStr == constants.NEURAL_NETWORK:
            print("Training with Neural Networks")
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(activation = 'logistic', max_iter = 50, verbose=True)  # Sigmoid Activation Function
            param_grid = {'learning_rate_init': [0.001, 0.002, 0.0025]}

        else:
            print("unknown model")
            return

        finalModel = HalvingGridSearchCV(model, param_grid = param_grid, n_jobs = -1, min_resources = 'exhaust', factor = 3)
        finalModel.fit(self.scaled_x_train, self.y_train)
        self.predictMetrics(finalModel)

        # save model
        pickle.dump(finalModel, open(f"{constants.MODELS_PATH}{modelStr}.mdl", 'wb'))

    def predictMetrics(self, model):
        preds = model.predict(self.scaled_x_test)

        if self.labelEncoder is None:
            self.load_label_encoder()
        
        preds = self.labelEncoder.inverse_transform(preds)
        y_test_inversed = self.labelEncoder.inverse_transform(self.y_test)
        print(classification_report(y_test_inversed, preds))
        print(confusion_matrix(y_test_inversed, preds))
        return preds

    def load_predict(self, modelStr, toPred=None):
        if self.modelStr != modelStr and modelStr not in constants.LIST_MODELS:
            print("unknown model")
            return
        if self.labelEncoder is None:
            self.load_label_encoder()

        if self.modelStr != modelStr:
            self.modelStr = modelStr
            self.model = pickle.load(open(f"{constants.MODELS_PATH}{modelStr}.mdl", 'rb'))

        if toPred is None:
            preds = self.predictMetrics(self.model)
        else:
            toPred = [self.cleaner.clean_text(tweet) for tweet in toPred]
            toPredtfidf = self.tfidf.transform(toPred).toarray()
            toPredScaled = self.scale.transform(toPredtfidf)
            preds = self.model.predict(toPredScaled)
            preds = self.labelEncoder.inverse_transform(preds)

        return preds
