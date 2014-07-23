__author__ = 'vignesh'


import random
from scikits.learn import linear_model
import numpy as np

class LinearRegression(object):

    trainX = None
    trainY = None
    testX = None
    train_selected = None
    test_selected = None
    model = None
    test_feature_set = None
    fixmv = None


    def __init__(self, featureset, y, testX, fixmv= None):
        self.trainX = featureset
        self.trainY = y
        self.testX = testX
        #self.select_features(
        self.fixmv = fixmv



    def set_train_selected(self,train_selected):
        self.train_selected = train_selected


    def set_test_selected(self, test_selected):
        self.test_selected = test_selected


    def select_features(self, num_features = None):
        """
        Need to write code to perform feature selection
        :return:

        Selected features based on some algorithm
        """
        if num_features == None:
            n = 5

        else:
            n = num_features
        feature_len = self.trainX.shape[1]
        choices = [random.choice(range(feature_len)) for i in range(n)]
        #self.selected_features = [self.trainX[c] for c in choices]
        self.train_selected = self.trainX[choices]
        self.test_selected = self.testX[choices]

    def build_regression_model(self):
        """
        Perform simple linear regression
        :return:
        """

        # Some basic house keeping

        print self.train_selected.shape
        print len(self.trainY)

        # Speeding up things by converting the df into numpy arrays a
        # and converting nans to 0
        if self.fixmv == True:
            self.train_selected = self.train_selected.fillna(0)
        self.model = linear_model.LinearRegression()
        #self.model = linear_model.Ridge()
        self.model.fit(self.train_selected, self.trainY)

    def predict(self):

        """

        :return:

        returns predicted results
        """

        #self.test_selected = self.test_selected.fillna(0)

        return self.model.predict(self.test_selected)

