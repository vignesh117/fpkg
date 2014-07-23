__author__ = 'vignesh'
import pandas as pd
from read_document import Document

class TrainDocument(Document):


    def __init__(self, docpath):
        super(TrainDocument, self).__init__(docpath=docpath)


class TestDocument(Document):


    def __init__(self,docpath):
        super(TestDocument,self).__init__(docpath=docpath)


def main():

    trainfile = '../resources/train.csv'
    testfile = '../resources/test.csv'

    traindoc = TrainDocument(docpath = trainfile)
    testdoc = TestDocument(testfile)


    print traindoc.get_feature_names()[:10]

