__author__ = 'vignesh'

import pandas as pd


class Document(object):

    docpath = None
    docstring = None
    docx = None
    docy = None
    def __init__(self, docpath):
        self.docpath = docpath
        self.read_doc()
        self.fix_missing_values()


    def set_docpath(self,docpath):
        self.docpath = docpath


    def read_doc(self):
        print 'read doc'
        self.docstring = pd.read_csv(self.docpath)
        self.fix_missing_values()

        # Preparing the training and testing data



        cols = self.docstring.columns
        cols = list(cols)

        # Extracting test data for train file
        if 'target' in cols:

            self.docy = self.docstring.target
            del cols[1] # Deleting the

            self.docx = self.docstring[cols]

        else:
            self.docx = self.docstring

    def get_feature_names(self):
        return self.docstring.columns


    def fix_missing_values(self):

        # currently we just make the missing values as zero

        self.docstring = self.docstring.fillna(0)


    def get_only_numeric(self,data):
        return data._get_numeric_data()
