# /usr/bin/python
__author__ = 'vignesh'

import read_document
import simple_regression
import pca

def test_linear_regression():

    train_doc = read_document.Document('resources/train.csv')
    trainx = train_doc.docx
    trainy = train_doc.docy

    test_doc = read_document.Document('resources/test.csv')
    testx = test_doc.docx


    sample_doc = read_document.Document('resources/sampleSubmission.csv')
    sample = sample_doc.docx

    # Get only the numeric columns in traindata

    train_x_numeric = train_doc.get_only_numeric(trainx)
    test_x_numeric = train_doc.get_only_numeric(testx)

    # Perform training

    lm = simple_regression.LinearRegression(train_x_numeric,trainy, test_x_numeric, False)
    lm.select_features(5)  # Performing Feature Selection
    lm.build_regression_model()
    predicted = lm.predict()

    sample['target'] = predicted
    sample.to_csv('submission.csv',index=False)
    print predicted


def test_pca_linear_regression():

    print 'Reading training data'
    print '======================='
    train_doc = read_document.Document('resources/train.csv')
    # Fixing missing values
    print 'Fixing Missing Values'
    print '======================='
    # The missing values fixing is done in the initialization itself

    # Getting the training and test data
    trainx = train_doc.docx
    trainy = train_doc.docy

    print 'Reading Test data '
    print '====================='
    test_doc = read_document.Document('resources/test.csv')
    testx = test_doc.docx

    sample_doc = read_document.Document('resources/sampleSubmission.csv')
    sample = sample_doc.docx

    # Get only the numeric columns in traindata

    train_x_numeric = train_doc.get_only_numeric(trainx)
    test_x_numeric = train_doc.get_only_numeric(testx)

    # Performing PCA

    print 'Performing PCA'
    print '======================='

    model = pca.PCAMODEL(15, train_x_numeric, trainy)
    model.build_model()

    # Transforming the data set to PCA diamensions

    train_pca_x = model.reduce_dim(train_x_numeric)
    test_pca_x = model.reduce_dim(test_x_numeric)

    # Perform simple linear regression using this data

    print 'Building Regression Model'
    print '============================='


    lm = simple_regression.LinearRegression(train_pca_x, trainy, test_pca_x, False)
    lm.set_train_selected(train_pca_x)
    lm.set_test_selected(test_pca_x)
    lm.build_regression_model()
    predicted = lm.predict()

    sample['target'] = predicted
    sample.to_csv('submission_pca.csv',index=False)
    print predicted

test_linear_regression()
# test_pca_linear_regression()



