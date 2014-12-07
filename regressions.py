from sklearn.svm import SVR,SVC
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import *
import json
from pprint import pprint
import time
import tweetParse
from textblob import TextBlob
import collections
import featureExtractor

def ComputeError(y,pred_y):
    totalSquareError = 0.0
    totalCount = 0
    zeroSquareError = 0.0
    zeroCorrectCount = 0.0
    zeroCount = 0
    nonZeroSquareError = 0.0
    nonZeroCorrectCount = 0.0
    nonZeroCount = 0
    for idx,pred in enumerate(pred_y):
        totalCount+=1
        totalSquareError += (pred-y[idx])**2
        if y[idx]==0:
            zeroCount+=1
            zeroSquareError += (pred-y[idx])**2
            if pred==0: zeroCorrectCount+=1
        else:
            nonZeroCount+=1
            nonZeroSquareError += (pred-y[idx])**2
            if pred!=0: nonZeroCorrectCount+=1
    print 'Zero Correct Percent: ',zeroCorrectCount/zeroCount
    print 'Non Zero Correct Percent: ',nonZeroCorrectCount/zeroCount
    return (totalSquareError/totalCount,zeroSquareError/zeroCount,nonZeroSquareError/nonZeroCount)


def test():
    n_samples, n_features = 10,5
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)
    print y
    print X
    clf = SVR(C=1.0,epsilon=0.2)
    clf.fit(X,y)
    print clf.score(X,y)

# Least Squares Regression
def lsr(data,k,poly=False):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    for i in range(k):
        print "Fold: ",i
        train_features = []
        train_y = []
        train_classifier_features = []
        train_classifier_y = []
        train_non_zero_features = []
        train_non_zero_y = []
        for j in range(k):
            if i!=j:
                (features,y),(classifier_features,classifier_y),(non_zero_features,non_zero_y) = data[j]
                train_features.extend(features)
                train_y.extend(y)
                train_classifier_features.extend(classifier_features)
                train_classifier_y.extend(classifier_y)
                train_non_zero_features.extend(non_zero_features)
                train_non_zero_y.extend(non_zero_y)
        
        y = np.array(train_y)
        X = np.matrix(train_features)
        if poly:
            X= PolynomialFeatures(interaction_only=True).fit_transform(X)
        test_y = np.array(data[i][0][1])
        test_X = np.matrix(data[i][0][0])
        if poly:
            test_X = PolynomialFeatures(interaction_only=True).fit_transform(X)
        clf = LinearRegression()        
        clf.fit(X,y)
        pred_y = [int(round(y)) for y in clf.predict(test_X)]
        (total_error,zero_error,non_zero_error) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        print "For Fold Number: ",i,", we have Error: ",total_error
        (naive_error,zero_naive_error,non_zero_naive_error) = ComputeError(test_y,[0]*len(test_y))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        print "Naive Baseline Error: ",naive_error
# NOTE: removing the 6th fold errors because they are drastically higher than the others    
    del errors[6]
    del zeroErrors[6]
    del nonZeroErrors[6]
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))

# Ridge Regression
def rr(data,k,alpha,poly=False):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    for i in range(k):
        print "Fold: ",i
        train_features = []
        train_y = []
        train_classifier_features = []
        train_classifier_y = []
        train_non_zero_features = []
        train_non_zero_y = []
        for j in range(k):
            if i!=j:
                (features,y),(classifier_features,classifier_y),(non_zero_features,non_zero_y) = data[j]
                train_features.extend(features)
                train_y.extend(y)
                train_classifier_features.extend(classifier_features)
                train_classifier_y.extend(classifier_y)
                train_non_zero_features.extend(non_zero_features)
                train_non_zero_y.extend(non_zero_y)
        
        y = np.array(train_y)
        X = np.matrix(train_features)
        if poly:
            X= PolynomialFeatures(interaction_only=True).fit_transform(X)
        test_y = np.array(data[i][0][1])
        test_X = np.matrix(data[i][0][0])
        if poly:
            test_X = PolynomialFeatures(interaction_only=True).fit_transform(X)
        clf = Ridge(alpha = alpha)
        clf.fit(X,y)
        pred_y = [int(round(y)) for y in clf.predict(test_X)]
        (total_error,zero_error,non_zero_error) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        print "For Fold Number: ",i,", we have Error: ",total_error
        (naive_error,zero_naive_error,non_zero_naive_error) = ComputeError(test_y,[0]*len(test_y))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        print "Naive Baseline Error: ",naive_error
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))

# Stochastic Gradient Descent
def sgd(data,k,poly=False):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    for i in range(k):
        print "Fold: ",i
        train_features = []
        train_y = []
        train_classifier_features = []
        train_classifier_y = []
        train_non_zero_features = []
        train_non_zero_y = []
        for j in range(k):
            if i!=j:
                (features,y),(classifier_features,classifier_y),(non_zero_features,non_zero_y) = data[j]
                train_features.extend(features)
                train_y.extend(y)
                train_classifier_features.extend(classifier_features)
                train_classifier_y.extend(classifier_y)
                train_non_zero_features.extend(non_zero_features)
                train_non_zero_y.extend(non_zero_y)
        
        y = np.array(train_y)
        X = np.matrix(train_features)
        if poly:
            X= PolynomialFeatures(interaction_only=True).fit_transform(X)
        test_y = np.array(data[i][0][1])
        test_X = np.matrix(data[i][0][0])
        if poly:
            test_X = PolynomialFeatures(interaction_only=True).fit_transform(X)
        clf = SGDRegressor()
        clf.fit(X,y)
        pred_y = [int(round(y)) for y in clf.predict(test_X)]
        (total_error,zero_error,non_zero_error) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        print "For Fold Number: ",i,", we have Error: ",total_error
        (naive_error,zero_naive_error,non_zero_naive_error) = ComputeError(test_y,[0]*len(test_y))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        print "Naive Baseline Error: ",naive_error
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))


# Straight SVR
def svr(data,k):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    for i in range(k):
        print "Fold: ",i
        train_features = []
        train_y = []
        train_classifier_features = []
        train_classifier_y = []
        train_non_zero_features = []
        train_non_zero_y = []
        for j in range(k):
            if i!=j:
                (features,y),(classifier_features,classifier_y),(non_zero_features,non_zero_y) = data[j]
                train_features.extend(features)
                train_y.extend(y)
                train_classifier_features.extend(classifier_features)
                train_classifier_y.extend(classifier_y)
                train_non_zero_features.extend(non_zero_features)
                train_non_zero_y.extend(non_zero_y)
        
        y = np.array(train_y)
        X = np.matrix(train_features)
        test_y = np.array(data[i][0][1])
        test_X = np.matrix(data[i][0][0])
#        clf = SVR(C=1.0,epsilon=0.2)
#        clf = SVR(C=50)
        clf = SVR(kernel='poly',C=1e3,degree=2)
        clf.fit(X,y)
        pred_y = [int(round(y)) for y in clf.predict(test_X)]
        (total_error,zero_error,non_zero_error) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        print "For Fold Number: ",i,", we have Error: ",total_error
        (naive_error,zero_naive_error,non_zero_naive_error) = ComputeError(test_y,[0]*len(test_y))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        print "Naive Baseline Error: ",naive_error
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))


# SVC / SVR Hybrid
def svrSvcHybrid(data,k):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    for i in range(k):
        print "Fold: ",i
        train_features = []
        train_y = []
        train_classifier_features = []
        train_classifier_y = []
        train_non_zero_features = []
        train_non_zero_y = []
        testFeatures = []
        testY = []
        testClassifier_features = []
        testClassifier_y = []
        testNon_zero_features = []
        testNon_zero_y = []
        for j in range(k):
            if i!=j:
                (features,y),(classifier_features,classifier_y),(non_zero_features,non_zero_y) = data[j]
                train_features.extend(features)
                train_y.extend(y)
                train_classifier_features.extend(classifier_features)
                train_classifier_y.extend(classifier_y)
                train_non_zero_features.extend(non_zero_features)
                train_non_zero_y.extend(non_zero_y)
            else:
                (testFeatures,testY),(testClassifier_features,testClassifier_y),(testNon_zero_features,testNon_zero_y) = data[j]

               
    # Train the Classifier
        y = np.array(classifier_y)
        X = np.matrix(classifier_features)
        test_y = np.array(testClassifier_y)
        test_X = np.matrix(testClassifier_features)
        classifier = SVC(C=2.0)
        classifier.fit(X,y)
    # Train the Regression
        y = np.array(non_zero_y)
        X = np.matrix(non_zero_features)
        test_y = np.array(non_zero_y)
        test_X = np.matrix(non_zero_features)
        clf = SVR(C=1.0,epsilon=0.2)
#        clf = SVR(C=50)
#        clf = SVR(kernel='poly',C=1e3,degree=2)
        clf.fit(X,y)

    # Use Classifier to Predict 0 or Non-Zero
        predictions = classifier.predict(testFeatures).tolist()

    # If Classifier outputs non-zero, run SVR
        for idx,val in enumerate(predictions):
            if val!=0:
                pred = int(round(clf.predict(testFeatures[idx])))
                predictions[idx]=pred

        (total_error,zero_error,non_zero_error) = ComputeError(test_y,predictions)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        print "For Fold Number: ",i,", we have Error: ",total_error
        (naive_error,zero_naive_error,non_zero_naive_error) = ComputeError(test_y,[0]*len(test_y))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        print "Naive Baseline Error: ",naive_error
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))

# SVC to determine 0's, SVR on non_zeros
# Training
#y = np.array(classifier_y_list)
#X = np.matrix(classifier_feature_list)
#test_y = np.array(test_y_list)
#test_X = np.matrix(test_feature_list)
#classifier = SVC(C=2.0)
#classifier.fit(X,y)
#
## Train SVR for non-zero tweets
#reg_y = np.array(non_zero_y_list)
#reg_X = np.matrix(non_zero_feature_list)
#regression = SVR(C=1.0,epsilon=0.2)
#regression.fit(reg_X,reg_y)
#
## Actual y values
#print "Actual Values: ",test_y_list
#
## Run classifier on Test Set
#predictions = classifier.predict(test_X).tolist()
#print 'Classifier Predictions: ',predictions
#
## If Classifier outputs non-zero, run SVR
#for idx,val in enumerate(predictions):
#    if val!=0:
#        pred = int(round(regression.predict(np.matrix(classifier_feature_list[idx]))))
#        predictions[idx] = pred
#print 'Full Predictions: ',predictions
#
##totalSquareError = 0.0
##count = 0
##for idx,pred in enumerate(pred_output):
##    count+=1
##    totalSquareError += (pred-test_y_list[idx])**2
##print 'Regular SVR Error: ',totalSquareError/count
#
#totalSquareError = 0.0
#count = 0
#for idx,pred in enumerate(predictions):
#    count+=1
#    totalSquareError += (pred-test_y_list[idx])**2
#print 'SVC/SVR Hybrid Error: ',totalSquareError/count


def main(k):
    data = featureExtractor.FeatureExtractor(k,languageModel=False)
    if svr_regression: svr(data,k)
    if lsr_regression: lsr(data,k,poly=False)
    if rr_regression: rr(data,k,.5,poly=False)
    if sgd_regression: sgd(data,k,poly=False)
    if svr_svc: svrSvcHybrid(data,k)

svr_regression = False
lsr_regression = False
rr_regression = False
sgd_regression = False
svr_svc = True
main(10)
