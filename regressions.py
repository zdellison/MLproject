from sklearn.svm import SVR,SVC
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, LogisticRegression
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
#    print 'Zero Correct Percent: ',zeroCorrectCount/zeroCount
#    print 'Non Zero Correct Percent: ',nonZeroCorrectCount/zeroCount
    if totalCount == 0: totalCount=1
    if zeroCount == 0: 
        zeroCorrectCount = 1
        zeroCount = 1.0
    if nonZeroCount==0: 
        nonZeroCorrectCount = 1.0
        nonZeroCount=1
    return (totalSquareError/totalCount,zeroSquareError/zeroCount,nonZeroSquareError/nonZeroCount,zeroCorrectCount/zeroCount,nonZeroCorrectCount/zeroCount)


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
    zeroPercent = []
    nonZeroPercent = []
    train_errors = []
    train_zeroErrors = []
    train_nonZeroErrors = []
    train_zeroPercent = []
    train_nonZeroPercent = []
    for i in range(k):
        print "-------- Fold: ",i,' ----------'
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

    # Compute Train Error:
        train_reg_pred = [int(round(x)) for x in clf.predict(X)]
        (train_total_error,train_zero_error,train_non_zero_error,train_zero_percent,train_nonZero_percent) = ComputeError(y,train_reg_pred)

        pred_y = [int(round(y)) for y in clf.predict(test_X)]
        (total_error,zero_error,non_zero_error,zero_percent,nonZero_percent) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        train_errors.append(train_total_error)
        train_zeroErrors.append(train_zero_error)
        train_nonZeroErrors.append(train_non_zero_error)
        print '----------- Train Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",train_total_error
        print "Percent of Zero Retweets predicted correctly: ",train_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",train_nonZero_percent
        print '----------- Test Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",total_error
        print "Percent of Zero Retweets predicted correctly: ",zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",nonZero_percent
        (naive_error,zero_naive_error,non_zero_naive_error,naive_zero_percent,naive_nonZero_percent) = ComputeError(data[i][0][1],[0]*len(data[i][0][1]))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        zeroPercent.append(zero_percent)
        nonZeroPercent.append(nonZero_percent)
        train_zeroPercent.append(train_zero_percent)
        train_nonZeroPercent.append(nonZero_percent)
        print "Naive Baseline Error: ",naive_error
# NOTE: removing the 6th fold errors because they are drastically higher than the others    
    del errors[6]
    del zeroErrors[6]
    del nonZeroErrors[6]
    print "==================== SUMMARY =================="
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Percent Zero Retweets predicted correctly: ",sum(zeroPercent)/float(len(zeroPercent))
    print "Percent Non-Zero Retweets predicted correctly: ",sum(nonZeroPercent)/float(len(nonZeroPercent))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))
    print "Total Train Error: ",float(sum(train_errors))/float(len(train_errors))
    print "Train Zero Errors: ",float(sum(train_zeroErrors))/float(len(train_zeroErrors))
    print "Train Non-Zero Errors: ",float(sum(train_nonZeroErrors))/float(len(train_nonZeroErrors))
    print "Train Zero Retweet Percent Correct: ",float(sum(train_zeroPercent))/float(len(train_zeroPercent))
    print "Train Non Zero Retweet Percent Correct: ",float(sum(train_nonZeroPercent))/float(len(train_nonZeroPercent))


# Ridge Regression
def rr(data,k,alpha,poly=False):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    zeroPercent = []
    nonZeroPercent = []
    for i in range(k):
        print "-------- Fold: ",i,' ----------'
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
        (total_error,zero_error,non_zero_error,zero_percent,nonZero_percent) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        print "For Fold Number: ",i,", we have Error: ",total_error
        print "Percent of Zero Retweets predicted correctly: ",zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",nonZero_percent        
        (naive_error,zero_naive_error,non_zero_naive_error,naive_zero_percent,naive_nonZero_percent) = ComputeError(test_y,[0]*len(test_y))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        zeroPercent.append(zero_percent)
        nonZeroPercent.append(nonZero_percent)
        print "Naive Baseline Error: ",naive_error
    print "==================== SUMMARY =================="
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Percent Zero Retweets predicted correctly: ",sum(zeroPercent)/float(len(zeroPercent))
    print "Percent Non-Zero Retweets predicted correctly: ",sum(nonZeroPercent)/float(len(nonZeroPercent))
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
    zeroPercent = []
    nonZeroPercent = []
    for i in range(k):
        print "-------- Fold: ",i,' ----------'
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
        (total_error,zero_error,non_zero_error,zero_percent,nonZero_percent) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        print "For Fold Number: ",i,", we have Error: ",total_error
        print "Percent of Zero Retweets predicted correctly: ",zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",nonZero_percent
        (naive_error,zero_naive_error,non_zero_naive_error,naive_zero_percent,naive_nonZero_percent) = ComputeError(test_y,[0]*len(test_y))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        zeroPercent.append(zero_percent)
        nonZeroPercent.append(nonZero_percent)
        print "Naive Baseline Error: ",naive_error
    print "==================== SUMMARY =================="
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Percent Zero Retweets predicted correctly: ",sum(zeroPercent)/float(len(zeroPercent))
    print "Percent Non-Zero Retweets predicted correctly: ",sum(nonZeroPercent)/float(len(nonZeroPercent))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))


# Logistic Regression /  Least Squares Regression Hybrid
def logRlsrHybrid(data,k):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    predZeroPercent = []
    predNonZeroPercent = []
    zeroPercent = []
    nonZeroPercent = []
    train_errors = []
    train_zeroErrors = []
    train_nonZeroErrors = []
    train_predZeroPercent = []
    train_predNonZeroPercent = []
    train_zeroPercent = []
    train_nonZeroPercent = []
    for i in range(k):
        print "-------- Fold: ",i,' ----------'
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

               
    # Train the Classifier
        y = np.array(classifier_y)
        X = np.matrix(classifier_features)
        classifier = LogisticRegression()
        classifier.fit(X,y)
        train_class_pred = classifier.predict(X)
        (train_pred_total_error,train_pred_zero_error,train_pred_non_zero_error,train_pred_zero_percent,train_pred_nonZero_percent) = ComputeError([0 if x==0 else 1 for x in y],train_class_pred)

    # Train the Regression
        y = np.array(non_zero_y)
        X = np.matrix(non_zero_features)
        clf = LinearRegression()
        clf.fit(X,y)
        train_reg_pred = clf.predict(X)
        (train_total_error,train_zero_error,train_non_zero_error,train_zero_percent,train_nonZero_percent) = ComputeError(y,train_reg_pred)

    # Use Classifier to Predict 0 or Non-Zero
        predictions = classifier.predict(data[i][0][0]).tolist()

        (pred_total_error,pred_zero_error,pred_non_zero_error,pred_zero_percent,pred_nonZero_percent) = ComputeError([0 if x==0 else 1 for x in data[i][0][1]],predictions)
        print '-------- Train Classifier Errors  -----------'
        print "Classifier Errors: Total Predictor Error: ",train_pred_total_error
        print "Zero Predictor Errors: ",train_pred_zero_error
        print "Non-Zero Predictor Errors: ",train_pred_non_zero_error
        print "Percent of Zero Retweets predicted correctly: ",train_pred_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",train_pred_nonZero_percent
        print '-------- Test Classifier Errors  -----------'
        print "Classifier Errors: Total Predictor Error: ",pred_total_error
        print "Zero Predictor Errors: ",pred_zero_error
        print "Non-Zero Predictor Errors: ",pred_non_zero_error
        print "Percent of Zero Retweets predicted correctly: ",pred_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",pred_nonZero_percent

    # If Classifier outputs non-zero, run SVR
        for idx,val in enumerate(predictions):
            if val!=0:
                pred = int(round(clf.predict(data[i][0][0][idx])))
                predictions[idx]=pred

        (total_error,zero_error,non_zero_error,zero_percent,nonZero_percent) = ComputeError(data[i][0][1],predictions)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        train_errors.append(train_total_error)
        train_zeroErrors.append(train_zero_error)
        train_nonZeroErrors.append(train_non_zero_error)
        print '----------- Train Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",train_total_error
        print "Percent of Zero Retweets predicted correctly: ",train_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",train_nonZero_percent
        print '----------- Test Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",total_error
        print "Percent of Zero Retweets predicted correctly: ",zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",nonZero_percent
        (naive_error,zero_naive_error,non_zero_naive_error,naive_zero_percent,naive_nonZero_percent) = ComputeError(data[i][0][1],[0]*len(data[i][0][1]))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        predZeroPercent.append(pred_zero_percent)
        predNonZeroPercent.append(pred_nonZero_percent)
        zeroPercent.append(zero_percent)
        nonZeroPercent.append(nonZero_percent)
        train_predZeroPercent.append(train_pred_zero_percent)
        train_predNonZeroPercent.append(train_pred_nonZero_percent)
        train_zeroPercent.append(train_zero_percent)
        train_nonZeroPercent.append(nonZero_percent)
        print "Naive Baseline Error: ",naive_error
    print "==================== SUMMARY =================="
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Percent Zero Retweets predicted correctly: ",sum(zeroPercent)/float(len(zeroPercent))
    print "Percent Non-Zero Retweets predicted correctly: ",sum(nonZeroPercent)/float(len(nonZeroPercent))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))
    print "Predictor Zero Retweets predicted correctly: ",sum(predZeroPercent)/float(len(predZeroPercent))
    print "Predictor Non-Zero Retweets predicted correctly: ",sum(predNonZeroPercent)/float(len(predNonZeroPercent))
    print "Total Train Error: ",float(sum(train_errors))/float(len(train_errors))
    print "Train Zero Errors: ",float(sum(train_zeroErrors))/float(len(train_zeroErrors))
    print "Train Non-Zero Errors: ",float(sum(train_nonZeroErrors))/float(len(train_nonZeroErrors))
    print "Train Predictor Zero retweets Percent Correct: ",float(sum(train_predZeroPercent))/float(len(train_predZeroPercent))
    print "Train Predictor Non-Zero retweets Percent Correct: ",float(sum(train_predNonZeroPercent))/float(len(train_predNonZeroPercent))
    print "Train Zero Retweet Percent Correct: ",float(sum(train_zeroPercent))/float(len(train_zeroPercent))
    print "Train Non Zero Retweet Percent Correct: ",float(sum(train_nonZeroPercent))/float(len(train_nonZeroPercent))


# Straight SVR
def svr(data,k):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    zeroPercent = []
    nonZeroPercent = []
    train_errors = []
    train_zeroErrors = []
    train_nonZeroErrors = []
    train_zeroPercent = []
    train_nonZeroPercent = []
    for i in range(k):
        print "-------- Fold: ",i,' ----------'
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

    # Compute Train Error:
        train_reg_pred = [int(round(x)) for x in clf.predict(X)]
        (train_total_error,train_zero_error,train_non_zero_error,train_zero_percent,train_nonZero_percent) = ComputeError(y,train_reg_pred)

        pred_y = [int(round(y)) for y in clf.predict(test_X)]
        (total_error,zero_error,non_zero_error,zero_percent,nonZero_percent) = ComputeError(test_y,pred_y)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        train_errors.append(train_total_error)
        train_zeroErrors.append(train_zero_error)
        train_nonZeroErrors.append(train_non_zero_error)
        print '----------- Train Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",train_total_error
        print "Percent of Zero Retweets predicted correctly: ",train_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",train_nonZero_percent
        print '----------- Test Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",total_error
        print "Percent of Zero Retweets predicted correctly: ",zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",nonZero_percent
        (naive_error,zero_naive_error,non_zero_naive_error,naive_zero_percent,naive_nonZero_percent) = ComputeError(data[i][0][1],[0]*len(data[i][0][1]))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        zeroPercent.append(zero_percent)
        nonZeroPercent.append(nonZero_percent)
        train_zeroPercent.append(train_zero_percent)
        train_nonZeroPercent.append(nonZero_percent)
        print "Naive Baseline Error: ",naive_error
    print "==================== SUMMARY =================="
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Percent Zero Retweets predicted correctly: ",sum(zeroPercent)/float(len(zeroPercent))
    print "Percent Non-Zero Retweets predicted correctly: ",sum(nonZeroPercent)/float(len(nonZeroPercent))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))
    print "Total Train Error: ",float(sum(train_errors))/float(len(train_errors))
    print "Train Zero Errors: ",float(sum(train_zeroErrors))/float(len(train_zeroErrors))
    print "Train Non-Zero Errors: ",float(sum(train_nonZeroErrors))/float(len(train_nonZeroErrors))
    print "Train Zero Retweet Percent Correct: ",float(sum(train_zeroPercent))/float(len(train_zeroPercent))
    print "Train Non Zero Retweet Percent Correct: ",float(sum(train_nonZeroPercent))/float(len(train_nonZeroPercent))



# SVC / SVR Hybrid
def svrSvcHybrid(data,k):
    errors = []
    zeroErrors = []
    nonZeroErrors = []
    naiveErrors = []
    zeroNaiveErrors = []
    nonZeroNaiveErrors = []
    predZeroPercent = []
    predNonZeroPercent = []
    zeroPercent = []
    nonZeroPercent = []
    train_errors = []
    train_zeroErrors = []
    train_nonZeroErrors = []
    train_predZeroPercent = []
    train_predNonZeroPercent = []
    train_zeroPercent = []
    train_nonZeroPercent = []

    for i in range(k):
        print "-------- Fold: ",i,' ----------'
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

               
    # Train the Classifier
        y = np.array(classifier_y)
        X = np.matrix(classifier_features)
#        classifier = SVC(C=1.0)
        classifier = SVC(C=1.0,kernel='rbf',gamma=0.1)
#        classifier = SVC(C=1.0,kernel='sigmoid')
        classifier.fit(X,y)
        train_class_pred = classifier.predict(X)
        (train_pred_total_error,train_pred_zero_error,train_pred_non_zero_error,train_pred_zero_percent,train_pred_nonZero_percent) = ComputeError([0 if x==0 else 1 for x in y],train_class_pred)
    # Train the Regression
        y = np.array(non_zero_y)
        X = np.matrix(non_zero_features)
#        clf = SVR(C=1.0,epsilon=0.2)
#        clf = SVR(C=50)
        clf = SVR(kernel='poly',C=1e3,degree=2)
        clf.fit(X,y)
        train_reg_pred = clf.predict(X)
        (train_total_error,train_zero_error,train_non_zero_error,train_zero_percent,train_nonZero_percent) = ComputeError(y,train_reg_pred)

    # Use Classifier to Predict 0 or Non-Zero
        predictions = classifier.predict(data[i][0][0]).tolist()

        (pred_total_error,pred_zero_error,pred_non_zero_error,pred_zero_percent,pred_nonZero_percent) = ComputeError([0 if x==0 else 1 for x in data[i][0][1]],predictions)
        print '-------- Train Classifier Errors  -----------'
        print "Classifier Errors: Total Predictor Error: ",train_pred_total_error
        print "Zero Predictor Errors: ",train_pred_zero_error
        print "Non-Zero Predictor Errors: ",train_pred_non_zero_error
        print "Percent of Zero Retweets predicted correctly: ",train_pred_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",train_pred_nonZero_percent
        print '-------- Test Classifier Errors  -----------'
        print "Classifier Errors: Total Predictor Error: ",pred_total_error
        print "Zero Predictor Errors: ",pred_zero_error
        print "Non-Zero Predictor Errors: ",pred_non_zero_error
        print "Percent of Zero Retweets predicted correctly: ",pred_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",pred_nonZero_percent

    # If Classifier outputs non-zero, run SVR
        for idx,val in enumerate(predictions):
            if val!=0:
                pred = int(round(clf.predict(data[i][0][0][idx])))
                predictions[idx]=pred

        (total_error,zero_error,non_zero_error,zero_percent,nonZero_percent) = ComputeError(data[i][0][1],predictions)
        errors.append(total_error)
        zeroErrors.append(zero_error)
        nonZeroErrors.append(non_zero_error)
        train_errors.append(train_total_error)
        train_zeroErrors.append(train_zero_error)
        train_nonZeroErrors.append(train_non_zero_error)
        print '----------- Train Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",train_total_error
        print "Percent of Zero Retweets predicted correctly: ",train_zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",train_nonZero_percent
        print '----------- Test Regression Errors ------------'
        print "For Fold Number: ",i,", we have Error: ",total_error
        print "Percent of Zero Retweets predicted correctly: ",zero_percent
        print "Percent of Non-Zero Retweets predicted correctly: ",nonZero_percent
        (naive_error,zero_naive_error,non_zero_naive_error,naive_zero_percent,naive_nonZero_percent) = ComputeError(data[i][0][1],[0]*len(data[i][0][1]))
        naiveErrors.append(naive_error)
        zeroNaiveErrors.append(zero_naive_error)
        nonZeroNaiveErrors.append(non_zero_naive_error)
        predZeroPercent.append(pred_zero_percent)
        predNonZeroPercent.append(pred_nonZero_percent)
        zeroPercent.append(zero_percent)
        nonZeroPercent.append(nonZero_percent)
        train_predZeroPercent.append(train_pred_zero_percent)
        train_predNonZeroPercent.append(train_pred_nonZero_percent)
        train_zeroPercent.append(train_zero_percent)
        train_nonZeroPercent.append(nonZero_percent)
        print "Naive Baseline Error: ",naive_error
    print "==================== SUMMARY =================="
    print "Total Error: ",float(sum(errors))/float(len(errors))
    print "Percent Zero Retweets predicted correctly: ",sum(zeroPercent)/float(len(zeroPercent))
    print "Percent Non-Zero Retweets predicted correctly: ",sum(nonZeroPercent)/float(len(nonZeroPercent))
    print "Zero Errors: ",float(sum(zeroErrors))/float(len(zeroErrors))
    print "Non Zero Errors: ",float(sum(nonZeroErrors))/float(len(nonZeroErrors))
    print "Naive Predictor Error: ",float(sum(naiveErrors))/float(len(naiveErrors))
    print "Naive Predictor Zero Errors: ",float(sum(zeroNaiveErrors))/float(len(zeroNaiveErrors))
    print "Naive Predictor Non Zero Errors: ",float(sum(nonZeroNaiveErrors))/float(len(nonZeroNaiveErrors))
    print "Predictor Zero Retweets predicted correctly: ",sum(predZeroPercent)/float(len(predZeroPercent))
    print "Predictor Non-Zero Retweets predicted correctly: ",sum(predNonZeroPercent)/float(len(predNonZeroPercent))
    print "Total Train Error: ",float(sum(train_errors))/float(len(train_errors))
    print "Train Zero Errors: ",float(sum(train_zeroErrors))/float(len(train_zeroErrors))
    print "Train Non-Zero Errors: ",float(sum(train_nonZeroErrors))/float(len(train_nonZeroErrors))
    print "Train Predictor Zero retweets Percent Correct: ",float(sum(train_predZeroPercent))/float(len(train_predZeroPercent))
    print "Train Predictor Non-Zero retweets Percent Correct: ",float(sum(train_predNonZeroPercent))/float(len(train_predNonZeroPercent))
    print "Train Zero Retweet Percent Correct: ",float(sum(train_zeroPercent))/float(len(train_zeroPercent))
    print "Train Non Zero Retweet Percent Correct: ",float(sum(train_nonZeroPercent))/float(len(train_nonZeroPercent))


def main(k):
    data = featureExtractor.FeatureExtractor(k,languageModel=False)
    if svr_regression: 
        print "============================================================"
        print "                 Support Vector Regression                   "
        print "============================================================"
        svr(data,k)
    if lsr_regression: 
        print "============================================================"
        print "                 Least Squares Regression                   "
        print "============================================================"
        lsr(data,k,poly=False)
    if rr_regression: 
        print "============================================================"
        print "                 Ridge Regression                          "
        print "============================================================"
        rr(data,k,.5,poly=False)
    if sgd_regression: 
        print "============================================================"
        print "                 Stochastic Gradient Descent                  "
        print "============================================================"
        sgd(data,k,poly=False)
    if svr_svc: 
        print "============================================================"
        print "                 SVC-SVR Hybrid                           "
        print "============================================================"
        svrSvcHybrid(data,k)
    if logr_lsr: 
        print "============================================================"
        print "              Logistic Regression-LSR Hybrid               "
        print "============================================================"
        logRlsrHybrid(data,k)

svr_regression = True
lsr_regression = True
rr_regression = False
sgd_regression = False
svr_svc = True
logr_lsr = True

main(10)
