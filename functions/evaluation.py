# Evaluation function

def evaluation(y_true, y_pred):

    '''
    Shows Accuracy, Precision, Recall, and F1-Score evaluation metrics
    '''

    print('Evaluation Metrics:')
    print('Accuracy: ' + str(metrics.accuracy_score(y_true, y_pred)))
    print('Precision: ' + str(metrics.precision_score(y_true, y_pred)))
    print('Recall: ' + str(metrics.recall_score(y_true, y_pred)))
    print('F1 Score: ' + str(metrics.f1_score(y_true, y_pred)))


# Cross-validation evaluation

def cross_validation(model, X_train, y_train, x):
    '''
    Prints cross-validation metrics for evaluation
    '''

    scores = cross_val_score(model, X_train, y_train, cv=x)
    print('Cross-Validation Accuracy Scores:', scores)    
    print('\nMin: ', round(scores.min(), 6))
    print('Max: ', round(scores.max(), 6))
    print('Mean: ', round(scores.mean(), 6)) 
    print('Range: ', round(scores.max() - scores.min(), 6))