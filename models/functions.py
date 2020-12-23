# Sklearn Packages
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix, roc_curve, auc, classification_report
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

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
    print('\nCross-Validation Accuracy Scores:', scores)    
    print('Min: ', round(scores.min(), 6))
    print('Max: ', round(scores.max(), 6))
    print('Mean: ', round(scores.mean(), 6)) 
    print('Range: ', round(scores.max() - scores.min(), 6))
    
    
# Creating dictionary with all metrics
def evaluation_dict(accuracy, precision, recall, f1, y_test, y_pred, model_name):

    '''
    This function adds the results to a dictionary so that we can create a DataFrame with the results
    '''
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    
    metric_dict[model_name] = {
                                                    'Accuracy': accuracy,
                                                    'Precision': precision,
                                                    'Recall': recall,
                                                    'F1 Score': f1 }