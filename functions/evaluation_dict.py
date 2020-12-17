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