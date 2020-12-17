    # Evaluation DataFrame
def df_metrics():
    evaluation_df = pd.DataFrame.from_dict(metric_dict, orient='index')
    evaluation_df = evaluation_df.sort_values(by='Accuracy', ascending=False)
    return evaluation_df
    