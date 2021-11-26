# Import metrics libraries
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score

def col_name_cnvt(df):
    '''
    Take the columns needed,
    and remove space and units, and replace column names with lowercase.
    It does an inplace operation.
    
    Input: a dataframe
    Return: a new dataframe.
    '''
    data = df.iloc[:, 2:-5]
    col_labels = data.columns
    col_labels = [label.split(' [')[0].lower() for label in col_labels]
    col_labels = ['_'.join(label.split()) for label in col_labels]
    data.columns = col_labels
    return data

def soft_to_hard(model, X_test, cutoff):
    '''convert soft decision to hard decision with given cutoff threshold'''    
    y_probs = model.predict_proba(X_test)
    y_hats = [1 if y_prob[1] > cutoff else 0 for y_prob in y_probs]
    return y_hats

def evaluate(y_test, y_pred, cm=True):
    '''
    print recall and precision, and display confusion matrix if cm=True
    return a list of scores
    ''' 
    # accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)    
    if cm:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred);
    return {'recall':recall, 'precision':precision, 'f1':f1}

def dl_evaluate(estomator, X_test, y_test, cutoff):
    '''used to evaluate deep learning network'''
    y_probs = estomator.predict(X_test)
    y_pred = [1 if y_prob > cutoff else 0 for y_prob in y_probs]
    return evaluate(y_test, y_pred)


# Import pipeline making libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline

def preprocessing(X, scaler=None):
    '''
    Preprocess an input dataframe - categorical values being one-hot encoded, numerical values being scaled
    Input: a dataframe
    Return: a preprocessed dataframe
    '''
    
    # one hot encoding
    data = col_name_cnvt(X)
    data = data.reset_index(drop=True)
    cat_cols = pd.get_dummies(data['type'])
    
    # scaling
    num_cols = data.iloc[:, 1:]  #select all numerical columns
    
    if not scaler:
        scaler = StandardScaler()
    nums = scaler.fit_transform(num_cols)    
    # concate them in a new dataframe
    sc_num_cols = pd.DataFrame(nums)    
    sc_num_cols.columns = num_cols.columns
    df = pd.concat([cat_cols, sc_num_cols], axis=1)    
    return df, scaler    

def create_pipe(clf):
    '''
    1. This function makes data ready using col_name_cnvt()
    2. One-hot encode the categorical columns and scale the numerical columns
    3. Make a pipeline
    Input: An estimator
    Output: A pipeline object

    '''
    process_cols = FunctionTransformer(col_name_cnvt) # col_name_cnvt() prepares the data ready
    ohe = OneHotEncoder()
    scaler = StandardScaler()

    cat_cols = make_column_selector(dtype_exclude='number')
    num_cols = make_column_selector(dtype_include='number')

    preprocessor = make_column_transformer(
        (ohe, cat_cols),
        (scaler, num_cols)
    )    

    pipe = Pipeline([
        ('proc_cols', process_cols),
        ('prep', preprocessor),
        ('model', clf)
    ])

    return pipe


import numpy as np
from sklearn.model_selection import StratifiedKFold

def k_fold(model, X_train, y_train, k=10, cutoff=None):
    '''
    Cross validate the model.
    Input: model - estimator to be validated
           X_train, y_train - data to be splited
           k - number of fold, default 10
           cutoff - specified when using soft decision rule
    Return: A dictionary of scores of recall, decision, and f1
    '''
    k_fold = k        
    kf = StratifiedKFold(n_splits=k_fold)

    # do the split and train
    scores = []
    for train_idx, test_idx in kf.split(X_train, y_train):    
        X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model.fit(X_train_fold, y_train_fold)        
        y_hats = soft_to_hard(model, X_test_fold, cutoff=cutoff)
        score = evaluate(y_test_fold, y_hats, cm=False)
        scores.append(score)

    # calculate the average recall, precision, and f1
    recall, precision, f1 = [], [], []    
    for score in scores:    
        recall.append(score['recall'])
        precision.append(score['precision'])
        f1.append(score['f1'])

    return {'Mean Recall': np.array(recall).mean(),
            'Mean Precision': np.array(precision).mean(),
            'Mean F1': np.array(f1).mean()
           }


from tensorflow.keras.losses import msle, mae
import matplotlib.pyplot as plt

def dist_plot(model, X_normal_test, X_anomaly_test, threshold):
    '''
    Plot the distribution of normal data and anomaly data in the same figure
    Input: 
       model: estimator
       X_normal_test: New normal data
       X_anomaly_test: New anomaly data
       threshold: The average loss found by `find_threshold()` function

    Return: None
    '''
    reconstructions_normal = model.predict(X_normal_test)
    train_loss = mae(reconstructions_normal, X_normal_test)
    reconstructions_anomaly = model.predict(X_anomaly_test)
    train_loss_a = mae(reconstructions_anomaly, X_anomaly_test)    
    plt.hist(train_loss[None, :], bins=50, label='normal')
    plt.hist(train_loss_a[None, :], bins=50, label='anomaly')
    plt.axvline(threshold, 
                color='r', linewidth=2, linestyle='--',
                label='threshold')
    plt.legend()
    plt.show();

def find_threshold(model, X_normal_test):
    '''
    Find and return the threshold of minimum loss
    given the model and normal data
    '''
    reconstructions = model.predict(X_normal_test)
    
    # provides losses of individual instances
    train_loss = mae(reconstructions, X_normal_test)    
    
    # threshold for anomaly scores 
    # using 2 stadard deviation
    threshold = np.mean(train_loss) + 2*np.std(train_loss)
    
    return threshold

def get_predictions(model, X_test, threshold):
    '''
    Make prediction on new unseen data using the "threshold" given
    Input: 
       model: predictive estomator
       X_test: test data
       threshold: threshold used for prediction
    Return: the predicted labels
    '''
    predictions = model.predict(X_test)
    
    # provides losses of individual instances
    errors = mae(predictions, X_test)
    
    # 1 = anomaly, 0 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 1 if x == True else 0)
    
    return preds