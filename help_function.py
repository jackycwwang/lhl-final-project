# Import metrics libraries
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score


def col_name_cnvt(df):
    '''
    Take the columns needed,
    and remove space and units, and replace column names with lowercase.
    It does a inplace operation.
    
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
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)    
    if cm:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred);
    return {'recall':recall, 'precision':precision, 'f1':f1}


# Import pipeline making libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline


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
from sklearn.model_selection import StratifiedKFold, KFold

def k_fold(model, X_train, y_train, k=20, cutoff=0.5):
    k_fold = k        
    kf = KFold(n_splits=k_fold)

    # do the split and train
    scores = []
    for train_idx, test_idx in kf.split(X_train):    
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
    print('Mean Recall: ', np.array(recall).mean())
    print('Mean Precision: ', np.array(precision).mean())
    print('Mean F1: ', np.array(f1).mean())