# Import metrics libraries
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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

def evaluate(y_test, y_pred):
    '''print recall and precision, and display confusion matrix'''
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred);


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
