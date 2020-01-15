from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot  as plt
import seaborn as sns
import pandas as pd

def cast_columns(table, table_name, col_classes):
    '''
    Function to cast column types of a pd.DataFrame
    
    :param table: pd.DataFrame which columns need to be casted
    :param table_name: name of the table in col_classes dictionary    
    :param col_classes: dictionary containing column names and column types of 
    each table
    :return: modified pd.DataFrame
    '''
    numeric_columns = col_classes[table_name]["numeric"]
    categorical_columns = col_classes[table_name]["categorical"]

    for numeric_column in numeric_columns:
        table[numeric_column] = table[numeric_column].astype(np.float64).values

    for categorical_column in categorical_columns:
        table[categorical_column] = table[categorical_column].astype(str).values
    return table

def expand_data(table , id_column, id_date, t_length=None):
    '''
    Given a pd.DataFrame, expand it in such a way that each ID is repeated the 
    same number of times. Padding with NAs or deleting exceeded IDs when 
    necessary
    
    :param table: pd.DataFrame to be modified
    :param id_column: name of the column conmtaining the IDs
    :param id_date: column containing time-based or ordinal data column on which data
    will be sorted
    :param t_length: number of times each id needs to appear
    :return: modified pd.DataFrame
    '''
    if t_length==None:
        t_length = table.groupby([id_column])[id_column].count().max()
    unique_ids = pd.unique(table[id_column])
    repeated_ids = np.repeat(unique_ids, t_length)
    repeated_rank = np.tile(np.arange(1,t_length+1), len(unique_ids))
    all_comb_df = pd.DataFrame({id_column: repeated_ids,
                                    'rank': repeated_rank})

    cartesian_data = pd.merge(all_comb_df, table, on=["rank", id_column], how='left').fillna(0)
    cartesian_data = cartesian_data.sort_values(by=[id_column, id_date])
    return cartesian_data

def encoders_fit(table, table_name, col_classes, encoders):
    '''
    Fits LabelEncoder objects and stores them in a dictionary
    
    :param table: pd.DataFrame containing categorical variables that need to be 
    encoded
    :param table_name: number of the table. It will be used to store the 
    LabelEncoder objects in the dictionary
    :param col_classes: dictionary containing columns and dtypes of all tables
    :param encoders: dinctionary of LabelEncoder objects
    '''
    encoders_i = {}
    for char_col in col_classes[table_name]["categorical"]:
        encoder = LabelEncoder()
        encoder.fit(table[char_col].values.tolist()+["MISSING"] )
        encoders_i[char_col] = encoder
    encoders[table_name] = encoders_i
    return encoders

def encoders_fit_nparray(np_data, table_name, col_classes, encoders):
    '''
    Fits LabelEncoder objects and stores them in a dictionary
    
    :param np_data: np.array containing categorical variables that need to be 
    encoded
    :param table_name: number of the table. It will be used to store the 
    LabelEncoder objects in the dictionary
    :param col_classes: dictionary containing columns and dtypes of all tables
    :param encoders: dinctionary of LabelEncoder objects
    '''
    encoders_i = {}
    if len(np_data.shape)==3:
        np_data = np_data.reshape([-1,np_data.shape[2]])

    for i, char_col in enumerate(col_classes[table_name]["categorical"]):
        encoder = LabelEncoder()
        encoder.fit(list(set(np_data[:,i]))+["MISSING"] )
        encoders_i[char_col] = encoder
    encoders[table_name] = encoders_i
    return encoders

def encoders_transform(table, table_name, col_classes, encoders):
    '''
    Given a dictionary of pre-fitted LabelEncoder objects, use them to encode
    categorical variables in a np.array
    
    :param table: pd.DataFrame containing the categorical variables that need to 
    be encoded
    :param table_name: name of the table from which the np.array was created
    :param col_classes: dictionary containing columns and dtypes of all tables
    :param encoders: dictionary of LabelEncoder objects
    '''
    for char_col in col_classes[table_name]["categorical"]:
        encoder = encoders[table_name][char_col]
        table[char_col] = encoder.transform(table[char_col] )
    return table

def encoders_transform_nparray(np_data, table_name, col_classes, encoders):
    '''
    Given a dictionary of pre-fitted LabelEncoder objects, use them to encode
    categorical variables in a np.array
    
    :param np_data: np.array containing the categorical variables that need to 
    be encoded
    :param table_name: name of the table from which the np.array was created
    :param col_classes: dictionary containing columns and dtypes of all tables
    :param encoders: dictionary of LabelEncoder objects
    '''
    for i, char_col in enumerate(col_classes[table_name]["categorical"]):
        encoder = encoders[table_name][char_col]
        
        if len(np_data.shape)==2:
            np_data[np.isin( np_data[:,i] , encoder.classes_), i] = "MISSING"
            np_data[:,i] = encoder.transform(np_data[:,i])
            
        if len(np_data.shape)==3:
            original_shape = np_data.shape
            np_data = np_data.reshape([-1,np_data.shape[2]])
            np_data[np.isin( np_data[:,i] , encoder.classes_), i] = "MISSING"
            np_data[:,i] = encoder.transform(np_data[:,i])
            np_data = np_data.reshape(original_shape)
            
    return np_data

class Encoders():
    '''
    Class to manage label encoding
    '''
    def __init__(self, col_classes):
        ''''
        :param col_classes: dictionary containing columnd and dtypes of all tables
        '''
        self.col_classes = col_classes
        self.encoders = {}

    def fit(self, np_data, table_name):
        '''
        Fits LabelEncoder objects and stores them in self.encoders        
        :param np_data: np.array containing categorical variables that need to 
        be encoded
        :param table_name: number of the table. It will be used to store the 
        '''
        encoders_i = {}
        if len(np_data.shape)==3:
            np_data = np_data.reshape([-1,np_data.shape[2]])
        for i, char_col in enumerate(self.col_classes[table_name]["categorical"]):
            encoder = LabelEncoder()
            encoder.fit(list(set(np_data[:,i]))+["MISSING"] )
            encoders_i[char_col] = encoder
        self.encoders[table_name] = encoders_i

    def transform(self, np_data, table_name):
        '''
        Use pre-fitted LabelEncoder objects in self.encoders to encode
        categorical variables in a np.array        
        :param np_data: np.array containing the categorical variables that need 
        to be encoded
        :param table_name: name of the table from which the np.array was 
        created
        '''
        for i, char_col in enumerate(self.col_classes[table_name]["categorical"]):
            encoder = self.encoders[table_name][char_col]
            if len(np_data.shape)==2:
                np_data[~np.isin( np_data[:,i] , encoder.classes_), i] = "MISSING"
                np_data[:,i] = encoder.transform(np_data[:,i])

            if len(np_data.shape)==3:
                original_shape = np_data.shape
                np_data = np_data.reshape([-1,np_data.shape[2]])
                np_data[~np.isin( np_data[:,i] , encoder.classes_), i] = "MISSING"
                np_data[:,i] = encoder.transform(np_data[:,i])
                np_data = np_data.reshape(original_shape)
        return np_data
    

    
def do_nothing(x_data):
    '''For normalizer object, in case we do not want 
    to apply any normalization'''
    return x_data  
    
    
class Batcher():
    '''
    Batcher class. Given a list of np.arrays of same 0-dimension, returns a 
    a list of batches for these elements
    '''
    def __init__(self, data, batch_size, shuffle_on_reset=False):
        '''
        :param data: list containing np.arrays
        :param batch_size: size of each batch
        :param shuffle_on_reset: flag to shuffle data
        '''
        self.data = data
        self.batch_size = batch_size
        self.shuffle_on_reset = shuffle_on_reset
        
        if type(data) == list:
            self.data_size = data[0].shape[0]
        else:
            self.data_size = data.shape[0]
        self.n_batches = int(np.ceil(self.data_size/self.batch_size))
        self.I = np.arange(0, self.data_size, dtype=int)
        if shuffle_on_reset:
            np.random.shuffle(self.I)
        self.current = 0
        
    def shuffle(self):
        '''
        Re-shufle the data
        '''
        np.random.shuffle(self.I)
        
    def reset(self):
        '''
        Reset iteration counter
        '''
        if self.shuffle_on_reset:
            self.shuffle()
        self.current = 0
        
    def next(self):
        '''
        Get next batch
        :return: list of np.arrays
        '''
        I_select = self.I[(self.current*self.batch_size):((self.current+1)*self.batch_size)]
        batch = []
        for elem in self.data:
            batch.append(elem[I_select])
        
        if(self.current<self.n_batches-1):
            self.current = self.current+1
        else:
            self.reset()
            
        return batch
    
def train_dev_test_split(data, train_size=0.9, random_state=1):
    '''
    Given a list of np.arrays, sample them in train, dev and test, where dev 
    and test sizes will be the same
    
    :param data: list of np.arrays
    :param train_size: proportion of train data. Dev and test sizes will be 
    (1-train_size)/2
    :param random_state: seed
    '''
    I = np.arange(0, data[0].shape[0])
    I_train, I_dev_test = train_test_split(I, test_size=1-train_size, random_state=random_state)
    I_dev, I_test = train_test_split(I_dev_test, test_size=0.5, random_state=random_state)
    
    train_split = []
    dev_split = []
    test_split= []

    for elem in data:
        train_split.append(elem[I_train])
        dev_split.append(elem[I_dev])
        test_split.append(elem[I_test])
    
    return  train_split, dev_split, test_split

def plot_roc(labels, preds):
    '''
    Plots roc curve
    
    :param labels: list-like object containing the labels
    :preds: list-like object containit the predictions
    '''
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
   
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def get_uplift(y_true,y_score, N=100, plot=None):
    '''
    Get uplift table
    
    :param y_true: labels
    :param y_score: scores
    :param N: number of buckets
    :param plot: if not none, string containing name of the plot
    :return: pd.DataFrame with several metrics
    '''
    uplift_df = pd.DataFrame({"y_true":np.reshape(y_true, (-1)),
                              "y_score":np.reshape(y_score, (-1))})

    uplift_df = uplift_df.sort_values(by=["y_score"], ascending=False)
    uplift_df["rank"] = 1+np.arange(uplift_df.shape[0])
    uplift_df["n_tile"] = np.ceil(N* uplift_df["rank"] / uplift_df.shape[0])

    uplift_result = uplift_df.groupby(["n_tile"])["y_true"].agg({'N':'count','captured':"sum", "perc_captured":"mean"})
    uplift_result['N_acum'] = uplift_result['N'].cumsum()
    uplift_result['captured_acum'] = uplift_result['captured'].cumsum()
    uplift_result['perc_captured_acum'] = uplift_result['captured_acum'] / uplift_result['N_acum']
    uplift_result['uplift'] = uplift_result['perc_captured'] / np.mean(y_score)
    uplift_result['uplift_acum'] = uplift_result['perc_captured_acum'] / np.mean(y_true)
    uplift_result['n_tile'] = 1+np.arange(uplift_result.shape[0])
    
    if plot:
        uplift_result.plot.bar(x=["n_tile"], y=[plot])
        plt.show()
        
    return uplift_result[['n_tile','N', 'N_acum', 'captured', 'captured_acum', 'perc_captured', 
                          'perc_captured_acum', 'uplift', 'uplift_acum']]

def print_distribution(scores, bins=250):
    '''
    Prints the distribution of a continous variable
    '''
    sns.set(color_codes=True)
    print("avg_value: {}\nmax_value: {}\nmin_value: {}".format(np.mean(scores), 
                                                               np.max(scores),
                                                               np.min(scores)))
    sns.distplot(scores,bins=bins)
    plt.show()
    
    
def sigmoid(x):
    '''
    Wrapper function for sigmoid in np.arrays
    '''
    return 1 / (1 + np.exp(-x))

class Normalizer():
    '''
    Class for normalization. Applies a basic normalization to center data 
    around zero and modify sd to 1, and then applies a second cunstom 
    normalization
    '''
    def __init__(self, norm_function=sigmoid, cube=False):
        '''
        :param norm_function: normalization function to be applied after basic 
        normalization
        :param cube: for the case of time series data, where it comes in 
        3D-shape 
        '''
        self.norm_function = norm_function
        self.cube = cube
    
    def normalize_train(self, x):
        '''
        Apply normalization on train data
        :param x: np.array to be normalized
        :return: normalized np.array
        '''
        if self.cube == False:
            self.col_means = np.mean(x,axis=0)
            self.col_sds = np.std(x,axis=0)
            res = self.norm_function((x-self.col_means)/self.col_sds)
        if self.cube == True: 
            x_copy = x.copy()
            x_copy = x_copy.reshape([-1,x.shape[2]])
            self.col_means = np.mean(x_copy,axis=0) 
            self.col_sds = np.std(x_copy,axis=0)
            res = self.norm_function((x_copy-self.col_means)/self.col_sds)
            res = np.reshape(res,x.shape)
        return res
    
    def normalize_predict(self, x):
        '''
        Apply normalization on predict data using statistics used to normalize
        train data
        :param x: np.array to be normalized
        :return: normalized np.array
        '''
        if self.cube == False:
            res = self.norm_function((x-self.col_means)/self.col_sds)
        if self.cube == True:
            x_copy = x.copy()
            x_copy = x_copy.reshape([-1,x.shape[2]])
            res = self.norm_function((x_copy-self.col_means)/self.col_sds)
            res = np.reshape(res,x.shape)
            
        return res
    
def get_target_weights(target_array, target_weight=1):
    '''
    Get target weights for unbalenced data
    '''
    weights = np.ones(target_array.shape)
    weights[np.where(target_array==1)] = target_weight
    return weights

def filter_id_values(col_list):
    '''
    Filter ID columns from a list of column names
    
    :param col_list: list of column names
    :return: filtered list
    '''
    id_cols = ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]
    return list(filter( lambda x : x not in id_cols, col_list))

def update_best(best_loss, best_auc, new_loss, new_auc):
    '''
    Helper function keep track of best training iteration
    :param best_loss: best value of loss function so far
    :param best_auc: best value of AUC so far
    :param new_loss: new value of loss function to be compared with previous 
    best
    :param new_auc: new value of AUC to be compared with previous best
    '''
    flag_new_best = False
    if new_loss < best_loss:
        best_loss = new_loss
        flag_new_best = True
    if new_auc > best_auc:
        best_auc = new_auc
        flag_new_best = True
    return flag_new_best, best_loss, best_auc


def one_hot_encoding(df, str_col, labels=None, drop=True, add_other=True):
    """
    Generates onh-hot-encoding variables

    :param df: table to modify (type: pd.DataFrame)
    :param str_col: name of column to one-hot-encode (type: str)
    :param labels: possible labels (type: str)
    :param drop: flag to drop str_col after computing one-hot-encoding (type: bool)
    :param add_other: flag to add an "other" column for those values not in labels (type: bool)
    :return: modified table (type: pd.DataFrame)
    """

    if labels is None:
        logger.warning(
            "[ONE-HOT-ENCODING] No label list especified, inferring from data"
        )
        labels = df[str_col].unique()
    logger.info(f"[ONE-HOT-ENCODING] Labels are: {labels}")

    logger.info(f"[ONE.HOT-ENCODING] Generating new variables")
    for label in labels:
        df[f"{str_col}_{label}"] = np.where(df[str_col] == label, 1, 0)

    if add_other:
        logger.info(f"[ONE-HOT-ENCODING] Including variable for missing label")
        df[f"{str_col}_OTHER"] = np.where(~df[str_col].isin(labels), 1, 0)

    if drop:
        logger.info(f"[ONE-HOT-ENCODING] Dropping original categorical column")
        df.drop(columns=str_col, axis=1, inplace=True)

    return df