from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot  as plt
import seaborn as sns
import pandas as pd

def cast_columns(table, table_name, col_classes):
    numeric_columns = col_classes[table_name]["numeric"]
    categorical_columns = col_classes[table_name]["categorical"]

    for numeric_column in numeric_columns:
        table[numeric_column] = table[numeric_column].astype(np.float64).values

    for categorical_column in categorical_columns:
        table[categorical_column] = table[categorical_column].astype(str).values
    return table

def expand_data(table , id_column, id_date, t_length=None):
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
    encoders_i = {}
    for char_col in col_classes[table_name]["categorical"]:
        encoder = LabelEncoder()
        encoder.fit(table[char_col].values.tolist()+["MISSING"] )
        encoders_i[char_col] = encoder
    encoders[table_name] = encoders_i
    return encoders

def encoders_fit_nparray(np_data, table_name, col_classes, encoders):
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
    for char_col in col_classes[table_name]["categorical"]:
        encoder = encoders[table_name][char_col]
        table[char_col] = encoder.transform(table[char_col] )
    return table

def encoders_transform_nparray(np_data, table_name, col_classes, encoders):
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
    def __init__(self, col_classes):
        self.col_classes = col_classes
        self.encoders = {}

    def fit(self, np_data, table_name):
        encoders_i = {}
        if len(np_data.shape)==3:
            np_data = np_data.reshape([-1,np_data.shape[2]])
        for i, char_col in enumerate(self.col_classes[table_name]["categorical"]):
            encoder = LabelEncoder()
            encoder.fit(list(set(np_data[:,i]))+["MISSING"] )
            encoders_i[char_col] = encoder
        self.encoders[table_name] = encoders_i

    def transform(self, np_data, table_name):
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
    
class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.999, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)
    
def do_nothing(x_data):
    # For normalizer object, in case we don't wantr to apply any normalization
    return x_data  
    
    
class Batcher():
    def __init__(self, data, batch_size, shuffle_on_reset=False):
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
        np.random.shuffle(self.I)
        
    def reset(self):
        if self.shuffle_on_reset:
            self.shuffle()
        self.current = 0
        
    def next(self):
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



def plot_roc(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
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
    sns.set(color_codes=True)
    print("avg_value: {}\nmax_value: {}\nmin_value: {}".format(np.mean(scores), 
                                                               np.max(scores),
                                                               np.min(scores)))
    sns.distplot(scores,bins=bins)
    plt.show()
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Normalizer():
    def __init__(self, norm_function=sigmoid, cube=False):
        self.norm_function = norm_function
        self.cube = cube
    
    def normalize_train(self, x):
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
        if self.cube == False:
            res = self.norm_function((x-self.col_means)/self.col_sds)
        if self.cube == True:
            x_copy = x.copy()
            x_copy = x_copy.reshape([-1,x.shape[2]])
            res = self.norm_function((x_copy-self.col_means)/self.col_sds)
            res = np.reshape(res,x.shape)
            
        return res
    
def get_target_weights(target_array, target_weight=1):
    weights = np.ones(target_array.shape)
    weights[np.where(target_array==1)] = target_weight
    return weights


def expand_data(dataset , id_column, id_date, all_ids, t_length=None):
    dataset = dataset.sort_values(by=[id_date], ascending=False)
    dataset = pd.merge(pd.DataFrame({id_column : all_ids}),dataset, on=[id_column], how='left')
    dataset["incremental"] = np.arange(dataset.shape[0])
    dataset["rank"] = dataset.groupby(id_column)['incremental'].rank(ascending=True)
    
    if t_length==None:
        t_length = dataset.groupby([id_column])[id_column].count().max()
    unique_ids = pd.unique(dataset[id_column])
    repeated_ids = np.repeat(unique_ids, t_length)
    repeated_rank = np.tile(np.arange(1,t_length+1), len(unique_ids))
    all_comb_df = pd.DataFrame({id_column: repeated_ids,
                                    'rank': repeated_rank})

    cartesian_data = pd.merge(all_comb_df, dataset, on=["rank", id_column], how='left').fillna(0)
    cartesian_data = cartesian_data.sort_values(by=[id_column, "rank"] , ascending=[True, False])
    
    return cartesian_data   


def filter_id_values(col_list):
    id_cols = ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]
    return list(filter( lambda x : x not in id_cols, col_list))

def update_best(best_loss, best_auc, new_loss, new_auc):
    flag_new_best = False
    if new_loss < best_loss:
        best_loss = new_loss
        flag_new_best = True
    if new_auc > best_auc:
        best_auc = new_auc
        flag_new_best = True
    return flag_new_best, best_loss, best_auc
