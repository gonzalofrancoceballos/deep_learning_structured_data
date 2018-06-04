import tensorflow as tf
import os
import shutil
import tensorflow.contrib.layers as layers


class BatchNorm(object):
    '''
    Object for batch normalization
    '''
    def __init__(self, epsilon=1e-5, momentum=0.999, name="batch_norm"):
        '''
        :param epsilon: epsilon parameter of tf batch_norm object
        :param momentum: momentum parameter of tf batch_norm object        
        :param name: name of the tensorflow element in the graph 
        '''
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
        

class LayerNorm(object):
    '''
    Object for layer normalization
    '''
    def __init__(self,  name="layer_norm"):
        '''
        :param name: name of the tensorflow element in the graph 
        '''
        with tf.variable_scope(name):
            self.name = name

    def __call__(self, x, train=True):
        '''
        :param x: tf tensor that contiains the batch that will travel through the layer
        :param train: if true, batch normalization for train stage. If false, 
        batch normalization for predict stage
        :return: layer normalization function
        '''
        return tf.contrib.layers.layer_norm(inputs=x,
                                            center=True,
                                            scale=True,
                                            activation_fn=None,
                                            reuse=None,
                                            variables_collections=None,
                                            outputs_collections=None,
                                            trainable=train,
                                            begin_norm_axis=1,
                                            begin_params_axis=-1,
                                            scope=self.name)

        
def pseudo_res_net(x, units_sizes, name="ResNet", concat_axis=2, activation=tf.nn.relu, dropout=None):
    '''
    Code that implements a pseudo resnet. Input and output of the block are 
    concatenated, instead of added
    
    :param unit_sizes: list or array containing the sized each dense layer
    :param name: name of the tensorflow element in the graph 
    :param concat_axis: axis along which inout and output will be concatenated
    :param activation: activation function to use after each dense layer
    :param dropout: if not none, dropout will be added with keep_prob equal to 
    this parameter
    :return: tf operation containing the psudo res_net block
    '''
    output = x
    i=0
    with tf.variable_scope(name):
        for units_size in units_sizes:
            output = layers.linear(output, num_outputs=units_size, activation_fn=activation)
            if dropout is not None:
                output = tf.nn.dropout(output, dropout, name= 'do_{}_{}'.format(name,i))
            i=i+1
        return tf.concat([output, x], axis=concat_axis, name="concat_{}_{}".format(name, i))         


def get_embedding_placeholders(col_classes, table_name, name, time_window=None):
    '''
    Wrapper function that encapsulates the generation of the elements of a 
    tf.session placeholder feed_dict associated to categorical variables that will be fed 
    to embedding layers
    
    :param col_classes: dictionary containing column names and column types of 
    each table
    :param table_name: string containing the name of the table that the 
    categorical variables belong to
    :param name: name of the tensorflow element in the graph 
    :param time_window: length of the time dimension, in case data is a time 
    series
    :return: dictionary containing the embedding placeholders
    '''
    embedding_dictionary = dict()
    if time_window:
        for char_col in col_classes[table_name]["categorical"]:
            embedding_dictionary[char_col] = tf.placeholder(dtype=tf.int64, 
                                                        shape=((None,time_window,1)), 
                                                        name="{}_{}".format(name,char_col))
    else:
        for char_col in col_classes[table_name]["categorical"]:
            embedding_dictionary[char_col] = tf.placeholder(dtype=tf.int64, 
                                                        shape=((None,1)), 
                                                        name="{}_{}".format(name,char_col))
    return embedding_dictionary

        
def embedding_intake(emb_placeholders, char_cols, enconders_dict, embedding_sizes_dict, name, time_series=False):
    """
    Wrapper function that encapsulates the input of categorical variables of a 
    especific table
    
    :param emb_placeholders: dictionary ocntaining placeholders
    :param char_cols: list of names of categorical columns in the table
    :param encoders_dict: dictionary containing the encoder object used to 
    label-encode the categorical variables
    :param embedding_sizes_dict: dictionary containing the size of the output 
    of the embedding for each categorical variable
    :param name: name of the tensorflow element in the graph 
    :time_series: flag whether data is time series (3D cube) or not (2D table)
    :return: tf operation containing the concatenated input after embedding
    """
    embedding_vars = dict()
    for char_col in char_cols:
        embedding_vars[char_col] = tf.get_variable(name= "emb_tensor_" + char_col + "_" + name,
                                                   shape= [len(enconders_dict[char_col].classes_), 
                                                           embedding_sizes_dict[char_col]])
    # Embedding layers
    embedding_layer = dict()
    for char_col in char_cols:
        embedding_layer[char_col] = tf.nn.embedding_lookup(embedding_vars[char_col], 
                                                           emb_placeholders[char_col],
                                                           name= "embedding_lookup_" + char_col + "_" + name)
    # Concatenate embeddings
    embedding_list = []
    for char_col in char_cols:
        embedding_list.append(embedding_layer[char_col])

    if time_series == False:
        concat_embedding = tf.concat(embedding_list, axis=2, name="emb_concat_" + name)
        concat_embedding = tf.squeeze(concat_embedding, axis=1, name="squeeze_concat_" + name)
    if time_series == True:
        concat_embedding = tf.concat(embedding_list, axis=3, name="emb_concat_" + name)
        concat_embedding = tf.squeeze(concat_embedding, axis=2, name="squeeze_concat_" + name)        
    return concat_embedding

def dense_block(d_input, num_outputs, activation, name, dropout_rate=1):
    '''
    Wrapper function that a simple dense block consisting of 
    dropout--+--layer_norm--+--dense
    
    :param d_input: tf tensor containing the input to the block
    :param num_outputs: since of the dense layer
    :param activation: activation in the dense layer
    :param name: name of the tensorflow element in the graph 
    :param dropout_rate: keep_prob of the dropout layer
    :return: tf operation containing output of the block
    '''
    d_output = tf.nn.dropout(d_input, keep_prob=dropout_rate, name="do_{}.".format(name))
    d_output = LayerNorm(name="do_{}.".format(name))(d_output)
    d_output = layers.linear(d_output, num_outputs=num_outputs, activation_fn=activation)
    return d_output


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def swish(x, name=None):
    '''
    Famous Google activation function
    
    :param x: tf tensor input to the function
    :param name: name of the tensorflow element in the graph 
    :return: tf tensor
    '''
    return tf.multiply(x , tf.nn.sigmoid(x), name=name)


def get_tensorflow_configuration(device="0", memory_fraction=1):
    """
    Function for selecting the GPU to use and the amount of memory the process is allowed to use
    :param device: which device should be used (str)
    :param memory_fraction: which proportion of memory must be allocated (float)
    :return: config to be passed to the session (tf object)
    """
    device = str(device)

    if device:
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        config.gpu_options.visible_device_list = device
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    return config

def start_tensorflow_session(device="0", memory_fraction=1):
    """
    Starts a tensorflow session taking care of what GPU device is going to be used and
    which is the fraction of memory that is going to be pre-allocated.
    :device: string with the device number (str)
    :memory_fraction: fraction of memory that is going to be pre-allocated in the specified
    device (float [0, 1])
    :return: configured tf.Session
    """
    return tf.Session(config=get_tensorflow_configuration(device=device, memory_fraction=memory_fraction))

def get_summary_writer(session, logs_path, project_id, version_id):
    """
    For Tensorboard reporting
    :param session: opened tensorflow session (tf.Session)
    :param logs_path: path where tensorboard is looking for logs (str)
    :param project_id: name of the project for reporting purposes (str)
    :param version_id: name of the version for reporting purposes (str)
    :return summary_writer: the tensorboard writer
    """
    path = os.path.join(logs_path,"{}_{}".format(project_id, version_id)) 
    if os.path.exists(path):
        shutil.rmtree(path)
    summary_writer = tf.summary.FileWriter(path, graph_def=session.graph_def)  #graph_def=session.graph
    return(summary_writer)

        
