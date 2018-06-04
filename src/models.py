import model_utils as M
import utils as U
import tensorflow as tf

class DLModel(object):
    def __init__(self, batch_size, col_classes, learning_rate, encoders, embedding_sizes, t_windows_dic, activation=tf.nn.relu):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_sizes = embedding_sizes
        self.activation = activation
        self.encoders = encoders
        self.col_classes = col_classes
        self.t_window_bureau = t_windows_dic["bureau"]
        self.t_window_prev_app = t_windows_dic["previous_application"]
        self.set_graph()
        self.model_saver = tf.train.Saver()
    
    def set_graph(self):
        tf.reset_default_graph()
        self.placeholders = M.NameSpacer(**self.get_placeholders())
        self.forward =  M.NameSpacer(**self.get_forward_operations())
        self.losses =  M.NameSpacer(**self.get_loss())
        self.optimizers = M.NameSpacer(**self.get_optimizer())
        self.metrics = M.NameSpacer(**self.get_metrics())
        self.summaries = M.NameSpacer(**self.get_summaries())
        
    def get_placeholders(self):
        with tf.variable_scope("Placeholders"):
            # train_application
            numeric_input = tf.placeholder(dtype=tf.float32, 
                                           shape=(None, 2*len(U.filter_id_values(self.col_classes["application_train"]["numeric"]))), 
                                           name="ph_numeric")  
            numeric_input_bureau = tf.placeholder(dtype=tf.float32, 
                                           shape=(None, self.t_window_bureau, 2*len(U.filter_id_values(self.col_classes["bureau"]["numeric"]))), 
                                           name="ph_numeric_bureau")
            numeric_input_prev_app = tf.placeholder(dtype=tf.float32, 
                                           shape=(None, self.t_window_prev_app, 2*len(U.filter_id_values(self.col_classes["previous_application"]["numeric"]))), 
                                           name="ph_numeric_prev_app")
            target = tf.placeholder(dtype=tf.float32, 
                                    shape=((None,1)),
                                    name="ph_target")
            
            # Embeddings
            embedding = M.get_embedding_placeholders(self.col_classes, table_name="application_train", name="ph_categorical")
            embedding_bureau = M.get_embedding_placeholders(self.col_classes, table_name="bureau", name="ph_categorical_bureau", time_window=self.t_window_bureau)
            embedding_prev_app = M.get_embedding_placeholders(self.col_classes, table_name="previous_application", name="ph_categorical_bureau", time_window=self.t_window_prev_app)


            dropout_1 = tf.placeholder_with_default(1.0, shape=())
            dropout_2 = tf.placeholder_with_default(1.0, shape=())
            dropout_3 = tf.placeholder_with_default(1.0, shape=())

            loss_weights =  tf.placeholder(dtype=tf.float32, shape=(None, 1))
            
            placeholders = {"numeric_input" : numeric_input,
                            "numeric_input_bureau" : numeric_input_bureau,
                            "numeric_input_prev_app" : numeric_input_prev_app,
                            "target" : target,
                            "embedding" : embedding,
                            "embedding_bureau" : embedding_bureau,
                            "embedding_prev_app" : embedding_prev_app,
                            "dropout_1" : dropout_1,
                            "dropout_2" : dropout_2,
                            "dropout_3" : dropout_3,
                            "loss_weights" : loss_weights}
            
            return placeholders
    
    def get_forward_operations(self):
        with tf.variable_scope("Forward"):
            
            ########################
            ### APPLICATION FLOW
            ########################
            with tf.variable_scope("application_flow"):
                ## Embedding flow ##
                concat_embedding_app = M.embedding_intake(emb_placeholders=self.placeholders.embedding, 
                                                        char_cols=self.col_classes["application_train"]["categorical"], 
                                                        enconders_dict=self.encoders["application_train"], 
                                                        embedding_sizes_dict=self.embedding_sizes["application_train"], 
                                                        name="application_train",
                                                        time_series=False)
                concat_all = tf.concat([self.placeholders.numeric_input, concat_embedding_app], axis=1)
                concat_all = M.dense_block(concat_all, num_outputs=16, activation=self.activation, dropout_rate=self.placeholders.dropout_1, name="concat_all_1")
                concat_all = M.dense_block(concat_all, num_outputs=16, activation=self.activation, dropout_rate=self.placeholders.dropout_2, name="concat_all_2")
                
            ########################
            ### BUREAU FLOW
            ########################
            with tf.variable_scope("bureau_flow"):
                ## Embedding flow ##
                concat_embedding_bureau = M.embedding_intake(emb_placeholders=self.placeholders.embedding_bureau, 
                                                        char_cols=self.col_classes["bureau"]["categorical"], 
                                                        enconders_dict=self.encoders["bureau"], 
                                                        embedding_sizes_dict=self.embedding_sizes["bureau"], 
                                                        name="bureau",
                                                        time_series=True)

                # Merge numeric and embedding flows
                concat_bureau = tf.concat([self.placeholders.numeric_input_bureau, concat_embedding_bureau],axis=2)
                concat_bureau = tf.nn.dropout(concat_bureau, self.placeholders.dropout_1, name="do_concat_bureau")            
                concat_bureau = M.LayerNorm(name="ln_concat_bureau")(concat_bureau)
                
                # LSTM definition

                bureau_cell = tf.nn.rnn_cell.BasicLSTMCell(16, activation=tf.nn.relu)
                batch_size    = tf.shape(concat_bureau)[1]
                initial_state = bureau_cell.zero_state(batch_size, tf.float32)

                # Passing data through rnn
                rnn_outputs_bureau, rnn_states_bureau = tf.nn.dynamic_rnn(bureau_cell, concat_bureau, initial_state=initial_state, time_major=True)
                rnn_output_bureau = rnn_outputs_bureau[:,-1,:]
                
                rnn_output_bureau = M.dense_block(rnn_output_bureau, num_outputs=16, activation=self.activation, dropout_rate=self.placeholders.dropout_1, name="concat_bureau_1")
                rnn_output_bureau = M.dense_block(rnn_output_bureau, num_outputs=16, activation=self.activation, dropout_rate=self.placeholders.dropout_2, name="concat_bureau_2")
            
            ########################
            ### PREV_APP FLOW
            ########################
            with tf.variable_scope("prev_app_flow"):
                ## Embedding flow ##
                concat_embedding_prev_app = M.embedding_intake(emb_placeholders=self.placeholders.embedding_prev_app, 
                                                             char_cols=self.col_classes["previous_application"]["categorical"], 
                                                             enconders_dict=self.encoders["previous_application"], 
                                                             embedding_sizes_dict=self.embedding_sizes["previous_application"], 
                                                             name="prev_app",
                                                             time_series=True)

                # Merge numeric and embedding flows
                concat_prev_app = tf.concat([self.placeholders.numeric_input_prev_app, concat_embedding_prev_app],axis=2)
                concat_prev_app = tf.nn.dropout(concat_prev_app, self.placeholders.dropout_1, name="do_concat_prev_app")            
                concat_prev_app = M.LayerNorm(name="ln_concat_prev_app")(concat_prev_app)

                # LSTM definition
                prev_app_cell = tf.nn.rnn_cell.BasicLSTMCell(16, activation=tf.nn.relu)
                batch_size_prev_app    = tf.shape(concat_prev_app)[1]
                initial_state_prev_app = prev_app_cell.zero_state(batch_size_prev_app, tf.float32)

                # Passing data through rnn
                rnn_outputs_prev_app, rnn_states_prev_app = tf.nn.dynamic_rnn(prev_app_cell, concat_prev_app, initial_state=initial_state_prev_app, time_major=True)
                rnn_output_prev_app = rnn_outputs_prev_app[:,-1,:]
                rnn_output_prev_app = M.dense_block(rnn_output_prev_app, num_outputs=16, activation=self.activation, dropout_rate=self.placeholders.dropout_1, name="concat_prev_app_1")
                rnn_output_prev_app = M.dense_block(rnn_output_prev_app, num_outputs=16, activation=self.activation, dropout_rate=self.placeholders.dropout_1, name="concat_prev_app_2")
            
            ########################
            ### MAIN FLOW
            ########################
            with tf.variable_scope("main_flow"):
                # Concat flows
                concat_main_flow = tf.concat([concat_all, rnn_output_bureau, rnn_output_prev_app], axis=1)                
                dense_output_3 = M.dense_block(concat_main_flow, num_outputs=32, activation=self.activation, dropout_rate=self.placeholders.dropout_2, name="dense_output_3")
                dense_output_4 = M.dense_block(dense_output_3, num_outputs=32, activation=self.activation, dropout_rate=self.placeholders.dropout_2, name="dense_output_4")
                dense_output_5 = M.dense_block(dense_output_4, num_outputs=8, activation=self.activation, dropout_rate=self.placeholders.dropout_2, name="dense_output_5")
                dense_output_6 = M.dense_block(dense_output_5, num_outputs=8, activation=self.activation, dropout_rate=self.placeholders.dropout_3, name="dense_output_6")
                dense_output_7 = M.dense_block(dense_output_6, num_outputs=2, activation=self.activation, dropout_rate=self.placeholders.dropout_3, name="dense_output_7")
               
                # Output layer
                pred = tf.layers.linear(dense_output_7, num_outputs=1, activation_fn=tf.nn.sigmoid)

            return {"pred" : pred}

    def get_loss(self):
        with tf.variable_scope("Loss"):
            logloss = tf.reduce_mean(tf.losses.log_loss(labels = self.placeholders.target,
                                                        predictions=self.forward.pred,
                                                        weights=self.placeholders.loss_weights)) 
            
            return {"logloss" : logloss}
    
    def get_optimizer(self):
        with tf.variable_scope("Optimizer"):
            train_fn = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.losses.logloss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)            
            grads = tf.gradients(self.losses.logloss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, 50)
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars)
            
            return {"optimizer" : train_fn,
                    "optimizer_clip":train_op}

    def get_metrics(self):
        with tf.variable_scope("Accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.abs(self.placeholders.target - self.forward.pred) < 0.5, tf.float32))
            auc, auc_update_op = tf.metrics.auc(labels=self.placeholders.target, predictions=self.forward.pred,num_thresholds=1000)
            return {"accuracy" : accuracy, 
                    "auc" : auc,
                    "auc_update_op":auc_update_op}
        
    def get_summaries(self):
        with tf.variable_scope("Summaries"):
            loss_train = tf.summary.scalar(name="loss_train", tensor=self.losses.logloss)
            loss_dev = tf.summary.scalar(name="loss_dev", tensor=self.losses.logloss)
            loss_test = tf.summary.scalar(name="loss_test", tensor=self.losses.logloss)
            accuracy_train = tf.summary.scalar(name="accuracy_train", tensor=self.metrics.accuracy)
            accuracy_dev = tf.summary.scalar(name="accuracy_dev", tensor=self.metrics.accuracy)
            accuracy_test = tf.summary.scalar(name="accuracy_test", tensor=self.metrics.accuracy)
            auc_train = tf.summary.scalar(name="auc_train", tensor=self.metrics.auc)
            auc_dev = tf.summary.scalar(name="auc_dev", tensor=self.metrics.auc)
            auc_test = tf.summary.scalar(name="auc_test", tensor=self.metrics.auc)
            
            return({"loss_train": loss_train,
                    "loss_dev": loss_dev,
                    "loss_test": loss_test,
                    "accuracy_train": accuracy_train,
                    "accuracy_dev": accuracy_dev,
                    "accuracy_test": accuracy_test,
                    "auc_train": auc_train,
                    "auc_dev": auc_dev,
                    "auc_test": auc_test})
    