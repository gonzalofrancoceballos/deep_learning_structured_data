# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import model_utils as M
import utils as U
import constants as C


def train_tast(dl_model, processed_train_outputs,  n_epoch, seed, device,
               dropout_rate_1, dropout_rate_2,dropout_rate_3, batch_size,
               log_dir, project_name, version):
    
    [target_train, target_dev, target_test]= processed_train_outputs["targets"]
    [categorical_input_train, categorical_input_dev, categorical_input_test,
     categorical_input_train_bureau, categorical_input_dev_bureau, categorical_input_test_bureau,
     categorical_input_train_prev_app,categorical_input_dev_prev_app, categorical_input_test_prev_app] = processed_train_outputs["categorical_inputs"]    
    
    [numeric_input_train, numeric_input_dev, numeric_input_test,
     numeric_input_train_bureau, numeric_input_dev_bureau, numeric_input_test_bureau,
     numeric_input_train_prev_app, numeric_input_dev_prev_app, numeric_input_test_prev_app] = processed_train_outputs["normalized_data"]
    
    BATCH_SIZE = batch_size
    N_EPOCH = n_epoch
    DROPUT_RATE_1 = dropout_rate_1
    DROPUT_RATE_2 = dropout_rate_2
    DROPUT_RATE_3 = dropout_rate_3
    SEED = seed
    DEVICE = device
    LOG_DIR = log_dir
    PROJECT_NAME = project_name
    VERSION = version


    train_batcher = U.Batcher([numeric_input_train, 
                         target_train, 
                         categorical_input_train, 
                         numeric_input_train_bureau, categorical_input_train_bureau,
                         numeric_input_train_prev_app, categorical_input_train_prev_app],  
                        BATCH_SIZE, shuffle_on_reset=True)



    dev_auc = []
    train_log = []
    best_loss = 1000
    best_auc = 0
    
    with M.start_tensorflow_session(device=DEVICE) as sess:
        print("target_train: {} | target_dev: {} | targt_test: {}".format(np.mean(target_train), np.mean(target_dev), np.mean(target_test),))
        tf.set_random_seed(SEED)
    
        # Summaries
        fw = M.get_summary_writer(sess, LOG_DIR , PROJECT_NAME , VERSION)
        
        # Initialize variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        epoch = 1
        iter_i=1
        while epoch <= N_EPOCH:
            train_batcher.reset()
            loss_train = []
            for i in range(train_batcher.n_batches):
                
                # Get next batch
                batch_numeric_input_train, batch_target_train, batch_categorical_input_train, batch_numeric_input_train_bu, batch_categorical_input_train_bu, batch_numeric_input_train_prev_app, batch_categorical_input_train_prev_app = train_batcher.next()
                target_train_w = U.get_target_weights(batch_target_train, target_weight=1)            
                # Creating feed_dict
    
                feed_dict_train={dl_model.placeholders.numeric_input : batch_numeric_input_train,
                                 dl_model.placeholders.numeric_input_bureau : batch_numeric_input_train_bu,
                                 dl_model.placeholders.numeric_input_prev_app : batch_numeric_input_train_prev_app,
                                 dl_model.placeholders.target : batch_target_train,
                                 dl_model.placeholders.dropout_1 : DROPUT_RATE_1,
                                 dl_model.placeholders.dropout_2 : DROPUT_RATE_2,
                                 dl_model.placeholders.dropout_3 : DROPUT_RATE_3,
                                 dl_model.placeholders.loss_weights : target_train_w}
                
                for i in range(len(C.col_classes["application_train"]["categorical"])):
                    feed_dict_train[dl_model.placeholders.embedding[C.col_classes["application_train"]["categorical"][i]] ] = batch_categorical_input_train[:,i].reshape([-1,1])
                
                for i in range(len(C.col_classes["bureau"]["categorical"])):
                    feed_dict_train[dl_model.placeholders.embedding_bureau[C.col_classes["bureau"]["categorical"][i]] ] = np.expand_dims(batch_categorical_input_train_bu[:,:,i], axis=2)
                    
                for i in range(len(C.col_classes["previous_application"]["categorical"])):
                    feed_dict_train[dl_model.placeholders.embedding_prev_app[C.col_classes["previous_application"]["categorical"][i]] ] = np.expand_dims(batch_categorical_input_train_prev_app[:,:,i], axis=2)
                    
                # Run forward prop
                _ ,pred,  loss, sum_acc_train, sum_loss_train  = sess.run([dl_model.optimizers.optimizer_clip,
                                                                           dl_model.forward.pred,
                                                                           dl_model.losses.logloss,
                                                                           dl_model.summaries.accuracy_train,
                                                                           dl_model.summaries.loss_train], 
                                                                          feed_dict= feed_dict_train)
                
                loss_train.append(loss)
                
                fw.add_summary(sum_acc_train, iter_i)
                fw.add_summary(sum_loss_train, iter_i)
                iter_i = iter_i+1
            
            
            
            
            target_dev_w = U.get_target_weights(target_dev, target_weight=1)
            
             # Evaluate performance on dev at the ned of epch        
            feed_dict_dev={dl_model.placeholders.numeric_input : numeric_input_dev,
                           dl_model.placeholders.numeric_input_bureau : numeric_input_dev_bureau,
                           dl_model.placeholders.numeric_input_prev_app : numeric_input_dev_prev_app,
                           dl_model.placeholders.target : target_dev,
                           dl_model.placeholders.loss_weights : target_dev_w}
    
            for i in range(len(C.col_classes["application_train"]["categorical"])):
                    feed_dict_dev[dl_model.placeholders.embedding[C.col_classes["application_train"]["categorical"][i]] ] = categorical_input_dev[:,i].reshape([-1,1])
            
            for i in range(len(C.col_classes["bureau"]["categorical"])):
                    feed_dict_dev[dl_model.placeholders.embedding_bureau[C.col_classes["bureau"]["categorical"][i]] ] = np.expand_dims(categorical_input_dev_bureau[:,:,i], axis=2)
                    
            for i in range(len(C.col_classes["previous_application"]["categorical"])):
                    feed_dict_dev[dl_model.placeholders.embedding_prev_app[C.col_classes["previous_application"]["categorical"][i]] ] = np.expand_dims(categorical_input_dev_prev_app[:,:,i], axis=2)
                    
            dev_pred, dev_loss, _ , sum_acc_dev, sum_auc_dev, sum_loss_dev = sess.run( [dl_model.forward.pred,
                                                                                        dl_model.losses.logloss,
                                                                                        dl_model.metrics.auc_update_op, 
                                                                                        dl_model.summaries.accuracy_dev,
                                                                                        dl_model.summaries.auc_dev,
                                                                                        dl_model.summaries.loss_dev], 
                                                                                      feed_dict=feed_dict_dev)
            
            
            avg_loss_train = np.mean(np.array(loss_train))
            dev_auc_i = U.roc_auc_score(y_true=target_dev, y_score=dev_pred)
            dev_auc.append(dev_auc_i)
            
            # Check new best
            flag_new_best, best_loss, best_auc = U.update_best(best_loss, best_auc, dev_loss, dev_auc_i)
            
            # Save model
            if flag_new_best:
                best_checkpoint = epoch
                save_path = dl_model.model_saver.save(sess=sess, save_path="models/dl_model_{}/model_{}".format(VERSION, VERSION), global_step=epoch)
            
            print('epoch: {} | train_loss: {} |  dev_loss: {} | dev_auc: {}'.format(epoch, avg_loss_train, dev_loss, dev_auc_i))
            train_log.append([epoch, avg_loss_train, dev_loss, dev_auc_i])
            # Print and save performance results
            fw.add_summary(sum_acc_dev, epoch)
            fw.add_summary(sum_auc_dev, epoch)
            fw.add_summary(sum_loss_dev, epoch)
            
            epoch = epoch +1
            
         # Evaluate performance on test at the end of epch
        target_test_w = U.get_target_weights(target_test, target_weight=1)
        feed_dict_test={dl_model.placeholders.numeric_input : numeric_input_test,
                        dl_model.placeholders.numeric_input_bureau : numeric_input_test_bureau,
                        dl_model.placeholders.numeric_input_prev_app : numeric_input_test_prev_app,
                        dl_model.placeholders.target : target_test,
                        dl_model.placeholders.loss_weights : target_test_w}
        for i in range(len(C.col_classes["application_train"]["categorical"])):
            feed_dict_test[dl_model.placeholders.embedding[C.col_classes["application_train"]["categorical"][i]] ] = categorical_input_test[:,i].reshape([-1,1])
        
        for i in range(len(C.col_classes["bureau"]["categorical"])):
            feed_dict_test[dl_model.placeholders.embedding_bureau[C.col_classes["bureau"]["categorical"][i]] ] = np.expand_dims(categorical_input_test_bureau[:,:,i], axis=2)
                
        for i in range(len(C.col_classes["previous_application"]["categorical"])):
            feed_dict_test[dl_model.placeholders.embedding_prev_app[C.col_classes["previous_applicaiton"]["categorical"][i]] ] = np.expand_dims(categorical_input_test_prev_app[:,:,i], axis=2)
        
        test_pred, test_loss, test_acc, test_auc= sess.run( [dl_model.forward.pred, 
                                                             dl_model.losses.logloss,
                                                             dl_model.metrics.accuracy,
                                                             dl_model.metrics.auc], 
                                                           feed_dict=feed_dict_test)
        
        return {"best_checkpoint" : best_checkpoint,
                "best_auc" : best_auc,
                "best_loss" :  best_auc,
                "train_log" : train_log,
                "save_path" : save_path}
    
