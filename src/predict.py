import utils as U
import constants as C
import model_utils as M
from models import DLModel
import numpy as np
import tensorflow as tf



def predict_task(processed_predict_data, data_encoders, t_windows_dic, version, 
                 best_checkpoint,batch_size=128, device='0'):
    
    
    [categorical_input_predict, 
     categorical_input_bureau_predict, 
     categorical_input_prev_app_predict]  = processed_predict_data["categorical_predict"]

    [numeric_input_predict, 
     numeric_input_bureau_predict, 
     numeric_input_prev_app_predict] = processed_predict_data["numeric_predict"]
    
    # Constants
    VERSION = version
    checkpoint = best_checkpoint
    
    # Instanciating Model object
    dl_model = DLModel(batch_size=32, 
                       col_classes=C.col_classes,
                       learning_rate=1e-4, 
                       encoders=data_encoders.encoders, 
                       embedding_sizes=C.embedding_sizes,
                       t_windows_dic=t_windows_dic,
                       activation= tf.nn.tanh)
    
    
    predict_batcher = U.Batcher([numeric_input_predict,  
                               categorical_input_predict, 
                               numeric_input_bureau_predict, categorical_input_bureau_predict,
                               numeric_input_prev_app_predict,categorical_input_prev_app_predict],  
                                batch_size, shuffle_on_reset=False)
    
    predictions = []
    # predictions = np.zeros([len(all_ids_test)])
    with M.start_tensorflow_session(device=device) as sess:
        dl_model.model_saver.restore(sess, "models/dl_model_"+str(VERSION)+"/model_"+str(VERSION)+"-" + str(checkpoint),)
        for i in range(predict_batcher.n_batches):
                    
            # Update progress bar
            # Get next batch
            batch_numeric_input_predict,  batch_categorical_input_predict, batch_numeric_input_predict_bu, batch_categorical_input_predict_bu, batch_numeric_input_predict_prev_app, batch_categorical_input_predict_prev_app = predict_batcher.next()
            
            # Creating feed_dict
    
            feed_dict_predict={dl_model.placeholders.numeric_input : batch_numeric_input_predict,
                               dl_model.placeholders.numeric_input_bureau : batch_numeric_input_predict_bu,
                               dl_model.placeholders.numeric_input_prev_app : batch_numeric_input_predict_prev_app}
    
            for i in range(len(C.col_classes["application_train"]["categorical"])):
                feed_dict_predict[dl_model.placeholders.embedding[C.col_classes["application_train"]["categorical"][i]] ] = batch_categorical_input_predict[:,i].reshape([-1,1])
    
            for i in range(len(C.col_classes["bureau"]["categorical"])):
                feed_dict_predict[dl_model.placeholders.embedding_bureau[C.col_classes["bureau"]["categorical"][i]] ] = np.expand_dims(batch_categorical_input_predict_bu[:,:,i], axis=2)
                
            for i in range(len(C.col_classes["previous_application"]["categorical"])):
                feed_dict_predict[dl_model.placeholders.embedding_prev_app[C.col_classes["previous_application"]["categorical"][i]] ] = np.expand_dims(batch_categorical_input_predict_prev_app[:,:,i], axis=2)
    
            # Run forward prop
            pred  = sess.run(dl_model.forward.pred, feed_dict= feed_dict_predict)
            predictions.append(pred)
            
    final_prediction=[]
    for pred_i in predictions:
        for elem in pred_i:
            final_prediction.append(elem)
    final_prediction = np.squeeze(np.array(final_prediction))
    
    return final_prediction
    