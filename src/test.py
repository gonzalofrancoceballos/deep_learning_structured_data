import utils as U
import model_utils as M
import constants as C
import numpy as np


def test_task(dl_model, processed_train_outputs , best_checkpoint, version, 
              batch_size, device='0'):
    checkpoint = best_checkpoint
    DEVICE = device
    VERSION = version
    BATCH_SIZE = batch_size
    
    [numeric_input_train, numeric_input_dev, numeric_input_test,
     numeric_input_train_bureau, numeric_input_dev_bureau, numeric_input_test_bureau,
     numeric_input_train_prev_app, numeric_input_dev_prev_app, numeric_input_test_prev_app] = processed_train_outputs["normalized_data"]
    
    [categorical_input_train, categorical_input_dev, categorical_input_test,
     categorical_input_train_bureau, categorical_input_dev_bureau, categorical_input_test_bureau,
     categorical_input_train_prev_app,categorical_input_dev_prev_app, categorical_input_test_prev_app] = processed_train_outputs["categorical_inputs"]
       
    [data_normalizer_app, data_normalizer_bur, data_normalizer_prev_app] = processed_train_outputs["data_normalizers"]
    [target_train, target_dev, target_test]= processed_train_outputs["targets"]
    [application_predict, bureau_predict, previous_application_predict] = processed_train_outputs["predict_data"]

    
    # Instantiating Batcher object
    predict_batcher = U.Batcher([numeric_input_test,  
                               categorical_input_test, 
                               numeric_input_test_bureau,categorical_input_test_bureau,
                               numeric_input_test_prev_app,categorical_input_test_prev_app],  BATCH_SIZE, shuffle_on_reset=False)
    
    predictions_test = []
    # predictions = np.zeros([len(all_ids_test)])
    with M.start_tensorflow_session(device=DEVICE) as sess:
        dl_model.model_saver.restore(sess, "models/dl_model_"+str(VERSION)+"/model_"+str(VERSION)+"-" + str(checkpoint),)
        for i in range(predict_batcher.n_batches):
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
            predictions_test.append(pred)
            
    final_prediction_test=[]
    for pred_i in predictions_test:
        for elem in pred_i:
            final_prediction_test.append(elem)
    final_prediction_test = np.squeeze(np.array(final_prediction_test))
    U.plot_roc(target_test, final_prediction_test)
    U.print_distribution(final_prediction_test)
    uplift = U.get_uplift(target_test, final_prediction_test, N=100, plot="uplift_acum")
    
    
    return {"prediction" : final_prediction_test, 
            "uplift" : uplift}
