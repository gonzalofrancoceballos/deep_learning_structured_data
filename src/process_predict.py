# -*- coding: utf-8 -*-
import numpy as np
import constants as C
import utils as U

def process_predict_task(processed_train_outputs, t_windows_dic, data_encoders):
    # =============================================================================
    # CONVERT TO np.array (Predict)
    # =============================================================================
    [numeric_input_train, numeric_input_dev, numeric_input_test,
     numeric_input_train_bureau, numeric_input_dev_bureau, numeric_input_test_bureau,
     numeric_input_train_prev_app, numeric_input_dev_prev_app, numeric_input_test_prev_app] = processed_train_outputs["normalized_data"]
    
    [categorical_input_train, categorical_input_dev, categorical_input_test,
     categorical_input_train_bureau, categorical_input_dev_bureau, categorical_input_test_bureau,
     categorical_input_train_prev_app,categorical_input_dev_prev_app, categorical_input_test_prev_app] = processed_train_outputs["categorical_inputs"]
       
    [data_normalizer_app, data_normalizer_bur, data_normalizer_prev_app] = processed_train_outputs["data_normalizers"]
    [target_train, target_dev, target_test]= processed_train_outputs["targets"]
    [application_predict, bureau_predict, previous_application_predict] = processed_train_outputs["predict_data"]

    ### APPLICATION PREDICT
    print("Processing application")
    # To np.array
    numeric_input_predict = application_predict[U.filter_id_values(C.col_classes["application_train"]["numeric"])].values
    categorical_input_predict = application_predict[C.col_classes["application_train"]["categorical"]].values
    
    # NA inputation
    numeric_input_na_flags_predict = np.zeros(numeric_input_predict.shape)
    numeric_input_na_flags_predict[ np.where(np.isnan(numeric_input_predict))] = 1
    numeric_input_predict[ np.where(np.isnan(numeric_input_predict))] = 0
    numeric_input_predict = np.concatenate([numeric_input_predict,numeric_input_na_flags_predict], axis=1)
    
    ### BUREAU
    print("Processing bureau")
    # To np.array
    numeric_input_bureau_predict = bureau_predict[U.filter_id_values(C.col_classes["bureau"]["numeric"])].values
    categorical_input_bureau_predict = bureau_predict[C.col_classes["bureau"]["categorical"]].values
    # NA inputation
    numeric_input_bureau_na_flags_predict = np.zeros(numeric_input_bureau_predict.shape)
    numeric_input_bureau_na_flags_predict[ np.where(np.isnan(numeric_input_bureau_predict))] = 1
    numeric_input_bureau_predict[ np.where(np.isnan(numeric_input_bureau_predict))] = 0
    numeric_input_bureau_predict = np.concatenate([numeric_input_bureau_predict,numeric_input_bureau_na_flags_predict], axis=1)
    # Reshape
    numeric_input_bureau_predict = np.reshape(numeric_input_bureau_predict, [-1, t_windows_dic["bureau"], numeric_input_bureau_predict.shape[1]])
    categorical_input_bureau_predict = np.reshape(categorical_input_bureau_predict, [-1, t_windows_dic["bureau"], categorical_input_bureau_predict.shape[1]])
    
    ### PREVIOUS APPLICATION
    print("Processing previous app")
    # To np.array
    numeric_input_prev_app_predict = previous_application_predict[U.filter_id_values(C.col_classes["previous_application"]["numeric"])].values
    categorical_input_prev_app_predict = previous_application_predict[C.col_classes["previous_application"]["categorical"]].values
    # NA inputation
    numeric_input_prev_app_na_flags_predict = np.zeros(numeric_input_prev_app_predict.shape)
    numeric_input_prev_app_na_flags_predict[ np.where(np.isnan(numeric_input_prev_app_predict))] = 1
    numeric_input_prev_app_predict[ np.where(np.isnan(numeric_input_prev_app_predict))] = 0
    numeric_input_prev_app_predict = np.concatenate([numeric_input_prev_app_predict,numeric_input_prev_app_na_flags_predict], axis=1)
    # Reshape
    numeric_input_prev_app_predict = np.reshape(numeric_input_prev_app_predict, [-1, t_windows_dic["previous_application"], numeric_input_prev_app_predict.shape[1]])
    categorical_input_prev_app_predict = np.reshape(categorical_input_prev_app_predict, [-1, t_windows_dic["previous_application"], categorical_input_prev_app_predict.shape[1]])# -*- coding: utf-8 -*-
    
    # ENCODING CATEGORICAL VARIABLES FOR PREDICT
    # Fitting
    categorical_input_predict = data_encoders.transform(categorical_input_predict, "application_train")
    categorical_input_bureau_predict = data_encoders.transform(categorical_input_bureau_predict, "bureau")
    categorical_input_prev_app_predict = data_encoders.transform(categorical_input_prev_app_predict, "previous_application")

    # Normalizing
    n_num_app = len(U.filter_id_values(C.col_classes["application_train"]["numeric"]))
    n_num_bur = len(U.filter_id_values(C.col_classes["bureau"]["numeric"]))
    n_num_prev_app = len(U.filter_id_values(C.col_classes["previous_application"]["numeric"]))
    
    numeric_input_predict[:, 0:n_num_app] =  U.data_normalizer_app.normalize_predict(numeric_input_predict[:, 0:n_num_app])
    numeric_input_bureau_predict[:,:, 0:n_num_bur] = U.data_normalizer_bur.normalize_predict(numeric_input_bureau_predict[:,:, 0:n_num_bur])
    numeric_input_prev_app_predict[:,:, 0:n_num_prev_app] = U.data_normalizer_prev_app.normalize_predict(numeric_input_prev_app_predict[:,:, 0:n_num_prev_app])
    
    
    categorical_predict = [categorical_input_predict, categorical_input_bureau_predict, categorical_input_prev_app_predict]
    numeric_predict = [numeric_input_predict, numeric_input_bureau_predict, numeric_input_prev_app_predict]
    
    processed_predict_data = {"categorical_predict" : categorical_predict,
                              "numeric_predict" : numeric_predict}
    
    return processed_predict_data