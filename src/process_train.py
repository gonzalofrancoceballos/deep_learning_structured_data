# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from src import utils as U
from src import constants as C

def process_train_task():
    # =============================================================================
    # Loading data
    # =============================================================================
    application_train = pd.read_csv("data/application_train.csv", sep=",")
    application_predict = pd.read_csv("data/application_test.csv", sep=",")
    bureau = pd.read_csv("data/bureau.csv", sep=",")
    previous_application = pd.read_csv("data/previous_application.csv", sep=",")
#    bureau_balance = pd.read_csv("data/bureau_balance.csv", sep=",")
#    credit_card_balance = pd.read_csv("data/credit_card_balance.csv", sep=",")
#    installments_payments = pd.read_csv("data/installments_payments.csv", sep=",")
#    pos_cash_balance = pd.read_csv("data/POS_CASH_balance.csv", sep=",")
    t_windows_dic = {}
    
    # =============================================================================
    # All unique ids in train
    # =============================================================================
    all_ids = application_train["SK_ID_CURR"].unique()
    all_ids_predict = application_predict["SK_ID_CURR"].unique()
    
    
    # =============================================================================
    # Processing bureau
    # =============================================================================
    # Get max time_window
    t_windows_dic["bureau"] = bureau.groupby(["SK_ID_CURR"])["SK_ID_CURR"].count().max()
    # Synthetic variables
    bureau["ACTIVE_LOAN"] = 0
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE']>0, 'ACTIVE_LOAN']  = 1
    # Sort by id and date
    bureau  = bureau.sort_values(by=["SK_ID_CURR", "DAYS_CREDIT"])
    # Expanding
    bureau_predict = U.expand_data(bureau, id_column="SK_ID_CURR", id_date="DAYS_CREDIT", all_ids=all_ids_predict, t_length=t_windows_dic["bureau"])
    bureau = U.expand_data(bureau, id_column="SK_ID_CURR", id_date="DAYS_CREDIT", all_ids=all_ids, t_length=t_windows_dic["bureau"])
    # Cast columns
    bureau = U.cast_columns(bureau, "bureau", C.col_classes)
    bureau_predict = U.cast_columns(bureau_predict, "bureau", C.col_classes)
    
    
    # =============================================================================
    # PROCESSING PREVIOUS APPLICATION
    # =============================================================================
    # Get max time_window
    t_windows_dic["previous_application"] = previous_application.groupby(["SK_ID_CURR"])["SK_ID_CURR"].count().max()
    # Sort by id and date
    previous_application  = previous_application.sort_values(by=["SK_ID_CURR", "DAYS_FIRST_DUE"])
    # Expanding
    previous_application_predict = U.expand_data(previous_application, id_column="SK_ID_CURR", id_date="DAYS_FIRST_DUE", all_ids=all_ids_predict, t_length=t_windows_dic["previous_application"])
    previous_application = U.expand_data(previous_application, id_column="SK_ID_CURR", id_date="DAYS_FIRST_DUE", all_ids=all_ids, t_length=t_windows_dic["previous_application"])
    # Cast columns
    previous_application = U.cast_columns(previous_application, "previous_application", C.col_classes)
    previous_application_predict = U.cast_columns(previous_application_predict, "previous_application", C.col_classes)
    
    
    # =============================================================================
    # PROCESSING APPLICATION
    # =============================================================================
    # Cast columns
    application_train = U.cast_columns(application_train, "application_train", C.col_classes)
    application_predict = U.cast_columns(application_predict, "application_train",C.col_classes)
    # Sorting data
    application_train  = application_train.sort_values(by=["SK_ID_CURR"])
    application_predict  = application_predict.sort_values(by=["SK_ID_CURR"])
    
    
    # =============================================================================
    # CONVERT TO np.array (TRAIN)
    # =============================================================================
    ### APPLICATION TRAIN
    print("Processing application train")
    ### APPLICATION TRAIN
    # To np.array
    numeric_input = application_train[U.filter_id_values(C.col_classes["application_train"]["numeric"])].values
    categorical_input = application_train[C.col_classes["application_train"]["categorical"]].values
    target = application_train[["TARGET"]].values
    # NA inputation
    numeric_input_na_flags = np.zeros(numeric_input.shape)
    numeric_input_na_flags[ np.where(np.isnan(numeric_input))] = 1
    numeric_input[ np.where(np.isnan(numeric_input))] = 0
    numeric_input = np.concatenate([numeric_input,numeric_input_na_flags], axis=1)
    
    ### BUREAU
    # To np.array
    numeric_input_bureau = bureau[U.filter_id_values(C.col_classes["bureau"]["numeric"])].values
    categorical_input_bureau = bureau[C.col_classes["bureau"]["categorical"]].values
    # NA inputation
    numeric_input_bureau_na_flags = np.zeros(numeric_input_bureau.shape)
    numeric_input_bureau_na_flags[ np.where(np.isnan(numeric_input_bureau))] = 1
    numeric_input_bureau[ np.where(np.isnan(numeric_input_bureau))] = 0
    numeric_input_bureau = np.concatenate([numeric_input_bureau,numeric_input_bureau_na_flags], axis=1)
    # Reshape
    numeric_input_bureau = np.reshape(numeric_input_bureau, [-1, t_windows_dic["bureau"], numeric_input_bureau.shape[1]])
    categorical_input_bureau = np.reshape(categorical_input_bureau, [-1, t_windows_dic["bureau"], categorical_input_bureau.shape[1]])
    
    ### PREVIOUS APPLICATION
    # To np.array
    numeric_input_prev_app = previous_application[U.filter_id_values(C.col_classes["previous_application"]["numeric"])].values
    categorical_input_prev_app = previous_application[C.col_classes["previous_application"]["categorical"]].values
    # NA inputation
    numeric_input_prev_app_na_flags = np.zeros(numeric_input_prev_app.shape)
    numeric_input_prev_app_na_flags[ np.where(np.isnan(numeric_input_prev_app))] = 1
    numeric_input_prev_app[ np.where(np.isnan(numeric_input_prev_app))] = 0
    numeric_input_prev_app = np.concatenate([numeric_input_prev_app,numeric_input_prev_app_na_flags], axis=1)
    # Reshape
    numeric_input_prev_app = np.reshape(numeric_input_prev_app, [-1, t_windows_dic["previous_application"], numeric_input_prev_app.shape[1]])
    categorical_input_prev_app = np.reshape(categorical_input_prev_app, [-1, t_windows_dic["previous_application"], categorical_input_prev_app.shape[1]])
    
    
    
    
    # =============================================================================
    #  TRAIN, DEV, TEST SPLIT
    # =============================================================================
    # Application_train
    [train_split, dev_split, test_split] = U.train_dev_test_split([numeric_input, 
                                                                 categorical_input, 
                                                                 target,
                                                                 numeric_input_bureau, 
                                                                 categorical_input_bureau,
                                                                 numeric_input_prev_app, 
                                                                 categorical_input_prev_app])
    numeric_input_train, categorical_input_train, target_train, numeric_input_train_bureau, categorical_input_train_bureau, numeric_input_train_prev_app, categorical_input_train_prev_app = train_split
    numeric_input_dev, categorical_input_dev, target_dev,numeric_input_dev_bureau, categorical_input_dev_bureau, numeric_input_dev_prev_app, categorical_input_dev_prev_app = dev_split
    numeric_input_test, categorical_input_test, target_test,numeric_input_test_bureau, categorical_input_test_bureau, numeric_input_test_prev_app, categorical_input_test_prev_app = test_split
    
    
    
    
    # =============================================================================
    # ENCODING CATEGORICAL VARIABLES FOR TRAIN-DEV-TEST
    # =============================================================================
    # Fitting
    print("Fitting encoders")
    data_encoders = U.Encoders(C.col_classes)
    data_encoders.fit(categorical_input_train, "application_train")
    data_encoders.fit(categorical_input_train_bureau, "bureau")
    data_encoders.fit(categorical_input_train_prev_app, "previous_application")
    
    #Transforming application data
    print("Encoding application data")
    categorical_input_train = data_encoders.transform(categorical_input_train, "application_train")
    categorical_input_dev = data_encoders.transform(categorical_input_dev, "application_train")
    categorical_input_test = data_encoders.transform(categorical_input_test, "application_train")
    #Transforming bureau data
    print("Encodding bureau data")
    categorical_input_train_bureau = data_encoders.transform(categorical_input_train_bureau, "bureau")
    categorical_input_dev_bureau = data_encoders.transform(categorical_input_dev_bureau, "bureau")
    categorical_input_test_bureau = data_encoders.transform(categorical_input_test_bureau, "bureau")
    #Transforming prev_app data
    print("Encodding previous application data")
    categorical_input_train_prev_app = data_encoders.transform(categorical_input_train_prev_app, "previous_application")
    categorical_input_dev_prev_app = data_encoders.transform(categorical_input_dev_prev_app, "previous_application")
    categorical_input_test_prev_app = data_encoders.transform(categorical_input_test_prev_app, "previous_application")
    
    
    # =============================================================================
    # NORMALIZE DATA
    # =============================================================================
    n_num_app = len(U.filter_id_values(C.col_classes["application_train"]["numeric"]))
    n_num_bur = len(U.filter_id_values(C.col_classes["bureau"]["numeric"]))
    n_num_prev_app = len(U.filter_id_values(C.col_classes["previous_application"]["numeric"]))
    data_normalizer_app = U.Normalizer()
    data_normalizer_bur = U.Normalizer(cube=True)
    data_normalizer_prev_app =U.Normalizer(cube=True)
    # Applications
    numeric_input_train[:, 0:n_num_app] =  data_normalizer_app.normalize_train(numeric_input_train[:, 0:n_num_app])
    numeric_input_dev[:, 0:n_num_app] =  data_normalizer_app.normalize_predict(numeric_input_dev[:, 0:n_num_app])
    numeric_input_test[:, 0:n_num_app] =  data_normalizer_app.normalize_predict(numeric_input_test[:, 0:n_num_app])
    # Bureau
    numeric_input_train_bureau[:,:, 0:n_num_bur] = data_normalizer_bur.normalize_train(numeric_input_train_bureau[:,:, 0:n_num_bur])
    numeric_input_dev_bureau[:,:, 0:n_num_bur] =   data_normalizer_bur.normalize_predict(numeric_input_dev_bureau[:,:, 0:n_num_bur])
    numeric_input_test_bureau[:,:, 0:n_num_bur] =  data_normalizer_bur.normalize_predict(numeric_input_test_bureau[:,:, 0:n_num_bur])
    # Previous applications
    numeric_input_train_prev_app[:,:, 0:n_num_prev_app] = data_normalizer_prev_app.normalize_train(numeric_input_train_prev_app[:,:, 0:n_num_prev_app])
    numeric_input_dev_prev_app[:,:, 0:n_num_prev_app] =   data_normalizer_prev_app.normalize_predict(numeric_input_dev_prev_app[:,:, 0:n_num_prev_app])
    numeric_input_test_prev_app[:,:, 0:n_num_prev_app] =  data_normalizer_prev_app.normalize_predict(numeric_input_test_prev_app[:,:, 0:n_num_prev_app])
    
    normalized_data = [numeric_input_train,
                        numeric_input_dev,
                        numeric_input_test,
                        numeric_input_train_bureau,
                        numeric_input_dev_bureau,
                        numeric_input_test_bureau,
                        numeric_input_train_prev_app,
                        numeric_input_dev_prev_app,
                        numeric_input_test_prev_app] 
    targets = [target_train, target_dev, target_test]
    data_normalizers = [data_normalizer_app,
                        data_normalizer_bur,
                        data_normalizer_prev_app]
    
    categorical_inputs = [categorical_input_train,
                            categorical_input_dev,
                            categorical_input_test,
                            categorical_input_train_bureau,
                            categorical_input_dev_bureau,
                            categorical_input_test_bureau,
                            categorical_input_train_prev_app,
                            categorical_input_dev_prev_app,
                            categorical_input_test_prev_app]
    
    predict_data = [application_predict, bureau_predict, previous_application_predict]
    return {"normalized_data" : normalized_data,
            "categorical_inputs" : categorical_inputs,
            "targets" : targets,
            "predict_data" : predict_data,
            "data_normalizers" : data_normalizers,
             "data_encoders" : data_encoders,
             "t_windows_dic" : t_windows_dic}
    
    