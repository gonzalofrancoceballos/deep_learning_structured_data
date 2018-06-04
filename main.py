# -*- coding: utf-8 -*-

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf

from ipywidgets import FloatProgress
from IPython.display import display

from src import utils as U
from src import constants as C
from src import models as M

from src.process_train import process_train_task
from src.process_predict import process_predict_task
from src.train import train_tast
from src.test import test_task
from src.predict import predict_task
from src.models import DLModel


# =============================================================================
# Processing data
# =============================================================================
processed_train_outputs = process_train_task()

[numeric_input_train, numeric_input_dev, numeric_input_test,
 numeric_input_train_bureau, numeric_input_dev_bureau, numeric_input_test_bureau,
 numeric_input_train_prev_app, numeric_input_dev_prev_app, numeric_input_test_prev_app] = processed_train_outputs["normalized_data"]

[categorical_input_train, categorical_input_dev, categorical_input_test,
 categorical_input_train_bureau, categorical_input_dev_bureau, categorical_input_test_bureau,
 categorical_input_train_prev_app,categorical_input_dev_prev_app, categorical_input_test_prev_app] = processed_train_outputs["categorical_inputs"]
   
[data_normalizer_app, data_normalizer_bur, data_normalizer_prev_app] = processed_train_outputs["data_normalizers"]
[target_train, target_dev, target_test]= processed_train_outputs["targets"]
[application_predict, bureau_predict, previous_application_predict] = processed_train_outputs["predict_data"]

data_encoders = processed_train_outputs["data_encoders"]
t_windows_dic = processed_train_outputs["t_windows_dic"]



# =============================================================================
# Parameters
# =============================================================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
N_EPOCH = 300
DROPUT_RATE_1 = 0.9
DROPUT_RATE_2 = 0.5
DROPUT_RATE_3 = 0.9
SEED = 1
DEVICE = '7'
LOG_DIR = "/home/mck/tensorflow_logs"
PROJECT_NAME = "home_credit_risk"
VERSION = "V50_gonzalo_zeus"
    
dl_model = DLModel(batch_size=BATCH_SIZE, 
                   col_classes=C.col_classes,
                   learning_rate=1e-4, 
                   encoders=data_encoders.encoders, 
                   embedding_sizes=C.embedding_sizes,
                   t_windows_dic=t_windows_dic,
                   activation= tf.nn.relu)


# =============================================================================
# Train
# =============================================================================
train_results = train_tast(dl_model, processed_train_outputs,  
                           n_epoch=N_EPOCH, 
                           seed=SEED,
                           device=DEVICE,
                           dropout_rate_1=DROPUT_RATE_1, 
                           dropout_rate_2=DROPUT_RATE_2,
                           dropout_rate_3=DROPUT_RATE_3, 
                           batch_size=BATCH_SIZE,
                           log_dir, project_name, version)


# =============================================================================
# Test
# =============================================================================
test_results = test_task(dl_model, processed_train_outputs , best_checkpoint, version, batch_size, device='0')


# =============================================================================
# Process data for predict
# =============================================================================
processed_predict_data = process_predict_task(processed_train_outputs, t_windows_dic, data_encoders)
    

# =============================================================================
# Predictions
# =============================================================================
final_prediction = predict_task(processed_predict_data , data_encoders, t_windows_dic, 
                                version=VERSION, best_checkpoint=best_checkpoint,
                                batch_size=128, device=DEVICE)



# =============================================================================
# Subsmission
# =============================================================================
submission = pd.DataFrame({"SK_ID_CURR": application_predict["SK_ID_CURR"].astype(int).values,
                           "TARGET":final_prediction })

submission.to_csv("submission.csv", index=False)




    