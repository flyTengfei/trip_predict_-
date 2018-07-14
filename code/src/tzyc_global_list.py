# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:31:20 2017

@author: dell
"""
import sys
import datetime
"""
定义全局变量
"""
WORK_LIST=sys.argv[1]
RUN_TIME = str(datetime.datetime.now()).split('.')[0]
xlid=sys.argv[2]
xlid='1'
#WORK_LIST='F:/腾飞的/tzyc'
trainDataTZPath=WORK_LIST+'/Original/tzdata_train_data.csv'
trainDataZCPath=WORK_LIST+'/Original/zcdata_train_data.csv'

saveModelPath=WORK_LIST+'/Model'
model_name='/TZModel.model'
result_list='/Result/'
resultPath='result.csv'
resultAllPath=WORK_LIST+'/Result/result_all.csv'
updata_tz=WORK_LIST+'/Original/tz_updata.csv'
updata_zc=WORK_LIST+'/Original/zc_updata.csv'
new_tz=WORK_LIST+'/Original/new_tz.csv'
new_zc=WORK_LIST+'/Original/new_zc.csv'
input_dir = WORK_LIST+'/Original/predict.csv'
output_dir = WORK_LIST+'/Output'

