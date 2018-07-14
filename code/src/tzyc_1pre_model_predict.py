# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:44:40 2017

@author: dell
"""

import tzyc_global_list as tgl
import tzyc_mutil_model as tmm
import tzyc_tools_unit as ttu
import pandas as pd
import datetime 
import os
import sys

predict_date=sys.argv[3]
#predict_date = '2017-12-01'  # 获取模型预测起始日期，格式：年-月-日
d1 = datetime.datetime.strptime(predict_date, '%Y-%m-%d')  # str(d1) = '2017-01-01 00:00:00'
#log_dir = tgl.output_dir+'/PredictLog'+''.join(str(d1).split(' ')[0].split('-'))+'.csv'


if __name__ == '__main__':
    

    #获取所需预测的线路名称
    #xlmc='10kV大桥线'
    #xlmc='-1'
    xlid=str(sys.argv[2])
    if os.path.exists(tgl.saveModelPath+tgl.model_name)==True:
        print('模型已经存在，可直接预测')
        #读取数据
        input_=pd.read_csv(tgl.input_dir,encoding='utf-8',header=None)
        if not os.path.exists(tgl.input_dir):
            print_str = 'Error code 103：缺少预测文件，无法完整预测，请将跳闸数据文件上传至路径:' + tgl.input_dir + '！'
            ttu.err_state_write(print_str)
        #模型预测
        tmm.predictModel(tgl.input_dir,xlid)
        #模型记录
        tmm.modelrecord()
        #if xlmc == '-1':
         #   xlmc_all = pd.DataFrame(input_.iloc[:,0])
          #  log_write(xlmc_all.reset_index(drop=True), predict_date, log_dir)
        #else:
            #xlmc= pd.DataFrame({'线路名称':xlmc})
            #time=predict_data
            #log_write(xlmc, predict_date, log_dir)
        print_str='运行正常'
        ttu.rigth_state_write(print_str)
        sys.exit(0)
    elif os.path.exists(tgl.saveModelPath+tgl.model_name)==False:
        print('模型不存在！需要训练并保存模型')
        tmm.Dataload(tgl.trainDataTZPath,tgl.trainDataZCPath)
       #训练模型
        tmm.RFmodel()
       #保存模型
        tmm.SaveModel()
       #模型得分
        tmm.score()
       #模型记录
        tmm.modelrecord()
       #模型训练
        #time=predict_data
        tmm.predictModel(tgl.input_dir,xlid)
        #log_write(xlmc, predict_date, log_dir)
        #print('预测日志文件写于：'+log_dir)
        print_str='运行正常'
        ttu.rigth_state_write(print_str)
        sys.exit(0)
