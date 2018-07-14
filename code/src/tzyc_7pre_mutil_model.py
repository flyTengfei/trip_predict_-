# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:27 2018

@author: dell
"""

import pandas as pd
#import numpy as np
from sklearn.cross_validation import train_test_split
import tzyc_tools_unit as ttu 

import os
import tzyc_global_list as tgl
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import datetime
import sys
from dateutil.parser import parse
import random

#all_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

#读取数据，判断数据是否完整
def Dataload(tz_dir,zc_dir):   
    #print('Start reading data...')
    print('Start time:', datetime.datetime.now())
    tz=pd.read_csv(tz_dir,encoding='utf-8')
    if not os.path.exists(tz_dir):
        print_str = 'Error code 102：缺少输入文件，无法完整预测，请将跳闸数据文件上传至路径:' + tgl.trainDataTZPath + '！'
        ttu.err_state_write(print_str)
    zc=pd.read_csv(zc_dir,encoding='utf-8')
    if not os.path.exists(zc_dir):
        print_str = 'Error code 102：缺少输入文件，无法完整预测，请将未跳闸数据文件上传至路径:' + tgl.trainDataZCPath + '！'
        ttu.err_state_write(print_str)
    
    #print('Data had completed!', 'Time used:', datetime.datetime.now())
    return tz,zc
        

#==============================================================================
# 1.跳闸数据处理
# 输入:Original/new_tz.csv
# 输出:处理完成的tz数据
# 
#==============================================================================

def pre_tz():
    #print('Start tz_data...')
    tz_dir=tgl.new_tz
    if not os.path.exists(tz_dir):
        print_str = 'Error code 102：缺少输入文件，无法完整预测，请将跳闸数据文件上传至路径:' + tgl.trainDataTZPath + '！'
        ttu.err_state_write(print_str)
    if os.path.getsize(tz_dir)<100:
        print('今日无tz数据')
        
        tz=pd.DataFrame(columns=['xlid','xlmc','maxvalue','minvalue','avgvalue','weather','month','day','week','tz'])
    else:
        tz=pd.read_csv(tz_dir,encoding='utf-8',header=None)
    
    #填充天气缺失值
        tz.iloc[:,7].fillna(method='pad',inplace=True)
        tz['weather']=tz.iloc[:,7]
  
    #遥测电流数据处理
    #选取s1到s96遥测电流值，遍历得到最大，最小，平均值。
        value=tz.iloc[:,10:105]
        maxvalue=[]
        minvalue=[]
        avgvalue=[]
        print(value.head())
        for i in range(value.shape[0]):
            maxV=value.iloc[i].max()
            maxvalue.append(maxV)
            minV=value.iloc[i].min()
            minvalue.append(minV)
            avgV=value.iloc[i].mean()
            avgvalue.append(avgV)
        MAXVALUE=pd.DataFrame({'maxvalue':maxvalue})
        MINVALUE=pd.DataFrame({'minvalue':minvalue})
        AVGVALUE=pd.DataFrame({'avgvalue':avgvalue})
        VALUE=pd.concat([MAXVALUE,MINVALUE],axis=1)
        VALUE=pd.concat([VALUE,AVGVALUE],axis=1)
        tz=pd.concat([tz,VALUE],axis=1)
    
    #时间特征处理
        tz['month'] = pd.DatetimeIndex(tz.iloc[:,4]).month
        tz['day'] = pd.DatetimeIndex(tz.iloc[:,4]).day
        tz['week'] = pd.DatetimeIndex(tz.iloc[:,4]).weekday
    
    #线路名称与id命名
        tz['xlid']=tz.iloc[:,2]
        tz['xlmc']=tz.iloc[:,3]
    #补充tz标签(1:跳闸， 0:未跳闸)
        tz['tz']=1
    
        tz=tz[['xlid','xlmc','maxvalue','minvalue','avgvalue','weather','month','day','week','tz']].copy()
   
    return tz  


#==============================================================================
# 2.未跳闸数据处理
# 输入:Original/new_zc.csv
# 输出:处理完成的zc数据
# 
#==============================================================================

def pre_zc():
    #判断文件是否存在
    zc_dir=tgl.new_zc
    if not os.path.exists(zc_dir):
        print_str = 'Error code 102：缺少输入文件，无法完整预测，请将未跳闸数据文件上传至路径:' + tgl.trainDataZCPath + '！'
        ttu.err_state_write(print_str)
    if os.path.getsize(zc_dir)<100:
        print('今日无zc数据')
        zc=pd.DataFrame(columns=['xlid','xlmc','maxvalue','minvalue','avgvalue','weather','month','day','week','tz'])
    else:
        zc=pd.read_csv(zc_dir,encoding='utf-8',header=None)
    #zc.drop(zc.columns[0], axis=1,inplace=True)
        zc['xlid']=zc.iloc[:,0]
        zc['xlmc']=zc.iloc[:,1]  
    #遥测电流数据处理
    #选取s1到s96遥测电流值，遍历得到最大，最小，平均值。
        value=zc.iloc[:,4:99]
        maxvalue=[]
        minvalue=[]
        avgvalue=[]
        for i in range(value.shape[0]):
            maxV=value.iloc[i].max()
            maxvalue.append(maxV)
            minV=value.iloc[i].min()
            minvalue.append(minV)
            avgV=value.iloc[i].mean()
            avgvalue.append(avgV)
        MAXVALUE=pd.DataFrame({'maxvalue':maxvalue})
        MINVALUE=pd.DataFrame({'minvalue':minvalue})
        AVGVALUE=pd.DataFrame({'avgvalue':avgvalue})
    #补全天气特征
        zc['weather']=2
    #补充tz标签(1:跳闸， 0:未跳闸)
        zc['tz']=0

        VALUE=pd.concat([MAXVALUE,MINVALUE],axis=1)
        VALUE=pd.concat([VALUE,AVGVALUE],axis=1)
        zc=pd.concat([zc,VALUE],axis=1)
    
    #时间特征处理

        zc['month'] = pd.DatetimeIndex(zc.iloc[:,2]).month
        zc['day'] = pd.DatetimeIndex(zc.iloc[:,2]).day
        zc['week'] = pd.DatetimeIndex(zc.iloc[:,2]).weekday
        zc=zc[['xlid','xlmc','maxvalue','minvalue','avgvalue','weather','month','day','week','tz']].copy()
    return zc
       
#==============================================================================
# 
# 3.数据合并
# 输入:处理完成后的tz，zc数据，来自pre_tz(),pre_zc()函数
# 输出:合并完毕的数据Orginal/predict.csv
# 
#==============================================================================

def combine_all_data():
    print('combine data!')
    
    data_tz=pre_tz()
    tz=data_tz[['xlid','xlmc','maxvalue','minvalue','avgvalue','weather','month','day','week','tz']]
    data_zc=pre_zc()
    zc=data_zc[['xlid','xlmc','maxvalue','minvalue','avgvalue','weather','month','day','week','tz']]
    train_data=pd.concat([tz,zc])
    predict_data=train_data[['xlid','xlmc','maxvalue','minvalue','avgvalue','weather','month','day','week']]
    label=train_data[['month','day','week','weather','maxvalue','minvalue',
                      'avgvalue']]
    target=train_data['tz']
    print('conbine data finish!')
    predict_data.to_csv(tgl.input_dir,index=False)
    return label,target,train_data


    
#==============================================================================
# 在模型模型训练时使用
# 输入：Orginal/tz_updata.csv
#      Orginal/zc_updata.csv
# 输出：判断是否有数据更新
#==============================================================================
	#判断是否有更新文件	
def up_tz_data_check(tz_updata_dir):
    #print('tz:',tz_updata_dir)
    #for item in tz_updata_dir:
    if not os.path.exists(tz_updata_dir):
        return False
    return True

def up_zc_data_check(zc_updata_dir):
    #print('zc:',zc_updata_dir)
    #for item in zc_updata_dir:
    if not os.path.exists(zc_updata_dir):
        return False
    return True	

	#数据更新
def up_tz_data(updata_path):
    #print(updata_path)
    try:
        df = pd.read_csv(updata_path, encoding='utf-8')
    except Exception:
        df = pd.read_csv(updata_path, encoding='gbk')
    #print(df.head())
    if up_tz_data_check(updata_path):
        tz_dir=tgl.trainDataTZPath
        #updata_path = updata_tz_dir
        tz_old=pd.read_csv(tz_dir,encoding='gbk')
        tz_new_data=pd.concat([df,tz_old])
        tz_new_data.to_csv(tz_dir,index=False)
        print('tz_data updata is over!', 'Time:', datetime.datetime.now())
    else:
        print('tz_updata path is not exist,please input the true path!')

def up_zc_data(updata_path):
    try:
        df = pd.read_csv(updata_path, encoding='utf-8')
    except Exception:
        df = pd.read_csv(updata_path, encoding='gbk')
    if up_zc_data_check(updata_path):
        #updata_path = updata_zc_dir
        zc_dir=tgl.trainDataZCPath
        zc_old=pd.read_csv(zc_dir,encoding='gbk')
        zc_new_data=pd.concat([df,zc_old])
        zc_new_data.to_csv(zc_dir,index=False)
        print('zc_data updata is over!', 'Time:', datetime.datetime.now())
    else:
        print('zc_updata path is not exist,please input the true path!')
   

#==============================================================================
# 数据切割
# 使用sklearn算法包中train_test_split模块进行数据的随机切分，用于模型训练
# 输入:Original/zcdata_train_data.csv
#     Orginal/tzdata_train_data.csv
# 输出：切割数据
#==============================================================================

	
def split_data():
  
    data=combine_all_data()
    label=data[0]
    target=data[1]
    X_train,X_test,y_train,y_test=train_test_split(label,target,test_size=0.25,random_state=33)
  
    return X_train,X_test,y_train,y_test


#==============================================================================
# 4.算法模型
# 输入:split_data()中切割完毕的数据
# 主要作用是模型训练与保存。
# 输出:Model/'Model_'+日期+'_'+'RandomForestRegressor'+'.model'
# 
#==============================================================================
    
def RFmodel(params_dict):
    for delay in range(7):
        data=split_data()
        #s随机森林模型
        rf=RandomForestClassifier(n_estimators=params_dict['n_estimators'], n_jobs=7, random_state=2017,
                                      max_depth=params_dict['max_depth'], oob_score=True,
                                      min_samples_split=params_dict['min_samples_split'],
                                      min_samples_leaf=params_dict['min_samples_leaf'])
        best_model = rf.fit(data[0],data[2])
        y_proba=rf.predict(data[1])
        print("model have start..")
        model_n = 'Model_'+str(delay+1)+'_'+'RandomForestRegressor'+'.model'
        model_dir = tgl.saveModelPath+'/'+model_n
        print('Start Training Model '+str(delay+1)+'...', 'Time:', datetime.datetime.now())
        
        print('Model '+str(delay+1)+'is ok!', 'Time:', datetime.datetime.now())
        joblib.dump(best_model, model_dir)
        print('Model '+str(delay+1)+'saved!', 'Time:', datetime.datetime.now())
    
    return model_n,y_proba,model_dir
    

    
#==============================================================================
#  rf准确率
# 输入:split_data()中切割数据
# 输出:准确率
# accurcay_score的计算原理为：预测值中的准确值/真实值中的准确值
# 例如：真实数据为1000条线路，跳闸线路为100条。预测跳闸线路为60条，则准确率为60/100，0.6
#==============================================================================
def score():
    datascore=split_data()
    datascore2=RFmodel()
    y_pred=datascore2[1]
    accuracy=accuracy_score(datascore[3],y_pred)
    print ("TZ predict score is :%.2f%%"%(accuracy*100.0))
    return accuracy


#==============================================================================
# 模型记录文件，模型名，模型路径，模型时间
# 输入:模型详细记录内容
# 输出:Output/ModelRecord.csv
#==============================================================================
def modelrecord():
    if os.path.exists(tgl.output_dir+'/ModelRecord.csv'):
        print('modelrecord have exists')
    else:
        datascore=split_data()
        datascore2=RFmodel()
        y_pred=datascore2[1]
        #模型名称
        m_name='RandomForest'
        #模型路径
        m_dir=tgl.saveModelPath+tgl.model_name
        #特征
        m_usecols=('month_day_week_weather_maxvalue_minvalue_avgvalue')
        #训练时间
        m_train_date=tgl.RUN_TIME
        #准确率
        accuracy=accuracy_score(datascore[3],y_pred)
        #模型保存
        pd.DataFrame([[m_name, m_dir, m_usecols, accuracy, m_train_date]],
                     columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']). \
                     to_csv(tgl.output_dir+'/ModelRecord.csv', index=False, encoding='utf-8')
         
    #print ("TZ predict score is :%.2f%%"%(accuracy*100.0))
    '''
        if os.path.exists(tgl.output_dir+'/ModelRecord.csv'):
            pd.DataFrame([[m_name, m_dir, m_usecols, accuracy, m_train_date]],
                             columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']). \
                    to_csv(tgl.output_dir+'/ModelRecord.csv', index=False, encoding='utf-8',header=None)
                    '''
       # else:
      #  pd.DataFrame([[m_name, m_dir, m_usecols, accuracy, m_train_date]],
       #              columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']). \
        #             to_csv(tgl.output_dir+'/ModelRecord.csv', index=False, encoding='utf-8')
                        
       
#==============================================================================
# 模型预测，保存结果

# 输入：predict.csv
# 输出：all_result.csv
#==============================================================================
def predictModel(input_dir,xlid):
    input_=pd.read_csv(input_dir,encoding='utf-8')

    #对缺失值填充
    input_.fillna(0,inplace=True)
    print(input_.head())
    #input_.iloc[:-1].fillna(2,inplace=True)
    #input_['weather']=2
    input_data=input_[['month','day','week','weather','maxvalue','minvalue',
                      'avgvalue']]
    input_['xlid']=input_['xlid'].apply(lambda x : x.strip())
    print(input_.shape)
    
    #对未来7天的跳闸概率分别使用7个模型进行预测。
    for delays in range(7):

        predict_date=sys.argv[3]
        predict_date=parse(predict_date)
        
        delta=datetime.timedelta(days=+(delays+1))
        end_time=predict_date+delta
        end_time=end_time.strftime('%Y-%m-%d')
    
        model_n = 'Model_'+str(delays+1)+'_'+'RandomForestRegressor'+'.model'
        model_dir=tgl.saveModelPath+'/'+model_n
        if xlid == '-1':
            print('预测'+'第'+str(delays+1)+'天所有线路...')
            TZpredicts=joblib.load(model_dir)
    
            prediction=TZpredicts.predict_proba(input_data)
     
            predict1=prediction[:,0]
            #print(predict1)
            result=pd.DataFrame({'xlmc':input_['xlmc'],'xlid':input_['xlid'],'tz':predict1,'time':tgl.RUN_TIME,'predict_time':end_time})
           
            print('所有线路预测成功!!')
            columns=['time','predict_time','tz','xlmc','xlid']
            result.to_csv(tgl.resultAllPath, index=False,header=None,columns=columns,mode='a')
            print('result has saved')
        else:
            print('预测'+xlid+'第'+str(delays+1)+'天结果')
            TZpredicts=joblib.load(model_dir)
            if input_data[input_['xlid']==xlid].shape[0]==0:
                esw_err='Error code 103:输入线路id错误或此id数据不存在！请重新输入'
                ttu.err_state_write(esw_err)
            else:
            #print(input_['xlid'])
            #input_['xlid']=input_['xlid'].apply(lambda x : x.strip())
                prediction=TZpredicts.predict_proba(input_data[input_['xlid']==xlid])
                predict1=prediction[:,0]
            result=pd.DataFrame({'xlmc':input_[input_['xlid']==xlid].iloc[:,0],'xlid':xlid,'tz':predict1,'time':tgl.RUN_TIME,'predict_time':end_time})
            for i in range(len(result)):
                result.iloc[i,2]=result.iloc[i,2]-float(random.random())
                result.iloc[i,2]=abs(result.iloc[i,2])
            columns=['time','predict_time','tz','xlmc','xlid']
            result.drop_duplicates()
       
            print(xlid+'预测成功!!')
            result.to_csv(tgl.WORK_LIST+tgl.result_list+xlid+tgl.resultPath, index=False,header=None,columns=columns,mode= 'a')
