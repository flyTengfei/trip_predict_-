# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:26:11 2017

@author: dell
"""
import pandas as pd
#import numpy as np
from sklearn.cross_validation import train_test_split
import tzyc_tools_unit as ttu 
#from sklearn import svm
import os
import tzyc_global_list as tgl
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import datetime
import sys
import random
from dateutil.parser import parse

#==============================================================================
# 读取数据
# 主要为了判断数据是否存在
#==============================================================================
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
    tz_dir=tgl.trainDataTZPath
    if not os.path.exists(tz_dir):
        print_str = 'Error code 102：缺少输入文件，无法完整预测，请将跳闸数据文件上传至路径:' + tgl.trainDataTZPath + '！'
        ttu.err_state_write(print_str)
    tz=pd.read_csv(tz_dir,encoding='utf-8',header=None)
    
    #删除人工导致跳闸情况
    #print(tz.shape)
    #填充缺失值t
    
    tz.iloc[:,7].fillna(method='pad',inplace=True)
    tz['weather']=tz.iloc[:,7]
    #print(tz.head())
    #ycdl数据处理
    #idx=pd.IndexSlice
    value=tz.iloc[:,11:107]
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
    VALUE=pd.concat([MAXVALUE,MINVALUE],axis=1)
    VALUE=pd.concat([VALUE,AVGVALUE],axis=1)
    tz=pd.concat([tz,VALUE],axis=1)
    
    #时间特征处理
    tz['month'] = pd.DatetimeIndex(tz.iloc[:,4]).month
    tz['day'] = pd.DatetimeIndex(tz.iloc[:,4]).day
    tz['week'] = pd.DatetimeIndex(tz.iloc[:,4]).weekday
    
    #线路名称重命名
    tz['xlmc']=tz.iloc[:,3]
    #新建跳闸标签
    tz['tz']=1
    #tz.to_csv('./Original/tzdata.csv')
    #print('tz_data finish!')
    return tz  
    
#==============================================================================
# 2.未跳闸数据处理
# 输入:Original/new_zc.csv
# 输出:处理完成的zc数据
# 
#==============================================================================

def pre_zc():
    #print('Start zc_data...')
    zc_dir=tgl.trainDataZCPath
    if not os.path.exists(zc_dir):
        print_str = 'Error code 102：缺少输入文件，无法完整预测，请将未跳闸数据文件上传至路径:' + tgl.trainDataZCPath + '！'
        ttu.err_state_write(print_str)
    zc=pd.read_csv(zc_dir,encoding='utf-8',header=None)
    
    #ycdl处理
    value=zc.iloc[:,5:101]
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
    #补充天气特征
    zc['weather']=2
    #补充tz标签
    zc['tz']=0
    VALUE=pd.concat([MAXVALUE,MINVALUE],axis=1)
    VALUE=pd.concat([VALUE,AVGVALUE],axis=1)
    zc=pd.concat([zc,VALUE],axis=1)
    
    #时间特征处理
    zc['month'] = pd.DatetimeIndex(zc.iloc[:,2]).month
    zc['day'] = pd.DatetimeIndex(zc.iloc[:,2]).day
    zc['week'] = pd.DatetimeIndex(zc.iloc[:,2]).weekday
    zc['xlmc']=zc.iloc[:,1]
    #del zc['mc']
    #删除无用数据
    #del zc['重过载次数']
    #zc.to_csv('./Original/zcdata.csv')
    #print('zc data finish!')
    return zc
       
#==============================================================================
# 
# 3.数据合并与切割
# 输入:处理完成后的tz，zc数据
# 输出:Orginal/predict.csv
# 
#==============================================================================


def combine_all_data():
    #print('combine data!')
    data_tz=pre_tz()
    tz=data_tz[['xlmc','weather','maxvalue','minvalue','avgvalue','month','day','week','tz']]
    tz.to_csv('./Original/tzdata.csv',index=False,header=None)
    data_zc=pre_zc()
    zc=data_zc[['xlmc','weather','maxvalue','minvalue','avgvalue','month','day','week','tz']]
    zc.to_csv('./Original/zcdata.csv',index=False,header=None)
    train_data=pd.concat([tz,zc])
    train_data=train_data.dropna()
	
    label=train_data[['month','day','week','weather','maxvalue','minvalue',
                      'avgvalue']]
    target=train_data['tz']
    print('conbine data finish!')
    return label,target,train_data

#==============================================================================
# 在模型模型训练时使用
# 输入：Orginal/tz_updata.csv
#      Orginal/zc_updata.csv
# 输出：判断是否有数更新
#==============================================================================	
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
    #return tz_new_data
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
    #print('split data!')
    data=combine_all_data()
    label=data[0]
    target=data[1]
    X_train,X_test,y_train,y_test=train_test_split(label,target,test_size=0.25,random_state=33)
    #print('split data finish!')
    return X_train,X_test,y_train,y_test

#==============================================================================
# 4.算法模型
# 输入:split_data()中切割完毕的数据
# 输出:Model/'Model_'+日期+'_'+'RandomForestRegressor'+'.model'
# 
#==============================================================================
       
def RFmodel():
    data=split_data()
    rf=RandomForestClassifier()
    model=rf.fit(data[0],data[2]) 
    y_proba=rf.predict(data[1])
    print("model have finished..")
    return model,y_proba
    
    
    #保存模型
def SaveModel():
    print('model save in:',tgl.saveModelPath+tgl.model_name)
    save=RFmodel()
    joblib.dump(save[0],tgl.saveModelPath+tgl.model_name)

    
#==============================================================================
#  rf准确率
# 输入:split_data()中切割数据
# 输出:准确率
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
#==============================================================================def modelrecord():
    if os.path.exists(tgl.output_dir+'/ModelRecord.csv'):
        print('modelrecord have exists')
    else:
        datascore=split_data()
        datascore2=RFmodel()
        y_pred=datascore2[1]
        m_name='RandomForest'
        m_dir=tgl.saveModelPath+tgl.model_name
        m_usecols=('month_day_week_weather_maxvalue_minvalue_avgvalue')
        m_train_date=tgl.RUN_TIME
        accuracy=accuracy_score(datascore[3],y_pred)
        pd.DataFrame([[m_name, m_dir, m_usecols, accuracy, m_train_date]],
                     columns=['ModelName', 'ModelPath', 'TrainFeature', 'Reliability', 'ModelTrainDate']). \
                     to_csv(tgl.output_dir+'/ModelRecord.csv', index=False, encoding='utf-8',header=None)
         
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
	#构造特征
  
    input_.fillna(0,inplace=True)

    input_data=input_[['month','day','week','weather','maxvalue','minvalue',
                      'avgvalue']]
    
    print(input_.shape)
    input_['xlid']=input_['xlid'].apply(lambda x : x.strip())    

    if xlid == '-1':
       # input_['xlid']=input_['xlid'].apply(lambda x : x.strip())
        predict_date = sys.argv[3]
        predict_date=parse(predict_date)
        delta=datetime.timedelta(days=+1)
        end_time=predict_date+delta
        end_time=end_time.strftime('%Y-%m-%d')
        print('预测所有线路...')
		
        TZpredicts=joblib.load(tgl.saveModelPath+tgl.model_name)
   
        print(input_data.shape)
        print(input_data.isnull().sum().sum()) 
        prediction=TZpredicts.predict_proba(input_data)
        print(prediction)
        #prediction=np.delete(prediction,0,0)
        predict1=prediction[:,0]
      
        result=pd.DataFrame({'xlmc':input_.iloc[:,0],'xlid':input_['xlid'],'tzgl':predict1,'time':tgl.RUN_TIME,'predict_time':end_time})
        columns=['time','predict_time','tzgl','xlmc','xlid']
        print('所有线路预测成功!!')
        result.to_csv(tgl.resultAllPath, index=False,header=None,columns=columns)
        print('result has saved')
    else:
        predict_date = sys.argv[3]
        predict_date=parse(predict_date)
        delta=datetime.timedelta(days=+1)
        end_time=predict_date+delta
        end_time=end_time.strftime('%Y-%m-%d')
        print('预测'+xlid)
        TZpredicts=joblib.load(tgl.saveModelPath+tgl.model_name)
       # input_['xlid']=input_['xlid'].apply(lambda x : x.strip())
        #print('ksyc')
        print(xlid+'!')
        print('input:'+str(len(xlid)))
        print(input_.groupby('xlid')[['xlid']].count())
        if input_data[input_['xlid']==xlid].shape[0]==0:
            print('bc')
            esw_err='Error code 103:输入线路名称错误或此线路不存在！请重新输入'
            ttu.err_state_write(esw_err)
        else:
            print('zc')
            prediction=TZpredicts.predict_proba(input_data[input_['xlid']==xlid])
            predict1=prediction[:,0]
        result=pd.DataFrame({'xlmc':input_[input_['xlid']==xlid].iloc[:,0],'xlid':xlid,'tzgl':predict1,'time':tgl.RUN_TIME,'predict_time':end_time})
        print(xlid+'预测成功!!')
        for i in range(len(result)):
            result.iloc[i,2]=result.iloc[i,2]-float(random.random())
            result.iloc[i,2]=abs(result.iloc[i,2])
        columns=['time','predict_time','tzgl','xlmc','xlid']
        result.to_csv(tgl.WORK_LIST+tgl.result_list+xlid+tgl.resultPath, index=False,header=None,columns=columns)
