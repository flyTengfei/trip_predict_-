# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:13:02 2017

@author: NCU
"""
#import sys
import pandas as pd
import tzyc_tools_unit as ttu
#import datetime
import os
import tzyc_global_list as tgl

def pre_data():
    updata_path1=tgl.new_tz
    ttu.check_file_exists(updata_path1)
    if os.path.getsize(updata_path1)==0:
        print('今日无跳闸数据')
    else:
   # try:
        tz = pd.read_csv(updata_path1)
    #except Exception:
    #    tz = pd.read_csv(updata_path1, encoding='gbk')
    #print(tz.shape)
    #保存天气
        tz_weather=tz.iloc[:,7]
        tz_xlid=tz.iloc[:,2]
    #删除无用列
        tz.drop(tz.columns[0:3], axis=1,inplace=True)
    
        tz.drop(tz.columns[2:6], axis=1,inplace=True)
    #print(tz.shape)
    #print(tz.iloc[:,1]) 
        tz['month'] = pd.DatetimeIndex(tz.iloc[:,1]).month
        tz['day'] = pd.DatetimeIndex(tz.iloc[:,1]).day
        tz['week'] = pd.DatetimeIndex(tz.iloc[:,1]).weekday
        tz['xlid']=tz_xlid
        tz['weather']=tz_weather
    #tz.drop(tz.columns[2:4], axis=1,inplace=True)
        tz.drop(tz.columns[1], axis=1,inplace=True)
        tz_news=tz
        print(tz_news.shape)
        tz_news.to_csv(tgl.updata_tz,index=None,header=None)
        tz_new=pd.read_csv(tgl.updata_tz,header=None,encoding='utf-8')
    updata_path2=tgl.new_zc
    ttu.check_file_exists(updata_path2)
   # try:
    zc = pd.read_csv(updata_path2,error_bad_lines=False)
    zc_xlid=zc.iloc[:,0]
    #except Exception:
     #   zc = pd.read_csv(updata_path2, encoding='gbk')
    zc.drop(zc.columns[0], axis=1,inplace=True)
    print(zc.iloc[:,1]) 
    zc['month'] = pd.DatetimeIndex(zc.iloc[:,1]).month
    zc['day'] = pd.DatetimeIndex(zc.iloc[:,1]).day
    zc['week'] = pd.DatetimeIndex(zc.iloc[:,1]).weekday
    zc['xlid'] = zc_xlid
    zc.drop(zc.columns[1],axis=1,inplace=True)
    zc_news=zc
    print(zc_news.head())
    #zc.fillna(method='pad',inplace=True)
    zc_news.to_csv(tgl.updata_zc,index=None,header=None)
    zc_new=pd.read_csv(tgl.updata_zc,header=None,encoding='utf-8')
    if os.path.getsize(updata_path1)==0:
        all_new=zc_new
        all_new['weather']=2
        print(all_new.shape)
    else:
        all_new=pd.concat([tz_new,zc_new],axis=0)
    #all_new['weather']=tz_weather
        all_new.iloc[:,-1].fillna(2,inplace=True)
        print(all_new.shape)
    all_new.to_csv(tgl.input_dir,index=None,header=None)
    os.remove(tgl.updata_zc)
    os.remove(tgl.updata_tz)       
if __name__ == '__main__':
    pre_data()
