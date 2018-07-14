# -*- coding: utf-8 -*—

"""
该单元中定义跳闸预测流程中用到的工具单元

"""

import os
import sys
import datetime
import pandas as pd


import tzyc_global_list as tgl

#==============================================================================
# 读取csv文件函数
#==============================================================================
def csv_reader(cr_dir, cr_usecols):
    try:
        return pd.read_csv(cr_dir, usecols=cr_usecols, encoding='utf-8')
    except Exception:
        return pd.read_csv(cr_dir, usecols=cr_usecols, encoding='gbk')


#==============================================================================
# 状态表报错函数单元
#==============================================================================
def err_state_write(esw_err):
    state_ = pd.DataFrame([[tgl.RUN_TIME, 1, tgl.xlid,esw_err,'auto_predict.sh',
                            str(datetime.datetime.now()).split('.')[0]]])
    state_.to_csv(tgl.output_dir+'/job_state.csv', encoding='utf-8', header=None, index=False)
    sys.exit(1)

#==============================================================================
# 状态表正确状态函数单元
#==============================================================================
def rigth_state_write(esw_err):
    state_ = pd.DataFrame([[tgl.RUN_TIME, 0, tgl.xlid, esw_err,'auto_predict.sh',
                            str(datetime.datetime.now()).split('.')[0]]])
    state_.to_csv(tgl.output_dir+'/job_state.csv', encoding='utf-8', header=None, index=False)
    sys.exit(0)

#==============================================================================
# 检查文件是否存在函数单元
#==============================================================================
def check_file_exists(cfe_dir):
    if not os.path.exists(cfe_dir):
        err = cfe_dir+'文件丢失异常！'
        print(err, 'Time:', datetime.datetime.now())
        err_state_write(err)





