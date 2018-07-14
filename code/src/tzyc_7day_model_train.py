# -*- coding: utf-8 -*—

import sys
import tzyc_7pre_mutil_model as t7mm

#模型训练
def model_train():
    #run_time = sys.argv[2]
    
    # 随机森林
    n_estimators=sys.argv[2]
    #n_estimators=30
    #max_depth = sys.argv[3]
    max_depth=20
    
    #min_samples_split = sys.argv[4]
    min_samples_split=60
    #min_samples_leaf = sys.argv[5]
    min_samples_leaf=50
#每棵决策树的最大深度：推荐(15至25)
 
#样本分裂数：推荐(60至80)
    
#叶子节点最小样本数：推荐(40至60)
 

    train_dicts = {'n_estimators': int(n_estimators), 'max_depth': int(max_depth),
			   'model_name': 'RandomForestRegressor', 'min_samples_split': int(min_samples_split),
			   'min_samples_leaf': int(min_samples_leaf)}
   
    
       
    t7mm.RFmodel(train_dicts)


if __name__ == '__main__':
    model_train()
