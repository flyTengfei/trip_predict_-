basepath=$(cd `dirname $0`; pwd)
python_path=/anaconda3/bin/python3.6
#train_path=/fhyc/code/src/fhyc_model_train.py
train_path=/code/src/tzyc_7day_model_train.py

#ls_date 是执行时间
ls_date=`date +%Y-%m-%d`

#随机森林决策树数量：推荐(20至40)
n_estimators=$1
#每棵决策树的最大深度：推荐(15至25)
max_depth=$2
#样本分裂数：推荐(60至80)
min_samples_split=$3
#叶子节点最小样本数：推荐(40至60)
min_samples_leaf=$4

if [ $# != 4 ]
then 
echo '参数个数不对，请输入正确的参数，随机森林决策树数量：推荐(20至40)、每棵决策树的最大深度：推荐(15至25)、样本分裂数：推荐(40至60)、叶子节点最小样本数：推荐(60至80)'>>${basepath}/Log/log_train_${ls_date}_${n_estimators}_${max_depth}_${min_sample_split}_${min_sample_leaf}.log 2>&1

exit
fi


if [ -e ${basepath}${python_path} ]; then
python3path=${basepath}${python_path}
else
    echo 'python配置文件不存在、请输入python配置文件路径:'>>${basepath}/Log/log_train_${ls_date}_${version}_${n_estimators}_${max_depth}_${min_sample_split}_${min_sample_leaf}.log 2>&1
    exit
fi

if [ -e ${basepath}${train_path} ]; then
trainpath=${basepath}${train_path}
else
    echo '训练文件不存在，请输入训练文件路径:'>>${basepath}/Log/log_train_${ls_date}_${n_estimators}_${max_depth}_${min_sample_split}_${min_sample_leaf}.log 2>&1
    exit
fi

${python3path} ${trainpath} ${basepath} ${n_estimators} ${max_depth} ${min_samples_split} ${min_samples_leaf}



