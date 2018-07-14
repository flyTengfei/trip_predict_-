xlid=$1
predict_time=$2
#predict_time=$(date +%Y-%m-%d)
#predict_time=2017-12-01
basepath=$(cd `dirname $0`; pwd)
python_path=/anaconda3/bin/python3.6
impala_path=/code/src/tzyc_updata.py
predict_path=/code/src/tzyc_model_predict.py
data_time=$(date +%Y%m%d)

kinit -kt /home/ZH000005/ZH000005.keytab ZH000005
if [ $# != 2 ]
then
    '参数不全，请输入正确的参数:预测类型、预测日期'>>${basepath}/Log/log_predict_RF${data_time}message.log 2>&1
exit
fi


#hive -e "select * from tzyc.tz where rtime like \"$predict_time%\";">${basepath}/Original/new_tz.csv
#sed -i 's/\t/,/g' ${basepath}/Original/new_tz.csv
#sed -i 's/\r/\n/g' ${basepath}/Original/new_tz.csv
#hive -e "select * from tzyc.wtz where rday like \"$predict_time%\";">${basepath}/Original/new_zc.csv
#sed -i 's/\t/,/g' ${basepath}/Original/new_zc.csv
#sed -i 's/\r/\n/g' ${basepath}/Original/new_zc.csv
#sed -i 's/\r//g' ${basepath}/Original/new_zc.csv
#sed -i 'N;/"s/\n//' ${basepath}/Original/new_tz.csv
#sed 's/^M//g'  ${basepath}/Original/new_zc.csv
#if [ -e ${basepath}${python_path} ] && [ -e ${basepath}${impala_path} ]
#then
#  ${basepath}${python_path} ${basepath}${impala_path} ${basepath} ${predict_time}>>${basepath}/Log/log_split_data${data_time}message.log 2>&1

#else
#    echo 'python配置文件目录错误或者数据切割脚本文件目录错误'>>${basepath}/Log/log_split_RF${data_time}message.log 2>&1
#fi

if [ -e ${basepath}${python_path} ] && [ -e ${basepath}${predict_path} ]
then
  ${basepath}${python_path} ${basepath}${predict_path} ${basepath} ${xlid} ${predict_time}>${basepath}/Log/log_predict_RF${data_time}message.log 2>&1
else
    echo 'python配置文件目录错误或者预测脚本文件目录错误'>>${basepath}/Log/log_predict_RF${data_time}message.log 2>&1
fi

re=$?
echo $re
hive -e"load data local inpath \"$basepath/Output/job_state.csv\" into table tzyc.tzyc_forecast_job_state;"
impala-shell -B -i 10.234.239.15 -q"invalidate metadata tzyc.tzyc_forecast_job_state;";
if [ $re == 0 ]
then
while true
do
if [ $xlid == -1 ]
#if [ -e $basepath/Result/result_all.csv ]
then

hive -e"load data local inpath \"$basepath/Result/result_all.csv\" into table tzyc.tzyc_forecast_result;"
hive -e"load data local inpath \"$basepath/Output/ModelRecord.csv\" into table tzyc.tzyc_model_recode;"
hive -e"load data local inpath \"$basepath/Output/job_state.csv\" into table tzyc.tzyc_forecast_job_state;"
impala-shell -B -i 10.234.239.15 -q"invalidate metadata tzyc.tzyc_forecast_result;";
impala-shell -B -i 10.234.239.15 -q"invalidate metadata tzyc.tzyc_model_recode;";
#impala-shell -B -i 10.234.239.15 -q"invalidate metadata tzyc.tzyc_forecast_job_state;";
exit
else
hive -e"load data local inpath \"$basepath/Result/${xlid}result.csv\" into table tzyc.tzyc_forecast_result_tdxl;"
hive -e"load data local inpath \"$basepath/Output/ModelRecord.csv\" into table tzyc.tzyc_model_recode;"
#hive -e"load data local inpath \"$basepath/Output/job_state.csv\" into table tzyc.tzyc_forecast_job_state;"
impala-shell -B -i 10.234.239.15 -q"invalidate metadata tzyc.tzyc_forecast_result_tdxl;";
impala-shell -B -i 10.234.239.15 -q"invalidate metadata tzyc.tzyc_model_recode;";
#impala-shell -B -i 10.234.239.15 -q"invalidate metadata tzyc.tzyc_forecast_job_state;";
exit
fi
done
exit

else
sleep 1
exit
fi
