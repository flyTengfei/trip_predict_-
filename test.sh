#!/bin/sh

export JAVA_HOME=/usr/java/jdk1.7.0_67-cloudera
export CLASSPATH=.:/usr/java/jdk1.7.0_67-cloudera/lib/dt.jar:/usr/java/jdk1.7.0_67-cloudera/lib/tools.jar
export HADOOP_HOME=/opt/cloudera/parcels/CDH-5.7.0-1.cdh5.7.0.p0.45/lib/hadoop
export SQOOP_HOME=/opt/cloudera/parcels/CDH-5.7.0-1.cdh5.7.0.p0.45/lib/sqoop
export YARN_HOME=/opt/cloudera/parcels/CDH-5.7.0-1.cdh5.7.0.p0.45/lib/hadoop-yarn

datadate=`date +"%Y-%m-%d" -d "-1 days"`

echo "*********** begin : `date`**********" > /home/ZH000005/ncu/data_etl/logs/tzyc_run_$datadate.log

connect="jdbc:oracle:thin:@10.234.239.164:11521/PRSS"
username="pmcs_cx"
password="th_pmcs_cx"

impala_ip="10.234.239.15"
keytab_URL="/home/ZH000005/ZH000005.keytab"
keytab_USER="ZH000005"

kinit -kt ${keytab_URL} ${keytab_USER}

# pmcs.v_all_lineinfo
o_table_v_all_lineinfo="PMCS.V_ALL_LINEINFO"
split_by_v_all_lineinfo="obj_id"
hive_table_v_all_lineinfo="tzyc.v_all_lineinfo"
hive -e "drop table $hive_table_v_all_lineinfo"
$SQOOP_HOME/bin/sqoop import --connect $connect --username $username --password $password --table $o_table_v_all_lineinfo --split-by $split_by_v_all_lineinfo --hive-import --hive-table $hive_table_v_all_lineinfo --create-hive-table --fields-terminated-by '$' --hive-drop-import-delims --null-string '' --null-non-string '' -m 6 2>>/home/ZH000005/ncu/data_etl/logs/tzyc_run_$datadate.log

