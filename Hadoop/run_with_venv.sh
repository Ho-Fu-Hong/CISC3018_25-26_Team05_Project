#!/bin/bash
# Run Python scripts with the Hadoop virtual environment

# Activate virtual environment
source /home/Kelvin/abc/MyCISC3018-2025/hadoop_env/bin/activate

# Set environment variables
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export HADOOP_HOME=/home/Kelvin/abc/MyCISC3018-2025/hadoop-3.4.2
export SPARK_HOME=/home/Kelvin/abc/MyCISC3018-2025/spark-4.0.1-bin-hadoop3
export PATH=$PATH:$HADOOP_HOME/bin:$SPARK_HOME/bin

# Run the script passed as argument
python3 "$@"

# Deactivate (optional)
deactivate
