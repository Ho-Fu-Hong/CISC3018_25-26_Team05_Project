#!/bin/bash
# setup_hadoop_project.sh - Setup Hadoop in project directory

echo "=== Setting up Hadoop in project directory ==="
echo "Project path: $(pwd)"

# Set environment variables
export HADOOP_HOME="/home/Kelvin/abc/MyCISC3018-2025/hadoop-3.4.2"
export PATH="$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH"

# Create Hadoop directories
echo "Creating Hadoop directories..."
mkdir -p $HADOOP_HOME/namenode
mkdir -p $HADOOP_HOME/datanode
mkdir -p $HADOOP_HOME/tmp
mkdir -p $HADOOP_HOME/logs
mkdir -p $HADOOP_HOME/namesecondary
mkdir -p hdfs_storage
mkdir -p project_data

# Set permissions
chmod -R 755 $HADOOP_HOME
chmod -R 755 hdfs_storage
chmod -R 755 project_data

# Format HDFS (only if not already formatted)
echo "Checking if HDFS needs formatting..."
if [ ! -d "$HADOOP_HOME/namenode/current" ]; then
    echo "Formatting HDFS Namenode..."
    $HADOOP_HOME/bin/hdfs namenode -format -force
    if [ $? -eq 0 ]; then
        echo "✓ HDFS formatted successfully"
    else
        echo "✗ HDFS formatting failed"
        exit 1
    fi
else
    echo "✓ HDFS already formatted"
fi

echo ""
echo "=== Hadoop project setup complete! ==="
echo "To start Hadoop: $HADOOP_HOME/sbin/start-dfs.sh"
echo "Web UI: http://localhost:9870"
echo "Test with: $HADOOP_HOME/bin/hdfs dfs -ls /"

