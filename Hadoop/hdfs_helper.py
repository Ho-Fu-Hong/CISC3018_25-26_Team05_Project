"""
hdfs_helper.py - Simple HDFS integration for Agricultural Project
Place this in your project directory and import in Jupyter notebooks
"""

import os
import sys

class HDFSHelper:
    """Helper class for HDFS operations"""
    
    def __init__(self):
        self.hdfs_uri = "hdfs://localhost:9000"
        self.project_path = "/project/agriculture"
        
    def get_data_paths(self):
        """Return all available data paths"""
        return {
            # Raw data paths
            "yield_df": f"{self.hdfs_uri}{self.project_path}/raw/yield_df.csv",
            "yield": f"{self.hdfs_uri}{self.project_path}/raw/yield.csv",
            "temp": f"{self.hdfs_uri}{self.project_path}/raw/temp.csv",
            "rainfall": f"{self.hdfs_uri}{self.project_path}/raw/rainfall.csv",
            "pesticides": f"{self.hdfs_uri}{self.project_path}/raw/pesticides.csv",
            
            # Directory paths
            "raw_dir": f"{self.hdfs_uri}{self.project_path}/raw/",
            "processed_dir": f"{self.hdfs_uri}{self.project_path}/processed/",
            "results_dir": f"{self.hdfs_uri}{self.project_path}/results/",
            "scripts_dir": f"{self.hdfs_uri}{self.project_path}/scripts/"
        }
    
    def create_spark_session(self, app_name="AgriculturalProject"):
        """Create Spark session with HDFS configuration"""
        from pyspark.sql import SparkSession
        
        # Set environment variables
        os.environ['HADOOP_HOME'] = '/home/Kelvin/abc/MyCISC3018-2025/hadoop-3.4.2'
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
        
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.hadoop.fs.defaultFS", self.hdfs_uri) \
            .config("spark.sql.warehouse.dir", f"{self.hdfs_uri}/project/spark_warehouse") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()
        
        return spark
    
    def load_dataset(self, spark, dataset_name):
        """Load a specific dataset from HDFS"""
        paths = self.get_data_paths()
        
        if dataset_name in paths:
            path = paths[dataset_name]
            print(f"Loading {dataset_name} from: {path}")
            
            # Load CSV from HDFS
            df = spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(path)
            
            print(f"✓ Loaded {df.count()} records")
            return df
        else:
            print(f"Dataset '{dataset_name}' not found. Available: {list(paths.keys())}")
            return None

# Create a global instance
hdfs = HDFSHelper()

# Convenience functions
def setup_spark_for_project(app_name="CropYieldAnalysis"):
    """Quick setup for Spark with HDFS"""
    return hdfs.create_spark_session(app_name)

def load_all_datasets(spark):
    """Load all datasets from HDFS"""
    paths = hdfs.get_data_paths()
    
    datasets = {}
    for name, path in paths.items():
        if name.endswith(".csv") or name in ["yield_df", "yield", "temp", "rainfall", "pesticides"]:
            try:
                df = spark.read \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .csv(path)
                datasets[name] = df
                print(f"✓ Loaded {name}: {df.count()} records")
            except:
                print(f"✗ Could not load {name}")
    
    return datasets

# Print info when imported
print("=" * 60)
print("HDFS HELPER FOR AGRICULTURAL PROJECT")
print("=" * 60)
print("\nAvailable functions:")
print("1. hdfs.get_data_paths() - Get all HDFS paths")
print("2. setup_spark_for_project() - Create Spark session")
print("3. hdfs.load_dataset(spark, 'yield_df') - Load specific dataset")
print("4. load_all_datasets(spark) - Load all datasets")
print("\nExample usage:")
print('''
# In your Jupyter notebook:
from hdfs_helper import setup_spark_for_project, hdfs

# 1. Create Spark session
spark = setup_spark_for_project("YourAnalysis")

# 2. Load data
df_yield = hdfs.load_dataset(spark, "yield_df")
df_temp = hdfs.load_dataset(spark, "temp")

# 3. Use your existing crop_yield_prediction.py code
# ... your analysis here ...
''')
print("=" * 60)
