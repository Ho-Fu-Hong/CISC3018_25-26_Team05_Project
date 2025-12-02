#!/usr/bin/env python3
"""
Fixed version of crop yield prediction with HDFS
Adapts to actual column names in the data
"""

import os
import sys

# Add virtual environment if exists
venv_path = "/home/Kelvin/abc/MyCISC3018-2025/hadoop_env"
if os.path.exists(venv_path):
    site_packages = os.path.join(venv_path, "lib", "python3.12", "site-packages")
    sys.path.insert(0, site_packages)

# Set environment variables
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
os.environ['HADOOP_HOME'] = '/home/Kelvin/abc/MyCISC3018-2025/hadoop-3.4.2'
os.environ['SPARK_HOME'] = '/home/Kelvin/abc/MyCISC3018-2025/spark-4.0.1-bin-hadoop3'

print("=" * 70)
print("CROP YIELD PREDICTION WITH HDFS (FLEXIBLE VERSION)")
print("=" * 70)

def inspect_data(spark, hdfs_path):
    """Inspect the data to see actual column names"""
    print("\n1. Inspecting data structure...")
    
    # Load data
    df_spark = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(hdfs_path)
    
    print(f"   Total columns: {len(df_spark.columns)}")
    print(f"   Total rows: {df_spark.count():,}")
    
    print("\n   Column names:")
    for i, col in enumerate(df_spark.columns, 1):
        print(f"     {i:2}. {col}")
    
    print("\n   Data types:")
    for field in df_spark.schema.fields:
        print(f"     {field.name}: {field.dataType}")
    
    print("\n   Sample data (first 3 rows):")
    df_spark.show(3, truncate=50)
    
    return df_spark

def identify_columns(df_spark):
    """Identify relevant columns based on common patterns"""
    print("\n2. Identifying relevant columns...")
    
    all_columns = df_spark.columns
    column_mapping = {}
    
    # Common column patterns
    patterns = {
        'year': ['year', 'Year', 'YEAR', 'season_year'],
        'rainfall': ['rain', 'rainfall', 'Rainfall', 'precipitation', 'rain_fall', 'average_rain_fall'],
        'pesticide': ['pesticide', 'Pesticide', 'pesticides', 'pesticides_tonnes'],
        'temperature': ['temp', 'temperature', 'Temperature', 'avg_temp', 'average_temp'],
        'area': ['area', 'Area', 'AREA', 'country', 'Country', 'region', 'Region'],
        'crop': ['crop', 'Crop', 'item', 'Item', 'crop_type'],
        'yield': ['yield', 'Yield', 'production', 'Production', 'hg/ha_yield', 'yield_kg']
    }
    
    print("   Searching for column patterns...")
    for col in all_columns:
        col_lower = col.lower()
        
        for pattern_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern in col_lower:
                    column_mapping[pattern_name] = col
                    print(f"     Found {pattern_name}: '{col}'")
                    break
    
    # Print what we found
    print("\n   Identified columns:")
    for key, value in column_mapping.items():
        print(f"     {key}: {value}")
    
    return column_mapping

def prepare_data(df_spark, column_mapping):
    """Prepare data based on identified columns"""
    print("\n3. Preparing data...")
    
    # Create a new DataFrame with standardized column names
    from pyspark.sql.functions import col
    
    # Build select expression
    select_exprs = []
    for std_name, orig_name in column_mapping.items():
        select_exprs.append(col(orig_name).alias(std_name))
    
    # Also include any remaining columns
    used_columns = list(column_mapping.values())
    remaining_cols = [c for c in df_spark.columns if c not in used_columns]
    
    for col_name in remaining_cols[:5]:  # Include first 5 extra columns
        select_exprs.append(col(col_name))
    
    # Select columns
    df_prepared = df_spark.select(*select_exprs)
    
    print(f"   Prepared DataFrame: {df_prepared.count()} rows, {len(df_prepared.columns)} columns")
    print("   New column names:")
    for col in df_prepared.columns:
        print(f"     - {col}")
    
    return df_prepared

def run_analysis(df):
    """Run basic analysis on the data"""
    print("\n4. Running basic analysis...")
    
    # Convert to Pandas if it's a Spark DataFrame
    if hasattr(df, 'toPandas'):
        df_pandas = df.toPandas()
    else:
        df_pandas = df
    
    print(f"   DataFrame shape: {df_pandas.shape}")
    
    # Check for missing values
    missing = df_pandas.isnull().sum()
    print("\n   Missing values per column:")
    for col, count in missing.items():
        if count > 0:
            print(f"     {col}: {count} missing ({count/len(df_pandas)*100:.1f}%)")
    
    # Basic statistics for numeric columns
    numeric_cols = df_pandas.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\n   Numeric columns ({len(numeric_cols)}): {list(numeric_cols)}")
        
        # Show basic stats for first few numeric columns
        for col in numeric_cols[:3]:
            print(f"\n     {col}:")
            print(f"       Min: {df_pandas[col].min():.2f}")
            print(f"       Max: {df_pandas[col].max():.2f}")
            print(f"       Mean: {df_pandas[col].mean():.2f}")
            print(f"       Std: {df_pandas[col].std():.2f}")
    
    # Unique values for categorical columns
    categorical_cols = df_pandas.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\n   Categorical columns ({len(categorical_cols)}): {list(categorical_cols)}")
        
        for col in categorical_cols[:2]:
            unique_vals = df_pandas[col].nunique()
            print(f"\n     {col}: {unique_vals} unique values")
            if unique_vals < 10:
                print(f"       Values: {sorted(df_pandas[col].dropna().unique())}")
    
    return df_pandas

def save_results(spark, df, output_base="hdfs://localhost:9000/project/agriculture"):
    """Save results to HDFS"""
    print("\n5. Saving results to HDFS...")
    
    import subprocess
    
    # Create directories if they don't exist
    subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', '/project/agriculture/processed/'], 
                  check=False)
    subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', '/project/agriculture/results/'], 
                  check=False)
    
    # Save the prepared data
    if hasattr(df, 'write'):
        # It's a Spark DataFrame
        output_path = f"{output_base}/processed/prepared_data.parquet"
        df.write.mode("overwrite").parquet(output_path)
        print(f"   ✅ Prepared data saved as Parquet: {output_path}")
        
        # Also save a sample as CSV
        sample_path = f"{output_base}/results/data_sample.csv"
        df.limit(1000).write.mode("overwrite").csv(sample_path, header=True)
        print(f"   ✅ Sample saved as CSV: {sample_path}")
    else:
        # It's a Pandas DataFrame
        import pandas as pd
        
        # Save locally first, then upload
        local_path = "/tmp/prepared_data.csv"
        df.to_csv(local_path, index=False)
        
        hdfs_path = "/project/agriculture/processed/prepared_data.csv"
        subprocess.run(['hdfs', 'dfs', '-put', '-f', local_path, hdfs_path], check=False)
        print(f"   ✅ Prepared data saved as CSV: hdfs://localhost:9000{hdfs_path}")
        
        # Clean up
        import os
        os.remove(local_path)
    
    # Create a summary report
    report = f"""
HDFS DATA PROCESSING REPORT
===========================
Date: {pd.Timestamp.now()}
Data Source: /project/agriculture/raw/yield_df.csv
Total Records: {len(df) if hasattr(df, '__len__') else df.count()}
Total Columns: {len(df.columns)}

COLUMNS FOUND:
{'-' * 40}
"""
    
    if hasattr(df, 'toPandas'):
        df_pandas = df.toPandas()
    else:
        df_pandas = df
    
    for col in df_pandas.columns:
        dtype = df_pandas[col].dtype
        unique = df_pandas[col].nunique()
        missing = df_pandas[col].isnull().sum()
        report += f"{col}: {dtype} (Unique: {unique}, Missing: {missing})\n"
    
    # Save report
    report_path = "/tmp/data_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    hdfs_report_path = "/project/agriculture/results/data_report.txt"
    subprocess.run(['hdfs', 'dfs', '-put', '-f', report_path, hdfs_report_path], check=False)
    print(f"   ✅ Data report saved: hdfs://localhost:9000{hdfs_report_path}")
    
    # Clean up
    import os
    os.remove(report_path)

def main():
    """Main function"""
    try:
        from pyspark.sql import SparkSession
        
        # Create Spark session
        print("\nInitializing Spark with HDFS...")
        spark = SparkSession.builder \
            .appName("CropYieldHDFSFixed") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
            .getOrCreate()
        
        print(f"✅ Spark version: {spark.version}")
        
        # HDFS path
        hdfs_path = "hdfs://localhost:9000/project/agriculture/raw/yield_df.csv"
        
        # Step 1: Inspect data
        df_spark = inspect_data(spark, hdfs_path)
        
        # Step 2: Identify columns
        column_mapping = identify_columns(df_spark)
        
        if not column_mapping:
            print("\n⚠ Could not identify standard columns.")
            print("  Using all columns as-is...")
            df_prepared = df_spark
        else:
            # Step 3: Prepare data
            df_prepared = prepare_data(df_spark, column_mapping)
        
        # Step 4: Run analysis
        df_analysis = run_analysis(df_prepared)
        
        # Step 5: Save results
        save_results(spark, df_prepared)
        
        spark.stop()
        
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE!")
        print("=" * 70)
        print("\nOutput saved to HDFS:")
        print("• Prepared data: /project/agriculture/processed/")
        print("• Results: /project/agriculture/results/")
        print("• Data report: /project/agriculture/results/data_report.txt")
        print("\nWeb UI: http://localhost:9870")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
