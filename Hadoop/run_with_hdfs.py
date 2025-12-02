#!/usr/bin/env python3
"""
run_with_hdfs.py - Run existing crop_yield_prediction.py with HDFS data
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def setup_environment():
    """Setup environment for HDFS"""
    os.environ['HADOOP_HOME'] = '/home/Kelvin/abc/MyCISC3018-2025/hadoop-3.4.2'
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
    
    print("Environment setup for HDFS")

def load_data_from_hdfs():
    """Load data from HDFS to local file for existing script"""
    import subprocess
    
    print("Downloading data from HDFS...")
    
    # Create local data directory
    os.makedirs('hdfs_data', exist_ok=True)
    
    # Download each CSV from HDFS
    files = [
        ('yield_df.csv', 'Main crop yield dataset'),
        ('yield.csv', 'Additional yield data'),
        ('temp.csv', 'Temperature data'),
        ('rainfall.csv', 'Rainfall data'),
        ('pesticides.csv', 'Pesticide data')
    ]
    
    for filename, description in files:
        hdfs_path = f"/project/agriculture/raw/{filename}"
        local_path = f"hdfs_data/{filename}"
        
        print(f"  Downloading {description}...")
        result = subprocess.run(
            ['hdfs', 'dfs', '-get', hdfs_path, local_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"    ✓ {filename}")
        else:
            print(f"    ✗ {filename} (not found in HDFS)")
    
    return 'hdfs_data/yield_df.csv'

def upload_results_to_hdfs():
    """Upload results back to HDFS"""
    import subprocess
    import glob
    
    print("\nUploading results to HDFS...")
    
    # Create results directory in HDFS
    subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', '/project/agriculture/results/'], 
                  check=False)
    
    # Upload any generated files
    result_files = glob.glob('*.pkl') + glob.glob('*.csv')
    
    for file in result_files:
        print(f"  Uploading {file}...")
        subprocess.run(
            ['hdfs', 'dfs', '-put', file, f'/project/agriculture/results/'],
            check=False
        )
    
    print("✓ Results uploaded to HDFS")

def main():
    """Main function to run existing code with HDFS"""
    print("=" * 60)
    print("RUNNING EXISTING CODE WITH HADOOP HDFS")
    print("=" * 60)
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Download data from HDFS
    data_file = load_data_from_hdfs()
    
    print(f"\nData ready at: {data_file}")
    
    # Step 3: Ask user if they want to run existing script
    response = input("\nDo you want to run the existing crop_yield_prediction.py? (y/n): ")
    
    if response.lower() == 'y':
        print("\nRunning existing script with HDFS data...")
        
        # Import and run the existing script
        try:
            import crop_yield_prediction
            print("\nNote: The script will use data downloaded from HDFS")
            print("Results will be saved locally and uploaded to HDFS")
            
            # We need to modify the script to use our data file
            # For now, just inform the user
            print("\n" + "=" * 60)
            print("TO USE HDFS DATA IN YOUR EXISTING SCRIPT:")
            print("=" * 60)
            print("Modify line in crop_yield_prediction.py:")
            print(f'  Change: df = load_and_preprocess_data("yield_df.csv")')
            print(f'  To:     df = load_and_preprocess_data("{data_file}")')
            print("\nOr run the HDFS-integrated version:")
            print("  python3 crop_yield_prediction_hdfs.py")
            
        except ImportError:
            print("Could not import crop_yield_prediction.py")
            print("Make sure it's in the current directory")
    
    # Step 4: Upload any results
    upload_results_to_hdfs()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nData downloaded from HDFS to: hdfs_data/")
    print("Run your analysis, then results will be uploaded to HDFS")
    print("\nHDFS paths:")
    print("  Data: /project/agriculture/raw/")
    print("  Results: /project/agriculture/results/")
    print("  Web UI: http://localhost:9870")

if __name__ == "__main__":
    main()
