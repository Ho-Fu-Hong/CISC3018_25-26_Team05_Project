import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# IMPORTANT: Add these imports for HDFS
from pyspark.sql import SparkSession
import os

# Set plotting style
plt.style.use("ggplot")

def setup_spark_for_hdfs():
    """Setup Spark session with HDFS configuration"""
    spark = SparkSession.builder \
        .appName("CropYieldPrediction") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .config("spark.sql.warehouse.dir", "hdfs://localhost:9000/project/spark_warehouse") \
        .getOrCreate()
    return spark

def load_from_hdfs(spark, hdfs_path):
    """Load CSV from HDFS using Spark and convert to Pandas"""
    print(f"Loading data from HDFS: {hdfs_path}")
    
    # Load using Spark
    df_spark = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(hdfs_path)
    
    # Convert to Pandas (for scikit-learn compatibility)
    df = df_spark.toPandas()
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df

def save_to_hdfs(spark, df, hdfs_path, format="parquet"):
    """Save DataFrame to HDFS"""
    print(f"Saving to HDFS: {hdfs_path}")
    
    # Convert Pandas to Spark DataFrame
    df_spark = spark.createDataFrame(df)
    
    if format.lower() == "parquet":
        df_spark.write.mode("overwrite").parquet(hdfs_path)
    elif format.lower() == "csv":
        df_spark.write.mode("overwrite").csv(hdfs_path, header=True)
    
    print(f"✓ Saved to HDFS: {hdfs_path}")

def load_and_preprocess_data(spark, file_path):
    """Load and preprocess the data from HDFS"""
    df = load_from_hdfs(spark, file_path)
    
    # Remove unnecessary column
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Select relevant features
    selected_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 
                    'avg_temp', 'Area', 'Item', 'hg/ha_yield']
    df = df[selected_cols]
    
    return df

def explore_data(df):
    """Explore the dataset"""
    print("Dataset Information:")
    print(df.info())
    
    print("\nDataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDataset Description:")
    print(df.describe())
    
    # Calculate correlation only for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("\nCorrelation Matrix (Numerical Features Only):")
    print(df[numerical_cols].corr())
    
    print(f"\nNumber of Countries: {len(df['Area'].unique())}")
    print(f"Number of Crops: {len(df['Item'].unique())}")

def visualize_data(df):
    """Create visualizations for data exploration"""
    # Country distribution
    plt.figure(figsize=(15, 20))
    sns.countplot(y=df['Area'])
    plt.title('Country Distribution')
    plt.tight_layout()
    plt.savefig('/tmp/country_distribution.png')
    print("Saved: /tmp/country_distribution.png")
    plt.show()
    
    # Crop distribution
    plt.figure(figsize=(15, 20))
    sns.countplot(y=df['Item'])
    plt.title('Crop Distribution')
    plt.tight_layout()
    plt.savefig('/tmp/crop_distribution.png')
    print("Saved: /tmp/crop_distribution.png")
    plt.show()
    
    # Total yield by country (top 30 countries for better visualization)
    yield_by_country = df.groupby('Area')['hg/ha_yield'].sum().sort_values(ascending=False)
    top_countries = yield_by_country.head(30)
    
    plt.figure(figsize=(15, 12))
    sns.barplot(y=top_countries.index, x=top_countries.values)
    plt.title('Total Yield by Country (Top 30)')
    plt.tight_layout()
    plt.savefig('/tmp/yield_by_country.png')
    print("Saved: /tmp/yield_by_country.png")
    plt.show()
    
    # Total yield by crop
    yield_by_crop = df.groupby('Item')['hg/ha_yield'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(15, 12))
    sns.barplot(y=yield_by_crop.index, x=yield_by_crop.values)
    plt.title('Total Yield by Crop')
    plt.tight_layout()
    plt.savefig('/tmp/yield_by_crop.png')
    print("Saved: /tmp/yield_by_crop.png")
    plt.show()
    
    # Correlation heatmap for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('/tmp/correlation_heatmap.png')
    print("Saved: /tmp/correlation_heatmap.png")
    plt.show()

def prepare_features(df):
    """Prepare features and target variable"""
    X = df.drop('hg/ha_yield', axis=1)
    y = df['hg/ha_yield']
    return X, y

def create_preprocessor():
    """Create data preprocessor"""
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    scale = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('StandardScale', scale, [0, 1, 2, 3]),  # Standardize numerical features
            ('OneHotEncode', ohe, [4, 5])           # One-hot encode categorical features
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def train_and_evaluate_models(X_train_dummy, X_test_dummy, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=0),
        'KNN': KNeighborsRegressor(),
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_dummy, y_train)
        y_pred = model.predict(X_test_dummy)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'R2': r2}
        print(f"{name}: MAE: {mae:.2f}, R2 Score: {r2:.4f}")
    
    return results

def create_predictive_system(model, preprocessor):
    """Create predictive system"""
    def predict_yield(Year, average_rain_fall_mm_per_year, pesticides_tonnes, 
                     avg_temp, Area, Item):
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, 
                            avg_temp, Area, Item]], dtype=object)
        transform_features = preprocessor.transform(features)
        predicted_yield = model.predict(transform_features).reshape(-1, 1)
        return predicted_yield[0][0]
    
    return predict_yield

def save_models_to_hdfs(spark, model, preprocessor, model_name="dtr_model"):
    """Save trained model and preprocessor to HDFS"""
    # Save locally first
    local_model_path = f"/tmp/{model_name}.pkl"
    local_preprocessor_path = f"/tmp/{model_name}_preprocessor.pkl"
    
    pickle.dump(model, open(local_model_path, "wb"))
    pickle.dump(preprocessor, open(local_preprocessor_path, "wb"))
    
    # Upload to HDFS
    hdfs_model_path = f"hdfs://localhost:9000/project/agriculture/models/{model_name}.pkl"
    hdfs_preprocessor_path = f"hdfs://localhost:9000/project/agriculture/models/{model_name}_preprocessor.pkl"
    
    # Use Hadoop command to copy to HDFS
    import subprocess
    subprocess.run(['hdfs', 'dfs', '-put', local_model_path, 
                   '/project/agriculture/models/'], check=False)
    subprocess.run(['hdfs', 'dfs', '-put', local_preprocessor_path, 
                   '/project/agriculture/models/'], check=False)
    
    print(f"Model saved to HDFS: {hdfs_model_path}")
    print(f"Preprocessor saved to HDFS: {hdfs_preprocessor_path}")

def save_results_to_hdfs(spark, results_df, filename="model_results.csv"):
    """Save model results to HDFS"""
    # Save locally first
    local_path = f"/tmp/{filename}"
    results_df.to_csv(local_path, index=False)
    
    # Upload to HDFS
    hdfs_path = f"hdfs://localhost:9000/project/agriculture/results/{filename}"
    
    import subprocess
    subprocess.run(['hdfs', 'dfs', '-put', local_path, 
                   '/project/agriculture/results/'], check=False)
    
    print(f"Results saved to HDFS: {hdfs_path}")

def main():
    print("=" * 60)
    print("CROP YIELD PREDICTION WITH HADOOP HDFS INTEGRATION")
    print("=" * 60)
    
    # Step 1: Setup Spark with HDFS
    print("\n1. Setting up Spark with HDFS...")
    spark = setup_spark_for_hdfs()
    print(f"   Spark version: {spark.version}")
    
    # Step 2: Load data from HDFS
    print("\n2. Loading data from HDFS...")
    hdfs_data_path = "hdfs://localhost:9000/project/agriculture/raw/yield_df.csv"
    df = load_and_preprocess_data(spark, hdfs_data_path)
    
    # Step 3: Data exploration
    print("\n3. Exploring data...")
    explore_data(df)
    
    # Step 4: Data visualization
    print("\n4. Creating visualizations...")
    visualize_data(df)
    
    # Step 5: Prepare features
    print("\n5. Preparing features...")
    X, y = prepare_features(df)
    
    # Step 6: Split dataset
    print("\n6. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    )
    
    # Step 7: Create and train preprocessor
    print("\n7. Creating preprocessor...")
    preprocessor = create_preprocessor()
    X_train_dummy = preprocessor.fit_transform(X_train)
    X_test_dummy = preprocessor.transform(X_test)
    
    print(f"   Training set shape: {X_train_dummy.shape}")
    print(f"   Test set shape: {X_test_dummy.shape}")
    
    # Step 8: Train and evaluate models
    print("\n8. Training and evaluating models...")
    results = train_and_evaluate_models(X_train_dummy, X_test_dummy, y_train, y_test)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    print("\nModel Results Summary:")
    print(results_df)
    
    # Step 9: Save results to HDFS
    print("\n9. Saving results to HDFS...")
    save_results_to_hdfs(spark, results_df)
    
    # Step 10: Select best model (using Decision Tree as example)
    print("\n10. Training final model...")
    best_model = DecisionTreeRegressor(random_state=0)
    best_model.fit(X_train_dummy, y_train)
    
    # Step 11: Create predictive system
    print("\n11. Creating predictive system...")
    predict_function = create_predictive_system(best_model, preprocessor)
    
    # Test predictive system
    result = predict_function(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
    print(f"   Test Prediction (1990, Albania, Maize): {result:.2f}")
    
    # Step 12: Save models to HDFS
    print("\n12. Saving models to HDFS...")
    save_models_to_hdfs(spark, best_model, preprocessor, "crop_yield_model")
    
    # Step 13: Save processed data to HDFS
    print("\n13. Saving processed data to HDFS...")
    save_to_hdfs(spark, df, "hdfs://localhost:9000/project/agriculture/processed/crop_yield_cleaned.parquet")
    
    spark.stop()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutput saved to HDFS:")
    print("• Raw data: /project/agriculture/raw/")
    print("• Processed data: /project/agriculture/processed/")
    print("• Model results: /project/agriculture/results/")
    print("• Trained models: /project/agriculture/models/")
    print("• Visualizations: /tmp/*.png")
    print("\nHDFS Web UI: http://localhost:9870")
    print("=" * 60)

if __name__ == "__main__":
    main()
