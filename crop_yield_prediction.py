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

# Set plotting style
plt.style.use("ggplot")

def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    df = pd.read_csv(file_path)
    
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
    plt.show()
    
    # Crop distribution
    plt.figure(figsize=(15, 20))
    sns.countplot(y=df['Item'])
    plt.title('Crop Distribution')
    plt.tight_layout()
    plt.show()
    
    # Total yield by country (top 30 countries for better visualization)
    yield_by_country = df.groupby('Area')['hg/ha_yield'].sum().sort_values(ascending=False)
    top_countries = yield_by_country.head(30)
    
    plt.figure(figsize=(15, 12))
    sns.barplot(y=top_countries.index, x=top_countries.values)
    plt.title('Total Yield by Country (Top 30)')
    plt.tight_layout()
    plt.show()
    
    # Total yield by crop
    yield_by_crop = df.groupby('Item')['hg/ha_yield'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(15, 12))
    sns.barplot(y=yield_by_crop.index, x=yield_by_crop.values)
    plt.title('Total Yield by Crop')
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
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

def save_models(model, preprocessor, model_path="dtr.pkl", preprocessor_path="preprocesser.pkl"):
    """Save trained model and preprocessor"""
    pickle.dump(model, open(model_path, "wb"))
    pickle.dump(preprocessor, open(preprocessor_path, "wb"))
    print("Model and preprocessor saved successfully")

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("yield_df.csv")
    
    # Data exploration
    explore_data(df)
    
    # Data visualization
    visualize_data(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    )
    
    # Create and train preprocessor
    preprocessor = create_preprocessor()
    X_train_dummy = preprocessor.fit_transform(X_train)
    X_test_dummy = preprocessor.transform(X_test)
    
    print(f"Training set shape: {X_train_dummy.shape}")
    print(f"Test set shape: {X_test_dummy.shape}")
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train_dummy, X_test_dummy, y_train, y_test)
    
    # Select best model (using Decision Tree as example)
    best_model = DecisionTreeRegressor(random_state=0)
    best_model.fit(X_train_dummy, y_train)
    
    # Create predictive system
    predict_function = create_predictive_system(best_model, preprocessor)
    
    # Test predictive system
    result = predict_function(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
    print(f"\nPrediction Result: {result}")
    
    # Save models
    save_models(best_model, preprocessor)

if __name__ == "__main__":
    main()