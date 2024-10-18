
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Cloud's Data Preprocessing Module
def data_preprocessing_pipeline(data):
    # Handling missing values (example: drop rows with missing values)
    data_cleaned = data.dropna()

    # Normalizing numerical features
    scaler = StandardScaler()
    numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    data_cleaned[numerical_columns] = scaler.fit_transform(data_cleaned[numerical_columns])

    return data_cleaned

# Cloud's Feature Selection Module
def feature_selection_pipeline(X, y, num_features=5):
    # Using SelectKBest to select top features based on ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_selected = selector.fit_transform(X, y)

    return X_selected, selector.get_support(indices=True)  # Return selected features and indices

# Example usage with sample dataset (you can replace this with any dataset)
def main():
    # Simulate a sample dataset
    data = pd.DataFrame({
        'feature1': [1.2, 2.4, 3.6, 4.8, 5.0],
        'feature2': [2.3, 3.5, 6.5, 3.4, 1.2],
        'feature3': [5.1, 1.3, 3.2, 4.9, 2.1],
        'target': [1, 0, 1, 0, 1]
    })

    # Data Preprocessing
    processed_data = data_preprocessing_pipeline(data)

    # Splitting features and target
    X = processed_data.drop(columns=['target'])
    y = processed_data['target']

    # Feature Selection (select top 2 features)
    X_selected, selected_indices = feature_selection_pipeline(X, y, num_features=2)
    print(f"Selected Features (Indices): {selected_indices}")

if __name__ == "__main__":
    main()
