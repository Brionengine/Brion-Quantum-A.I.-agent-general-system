
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
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

# Cloud's AutoML Pipeline
def auto_ml_pipeline(X, y):
    # Splitting the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Defining models to try in AutoML
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Neural Network': MLPClassifier(max_iter=500)
    }

    # Hyperparameter grids for each model
    param_grids = {
        'Decision Tree': {'max_depth': [3, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100, 200]},
        'Neural Network': {'hidden_layer_sizes': [(64, 32), (128, 64)]}
    }

    # Dictionary to store the best model and its performance
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # Looping through models and hyperparameters
    for model_name, model in models.items():
        print(f"Training {model_name}...")

        # Using GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Getting the best model
        best_estimator = grid_search.best_estimator_

        # Making predictions on the test set
        y_pred = best_estimator.predict(X_test)

        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.4f}")

        # Keeping track of the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_estimator
            best_model_name = model_name

    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    return best_model, best_model_name, best_accuracy

# Full Automated Workflow
def cloud_automated_workflow(data):
    # Step 1: Data Preprocessing
    print("Running Data Preprocessing...")
    processed_data = data_preprocessing_pipeline(data)

    # Splitting features and target
    X = processed_data.drop(columns=['target'])
    y = processed_data['target']

    # Step 2: Feature Selection
    print("Running Feature Selection...")
    X_selected, selected_indices = feature_selection_pipeline(X, y, num_features=2)
    print(f"Selected Features (Indices): {selected_indices}")

    # Step 3: AutoML
    print("Running AutoML...")
    best_model, best_model_name, best_accuracy = auto_ml_pipeline(X_selected, y)
    print(f"AutoML completed. Best model: {best_model_name} with accuracy {best_accuracy:.4f}")

    return best_model

# Example usage with sample dataset (replace with real dataset)
def main():
    # Simulating a dataset
    data = pd.DataFrame({
        'feature1': [1.2, 2.4, 3.6, 4.8, 5.0, 3.1, 4.2, 5.3, 6.4, 2.2],
        'feature2': [2.3, 3.5, 6.5, 3.4, 1.2, 3.6, 4.7, 2.9, 1.8, 4.3],
        'feature3': [5.1, 1.3, 3.2, 4.9, 2.1, 3.6, 2.7, 4.8, 3.9, 2.4],
        'target': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    })

    # Running the full automated workflow
    best_model = cloud_automated_workflow(data)
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()
