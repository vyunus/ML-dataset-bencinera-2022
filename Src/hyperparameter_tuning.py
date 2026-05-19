from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def optimize_random_forest(X_train, y_train, cv=5):
    """
    Ejecuta validación cruzada para encontrar los hiperparámetros óptimos.
    """
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [2, 4, 6]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42), 
        param_grid=param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("Iniciando búsqueda exhaustiva de hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    print("\n--- Resultados de la Optimización ---")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Accuracy en Validación Cruzada: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_
