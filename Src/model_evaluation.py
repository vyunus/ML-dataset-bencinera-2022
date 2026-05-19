from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score

def evaluate_supervised_model(model, X_test, y_test):
    """Calcula y devuelve métricas clave para clasificación multiclase."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc_ovr': roc_auc_score(y_test, y_prob, multi_class='ovr')
    }
    
    print("--- Auditoría: Modelo Predictivo (Random Forest) ---")
    print(f"Accuracy (Exactitud): {metrics['accuracy']:.4f}")
    print(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC (OvR):        {metrics['roc_auc_ovr']:.4f}\n")
    
    return metrics

def evaluate_unsupervised_model(X_scaled, labels):
    """Calcula la densidad y separación de los clústeres."""
    sil_score = silhouette_score(X_scaled, labels)
    
    print("--- Auditoría: Clustering de Comportamiento (K-Means) ---")
    print(f"Silhouette Score: {sil_score:.4f}\n")
    
    return sil_score
