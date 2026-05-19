from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

def train_kmeans(X_scaled, n_clusters=4, random_state=42):
    """Entrena el modelo de clustering y devuelve el modelo y sus etiquetas."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def train_random_forest_base(X_train, y_train, random_state=42, n_estimators=100):
    """Entrena el modelo base de Random Forest para clasificación."""
    clf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    return clf
