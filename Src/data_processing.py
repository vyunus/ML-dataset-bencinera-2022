import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_sample_data(filepath, sample_size=5000, random_state=42):
    """Carga el dataset limpio y extrae una muestra representativa."""
    df = pd.read_csv(filepath).dropna()
    return df.sample(n=sample_size, random_state=random_state)

def prepare_unsupervised_data(df):
    """
    Prepara la matriz para K-Means (Comportamiento de carga).
    Usa 'cantidad_comp_15' (escalada) y 'nombre_prod' (dummies).
    """
    X_continua = df[['cantidad_comp_15']]
    X_categorica = df[['nombre_prod']]
    
    # Dummies asegurando formato numérico
    X_dummies = pd.get_dummies(X_categorica, drop_first=True).astype(int)
    
    # Escalamiento solo para la continua
    scaler = StandardScaler()
    X_continua_scaled = scaler.fit_transform(X_continua)
    
    # Concatenación
    X_real_scaled = pd.concat([
        pd.DataFrame(X_continua_scaled, columns=['cantidad_scaled'], index=X_dummies.index),
        X_dummies
    ], axis=1)
    
    return X_real_scaled

def prepare_supervised_data(df):
    """
    Prepara la matriz para Random Forest (Clasificación de Industria).
    Aplica el Principio de Parsimonia: usa SOLO 'nombre_prod'.
    """
    X = df[['nombre_prod']]
    y = df['industria']
    
    X_encoded = pd.get_dummies(X, drop_first=True).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
