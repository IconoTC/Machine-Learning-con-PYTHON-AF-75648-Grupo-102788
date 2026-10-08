from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Guardar modelo
joblib.dump(model, "model.pkl")

print("Modelo entrenado y guardado como model.pkl")
