import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos
data = pd.read_csv("loan_data.csv")

# Normalización de variables independientes
indep_vars = ["Loan amount", "Monthly income"]
data[indep_vars] = MinMaxScaler().fit_transform(data[indep_vars])

# Definir variables independientes y dependientes
X = data[indep_vars]
y = data[["Repaid"]].astype(float)

# Agregar columna de sesgo
X.insert(0, "X0", 1.)

# Dividir los datos en entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3)

class SimplePerceptron:
    learn_rate = 0.1

    def __init__(self):
        self.weights = None

    def sigmoid_function(self, x: float) -> float:
        """
        Función sigmoide para convertir la salida en una probabilidad entre [0,1]
        """
        return 1. / (1 + np.exp(-x))

    def forward_pass(self, X: np.ndarray) -> float:
        """
        Propagación hacia adelante para predecir la salida de un solo dato
        """
        weighted_sum = np.dot(X, self.weights)
        return self.sigmoid_function(weighted_sum)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 100):
        """
        Entrenamiento utilizando descenso de gradiente.
        """
        self.weights = np.random.uniform(-1, 1, X_train.shape[1])
        
        
        for epoch in range(n_epochs):
            for x, y in zip(X_train, y_train):
                y_predicted = self.forward_pass(x)
                error = y_predicted - y
                gradient = error * y_predicted * (1 - y_predicted) * x
                self.weights -= self.learn_rate * gradient
                
            print(f"Época {epoch+1}: Pesos actualizados - {self.weights}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicción de datos de prueba
        """
        return np.array([self.forward_pass(x) for x in X_test])

# Entrenar y probar el perceptrón
perceptron = SimplePerceptron()
#prueba=[16795,6373,1860,6306,1860,6306,1860,6306,1860,1860,6306,1860]
perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)
#y_pred = perceptron.predict(prueba)


# Mostrar primeras predicciones
print("Predicciones:", y_pred[:10])

# Convertir predicciones a valores binarios
y_pred = (y_pred >= 0.5).astype(int)
print("Predicciones discretizadas:", y_pred[:10])

# Calcular precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# ... (todo el código anterior permanece igual hasta accuracy_score)

# =============================================
# SECCIÓN NUEVA: PREDICCIÓN CON DATOS NUEVOS
# =============================================

# 1. Definimos los datos del nuevo solicitante (valores originales SIN normalizar)
nuevo_solicitante = {
    "Loan amount": [1000],  # Ejemplo: $2,500 dólares
    "Monthly income": [3000] # Ejemplo: $3,000 dólares mensuales
    
}
# 2. Creamos un DataFrame con estos datos
nuevos_datos = pd.DataFrame(nuevo_solicitante)

# 3. Normalizamos EXACTAMENTE igual que los datos de entrenamiento
# (Usamos el mismo scaler que ya se creó en data[indep_vars] = MinMaxScaler()...)
scaler = MinMaxScaler()
scaler.fit(data[indep_vars])  # Ajustamos con los datos originales
nuevos_datos[indep_vars] = scaler.transform(nuevos_datos[indep_vars])

# 4. Añadimos la columna de sesgo (X0 = 1.0)
nuevos_datos.insert(0, "X0", 1.0)

# 5. Hacemos la predicción
probabilidad = perceptron.predict(nuevos_datos.values)[0]
decision = "APROBAR" if probabilidad >= 0.5 else "RECHAZAR"

# 6. Mostramos resultados
print("\n=== PREDICCIÓN PARA NUEVO SOLICITANTE ===")
print(f"- Monto del préstamo: ${nuevo_solicitante['Loan amount'][0]:,}")
print(f"- Ingreso mensual: ${nuevo_solicitante['Monthly income'][0]:,}")
print(f"\nProbabilidad de pago: {probabilidad:.2%}")
print(f"Decisión: {decision}")

# =============================================
# (el resto del código original continúa)
# =============================================

# Mostrar primeras predicciones del test (original)
print("\nPredicciones:", y_pred[:10])
# ... (resto del código original)


"""
print("valor de x",X_test)

print("valor de Y",y_test)

print("valor de Y_pred",y_pred)"""