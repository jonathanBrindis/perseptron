import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar datos
data = pd.read_csv("loan_data.csv")

# Normalización de variables independientes
indep_vars = ["Loan amount", "Monthly income"]
scaler = MinMaxScaler()  # Creamos el scaler como objeto aparte para reusarlo
data[indep_vars] = scaler.fit_transform(data[indep_vars])

# Definir variables independientes y dependientes
X = data[indep_vars]
y = data[["Repaid"]].astype(float)

# Agregar columna de sesgo
X.insert(0, "X0", 1.)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

class SimplePerceptron:
    def __init__(self, learn_rate=0.1):
        self.learn_rate = learn_rate
        self.weights = None

    def sigmoid_function(self, x: float) -> float:
        return 1. / (1 + np.exp(-x))

    def forward_pass(self, X: np.ndarray) -> float:
        weighted_sum = np.dot(X, self.weights)
        return self.sigmoid_function(weighted_sum)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 20):
        self.weights = np.random.uniform(-1, 1, X_train.shape[1])
        
        for epoch in range(n_epochs):
            for x, y in zip(X_train, y_train):
                y_predicted = self.forward_pass(x)
                error = y_predicted - y
                gradient = error * y_predicted * (1 - y_predicted) * x
                self.weights -= self.learn_rate * gradient
            print(f"Época {epoch+1}: Pesos actualizados - {self.weights}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.array([self.forward_pass(x) for x in X_test])

# Entrenar el perceptrón
perceptron = SimplePerceptron(learn_rate=0.05)
perceptron.fit(X_train, y_train, n_epochs=20)

# Evaluar el modelo
y_pred = perceptron.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)

print("\n=== EVALUACIÓN DEL MODELO ===")
print("Predicciones de prueba (probabilidades):", y_pred[:10])
print("Predicciones discretizadas:", y_pred_binary[:10])
print("Precisión del modelo:", accuracy)

# =============================================
# PREDICCIÓN PARA MÚLTIPLES NUEVOS SOLICITANTES
# =============================================

def predecir_solicitantes(nuevos_solicitantes):
    """
    Función para predecir múltiples nuevos solicitantes
    
    Args:
        nuevos_solicitantes: Lista de diccionarios con Loan amount y Monthly income
        Ejemplo: [{"Loan amount": 1000, "Monthly income": 3000},
                 {"Loan amount": 500, "Monthly income": 2000}]
    """
    # Convertir a DataFrame
    nuevos_df = pd.DataFrame(nuevos_solicitantes)
    
    # Normalizar los datos (usando el mismo scaler de entrenamiento)
    nuevos_df[indep_vars] = scaler.transform(nuevos_df[indep_vars])
    
    # Agregar columna de sesgo
    nuevos_df.insert(0, "X0", 1.)
    
    # Hacer predicciones
    probabilidades = perceptron.predict(nuevos_df.values)
    decisiones = (probabilidades >= 0.5).astype(int)
    
    # Mostrar resultados
    print("\n=== RESULTADOS DE APROBACIÓN ===")
    for i, (prob, dec) in enumerate(zip(probabilidades, decisiones)):
        print(f"\nSolicitante {i+1}:")
        print(f"- Monto del préstamo: ${nuevos_solicitantes[i]['Loan amount']:,}")
        print(f"- Ingreso mensual: ${nuevos_solicitantes[i]['Monthly income']:,}")
        print(f"- Probabilidad de pago: {prob:.2%}")
        print(f"- Decisión: {'APROBADO' if dec == 1 else 'RECHAZADO'}")
    
    return probabilidades, decisiones

# Ejemplo de uso con múltiples solicitantes
nuevos_solicitantes = [
    {"Loan amount": 1000, "Monthly income": 3000},
    {"Loan amount": 500, "Monthly income": 2000},
    {"Loan amount": 5000, "Monthly income": 1500},
    {"Loan amount": 2000, "Monthly income": 4000}
]

predecir_solicitantes(nuevos_solicitantes)

# Información adicional del modelo
print("\n=== INFORMACIÓN DEL MODELO ===")
print("Pesos finales:", perceptron.weights)
print("Rango de normalización para Loan amount:", 
      f"{scaler.data_min_[0]:.2f} - {scaler.data_max_[0]:.2f}")
print("Rango de normalización para Monthly income:", 
      f"{scaler.data_min_[1]:.2f} - {scaler.data_max_[1]:.2f}")