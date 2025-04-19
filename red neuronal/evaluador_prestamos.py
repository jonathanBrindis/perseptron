import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimplePerceptron:
    learn_rate = 0.1

    def __init__(self):
        self.weights = None

    def sigmoid_function(self, x: float) -> float:
        """Función sigmoide para convertir la salida en una probabilidad entre [0,1]"""
        return 1. / (1 + np.exp(-x))

    def forward_pass(self, X: np.ndarray) -> float:
        """Propagación hacia adelante para predecir la salida de un solo dato"""
        weighted_sum = np.dot(X, self.weights)
        return self.sigmoid_function(weighted_sum)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 1000):
        """Entrenamiento utilizando descenso de gradiente."""
        self.weights = np.random.uniform(-1, 1, X_train.shape[1])
        
        for epoch in range(n_epochs):
            for x, y in zip(X_train, y_train):
                y_predicted = self.forward_pass(x)
                error = y_predicted - y
                gradient = error * y_predicted * (1 - y_predicted) * x
                self.weights -= self.learn_rate * gradient

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predicción de datos de prueba"""
        return np.array([self.forward_pass(x) for x in X_test])

class LoanEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.scaler = None
        self.data = None
        self.accuracy = 0.0
        self.model_info_label = None
        
        self.setup_ui()
        self.load_and_train_model()
        
    def setup_ui(self):
        """Configura la interfaz gráfica"""
        self.root.title("Evaluador de Fiabilidad Crediticia")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        
        # Estilo
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10, 'bold'))
        style.configure('TEntry', font=('Arial', 10))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        
        # Encabezado
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=10, fill='x')
        
        ttk.Label(
            header_frame, 
            text="EVALUADOR DE PRÉSTAMOS", 
            style='Header.TLabel'
        ).pack()
        
        ttk.Label(
            header_frame, 
            text="Ingrese los datos del solicitante para evaluar su fiabilidad",
            style='TLabel'
        ).pack(pady=5)
        
        # Formulario de entrada
        form_frame = ttk.LabelFrame(self.root, text="Datos del Solicitante")
        form_frame.pack(pady=10, padx=20, fill='x')
        
        # Ingreso mensual
        ttk.Label(form_frame, text="Ingreso Mensual ($):").grid(
            row=0, column=0, padx=5, pady=5, sticky='w')
        self.income_entry = ttk.Entry(form_frame)
        self.income_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Monto del préstamo
        ttk.Label(form_frame, text="Monto del Préstamo ($):").grid(
            row=1, column=0, padx=5, pady=5, sticky='w')
        self.loan_entry = ttk.Entry(form_frame)
        self.loan_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Botón de evaluación
        ttk.Button(
            self.root, 
            text="EVALUAR SOLICITANTE", 
            command=self.evaluate,
            style='TButton'
        ).pack(pady=15)
        
        # Resultados
        self.result_frame = ttk.LabelFrame(self.root, text="Resultado de la Evaluación")
        self.result_frame.pack(pady=10, padx=20, fill='x')
        
        # Probabilidad
        prob_header = ttk.Label(self.result_frame, text="Probabilidad de Pago:")
        prob_header.pack(anchor='w', padx=10, pady=(10, 0))
        
        self.prob_value = ttk.Label(
            self.result_frame, 
            text="0.00%", 
            font=('Arial', 24, 'bold')
        )
        self.prob_value.pack(pady=(0, 10))
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            self.result_frame, 
            orient='horizontal', 
            length=400, 
            mode='determinate'
        )
        self.progress.pack(pady=5, padx=10, fill='x')
        
        # Decisión
        self.decision_label = ttk.Label(
            self.result_frame, 
            text="Decisión: -", 
            font=('Arial', 12)
        )
        self.decision_label.pack(pady=10)
        
        # Información del modelo
        self.model_info_label = ttk.Label(
            self.root, 
            text="Modelo Perceptrón | Precisión: calculando...",
            style='TLabel'
        )
        self.model_info_label.pack(side='bottom', pady=5)
        
    def load_and_train_model(self):
        """Carga los datos y entrena el modelo"""
        try:
            # Cargar datos (en una aplicación real, esto sería un archivo externo)
            try:
                self.data = pd.read_csv("loan_data.csv")
            except:
                self.create_sample_data()
                self.data = pd.read_csv("loan_data.csv")
            
            # Normalización de variables
            indep_vars = ["Loan amount", "Monthly income"]
            self.scaler = MinMaxScaler()
            self.data[indep_vars] = self.scaler.fit_transform(self.data[indep_vars])
            
            # Definir variables
            X = self.data[indep_vars]
            y = self.data[["Repaid"]].astype(int)  # Cambiado a int
            X.insert(0, "X0", 1.)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42)
            
            # Entrenar modelo
            self.model = SimplePerceptron()
            self.model.fit(X_train, y_train)
            
            # Calcular precisión
            y_pred = self.model.predict(X_test)
            y_pred = (y_pred >= 0.5).astype(int)
            y_test = y_test.flatten()  # Aplanar y_test para que coincida con y_pred
            
            self.accuracy = accuracy_score(y_test, y_pred)
            
            # Actualizar la etiqueta de información del modelo
            self.model_info_label.config(text=f"Modelo Perceptrón | Precisión: {self.accuracy:.2%}")
            
            # Mensaje de depuración (puedes eliminarlo después)
            print(f"Precisión calculada: {self.accuracy:.2%}")
            print(f"Ejemplo predicciones: {y_pred[:10]}")
            print(f"Ejemplo valores reales: {y_test[:10]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {str(e)}")
            self.root.destroy()
    
    def create_sample_data(self):
        """Crea datos de ejemplo si no existe el archivo"""
        import random
        data = {
            "Loan amount": [random.randint(1000, 10000) for _ in range(1000)],
            "Monthly income": [random.randint(800, 5000) for _ in range(1000)],
            "Repaid": [random.randint(0, 1) for _ in range(1000)]
        }
        df = pd.DataFrame(data)
        df.to_csv("loan_data.csv", index=False)
        print("Archivo loan_data.csv creado con datos de ejemplo")
    
    def evaluate(self):
        """Evalúa al solicitante basado en los datos ingresados"""
        try:
            # Obtener valores del formulario
            income = float(self.income_entry.get())
            loan = float(self.loan_entry.get())
            
            if income <= 0 or loan <= 0:
                messagebox.showerror("Error", "Los valores deben ser mayores a cero")
                return
                
            # Preparar datos
            nuevo_solicitante = pd.DataFrame({
                "Loan amount": [loan],
                "Monthly income": [income]
            })
            
            # Normalizar
            nuevo_solicitante[["Loan amount", "Monthly income"]] = self.scaler.transform(
                nuevo_solicitante[["Loan amount", "Monthly income"]])
            nuevo_solicitante.insert(0, "X0", 1.0)
            
            # Predecir
            probabilidad = self.model.predict(nuevo_solicitante.values)[0]
            decision = "FIABLE" if probabilidad >= 0.5 else "NO FIALBE"
            
            # Actualizar UI con resultados
            self.prob_value.config(text=f"{probabilidad:.2%}")
            self.progress['value'] = probabilidad * 100
            
            self.decision_label.config(text=f"Decisión: {decision}")
            if decision == "FIABLE":
                self.decision_label.config(foreground='green')
            else:
                self.decision_label.config(foreground='red')
                
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un problema: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LoanEvaluatorApp(root)
    root.mainloop()