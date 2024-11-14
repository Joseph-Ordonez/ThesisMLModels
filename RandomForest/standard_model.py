# =======================================================================================
# Este es el modelo estandar de Random Forest, es decir, sin ningún ajuste para mejorar
# su rendimiento, eficiencia, precisión, etc. He usado este como punto de partida para 
# aplicar los ajustes que se ven en el otro modelo.
#
# NOTA: Este NO es el modelo que se utiliza en la app de flask, tampoco se debe eliminar
# para poder hacer consultas posteriores.
# =======================================================================================
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# CSV file with separator
df = pd.read_csv("..\\dataset.csv", sep=';')

print(df.head())

#Independent variable
X = df.drop(columns=['CARRERA'])

#Dependent variable
y = df["CARRERA"]

#Split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.1,
    random_state= 50,
)

#Feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Model
classifier = RandomForestClassifier()

#Fit the model
classifier.fit(X_train, y_train)

# #=================================
# #check the accuracy
# #=================================
# # Make predictions
# y_pred = classifier.predict(X_test)
# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# # print accuracy
# print(f"Precision del modelo: {accuracy:.2f}")
# #=================================

#Pickle file
pickle.dump(classifier, open("model.pkl", "wb"))
