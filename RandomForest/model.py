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
X = df[["MatematicaI", "MatematicaII", "ComunicacionI", "ComunicacionII", 
        "InglesI", "InglesII", "DesarrolloPersonal", "Historia", 
        "ArteyCultura", "Quimica", "Biologia", "Fisica", "Emprendimiento", 
        "KuderA1", "KuderA2", "KuderA3", "KuderA4", "KuderA5", "KuderA6", 
        "KuderB1", "KuderB2", "KuderB3", "KuderB4", "KuderB5", "KuderB6", 
        "KuderC1", "KuderC2", "KuderC3", "KuderC4", "KuderC5", "KuderC6", 
        "KuderD1", "KuderD2", "KuderD3", "KuderD4", "KuderD5", "KuderD6", 
        "KuderE1", "KuderE2", "KuderE3", "KuderE4", "KuderE5", "KuderE6", 
        "KuderF1", "KuderF2", "KuderF3", "KuderF4", "KuderF5", "KuderF6", 
        "KuderG1", "KuderG2", "KuderG3", "KuderG4", "KuderG5", "KuderG6", 
        "KuderH1", "KuderH2", "KuderH3", "KuderH4", "KuderH5", "KuderH6", 
        "KuderI1", "KuderI2", "KuderI3", "KuderI4", "KuderI5", "KuderI6", 
        "KuderJ1", "KuderJ2", "KuderJ3", "KuderJ4", "KuderJ5", "KuderJ6", 
        "CapsE1MR", "CapsE2SR", "CapsE3VR", "CapsE4NA", "CapsE5LU", "CapsE6WK", "CapsE7PSA", 
        "CapsP1MR", "CapsP2SR", "CapsP3VR", "CapsP4NA", "CapsP5LU", "CapsP6WK", "CapsP7PSA"]]

#Dependent variable
y = df["Career"]

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
