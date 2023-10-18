# PRETRAITEMENT DES DONNEES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Chargement du fichier csv à l'aide de pandas
house_data = pd.read_csv('house.csv')

# suppression des colonnes qui nous intéressent pas
house_data = house_data.drop(["sqft_living","id", "date" ,"sqft_lot",   "zipcode", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15" ], axis = 1)

# Affichage des 20 premieres colonnes
print(house_data.head(20))

# Vérification des colonnes pour etre sur qu'il n' y a pas de valeurs nuls
print(house_data.isnull().sum())

# Visualisation des données


# Un nuage de points pour voir la relation entre le grade d'une maison  et le prix :

plt.figure(figsize=(8, 6))
sns.scatterplot(data=house_data, x='grade', y='price')
plt.xlabel('grade')
plt.ylabel('Prix')
plt.title('Relation entre le grade et le prix')
plt.show()

# Utilisation du boxplot pour voir les valeurs abérrantes

sns.boxplot(x='grade', y='price', data=house_data)
plt.xlabel('Grade')
plt.ylabel('Prix')
plt.title('Boxplot du prix en fonction du grade')
plt.show()

# Supprimer les valeurs aberrantes basées sur un seuil du z-score (par exemple, seuil de z-score de 3)
from scipy import stats

z_scores = stats.zscore(house_data[['price', 'bedrooms', 'bathrooms','view', 'grade', 'waterfront', "yr_built","yr_renovated", "lat", "long", "condition","floors"]])
house_data = house_data[(z_scores < 3).all(axis=1)]

# Division du dataset en un ensemble d'entraînement et un ensemble de test
from sklearn.model_selection import train_test_split


# Sélection des variables indépendantes (X) et (y)
X = house_data['grade']
y = house_data['price']

# Remodeler X pour le convertir en une matrice 2D
X = X.values.reshape(-1, 1)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Création d'un modèle de regression linéaire

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# prédiction sur un ensemble de test
y_prediction = model.predict(X_test)

# Évaluation des performances du modèle en calculant l'erreur quadratique moyenne (RMSE) et le coefficient de détermination (R2) :
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
r2 = r2_score(y_test, y_prediction)

print("RMSE:", rmse)
print("R2:", r2)

# Tracer la régression linéaire
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_prediction, color='red', linewidth=2, label='Régression linéaire')
plt.xlabel('grade')
plt.ylabel('Prix')
plt.legend()
plt.show()

# Application d'une regression linéaire multiple à notre dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sélectionner des variables indépendantes (X) et la variable dépendante (y)
X = house_data[['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'lat', 'long']]
y = house_data['price']

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Création d'un modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Prédiction des valeurs sur l'ensemble de test
y_prediction = model.predict(X_test)


# Évaluation des performances du modèle en calculant l'erreur quadratique moyenne (RMSE) et le coefficient de détermination (R2) :
rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
r2 = r2_score(y_test, y_prediction)

print("RMSE:", rmse)
print("R2:", r2)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sélection de  la variable indépendante (X) et la variable dépendante (y)
X = house_data[['grade']]
y = house_data['price']

# Remodelement de X pour le convertir en une matrice 2D
X = X.values.reshape(-1, 1)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Création un modèle de régression linéaire
model = LinearRegression()

# Entraînement le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Prédiction les valeurs sur l'ensemble de test
y_prediction = model.predict(X_test)

# Création d'un modèle de régression linéaire polynomiale
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Division des données polynomiales en ensemble d'entraînement et de test
X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

# Création et entraînement du modèle de régression linéaire polynomiale
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Prédiction des valeurs polynomiales sur l'ensemble de test
y_poly_pred = poly_model.predict(X_poly_test)

# Évaluer le modèle polynômial
rmse = np.sqrt(mean_squared_error(y_test, y_poly_pred))
r2 = r2_score(y_test, y_poly_pred)

print(f'RMSE (Polynomial): {rmse}')
print(f'R2 (Polynomial): {r2}')
