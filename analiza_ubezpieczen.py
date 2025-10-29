# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Wczytanie danych
try:
    dane = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku 'insurance.csv'.")
    print("Upewnij się, że plik znajduje się w tym samym folderze co skrypt.")
    exit() # Zakończ program, jeśli nie ma danych

# Spojrzenie na dane
print(dane.head())
print("\n") 

dane.info()
print("\n")


# Wizualizacja Danych (EDA)

sns.set_theme(style="whitegrid")

# Analiza rozkładu kosztów (zmienna 'charges')
plt.figure(figsize=(10, 6))
sns.histplot(dane['charges'], kde=True, bins=40)
plt.title('Rozkład kosztów ubezpieczenia (charges)', fontsize=16)
plt.xlabel('Koszty ($)', fontsize=12)
plt.ylabel('Liczba osób', fontsize=12)
plt.show() # Pokaż wykres

# Wpływ palenia na koszty
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=dane)
plt.title('Wpływ palenia na koszty ubezpieczenia', fontsize=16)
plt.xlabel('Palacz', fontsize=12)
plt.ylabel('Koszty ($)', fontsize=12)
plt.show()

# Wpływ wieku na koszty
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=dane, hue='smoker', palette={'yes': 'red', 'no': 'blue'}, alpha=0.7)
plt.title('Zależność kosztów od wieku (z podziałem na palaczy)', fontsize=16)
plt.xlabel('Wiek', fontsize=12)
plt.ylabel('Koszty ($)', fontsize=12)
plt.show()

# Macierz korelacji dla zmiennych numerycznych
# Korelacja pokazuje, jak silnie dwie zmienne są ze sobą liniowo powiązane.
# Wartości bliskie 1 lub -1 oznaczają silną korelację.
# Wartości bliskie 0 oznaczają brak korelacji liniowej.
dane_numeryczne = dane.select_dtypes(include=np.number)
macierz_korelacji = dane_numeryczne.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(macierz_korelacji, annot=True, cmap='viridis', fmt=".2f")
plt.title('Macierz korelacji zmiennych numerycznych', fontsize=16)
plt.show()

# Przygotowanie Danych
# "One-Hot Encoding".
# 'sex' -> 'sex_male' (1 jeśli mężczyzna, 0 jeśli kobieta)
# 'smoker' -> 'smoker_yes' (1 jeśli palacz, 0 jeśli nie)
# 'region' -> 4 nowe kolumny, po jednej dla każdego regionu

dane_przetworzone = pd.get_dummies(dane, columns=['sex', 'smoker', 'region'], drop_first=True)

print("\nDane po przetworzeniu (One-Hot Encoding):")
print(dane_przetworzone.head())
print("\n")

# Podział danych 
# X to wszystkie kolumny OPRÓCZ 'charges'
X = dane_przetworzone.drop('charges', axis=1)

# To, co chcemy przewidzieć
y = dane_przetworzone['charges']

# Dzielimy dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Tworzymy instancję modelu regresji liniowej
model_regresji = LinearRegression()

# Model uczy się
model_regresji.fit(X_train, y_train)

# Ocena modelu 
# Predykcja na danych testowych
y_pred = model_regresji.predict(X_test)

# R^2 (R-kwadrat) - jak dobrze model wyjaśnia zmienność danych. Bliżej 1 = lepiej.
r2_score = metrics.r2_score(y_test, y_pred)

# MAE (Mean Absolute Error) - średni błąd bezwzględny. Mówi, o ile średnio myli się nasz model.
mae = metrics.mean_absolute_error(y_test, y_pred)

# RMSE (Root Mean Squared Error) - pierwiastek błędu średniokwadratowego.
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("Metryki oceny modelu:")
print(f'Współczynnik R-kwadrat (R^2): {r2_score:.4f}')
print(f'Średni błąd bezwzględny (MAE): {mae:.2f} $')
print(f'Pierwiastek błędu średniokwadratowego (RMSE): {rmse:.2f} $')
print("\n")

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Porównanie wartości rzeczywistych i przewidywanych', fontsize=16)
plt.xlabel('Wartości rzeczywiste (y_test)', fontsize=12)
plt.ylabel('Wartości przewidziane (y_pred)', fontsize=12)
plt.show()
