# === KROK 1: IMPORT BIBLIOTEK I WCZYTANIE DANYCH ===

# Importujemy potrzebne biblioteki
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawiamy styl wykresów, aby wyglądały ładniej
sns.set_theme(style="whitegrid")

# 1. Wczytanie danych
# Używamy pandas (pd) do wczytania naszego pliku CSV.
# Zakładamy, że plik 'insurance.csv' jest w tym samym folderze co skrypt.
try:
    dane = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku 'insurance.csv'.")
    print("Upewnij się, że plik znajduje się w tym samym folderze co skrypt.")
    exit() # Zakończ program, jeśli nie ma danych

# 2. Pierwsze spojrzenie na dane
print("--- Pierwsze 5 wierszy danych (head): ---")
print(dane.head())
print("\n") # Dodajemy pustą linię dla czytelności

# 3. Informacje o zbiorze danych
# Sprawdzamy typy danych (np. czy 'age' to liczba)
# oraz czy są jakieś brakujące wartości (non-null)
print("--- Informacje o danych (info): ---")
dane.info()
print("\n")

# 4. Statystyczne podsumowanie danych numerycznych
# Pokazuje średnią, medianę (50%), min, max dla kolumn liczbowych
print("--- Statystyki opisowe (describe): ---")
print(dane.describe())
print("\n")

print(">>> Krok 1 zakończony: Dane zostały wczytane i wstępnie przeanalizowane.")

# === KONIEC KROKU 1 ===

# === KROK 2: WIZUALIZACJA I PRZYGOTOWANIE DANYCH (PREPROCESSING) ===

# --- Część A: Wizualizacja Danych (EDA) ---
print(">>> Rozpoczynam Krok 2A: Wizualizacja Danych...")

# 1. Analiza rozkładu kosztów (naszej zmiennej docelowej 'charges')
plt.figure(figsize=(10, 6))
sns.histplot(dane['charges'], kde=True, bins=40)
plt.title('Rozkład kosztów ubezpieczenia (charges)', fontsize=16)
plt.xlabel('Koszty ($)', fontsize=12)
plt.ylabel('Liczba osób', fontsize=12)
plt.show() # Pokaż wykres

# Wniosek: Rozkład jest prawostronnie skośny. Większość ludzi ma niskie koszty,
# ale jest grupa osób z bardzo wysokimi kosztami.

# 2. Wpływ palenia na koszty
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=dane)
plt.title('Wpływ palenia na koszty ubezpieczenia', fontsize=16)
plt.xlabel('Palacz', fontsize=12)
plt.ylabel('Koszty ($)', fontsize=12)
plt.show()

# Wniosek: To jest kluczowy wniosek! Palacze mają ZNACZNIE wyższe koszty.

# 3. Wpływ wieku na koszty
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=dane, hue='smoker', palette={'yes': 'red', 'no': 'blue'}, alpha=0.7)
plt.title('Zależność kosztów od wieku (z podziałem na palaczy)', fontsize=16)
plt.xlabel('Wiek', fontsize=12)
plt.ylabel('Koszty ($)', fontsize=12)
plt.show()

# Wniosek: Widać wyraźny trend - im starsza osoba, tym wyższe koszty.
# Widać też trzy "chmury" punktów - najniższa dla niepalących, dwie wyższe dla palących.

# 4. Macierz korelacji dla zmiennych numerycznych
# Korelacja pokazuje, jak silnie dwie zmienne są ze sobą liniowo powiązane.
# Wartości bliskie 1 lub -1 oznaczają silną korelację.
# Wartości bliskie 0 oznaczają brak korelacji liniowej.
dane_numeryczne = dane.select_dtypes(include=np.number)
macierz_korelacji = dane_numeryczne.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(macierz_korelacji, annot=True, cmap='viridis', fmt=".2f")
plt.title('Macierz korelacji zmiennych numerycznych', fontsize=16)
plt.show()

# Wniosek: 'age' i 'bmi' mają najwyższą (choć umiarkowaną) pozytywną korelację z 'charges'.

print(">>> Krok 2A zakończony. Wykresy zostały wygenerowane.")

# --- Część B: Przygotowanie Danych (Preprocessing) ---
print(">>> Rozpoczynam Krok 2B: Przygotowanie danych...")

# Nasz model nie rozumie tekstu ('male', 'female', 'yes', 'no').
# Musimy zamienić kolumny kategoryczne na liczby.
# Użyjemy do tego metody "One-Hot Encoding".
# 'sex' -> 'sex_male' (1 jeśli mężczyzna, 0 jeśli kobieta)
# 'smoker' -> 'smoker_yes' (1 jeśli palacz, 0 jeśli nie)
# 'region' -> 4 nowe kolumny, po jednej dla każdego regionu

dane_przetworzone = pd.get_dummies(dane, columns=['sex', 'smoker', 'region'], drop_first=True)

# Wyjaśnienie 'drop_first=True':
# Unikamy tzw. współliniowości. Jeśli mamy kolumnę 'sex_male',
# to nie potrzebujemy już 'sex_female'. Jeśli 'sex_male' = 0, to wiemy,
# że to kobieta. To samo dla palaczy i regionów.

print("\n--- Dane po przetworzeniu (One-Hot Encoding): ---")
print(dane_przetworzone.head())
print("\n")

print(">>> Krok 2 zakończony: Dane zostały zwizualizowane i przygotowane do modelowania.")

# === KONIEC KROKU 2 ===

# === KROK 3: BUDOWA, TRENING I OCENA MODELU REGRESJI LINIOWEJ ===
print(">>> Rozpoczynam Krok 3: Budowa, trening i ocena modelu...")

# Importujemy potrzebne narzędzia z biblioteki scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np # Upewnijmy się, że numpy jest zaimportowany

# --- Część A: Podział danych ---

# 1. Definiujemy zmienne wejściowe (X) i zmienną docelową (y)
# X to wszystkie kolumny OPRÓCZ 'charges' - to są nasze cechy (features)
X = dane_przetworzone.drop('charges', axis=1)

# y to JEDYNIE kolumna 'charges' - to jest to, co chcemy przewidzieć
y = dane_przetworzone['charges']

# 2. Dzielimy dane na zbiór treningowy i testowy
# Zbiór treningowy (X_train, y_train) posłuży do "nauczenia" modelu.
# Zbiór testowy (X_test, y_test) posłuży do sprawdzenia, jak dobrze model sobie radzi na "nowych" danych.
# test_size=0.2 oznacza, że 20% danych trafi do zbioru testowego.
# random_state=42 zapewnia, że podział będzie zawsze taki sam, co gwarantuje powtarzalność wyników.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} wierszy")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]} wierszy")
print("\n")

# --- Część B: Tworzenie i trening modelu ---

# 1. Tworzymy instancję modelu regresji liniowej
model_regresji = LinearRegression()

# 2. Trenujemy (uczmy) model na danych treningowych
# Model "uczy się" zależności między cechami (X_train) a kosztami (y_train)
model_regresji.fit(X_train, y_train)

print(">>> Model został pomyślnie wytrenowany!")
print("\n")

# --- Część C: Ocena modelu ---

# 1. Robimy predykcje na danych testowych (tych, których model nie widział)
y_pred = model_regresji.predict(X_test)

# 2. Oceniamy jakość modelu za pomocą metryk
# Porównujemy prawdziwe wartości (y_test) z tym, co przewidział model (y_pred)

# R^2 (R-kwadrat) - jak dobrze model wyjaśnia zmienność danych. Bliżej 1 = lepiej.
# Wartość 0.75 oznacza, że nasz model wyjaśnia 75% zmienności w kosztach ubezpieczenia.
r2_score = metrics.r2_score(y_test, y_pred)

# MAE (Mean Absolute Error) - średni błąd bezwzględny. Mówi, o ile średnio myli się nasz model.
mae = metrics.mean_absolute_error(y_test, y_pred)

# RMSE (Root Mean Squared Error) - pierwiastek błędu średniokwadratowego. Podobne do MAE,
# ale mocniej "karze" za duże błędy.
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("--- Metryki oceny modelu: ---")
print(f'Współczynnik R-kwadrat (R^2): {r2_score:.4f}')
print(f'Średni błąd bezwzględny (MAE): {mae:.2f} $')
print(f'Pierwiastek błędu średniokwadratowego (RMSE): {rmse:.2f} $')
print("\n")

# 3. Wizualizacja wyników
# Wykres punktowy pokazujący prawdziwe wartości vs przewidywane wartości
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2) # Linia idealnej predykcji
plt.title('Porównanie wartości rzeczywistych i przewidywanych', fontsize=16)
plt.xlabel('Wartości rzeczywiste (y_test)', fontsize=12)
plt.ylabel('Wartości przewidziane (y_pred)', fontsize=12)
plt.show()

print(">>> Krok 3 zakończony: Model został zbudowany, wytrenowany i oceniony.")

# === KONIEC KROKU 3 ===