## Przewidywanie Kosztów Ubezpieczenia Medycznego


## 📝 Opis Projektu

Celem tego projektu jest zbudowanie modelu regresji liniowej, który przewiduje wysokość rocznych kosztów medycznych ponoszonych przez pacjentów. Model bazuje na zestawie cech opisujących pacjenta, takich jak wiek, płeć, wskaźnik BMI, liczba dzieci, status palacza oraz region zamieszkania.

---

## 📊 Zbiór Danych

Projekt wykorzystuje zbiór danych **"Medical Cost Personal Datasets"** dostępny na platformie Kaggle.

* **Źródło:** [Kaggle: Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)

Zbiór zawiera 1338 obserwacji i następujące kolumny:
* `age`: Wiek pacjenta.
* `sex`: Płeć pacjenta.
* `bmi`: Wskaźnik masy ciała (Body Mass Index).
* `children`: Liczba dzieci.
* `smoker`: Status palacza (tak/nie).
* `region`: Region zamieszkania w USA.
* `charges`: **(Zmienna docelowa)** Roczne koszty medyczne w dolarach.

---

## 🛠️ Użyte Technologie

* **Język:** Python 3.12
* **Biblioteki:**
    * **Pandas:** Do manipulacji i analizy danych.
    * **NumPy:** Do operacji numerycznych.
    * **Matplotlib & Seaborn:** Do wizualizacji danych i wyników.
    * **Scikit-learn:** Do budowy i oceny modelu uczenia maszynowego.

---

## 📈 Metodologia

Projekt został zrealizowany w następujących krokach:

1.  **Eksploracyjna Analiza Danych (EDA):**
    * Wczytanie i wstępne zapoznanie się z danymi.
    * Wizualizacja rozkładu zmiennej docelowej (`charges`).
    * Analiza korelacji pomiędzy zmiennymi numerycznymi.
    * Wizualizacja wpływu kluczowych cech (wiek, palenie, BMI) na koszty ubezpieczenia.

2.  **Przygotowanie Danych (Preprocessing):**
    * Konwersja zmiennych kategorycznych (np. `sex`, `smoker`) na format numeryczny za pomocą techniki **One-Hot Encoding**.

3.  **Budowa i Trening Modelu:**
    * Podział danych na zbiór treningowy (80%) i testowy (20%).
    * Zbudowanie i wytrenowanie modelu **Regresji Liniowej** na zbiorze treningowym.

4.  **Ocena Modelu:**
    * Wygenerowanie predykcji na niewidzianym wcześniej zbiorze testowym.
    * Ocena jakości modelu przy użyciu metryk:
        * **Współczynnik R-kwadrat ($R^2$)**
        * **Średni Błąd Bezwzględny (MAE)**
        * **Pierwiastek Błędu Średniokwadratowego (RMSE)**

---

## 🎯 Wyniki i Wnioski

Model regresji liniowej osiągnął satysfakcjonującą wydajność na zbiorze testowym:

* **Współczynnik R-kwadrat ($R^2$):** **~0.78** (oznacza, że model wyjaśnia około 78% zmienności w kosztach ubezpieczenia).
* **Średni Błąd Bezwzględny (MAE):** **~$4181** (średnia pomyłka modelu w przewidywaniu kosztów).

**Kluczowe wnioski z analizy:**
* **Palenie jest najważniejszym czynnikiem wpływającym na koszty ubezpieczenia.** Palacze płacą znacznie wyższe składki niż osoby niepalące, niezależnie od innych czynników.
* **Wiek ma silny, pozytywny wpływ na koszty** – im starsza osoba, tym statystycznie wyższe są jej koszty leczenia.
* Wskaźnik **BMI** również ma pozytywną korelację z kosztami, szczególnie u osób palących.

---

## 🚀 Jak Uruchomić Projekt

1.  Sklonuj repozytorium na swój lokalny komputer:
    ```bash
    git clone [LINK_DO_TWOJEGO_REPOZYTORIUM_NA_GITHUB]
    ```
2.  Przejdź do folderu projektu:
    ```bash
    cd Project-Regression
    ```
3.  Stwórz i aktywuj wirtualne środowisko:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # MacOS/Linux
    source venv/bin/activate
    ```
4.  Zainstaluj wymagane biblioteki:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
5.  Uruchom główny skrypt:
    ```bash
    python analysis_medical_cost.py
    ```
