import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
from sklearn.decomposition import PCA

# 1. Сбор данных через API Мосбиржи (MOEX ISS)
tickers = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'NVTK', 'PLZL', 'MGNT', 'MTSS', 'CHMF', 'TATN']
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365) # За последний год

print("Загрузка данных с Мосбиржи...")
df_list = []
for ticker in tickers:
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json?from={start_date}&till={end_date}"
    try:
        response = requests.get(url)
        data = response.json()
        columns = data['history']['columns']
        values = data['history']['data']
        if values:
            temp_df = pd.DataFrame(values, columns=columns)
            temp_df['TRADEDATE'] = pd.to_datetime(temp_df['TRADEDATE'])
            temp_df.set_index('TRADEDATE', inplace=True)
            series = temp_df['CLOSE'].rename(ticker)
            df_list.append(series)
    except Exception as e:
        print(f"Ошибка загрузки {ticker}")

# Объединяем и сохраняем в файл (для офлайн-запуска на защите)
df = pd.concat(df_list, axis=1).dropna()
df.to_csv("moex_quotes.csv")
print(f"Данные сохранены в 'moex_quotes.csv'. Размер: {df.shape}")

# 2. Предобработка: доходности и стандартизация
returns = df.pct_change().dropna()
# Стандартизация: вычитаем среднее, делим на ст. отклонение
Z = (returns - returns.mean()) / returns.std(ddof=1)
X_c = Z.values 

# 3. Ручной расчет PCA (без scikit-learn)
n, m = X_c.shape
# Матрица ковариаций
cov_matrix = (X_c.T @ X_c) / (n - 1)

# Собственные значения (eigenvalues) и собственные векторы (eigenvectors)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Сортировка по убыванию дисперсии
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Доля объясненной дисперсии
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

# Проекция на первые 2 главные компоненты (Scores)
W = eigenvectors[:, :2]
Y = X_c @ W 

# Факторные нагрузки для биплота (Loadings)
loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

# ==========================================
# 4. ВИЗУАЛИЗАЦИИ (Сохранение в файлы)
# ==========================================

# 4.1. График "Каменистой осыпи" (Scree Plot)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', color='b', label='Собственные значения')
plt.axhline(y=1, color='r', linestyle='--', label='Критерий Кайзера (lambda=1)')
plt.title('График "каменистой осыпи" (Scree Plot)')
plt.xlabel('Номер главной компоненты')
plt.ylabel('Собственное значение (Дисперсия)')
plt.xticks(range(1, len(eigenvalues)+1))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("1_scree_plot.png", dpi=300, bbox_inches='tight') # Сохраняем картинку
plt.close() # Закрываем фигуру, чтобы не висела в памяти

# 4.2. Проекция торговых дней (Scores)
plt.figure(figsize=(8, 5))
plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5, c='g', edgecolors='k')
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
plt.title('Проекция наблюдений (торговых дней) на плоскость PC1-PC2')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("2_pca_scores.png", dpi=300, bbox_inches='tight')
plt.close()

# 4.3. Визуализация Biplot
plt.figure(figsize=(10, 8))

# Масштабирование точек дней
scale_scores = np.max(np.abs(loadings)) / np.max(np.abs(Y))
plt.scatter(Y[:, 0] * scale_scores, Y[:, 1] * scale_scores, alpha=0.3, c='lightgrey', label='Торговые дни')

# Рисуем векторы нагрузок (акции)
colors = plt.cm.tab10(np.linspace(0, 1, m))
for i in range(m):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
              color=colors[i], alpha=0.8, head_width=0.03, head_length=0.03)
    # Подписи тикеров
    plt.text(loadings[i, 0]*1.05, loadings[i, 1]*1.05, df.columns[i], 
             color=colors[i], fontsize=12, fontweight='bold')

plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
plt.title('Biplot: Кластеризация российских акций (MOEX)')
plt.xlabel('PC1 (Нагрузки)')
plt.ylabel('PC2 (Нагрузки)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("3_biplot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Графики успешно сохранены в текущую папку: 1_scree_plot.png, 2_pca_scores.png, 3_biplot.png")

# ==========================================
# 5. СРАВНЕНИЕ С SCIKIT-LEARN (ВЕРИФИКАЦИЯ)
# ==========================================
pca_sk = PCA(n_components=2)
Y_sk = pca_sk.fit_transform(X_c)

print("\n--- СРАВНЕНИЕ РЕЗУЛЬТАТОВ (NumPy vs Scikit-Learn) ---")
print(f"Собственные значения (Custom) : {eigenvalues[:2]}")
print(f"Собственные значения (Sklearn): {pca_sk.explained_variance_}")

# Разница между модулями векторов (так как векторы могут иметь зеркальное отражение знаков)
diff_components = np.max(np.abs(np.abs(eigenvectors[:, :2].T) - np.abs(pca_sk.components_)))
print(f"Максимальная разница в модулях собственных векторов: {diff_components:.10e}")
if diff_components < 1e-10:
    print("ВЫВОД: Расчеты полностью совпадают с точностью до машинной погрешности.")