import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm

def calculate_confidence_interval(predictions):
    # Güven aralığı hesapla
    mean = np.mean(predictions)
    std_dev = np.std(predictions)
    z_value = norm.ppf(0.975)  # %95 güven aralığı için

    lower_bound = mean - z_value * std_dev
    upper_bound = mean + z_value * std_dev

    return lower_bound, upper_bound, std_dev

def predict_and_plot(model_chandler, model_joey, model_kararsiz, user_date):
    # Tarih öncesi 6 ay için rastgele 3 tarih seç
    random_dates = pd.to_datetime(np.random.choice(pd.date_range(end=user_date, periods=90), 3), format='%Y-%m-%d')

    # Tahminleri yap
    chandler_preds = []
    joey_preds = []
    kararsiz_preds = []

    for date in random_dates:
        user_data = pd.DataFrame({'Tarih': [date], 'Toplam': [np.nan], 'Kararsız': [np.nan]})
        user_data.set_index('Tarih', inplace=True)

        chandler_pred = model_chandler.predict(user_data.index.values.reshape(-1, 1))
        joey_pred = model_joey.predict(user_data.index.values.reshape(-1, 1))
        kararsiz_pred = model_kararsiz.predict(user_data.index.values.reshape(-1, 1))

        # Toplamın 100'ü geçmediğinden emin ol
        total_pred = chandler_pred + joey_pred + kararsiz_pred
        scale_factor = 100 / total_pred.sum()
        chandler_pred *= scale_factor
        joey_pred *= scale_factor
        kararsiz_pred *= scale_factor

        chandler_preds.append(chandler_pred[0])
        joey_preds.append(joey_pred[0])
        kararsiz_preds.append(kararsiz_pred[0])

    # Verilerin ortalamasını al
    chandler_pred_avg = np.mean(chandler_preds)
    joey_pred_avg = np.mean(joey_preds)
    kararsiz_pred_avg = np.mean(kararsiz_preds)

    # Güven aralığını hesapla
    chandler_lower, chandler_upper, chandler_std_dev = calculate_confidence_interval(chandler_preds)
    joey_lower, joey_upper, joey_std_dev = calculate_confidence_interval(joey_preds)
    kararsiz_lower, kararsiz_upper, kararsiz_std_dev = calculate_confidence_interval(kararsiz_preds)

    # Tarih öncesi 6 ay için çizgi grafiği oluştur
    six_months_ago = user_date - timedelta(days=30 * 6)
    date_range = pd.date_range(six_months_ago, user_date, freq='D')

    chandler_preds = model_chandler.predict(date_range.values.reshape(-1, 1))
    joey_preds = model_joey.predict(date_range.values.reshape(-1, 1))
    kararsiz_preds = model_kararsiz.predict(date_range.values.reshape(-1, 1))

    plt.plot(date_range, chandler_preds, label='Chandler')
    plt.plot(date_range, joey_preds, label='Joey')
    plt.plot(date_range, kararsiz_preds, label='Kararsız')

    plt.xlabel('Tarih')
    plt.ylabel('Oylar')
    plt.title('Chandler, Joey, ve Kararsız Oylar - Tarih Öncesi 6 Ay')
    plt.legend()
    plt.show()

    return (
        chandler_pred_avg, joey_pred_avg, kararsiz_pred_avg,
        chandler_lower, chandler_upper, chandler_std_dev,
        joey_lower, joey_upper, joey_std_dev,
        kararsiz_lower, kararsiz_upper, kararsiz_std_dev
    )


# Veriyi oku
df = pd.read_csv('veri_seti.csv')

# '–' karakterini NaN olarak değiştir
df.replace('–', np.nan, inplace=True)

# Eksik değerleri temizle
df = df.dropna()

# 'Tarih' sütununu oluştur
df['Tarih'] = pd.to_datetime(df[['Gün', 'Ay', 'Yıl']].astype(str).agg('-'.join, axis=1), format='%d-%m-%Y')

# Chandler ve Joey toplamlarını hesapla
df['Toplam'] = df['Chandler'] + df['Joey']

# Veri setini sadece tarih ve toplam oy sütunlarına indirge
df = df[['Tarih', 'Toplam', 'Kararsız']]

# Tarihi indeks olarak ayarla
df.set_index('Tarih', inplace=True)

while True:
    try:
        # Modelleme - RandomForestRegressor
        random_state_chandler = np.random.randint(1, 100)
        random_state_joey = np.random.randint(1, 100)
        random_state_kararsiz = np.random.randint(1, 100)

        model_chandler = RandomForestRegressor(n_estimators=100, random_state=random_state_chandler)
        model_joey = RandomForestRegressor(n_estimators=100, random_state=random_state_joey)
        model_kararsiz = RandomForestRegressor(n_estimators=100, random_state=random_state_kararsiz)

        # Modeli eğit
        model_chandler.fit(df.index.values.reshape(-1, 1), df['Toplam'])
        model_joey.fit(df.index.values.reshape(-1, 1), df['Toplam'])
        model_kararsiz.fit(df.index.values.reshape(-1, 1), df['Kararsız'])

        user_input = input("Tarih (GG-AA-YYYY formatında), 'repeat' yazarak tekrar et, veya 'exit' yazarak çıkabilirsiniz: ")

        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'repeat':
            continue

        user_date = datetime.strptime(user_input, "%d-%m-%Y")

        # Tahminleri yapmak için kullanıcı tarihinde bir veri noktası oluştur
        user_data = pd.DataFrame({'Tarih': [user_date], 'Toplam': [np.nan], 'Kararsız': [np.nan]})
        user_data.set_index('Tarih', inplace=True)

        # Tahminleri yap ve sonuçları göster
        (
            chandler_pred, joey_pred, kararsiz_pred,
            chandler_lower, chandler_upper, chandler_std_dev,
            joey_lower, joey_upper, joey_std_dev,
            kararsiz_lower, kararsiz_upper, kararsiz_std_dev
        ) = predict_and_plot(model_chandler, model_joey, model_kararsiz, user_date)

        # Sonuçları göster
        print("\nSonuçlar:")
        print(f"Tarih: {user_date}")
        print(f"Tahmini Chandler: {chandler_pred:.2f} Güven Aralığı: ({chandler_lower:.2f}, {chandler_upper:.2f}) Standart Sapma: {chandler_std_dev:.2f}")
        print(f"Tahmini Joey: {joey_pred:.2f} Güven Aralığı: ({joey_lower:.2f}, {joey_upper:.2f}) Standart Sapma: {joey_std_dev:.2f}")
        print(f"Tahmini Kararsız: {kararsiz_pred:.2f} Güven Aralığı: ({kararsiz_lower:.2f}, {kararsiz_upper:.2f}) Standart Sapma: {kararsiz_std_dev:.2f}")

    except ValueError:
        print("Geçersiz tarih formatı. Tekrar deneyin.")