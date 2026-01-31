import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Rastgele veri üretimi için seed ayarlama
np.random.seed(42)

# Zaman serisi oluşturma
start_time = datetime.now()
time_points = [start_time + timedelta(minutes=i) for i in range(1000)]

# Hat akım değerleri (normal ve arıza durumları simüle edilmiş)
data = []

for i, timestamp in enumerate(time_points):
    # Normal durumda akım değerleri
    hat1_base = 216.5
    hat2_base = 218.0
    hat3_base = 221.0
    hat4_base = 221.0
    hat5_base = 215

    # Rastgele gürültü ekleme
    noise = np.random.normal(0, 5, 5)

    # Bazı noktalarda arıza simülasyonu
    if i % 100 == 0 and i > 0:  # Her 100 veri noktasında arıza simülasyonu
        # Hat 1 arızası (250A üzeri)
        hat1_current = hat1_base + noise[0] + np.random.uniform(35, 50)
        hat2_current = hat2_base + noise[1]
        hat3_current = hat3_base + noise[2]
        hat4_current = hat4_base + noise[3]
        hat5_current = hat5_base + noise[4]
        fault_status = 1
    elif i % 150 == 0 and i > 0:  # Her 150 veri noktasında diğer hatlar arızası
        hat1_current = hat1_base + noise[0]
        hat2_current = hat2_base + noise[1] + np.random.uniform(15, 25)  # 235A üzeri
        hat3_current = hat3_base + noise[2] + np.random.uniform(15, 25)
        hat4_current = hat4_base + noise[3] + np.random.uniform(15, 25)
        hat5_current = hat5_base + noise[4] + np.random.uniform(10, 20)
        fault_status = 1
    else:
        # Normal durum
        hat1_current = max(0, hat1_base + noise[0])
        hat2_current = max(0, hat2_base + noise[1])
        hat3_current = max(0, hat3_base + noise[2])
        hat4_current = max(0, hat4_base + noise[3])
        hat5_current = max(0, hat5_base + noise[4])
        fault_status = 0

    data.append({
        'timestamp': timestamp,
        'hat1_akim': round(hat1_current, 2),
        'hat2_akim': round(hat2_current, 2),
        'hat3_akim': round(hat3_current, 2),
        'hat4_akim': round(hat4_current, 2),
        'hat5_akim': round(hat5_current, 2),
        'ariza_durumu': fault_status
    })

# DataFrame oluşturma
df = pd.DataFrame(data)

# CSV dosyasına kaydetme
df.to_csv('hat_akimi_verileri.csv', index=False, encoding='utf-8')

print("CSV dosyası başarıyla oluşturuldu!")
print(f"Toplam veri noktası: {len(df)}")
print(f"Arıza sayısı: {df['ariza_durumu'].sum()}")
print("\nİlk 10 satır:")
print(df.head(10))