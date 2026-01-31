



import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


class BiLSTMArızaİzlemeSistemi:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 BiLSTM Elektrik Hattı Arıza İzleme Sistemi")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1a1a1a')

        # Model ve veri değişkenleri
        self.df = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        self.monitoring = False
        self.current_index = 0
        self.prediction_history = []
        self.actual_history = []
        self.son_veriler = None

        # Hat sınır değerleri
        self.hat1_limit = 250
        self.other_hats_limit = 230

        # Model durumu
        self.model_trained = False
        self.model_accuracy = 0.0

        # GUI bileşenlerini oluştur
        self.create_widgets()
        self.setup_styles()
        # Telegram bot (optional) - initialize after GUI is ready
        self.telegram_bot = None

    def setup_styles(self):
        """GUI stil ayarları"""
        style = ttk.Style()
        style.theme_use('clam')

        # Özel renkler
        style.configure('Title.TLabel', font=('Arial', 5, 'bold'), background='#1a1a1a', foreground='#00ff41')
        style.configure('Status.TLabel', font=('Arial', 5), background='#1a1a1a', foreground='#ffffff')
        style.configure('Normal.TLabel', font=('Arial', 5), background='#00ff41', foreground='#000000')
        style.configure('Fault.TLabel', font=('Arial', 5), background='#ff3333', foreground='#ffffff')
        style.configure('AI.TLabel', font=('Arial', 5), background='#9966cc', foreground='#ffffff')

    def _init_telegram_if_available(self):
        """If TELEGRAM_BOT_TOKEN is set, start Telegram bot in background."""
        try:
            token = os.environ.get('TELEGRAM_BOT_TOKEN')
            if not token:
                self.log_message("ℹ️ TELEGRAM_BOT_TOKEN environment variable not set")
                return
            
            self.log_message(f"🔑 Bot token found: {token[:10]}...")
            
            try:
                from telegram_bot import TelegramBotManager  # Lazy import
                self.log_message("✅ Telegram module imported successfully")
            except Exception as e:
                self.log_message(f"❌ Telegram modülü yüklenemedi: {str(e)}")
                return
            
            self.telegram_bot = TelegramBotManager(token=token, app=self)
            self.telegram_bot.start()
            self.log_message("🤖 Telegram bot başlatıldı ve dinlemede")
            
        except Exception as e:
            self.log_message(f"❌ Telegram bot başlatılamadı: {str(e)}")
            import traceback
            self.log_message(f"🔍 Hata detayı: {traceback.format_exc()}")

    def create_widgets(self):
        """GUI bileşenlerini oluşturma"""
        # Ana başlık
        title_frame = tk.Frame(self.root, bg='#1a1a1a', height=80)
        title_frame.pack(fill='x', padx=10, pady=5)

        title_label = ttk.Label(title_frame, text="🤖 BiLSTM ELEKTRİK HATTI ARIZA İZLEME SİSTEMİ 🤖",
                                style='Title.TLabel')
        title_label.pack(pady=20)

        # Ana konteyner
        main_container = tk.PanedWindow(self.root, orient='horizontal', bg='#1a1a1a',
                                        sashwidth=5, sashrelief='raised')
        main_container.pack(fill='both', expand=True, padx=10, pady=5)

        # Sol panel - Kontroller ve Model Bilgileri
        left_panel = tk.Frame(main_container, bg='#2d2d2d')
        main_container.add(left_panel, minsize=400)

        # Sağ panel - Grafikler
        right_panel = tk.Frame(main_container, bg='#2d2d2d')
        main_container.add(right_panel, minsize=800)

        self.create_control_panel(left_panel)
        self.create_graph_panel(right_panel)

    def create_control_panel(self, parent):
        """Kontrol paneli oluşturma"""
        # Model Durumu
        model_frame = tk.LabelFrame(parent, text="🧠 BiLSTM Model Durumu", bg='#2d2d2d',
                                    fg='#00ff41', font=('Arial', 6, 'bold'))
        model_frame.pack(fill='x', padx=10, pady=5)

        self.model_status = tk.Label(model_frame, text="❌ Model Eğitilmedi",
                                     bg='#ff3333', fg='white', font=('Arial', 6, 'bold'))
        self.model_status.pack(pady=5, fill='x')

        self.accuracy_label = tk.Label(model_frame, text="Doğruluk: -",
                                       bg='#2d2d2d', fg='#ffffff', font=('Arial', 6))
        self.accuracy_label.pack(pady=2)

        self.training_progress = ttk.Progressbar(model_frame, length=200, mode='indeterminate')
        self.training_progress.pack(pady=5, fill='x')

        # Dosya İşlemleri
        file_frame = tk.LabelFrame(parent, text="📁 Veri İşlemleri", bg='#2d2d2d',
                                   fg='#00ff41', font=('Arial', 11, 'bold'))
        file_frame.pack(fill='x', padx=10, pady=5)

        tk.Button(file_frame, text="📂 CSV Dosyası Yükle", command=self.load_csv,
                  bg='#0066cc', fg='white', font=('Arial', 9, 'bold')).pack(pady=3, fill='x')

        tk.Button(file_frame, text="🎲 Örnek Veri Oluştur", command=self.create_sample_data,
                  bg='#9966cc', fg='white', font=('Arial', 9, 'bold')).pack(pady=3, fill='x')

        tk.Button(file_frame, text="🤖 Model Eğit", command=self.train_model,
                  bg='#cc6600', fg='white', font=('Arial', 9, 'bold')).pack(pady=3, fill='x')

        # Anlık Hat Durumu
        status_frame = tk.LabelFrame(parent, text="📊 Anlık Hat Durumu", bg='#2d2d2d',
                                     fg='#00ff41', font=('Arial', 11, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=5)

        # Hat değerleri için etiketler
        self.hat_labels = {}
        self.status_labels = {}
        self.rule_labels = {}
        self.ai_labels = {}

        hat_names = ['Hat 1', 'Hat 2', 'Hat 3', 'Hat 4', 'Hat 5']
        normal_values = [216.5, 218.0, 221.0, 221.0, 215.0]
        limits = [self.hat1_limit, self.other_hats_limit, self.other_hats_limit,
                  self.other_hats_limit, self.other_hats_limit]

        for i, (name, normal, limit) in enumerate(zip(hat_names, normal_values, limits)):
            # Ana hat frame
            hat_frame = tk.Frame(status_frame, bg='#2d2d2d', relief='ridge', bd=1)
            hat_frame.pack(fill='x', pady=2, padx=2)

            # Hat başlığı ve değer
            header_frame = tk.Frame(hat_frame, bg='#2d2d2d')
            header_frame.pack(fill='x')

            tk.Label(header_frame, text=f"{name}:", bg='#2d2d2d', fg='#ffffff',
                     font=('Arial', 10, 'bold'), width=8).pack(side='left')

            self.hat_labels[i] = tk.Label(header_frame, text="0.0 A", bg='#2d2d2d', fg='#00ff41',
                                          font=('Arial', 10, 'bold'), width=10)
            self.hat_labels[i].pack(side='left')

            tk.Label(header_frame, text=f"(Normal: {normal}A, Limit: {limit}A)",
                     bg='#2d2d2d', fg='#888888', font=('Arial', 8)).pack(side='right')

            # Durum etiketleri
            status_frame_inner = tk.Frame(hat_frame, bg='#2d2d2d')
            status_frame_inner.pack(fill='x', pady=2)

            tk.Label(status_frame_inner, text="Kural:", bg='#2d2d2d', fg='#ffffff',
                     font=('Arial', 8), width=6).pack(side='left')

            self.rule_labels[i] = tk.Label(status_frame_inner, text="NORMAL", bg='#00ff41', fg='#000000',
                                           font=('Arial', 8, 'bold'), width=8)
            self.rule_labels[i].pack(side='left', padx=(0, 5))

            tk.Label(status_frame_inner, text="AI:", bg='#2d2d2d', fg='#ffffff',
                     font=('Arial', 8), width=3).pack(side='left')

            self.ai_labels[i] = tk.Label(status_frame_inner, text="NORMAL", bg='#9966cc', fg='#ffffff',
                                         font=('Arial', 8, 'bold'), width=8)
            self.ai_labels[i].pack(side='left')

        # Sistem Durumu
        system_frame = tk.LabelFrame(parent, text="⚡ Sistem Durumu", bg='#2d2d2d',
                                     fg='#00ff41', font=('Arial', 11, 'bold'))
        system_frame.pack(fill='x', padx=10, pady=5)

        self.system_status = tk.Label(system_frame, text="✅ Sistem Hazır",
                                      bg='#00ff41', fg='#000000', font=('Arial', 11, 'bold'))
        self.system_status.pack(pady=5, fill='x')

        self.data_status = tk.Label(system_frame, text="📄 Veri: Yüklenmedi",
                                    bg='#2d2d2d', fg='#ffffff', font=('Arial', 9))
        self.data_status.pack(pady=2)

        # AI Tahmin Bilgileri
        ai_frame = tk.Frame(system_frame, bg='#2d2d2d')
        ai_frame.pack(fill='x', pady=5)

        tk.Label(ai_frame, text="AI Arıza Olasılığı:", bg='#2d2d2d', fg='#ffffff',
                 font=('Arial', 9)).pack(side='left')

        self.ai_probability = tk.Label(ai_frame, text="0.00%", bg='#2d2d2d', fg='#9966cc',
                                       font=('Arial', 9, 'bold'))
        self.ai_probability.pack(side='right')

        # Kontrol Butonları
        control_frame = tk.LabelFrame(parent, text="🎮 Kontroller", bg='#2d2d2d',
                                      fg='#00ff41', font=('Arial', 11, 'bold'))
        control_frame.pack(fill='x', padx=10, pady=5)

        self.start_button = tk.Button(control_frame, text="▶️ İzlemeyi Başlat",
                                      command=self.start_monitoring,
                                      bg='#00ff41', fg='#000000', font=('Arial', 9, 'bold'))
        self.start_button.pack(pady=3, fill='x')

        self.stop_button = tk.Button(control_frame, text="⏹️ İzlemeyi Durdur",
                                     command=self.stop_monitoring,
                                     bg='#ff3333', fg='white', font=('Arial', 9, 'bold'),
                                     state='disabled')
        self.stop_button.pack(pady=3, fill='x')

        tk.Button(control_frame, text="📈 Grafikleri Güncelle",
                  command=self.update_graphs,
                  bg='#ffaa00', fg='#000000', font=('Arial', 9, 'bold')).pack(pady=3, fill='x')

        # Hız kontrolü
        speed_frame = tk.Frame(control_frame, bg='#2d2d2d')
        speed_frame.pack(fill='x', pady=5)

        tk.Label(speed_frame, text="Hız:", bg='#2d2d2d', fg='#ffffff',
                 font=('Arial', 9)).pack(side='left')

        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(speed_frame, from_=0.1, to=5.0, resolution=0.1,
                               orient='horizontal', variable=self.speed_var,
                               bg='#2d2d2d', fg='#ffffff', highlightthickness=0)
        speed_scale.pack(side='right', fill='x', expand=True)

        # Arıza Geçmişi
        log_frame = tk.LabelFrame(parent, text="📋 Arıza & AI Analiz Geçmişi", bg='#2d2d2d',
                                  fg='#00ff41', font=('Arial', 11, 'bold'))
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        log_container = tk.Frame(log_frame, bg='#2d2d2d')
        log_container.pack(fill='both', expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_container, bg='#1a1a1a', fg='#00ff41',
                                font=('Consolas', 8), height=12)
        log_scroll = tk.Scrollbar(log_container, orient='vertical',
                                  command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.log_text.pack(side='left', fill='both', expand=True)
        log_scroll.pack(side='right', fill='y')

    def create_graph_panel(self, parent):
        """Grafik paneli oluşturma"""
        # Matplotlib figure
        self.fig = Figure(figsize=(14, 10), facecolor='#2d2d2d')

        # 4 subplot
        self.ax1 = self.fig.add_subplot(2, 2, 1, facecolor='#1a1a1a')  # Hat akımları
        self.ax2 = self.fig.add_subplot(2, 2, 2, facecolor='#1a1a1a')  # Arıza durumu
        self.ax3 = self.fig.add_subplot(2, 2, 3, facecolor='#1a1a1a')  # AI tahmin karşılaştırması
        self.ax4 = self.fig.add_subplot(2, 2, 4, facecolor='#1a1a1a')  # Model performans metrikleri

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        self.setup_initial_graphs()

    def setup_initial_graphs(self):
        """İlk grafik ayarları"""
        # Hat akımları
        self.ax1.set_title('Hat Akımları (A)', color='#00ff41', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Akım (A)', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.3, color='gray')

        # Arıza durumu
        self.ax2.set_title('Kural vs AI Arıza Tespiti', color='#00ff41', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Arıza Durumu', color='white')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.3, color='gray')

        # AI tahmin
        self.ax3.set_title('AI Arıza Olasılığı', color='#00ff41', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Olasılık', color='white')
        self.ax3.tick_params(colors='white')
        self.ax3.grid(True, alpha=0.3, color='gray')

        # Model performansı
        self.ax4.set_title('Model Performans Metrikleri', color='#00ff41', fontsize=12, fontweight='bold')
        self.ax4.set_ylabel('Değer', color='white')
        self.ax4.tick_params(colors='white')
        self.ax4.grid(True, alpha=0.3, color='gray')

        self.fig.tight_layout()
        self.canvas.draw()

    def create_sample_data(self):
        """Gerçekçi kademeli arıza patternleri ile veri oluşturma"""
        try:
            self.log_message("🎲 Gerçekçi kademeli arıza veri oluşturuluyor...")

            np.random.seed(42)
            n_samples = 1000

            data = []
            normal_values = [216.5, 218.0, 221.0, 221.0, 215.0]

            # Hat durumları ve arıza seviyeleri
            hat_fault_states = [0, 0, 0, 0, 0]  # 0=normal, 1=yükseliyor, 2=arıza, 3=slalom, 4=düşüyor
            fault_levels = [0.0, 0.0, 0.0, 0.0, 0.0]  # Kademeli arıza seviyeleri
            fault_timers = [0, 0, 0, 0, 0]  # Arıza süresi sayacı
            slalom_phase = [0.0, 0.0, 0.0, 0.0, 0.0]  # Slalom fazı

            for i in range(n_samples):
                hat_values = []
                overall_fault = 0

                for j, base_val in enumerate(normal_values):
                    # Normal gürültü
                    noise = np.random.normal(0, 2)
                    current_value = base_val + noise

                    # Hat 5 için daha yüksek arıza eğilimi
                    fault_probability = 0.015 if j == 4 else 0.008  # Hat 5: %1.5, diğerleri: %0.8

                    # Arıza başlatma kontrolü
                    if hat_fault_states[j] == 0 and np.random.random() < fault_probability:
                        hat_fault_states[j] = 1  # Yükselme başlat
                        fault_timers[j] = 0
                        self.log_message(f"📈 Hat {j + 1} kademeli arıza başlangıcı (veri noktası: {i})")

                    # Hat durum makinesi
                    if hat_fault_states[j] == 1:  # Yükseliyor
                        fault_timers[j] += 1
                        # 10-20 adımda kademeli yükselme
                        increase_duration = np.random.randint(10, 21) if fault_timers[j] == 1 else fault_timers[j]
                        progress = min(1.0, fault_timers[j] / increase_duration)

                        if j == 0:  # Hat 1
                            max_increase = np.random.uniform(35, 55)
                        else:  # Diğer hatlar
                            max_increase = np.random.uniform(18, 35)

                        fault_levels[j] = progress * max_increase

                        # Yükselme tamamlandı mı?
                        if progress >= 1.0:
                            hat_fault_states[j] = 2  # Arıza seviyesine ulaştı
                            fault_timers[j] = 0

                    elif hat_fault_states[j] == 2:  # Arıza seviyesinde
                        fault_timers[j] += 1
                        # 15-30 adım arıza seviyesinde bekle
                        stay_duration = np.random.randint(15, 31) if fault_timers[j] == 1 else fault_timers[j]

                        if fault_timers[j] >= stay_duration:
                            hat_fault_states[j] = 3  # Slalom fazına geç
                            fault_timers[j] = 0
                            slalom_phase[j] = 0.0

                    elif hat_fault_states[j] == 3:  # Slalom (±5A)
                        fault_timers[j] += 1
                        slalom_phase[j] += np.random.uniform(0.3, 0.7)  # Faz ilerlemesi

                        # Slalom hareketi (sinüs dalgası ±5A)
                        slalom_offset = 5.0 * np.sin(slalom_phase[j])
                        fault_levels[j] = fault_levels[j] + slalom_offset

                        # 20-40 adım slalom yap
                        slalom_duration = np.random.randint(20, 41) if fault_timers[j] == 1 else fault_timers[j]

                        if fault_timers[j] >= slalom_duration:
                            hat_fault_states[j] = 4  # Düşme fazına geç
                            fault_timers[j] = 0

                    elif hat_fault_states[j] == 4:  # Düşüyor
                        fault_timers[j] += 1
                        # 8-15 adımda normale dön
                        decrease_duration = np.random.randint(8, 16) if fault_timers[j] == 1 else fault_timers[j]
                        progress = min(1.0, fault_timers[j] / decrease_duration)

                        fault_levels[j] = fault_levels[j] * (1.0 - progress)

                        # Normale döndü mü?
                        if progress >= 1.0:
                            hat_fault_states[j] = 0  # Normal duruma dön
                            fault_levels[j] = 0.0
                            fault_timers[j] = 0
                            slalom_phase[j] = 0.0

                    # Final değer hesaplama
                    current_value += fault_levels[j]
                    hat_values.append(max(0, current_value))

                    # Arıza durumu belirleme
                    limit = self.hat1_limit if j == 0 else self.other_hats_limit
                    if current_value > limit:
                        overall_fault = 1

                data.append({
                    'timestamp': datetime.now() + timedelta(minutes=i),
                    'hat1_akim': round(hat_values[0], 2),
                    'hat2_akim': round(hat_values[1], 2),
                    'hat3_akim': round(hat_values[2], 2),
                    'hat4_akim': round(hat_values[3], 2),
                    'hat5_akim': round(hat_values[4], 2),
                    'ariza_durumu': overall_fault
                })

            self.df = pd.DataFrame(data)
            self.current_index = 0

            fault_count = self.df['ariza_durumu'].sum()
            self.data_status.config(text=f"📄 Veri: {len(self.df)} satır ({fault_count} arıza)", fg='#00ff41')
            self.log_message(f"✅ Gerçekçi kademeli veri oluşturuldu: {len(self.df)} veri, {fault_count} arıza")
            self.log_message(f"📊 Hat 5 arıza eğilimi: %1.5 (diğerleri: %0.8)")

        except Exception as e:
            messagebox.showerror("Hata", f"Örnek veri oluşturulurken hata: {str(e)}")
            self.log_message(f"❌ Örnek veri hatası: {str(e)}")

    def load_csv(self):
        """CSV dosyası yükleme"""
        file_path = filedialog.askopenfilename(
            title="CSV Dosyası Seçin",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            self.load_csv_from_path(file_path)

    def load_csv_from_path(self, file_path: str):
        """CSV dosyasını verilen path'ten yükle (telegram bot için de kullanılır)."""
        try:
            self.df = pd.read_csv(file_path)
            self.current_index = 0

            required_columns = ['hat1_akim', 'hat2_akim', 'hat3_akim', 'hat4_akim', 'hat5_akim']
            if not all(col in self.df.columns for col in required_columns):
                messagebox.showerror("Hata", "CSV dosyasında gerekli sütunlar bulunamadı!")
                return

            fault_count = self.df.get('ariza_durumu', pd.Series([0])).sum()
            self.data_status.config(text=f"📄 Veri: {len(self.df)} satır ({fault_count} arıza)", fg='#00ff41')
            self.log_message(f"✅ CSV yüklendi: {len(self.df)} veri noktası")

        except Exception as e:
            messagebox.showerror("Hata", f"Dosya yüklenirken hata: {str(e)}")
            self.log_message(f"❌ CSV yükleme hatası: {str(e)}")

    def prepare_data_for_training(self):
        """BiLSTM için veri hazırlama"""
        features = ['hat1_akim', 'hat2_akim', 'hat3_akim', 'hat4_akim', 'hat5_akim']
        X = self.df[features].values
        y = self.df['ariza_durumu'].values if 'ariza_durumu' in self.df.columns else np.zeros(len(self.df))

        # Normalize et
        X_scaled = self.scaler.fit_transform(X)

        # Sekans verisi oluştur
        X_sequences, y_sequences = [], []
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i - self.sequence_length:i])
            y_sequences.append(y[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        return train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42,
                                stratify=y_sequences if len(np.unique(y_sequences)) > 1 else None)

    def create_model(self):
        """BiLSTM model oluşturma"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True),
                          input_shape=(self.sequence_length, 5)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def train_model(self):
        """Model eğitme"""
        if self.df is None:
            messagebox.showwarning("Uyarı", "Önce veri yüklemelisiniz!")
            return

        def train_thread():
            try:
                self.log_message("🤖 BiLSTM model eğitimi başlatılıyor...")
                self.model_status.config(text="⏳ Model Eğitiliyor...", bg='#ffaa00')
                self.training_progress.start()

                # Veri hazırlama
                X_train, X_test, y_train, y_test = self.prepare_data_for_training()

                if len(X_train) < 50:
                    raise Exception("Eğitim için yeterli veri yok (minimum 50 örnek gerekli)")

                # Model oluştur
                self.model = self.create_model()

                # Eğitim
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
                )

                # Performans değerlendirme
                y_pred_prob = self.model.predict(X_test, verbose=0)
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                accuracy = accuracy_score(y_test, y_pred)

                self.model_accuracy = accuracy
                self.model_trained = True

                # GUI güncelleme
                self.root.after(0, self._update_model_status_success, accuracy)
                self.log_message(f"✅ Model eğitimi tamamlandı! Doğruluk: {accuracy:.3f}")

            except Exception as e:
                self.root.after(0, self._update_model_status_error, str(e))
                self.log_message(f"❌ Model eğitim hatası: {str(e)}")

            finally:
                self.root.after(0, lambda: self.training_progress.stop())

        threading.Thread(target=train_thread, daemon=True).start()

    def _update_model_status_success(self, accuracy):
        """Model eğitim başarı durumu güncelleme"""
        self.model_status.config(text="✅ Model Eğitildi & Hazır", bg='#00ff41', fg='#000000')
        self.accuracy_label.config(text=f"Doğruluk: {accuracy:.1%}", fg='#00ff41')

    def _update_model_status_error(self, error_msg):
        """Model eğitim hata durumu güncelleme"""
        self.model_status.config(text="❌ Model Eğitim Hatası", bg='#ff3333')
        messagebox.showerror("Model Eğitim Hatası", error_msg)

    def rule_based_fault_detection(self, hat_values):
        """Kural tabanlı arıza tespiti"""
        hat1, hat2, hat3, hat4, hat5 = hat_values
        faults = []

        if hat1 > self.hat1_limit:
            faults.append(f"Hat1: {hat1:.1f}A > {self.hat1_limit}A")

        for i, (val, limit) in enumerate([(hat2, self.other_hats_limit),
                                          (hat3, self.other_hats_limit),
                                          (hat4, self.other_hats_limit),
                                          (hat5, self.other_hats_limit)], 2):
            if val > limit:
                faults.append(f"Hat{i}: {val:.1f}A > {limit}A")

        return len(faults) > 0, faults

    def ai_fault_prediction(self, hat_values):
        """AI tabanlı arıza tahmini"""
        if not self.model_trained or self.model is None:
            return False, 0.0

        try:
            # Veriyi normalize et
            hat_values_scaled = self.scaler.transform([hat_values])

            # Sekans oluştur
            if self.son_veriler is None:
                self.son_veriler = np.tile(hat_values_scaled, (self.sequence_length, 1))
            else:
                self.son_veriler = np.vstack([self.son_veriler[1:], hat_values_scaled])

            # Tahmin yap
            X_input = self.son_veriler.reshape(1, self.sequence_length, 5)
            probability = self.model.predict(X_input, verbose=0)[0][0]
            prediction = probability > 0.5

            return prediction, probability

        except Exception as e:
            self.log_message(f"❌ AI tahmin hatası: {str(e)}")
            return False, 0.0

    def start_monitoring(self):
        """İzlemeyi başlatma"""
        if self.df is None:
            messagebox.showwarning("Uyarı", "Önce veri yüklemelisiniz!")
            return

        self.monitoring = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

        self.system_status.config(text="🔄 Gerçek Zamanlı İzleme Aktif", bg='#0066cc')
        self.log_message("▶️ Gerçek zamanlı izleme başlatıldı")

        # İzleme thread başlat
        self.monitor_thread = threading.Thread(target=self.monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """İzlemeyi durdurma"""
        self.monitoring = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

        self.system_status.config(text="⏹️ İzleme Durduruldu", bg='#ff3333')
        self.log_message("⏹️ Gerçek zamanlı izleme durduruldu")

    def monitoring_loop(self):
        """İzleme döngüsü"""
        while self.monitoring and self.current_index < len(self.df):
            try:
                # Mevcut veriyi al
                row = self.df.iloc[self.current_index]
                hat_values = [
                    row['hat1_akim'], row['hat2_akim'], row['hat3_akim'],
                    row['hat4_akim'], row['hat5_akim']
                ]

                # Kural tabanlı analiz
                rule_fault, rule_messages = self.rule_based_fault_detection(hat_values)

                # AI tabanlı analiz
                ai_fault, ai_probability = self.ai_fault_prediction(hat_values)

                # Geçmişe kaydet
                self.prediction_history.append(ai_probability)
                if 'ariza_durumu' in self.df.columns:
                    self.actual_history.append(row['ariza_durumu'])

                # GUI'yi güncelle
                self.root.after(0, self.update_monitoring_display, hat_values, rule_fault,
                                rule_messages, ai_fault, ai_probability)

                # Sonraki veri noktasına geç
                self.current_index += 1

                # Hıza göre bekle
                time.sleep(1.0 / max(0.1, self.speed_var.get()))

            except Exception as e:
                self.root.after(0, self.log_message, f"❌ İzleme hatası: {str(e)}")
                break

        # İzleme tamamlandı
        if self.monitoring:
            self.root.after(0, self.stop_monitoring)
            self.root.after(0, self.log_message, "✅ Tüm veriler işlendi - İzleme tamamlandı")

    def update_monitoring_display(self, hat_values, rule_fault, rule_messages, ai_fault, ai_probability):
        """İzleme ekranını güncelleme"""
        limits = [self.hat1_limit, self.other_hats_limit, self.other_hats_limit,
                  self.other_hats_limit, self.other_hats_limit]

        # Hat değerlerini güncelle
        for i, (value, limit) in enumerate(zip(hat_values, limits)):
            self.hat_labels[i].config(text=f"{value:.1f} A")

            # Kural tabanlı durum
            if value > limit:
                self.rule_labels[i].config(text="ARIZA", bg='#ff3333', fg='white')
            else:
                self.rule_labels[i].config(text="NORMAL", bg='#00ff41', fg='#000000')

            # AI tabanlı durum (genel AI tahminine göre)
            if ai_fault:
                self.ai_labels[i].config(text="ARIZA", bg='#9966cc', fg='white')
            else:
                self.ai_labels[i].config(text="NORMAL", bg='#006600', fg='white')

        # AI olasılığını güncelle
        self.ai_probability.config(text=f"{ai_probability * 100:.1f}%",
                                   fg='#ff3333' if ai_probability > 0.5 else '#00ff41')

        # Sistem durumu
        if rule_fault or ai_fault:
            if rule_fault and ai_fault:
                status_text = "🚨 KURAL + AI ARIZA TESPİTİ!"
                status_bg = '#ff0000'
            elif rule_fault:
                status_text = "⚠️ KURAL TABANLI ARIZA!"
                status_bg = '#ff6600'
            else:
                status_text = "🤖 AI ARIZA TAHMİNİ!"
                status_bg = '#9966cc'

            self.system_status.config(text=status_text, bg=status_bg, fg='white')

            # Detaylı log mesajı
            log_parts = []
            if rule_fault:
                log_parts.append(f"KURAL: {', '.join(rule_messages)}")
            if ai_fault:
                log_parts.append(f"AI: Arıza olasılığı {ai_probability * 100:.1f}%")

            alert_text = f"🚨 ARIZA: {' | '.join(log_parts)}"
            self.log_message(alert_text)
            # Telegram bildirimi
            if self.telegram_bot is not None:
                try:
                    self.telegram_bot.notify_fault(alert_text)
                except Exception:
                    pass

        else:
            self.system_status.config(text="✅ Sistem Normal Çalışıyor", bg='#00ff41', fg='#000000')

        # Grafikleri güncelle
        if self.current_index % 10 == 0:  # Her 10 veri noktasında grafik güncelle
            self.update_graphs()

    def update_graphs(self):
        """Grafikleri güncelleme"""
        if self.df is None:
            return

        try:
            # Görüntülenecek veri aralığı
            end_idx = min(self.current_index + 1, len(self.df))
            start_idx = max(0, end_idx - 100)  # Son 100 veri noktası

            current_data = self.df.iloc[start_idx:end_idx]

            if len(current_data) == 0:
                return

            # Grafikleri temizle
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()

            # 1. Hat Akımları Grafiği
            colors = ['#ff3333', '#0066cc', '#00ff41', '#ffaa00', '#9966cc']
            hat_names = ['Hat 1', 'Hat 2', 'Hat 3', 'Hat 4', 'Hat 5']

            for i, (name, color) in enumerate(zip(hat_names, colors)):
                col_name = f'hat{i + 1}_akim'
                if col_name in current_data.columns:
                    self.ax1.plot(range(len(current_data)), current_data[col_name],
                                  label=name, color=color, linewidth=2)

            # Limit çizgileri
            self.ax1.axhline(y=self.hat1_limit, color='#ff3333', linestyle='--', alpha=0.8,
                             label=f'Hat 1 Limit ({self.hat1_limit}A)')
            self.ax1.axhline(y=self.other_hats_limit, color='#ffaa00', linestyle='--', alpha=0.8,
                             label=f'Diğer Hatlar Limit ({self.other_hats_limit}A)')

            self.ax1.set_title('Hat Akımları (A)', color='#00ff41', fontsize=12, fontweight='bold')
            self.ax1.set_ylabel('Akım (A)', color='white')
            self.ax1.tick_params(colors='white')
            self.ax1.grid(True, alpha=0.3, color='gray')
            self.ax1.legend(facecolor='#2d2d2d', edgecolor='#00ff41', labelcolor='white', fontsize=8)

            # 2. Kural vs AI Arıza Tespiti
            if 'ariza_durumu' in current_data.columns:
                actual_faults = current_data['ariza_durumu'].values
                x_vals = range(len(current_data))

                # Gerçek arızalar
                fault_indices = [i for i, val in enumerate(actual_faults) if val == 1]
                if fault_indices:
                    self.ax2.scatter([x_vals[i] for i in fault_indices],
                                     [1.0] * len(fault_indices),
                                     c='#ff3333', s=50, marker='o', label='Gerçek Arıza', alpha=0.8)

            # AI tahminleri (son prediction_history'den)
            if self.prediction_history:
                pred_start = max(0, len(self.prediction_history) - len(current_data))
                pred_data = self.prediction_history[pred_start:pred_start + len(current_data)]

                if pred_data:
                    self.ax2.plot(range(len(pred_data)), pred_data,
                                  color='#9966cc', linewidth=2, label='AI Olasılık', alpha=0.8)

                    # AI arıza tahminleri (>0.5)
                    ai_fault_indices = [i for i, val in enumerate(pred_data) if val > 0.5]
                    if ai_fault_indices:
                        self.ax2.scatter(ai_fault_indices, [0.8] * len(ai_fault_indices),
                                         c='#9966cc', s=30, marker='^', label='AI Arıza Tahmini', alpha=0.8)

            self.ax2.set_title('Kural vs AI Arıza Tespiti', color='#00ff41', fontsize=12, fontweight='bold')
            self.ax2.set_ylabel('Arıza Durumu / Olasılık', color='white')
            self.ax2.tick_params(colors='white')
            self.ax2.grid(True, alpha=0.3, color='gray')
            self.ax2.set_ylim(-0.1, 1.1)
            self.ax2.legend(facecolor='#2d2d2d', edgecolor='#00ff41', labelcolor='white', fontsize=8)

            # 3. AI Arıza Olasılığı Trendi
            if self.prediction_history:
                recent_predictions = self.prediction_history[-50:]  # Son 50 tahmin
                self.ax3.plot(range(len(recent_predictions)), recent_predictions,
                              color='#9966cc', linewidth=3, alpha=0.8)
                self.ax3.fill_between(range(len(recent_predictions)), recent_predictions,
                                      alpha=0.3, color='#9966cc')
                self.ax3.axhline(y=0.5, color='#ff3333', linestyle='--', alpha=0.8,
                                 label='Arıza Eşiği (0.5)')

            self.ax3.set_title('AI Arıza Olasılığı Trendi', color='#00ff41', fontsize=12, fontweight='bold')
            self.ax3.set_ylabel('Olasılık', color='white')
            self.ax3.set_xlabel('Zaman', color='white')
            self.ax3.tick_params(colors='white')
            self.ax3.grid(True, alpha=0.3, color='gray')
            self.ax3.set_ylim(0, 1)
            self.ax3.legend(facecolor='#2d2d2d', edgecolor='#00ff41', labelcolor='white', fontsize=8)

            # 4. Model Performans Metrikleri
            if self.model_trained and len(self.prediction_history) > 10 and len(self.actual_history) > 10:
                # Son tahminlerin performansını hesapla
                recent_pred = np.array(self.prediction_history[-50:])
                recent_actual = np.array(self.actual_history[-50:]) if len(self.actual_history) >= 50 else np.array(
                    self.actual_history)

                if len(recent_pred) == len(recent_actual) and len(recent_actual) > 0:
                    recent_pred_binary = (recent_pred > 0.5).astype(int)

                    # Basit metrikler
                    accuracy = np.mean(recent_pred_binary == recent_actual)
                    precision = np.mean(recent_pred_binary[recent_actual == 1] == 1) if np.any(
                        recent_actual == 1) else 0
                    recall = np.mean(recent_pred_binary[recent_actual == 1] == 1) if np.any(recent_actual == 1) else 0

                    metrics = ['Doğruluk', 'Kesinlik', 'Hassasiyet']
                    values = [accuracy, precision, recall]
                    colors_bar = ['#00ff41', '#0066cc', '#ffaa00']

                    bars = self.ax4.bar(metrics, values, color=colors_bar, alpha=0.8)

                    # Bar üzerine değer yaz
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        self.ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                      f'{val:.2f}', ha='center', va='bottom', color='white', fontweight='bold')
            else:
                # Model eğitilmemişse bilgi göster
                self.ax4.text(0.5, 0.5, 'Model Eğitilmedi\nveya\nYeterli Veri Yok',
                              ha='center', va='center', transform=self.ax4.transAxes,
                              color='#ff3333', fontsize=12, fontweight='bold')

            self.ax4.set_title('Gerçek Zamanlı Model Performansı', color='#00ff41', fontsize=12, fontweight='bold')
            self.ax4.set_ylabel('Değer', color='white')
            self.ax4.tick_params(colors='white')
            self.ax4.grid(True, alpha=0.3, color='gray')
            self.ax4.set_ylim(0, 1.1)

            # Grafikleri yenile
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.log_message(f"❌ Grafik güncelleme hatası: {str(e)}")

    def log_message(self, message):
        """Log mesajı ekleme"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, full_message)
        self.log_text.see(tk.END)

        # Log boyutunu sınırla (son 1000 satır)
        if int(self.log_text.index('end-1c').split('.')[0]) > 1000:
            self.log_text.delete('1.0', '100.0')

    def run(self):
        """GUI'yi çalıştırma"""
        print("🚀 GUI başlatılıyor...")
        self.log_message("🤖 BiLSTM Elektrik Hattı Arıza İzleme Sistemi başlatıldı")
        self.log_message("📋 Adımlar: 1) Veri yükle/oluştur 2) Model eğit 3) İzlemeyi başlat")
        
        print("🤖 Telegram bot başlatılıyor...")
        # Initialize Telegram bot after GUI is ready
        self._init_telegram_if_available()
        
        print("🔄 GUI mainloop başlatılıyor...")
        self.root.mainloop()


# Ana program
if __name__ == "__main__":
    print("🤖 BiLSTM Elektrik Hattı Arıza İzleme Sistemi Başlatılıyor...")
    print("📱 GUI penceresi açılıyor...")

    app = BiLSTMArızaİzlemeSistemi()
    app.run()

