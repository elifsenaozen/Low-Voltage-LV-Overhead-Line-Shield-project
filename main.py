import serial, csv, time, threading, math, re
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext
from collections import deque

FLOAT_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

class SimpleArduinoReader:
    def __init__(self, port='COM6', baud=9600, csv_file='acs712_log.csv'):
        self.port, self.baud, self.timeout = port, baud, 3
        self.csv_file_path = csv_file

        # Kalibrasyon/filtre parametreleri
        self.deadband = 0.001      # ±1 mA altını 0 göster
        self.zero_offset = 0.0     # Tare ile değişir (default 0)
        self.max_abs_a = 5.5       # 5A sensör + marj
        self.hampel_win = 15       # outlier penceresi
        self.hampel_k = 3.0        # outlier eşiği
        self.ema_alpha = 0.25      # EMA yumuşatma

        # Depolar
        self.timestamps = deque(maxlen=500)
        self.currents_f = deque(maxlen=500)  # filtered
        self._recent_raw = deque(maxlen=128)
        self._ema_state = None

        # IO/thread
        self.is_running = False
        self.serial_thread = None
        self.ser = None
        self.csv_file = None
        self.csv_writer = None

        self._last_emit = 0  # ekrana/CSV’ye son yazılan an (Arduino 1.5 s aralığına göre)

        self.setup_gui()

    # ---------- GUI ----------
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Arduino Akım İzleyici (Sağlamlaştırılmış)")
        self.root.geometry("800x600")

        settings_frame = ttk.Frame(self.root)
        settings_frame.pack(pady=10, padx=10, fill='x')

        ttk.Label(settings_frame, text="COM Port:").grid(row=0, column=0, padx=5)
        self.port_entry = ttk.Entry(settings_frame, width=10)
        self.port_entry.insert(0, self.port)
        self.port_entry.grid(row=0, column=1, padx=5)

        ttk.Label(settings_frame, text="Baud:").grid(row=0, column=2, padx=5)
        self.baud_entry = ttk.Entry(settings_frame, width=10)
        self.baud_entry.insert(0, str(self.baud))
        self.baud_entry.grid(row=0, column=3, padx=5)

        ttk.Label(settings_frame, text="Ölü Bölge (A):").grid(row=0, column=4, padx=5)
        self.db_entry = ttk.Entry(settings_frame, width=10)
        self.db_entry.insert(0, f"{self.deadband:.3f}")
        self.db_entry.grid(row=0, column=5, padx=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        self.start_button = ttk.Button(control_frame, text="Başlat", command=self.start_reading)
        self.start_button.pack(side='left', padx=5)
        self.stop_button = ttk.Button(control_frame, text="Durdur", command=self.stop_reading, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        self.tare_button = ttk.Button(control_frame, text="Sıfırla (Tare)", command=self.tare_now, state='disabled')
        self.tare_button.pack(side='left', padx=5)

        self.status_label = ttk.Label(self.root, text="Hazır", foreground="green")
        self.status_label.pack(pady=5)

        ttk.Label(self.root, text="Akım:").pack(pady=(10, 0))
        self.data_text = scrolledtext.ScrolledText(self.root, height=16, width=70)
        self.data_text.pack(pady=10, padx=10, fill='both', expand=True)

        stats_frame = ttk.Frame(self.root)
        stats_frame.pack(pady=10, fill='x')
        self.stats_label = ttk.Label(stats_frame, text="İstatistik: Veri yok")
        self.stats_label.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ---------- Serial ----------
    def init_serial_connection(self):
        try:
            self.port = self.port_entry.get().strip()
            self.baud = int(self.baud_entry.get().strip())
            self.deadband = float(self.db_entry.get().strip())

            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(2.2)  # UNO reset + calibrate fırsatı
            self.ser.reset_input_buffer()  # reset sonrası çöpü temizle

            self.csv_file = open(self.csv_file_path, mode='w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'Current_filtered (A)', 'Current_raw (A)'])
            return True
        except Exception as e:
            self.update_status(f"HATA: {e}", "red")
            return False

    def serial_reader(self):
        while self.is_running:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                s = line.decode('utf-8', 'ignore').strip().replace(',', '.')
                m = FLOAT_RE.search(s)
                if not m:
                    continue
                try:
                    raw = float(m.group(0))
                except:
                    continue

                # DC: negatif gelirse (yön) pozitife çevir
                raw = abs(raw)

                # Fiziksel aralık filtresi
                if not (0.0 <= raw <= self.max_abs_a):
                    continue

                # Tare uygula
                raw_adj = raw - self.zero_offset

                # Hampel outlier reddi
                clean = self._hampel(raw_adj)

                # EMA yumuşatma
                if self._ema_state is None:
                    self._ema_state = clean
                else:
                    self._ema_state = self.ema_alpha*clean + (1-self.ema_alpha)*self._ema_state

                out = self._ema_state

                # Deadband
                if abs(out) < self.deadband:
                    out = 0.0

                now = time.time()
                # Arduino 1.5s'de bir gönderiyor → ekrana/CSV'ye 1.2s arayla yaz (yaklaşık eşleşsin)
                if now - self._last_emit >= 1.2:
                    ts_short = datetime.now().strftime('%H:%M:%S')
                    ts_full = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    self.timestamps.append(ts_short)
                    self.currents_f.append(out)

                    self.csv_writer.writerow([ts_full, f"{out:.6f}", f"{raw:.6f}"])
                    self.csv_file.flush()

                    self.root.after(0, self.update_gui, ts_short, out)
                    self._last_emit = now

            except Exception as e:
                if self.is_running:
                    self.root.after(0, self.update_status, f"Veri okuma hatası: {e}", "red")
                break

    # ---------- Filters ----------
    def _hampel(self, x):
        self._recent_raw.append(x)
        data = list(self._recent_raw)
        n = len(data)
        if n < 5:
            return x
        med = self._median(data)
        mad = self._median([abs(d-med) for d in data])
        sigma = 1.4826*mad if mad > 0 else 0.0
        if sigma > 0 and abs(x - med) > self.hampel_k*sigma:
            return med  # outlier → medyan
        return x

    @staticmethod
    def _median(arr):
        arr = sorted(arr)
        n = len(arr)
        if n == 0:
            return 0.0
        mid = n // 2
        return arr[mid] if n % 2 else 0.5*(arr[mid-1] + arr[mid])

    # ---------- UI helpers ----------
    def update_gui(self, timestamp, current):
        self.data_text.insert('end', f"{timestamp} | {current:.6f} A\n")
        self.data_text.see('end')

        if len(self.currents_f) > 0:
            vals = list(self.currents_f)
            avg = sum(vals)/len(vals)
            mn = min(vals); mx = max(vals)
            self.stats_label.config(text=f"Veri: {len(vals)} | Ort: {avg:.4f}A | Min: {mn:.4f}A | Max: {mx:.4f}A")

        # çok satır birikirse kırp
        try:
            if int(self.data_text.index('end-1c').split('.')[0]) > 1500:
                self.data_text.delete('1.0', '600.0')
        except:
            pass

    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)

    def tare_now(self):
        if len(self.currents_f) > 0:
            self.zero_offset = self.currents_f[-1]
            self.update_status(f"Yeni ofset: {self.zero_offset:.6f} A", "purple")

    def start_reading(self):
        if not self.init_serial_connection():
            return
        self.is_running = True
        self.serial_thread = threading.Thread(target=self.serial_reader, daemon=True)
        self.serial_thread.start()
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.tare_button.config(state='normal')
        self.update_status("Okuma başlatıldı... (yükü 3 sn sonra bağla)", "blue")

    def stop_reading(self):
        self.is_running = False
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2)
        for h in (self.ser, self.csv_file):
            try:
                if h: h.close()
            except:
                pass
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.tare_button.config(state='disabled')
        self.update_status("Okuma durduruldu", "orange")

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        if self.is_running:
            self.stop_reading()
        self.root.destroy()

def main():
    print("=== ACS712 Sağlam Okuyucu ===")
    try:
        SimpleArduinoReader().run()
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()
