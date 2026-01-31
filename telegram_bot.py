import os
import time
import tempfile
import threading
from typing import Optional, Set

try:
    import telebot  # pyTelegramBotAPI
except ImportError as e:
    raise RuntimeError(
        "pyTelegramBotAPI is required. Install with: pip install pyTelegramBotAPI"
    ) from e


class TelegramBotManager:
    """Manages Telegram bot integration and bridges commands to the GUI app.

    Commands:
      /start, /help
      /status
      /start_monitoring
      /stop_monitoring
      /set_speed <0.1-5.0>
      /train
      /create_sample
    CSV upload: send a .csv document to the bot.
    """

    def __init__(self, token: str, app) -> None:
        self.bot = telebot.TeleBot(token)
        self.app = app
        self.chat_ids: Set[int] = set()
        self._thread: Optional[threading.Thread] = None
        self._last_alert_ts: float = 0.0
        self._last_alert_text: str = ""

        self._register_handlers()

    # -------------------- Public API --------------------
    def start(self) -> None:
        """Start the bot in a daemon background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="TelegramBot", daemon=True)
        self._thread.start()

    def notify_fault(self, text: str, throttle_seconds: float = 15.0) -> None:
        """Notify all known chats about a fault event with throttling."""
        now = time.time()
        if not self.chat_ids:
            return
        # Throttle and deduplicate short bursts
        if (now - self._last_alert_ts) < throttle_seconds and text == self._last_alert_text:
            return
        self._last_alert_ts = now
        self._last_alert_text = text
        for chat_id in list(self.chat_ids):
            try:
                self.bot.send_message(chat_id, text)
            except Exception:
                # Ignore send errors to avoid crashing the GUI
                pass

    # -------------------- Internal --------------------
    def _run(self) -> None:
        self.bot.infinity_polling(skip_pending=True, timeout=60)

    def _register_handlers(self) -> None:
        @self.bot.message_handler(commands=["start", "help"])  # noqa: ANN001
        def _start_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            self.bot.reply_to(message, self._help_text())

        @self.bot.message_handler(commands=["status"])  # noqa: ANN001
        def _status_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            self.bot.reply_to(message, self._status_text())

        @self.bot.message_handler(commands=["start_monitoring"])  # noqa: ANN001
        def _start_mon_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            self.app.root.after(0, self.app.start_monitoring)
            self.bot.reply_to(message, "▶️ İzleme başlatılıyor...")

        @self.bot.message_handler(commands=["stop_monitoring"])  # noqa: ANN001
        def _stop_mon_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            self.app.root.after(0, self.app.stop_monitoring)
            self.bot.reply_to(message, "⏹️ İzleme durduruluyor...")

        @self.bot.message_handler(commands=["train"])  # noqa: ANN001
        def _train_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            self.app.root.after(0, self.app.train_model)
            self.bot.reply_to(message, "🤖 Model eğitimi başlatılıyor...")

        @self.bot.message_handler(commands=["create_sample"])  # noqa: ANN001
        def _sample_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            self.app.root.after(0, self.app.create_sample_data)
            self.bot.reply_to(message, "🎲 Örnek veri oluşturuluyor...")

        @self.bot.message_handler(commands=["set_speed"])  # noqa: ANN001
        def _speed_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            try:
                parts = (message.text or "").split()
                if len(parts) < 2:
                    raise ValueError("Eksik parametre")
                speed = float(parts[1])
                speed = max(0.1, min(5.0, speed))
                def _apply_speed():
                    self.app.speed_var.set(speed)
                self.app.root.after(0, _apply_speed)
                self.bot.reply_to(message, f"✅ Hız ayarlandı: {speed:.1f}x")
            except Exception:
                self.bot.reply_to(message, "❌ Kullanım: /set_speed 0.1-5.0")

        # CSV upload as document
        @self.bot.message_handler(content_types=["document"])  # noqa: ANN001
        def _csv_upload_handler(message):  # noqa: ANN001
            self.chat_ids.add(message.chat.id)
            try:
                doc = message.document
                if not doc or not (doc.file_name or "").lower().endswith(".csv"):
                    self.bot.reply_to(message, "❌ Lütfen .csv dosyası gönderin")
                    return
                file_info = self.bot.get_file(doc.file_id)
                downloaded = self.bot.download_file(file_info.file_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(downloaded)
                    tmp_path = tmp.name
                # Load into app on UI thread
                self.app.root.after(0, self.app.load_csv_from_path, tmp_path)
                self.bot.reply_to(message, "✅ CSV alındı ve yükleniyor...")
            except Exception as e:  # noqa: BLE001
                self.bot.reply_to(message, f"❌ CSV yükleme hatası: {e}")

    def _help_text(self) -> str:
        return (
            "Merhaba! Komutlar:\n"
            "/status — sistem durumu\n"
            "/start_monitoring — izlemeyi başlat\n"
            "/stop_monitoring — izlemeyi durdur\n"
            "/set_speed 1.0 — hız (0.1-5.0)\n"
            "/train — modeli eğit\n"
            "/create_sample — örnek veri oluştur\n"
            "CSV yüklemek için .csv dosyasını belge olarak gönderin."
        )

    def _status_text(self) -> str:
        try:
            df_len = 0 if self.app.df is None else len(self.app.df)
            trained = "Eğitildi" if self.app.model_trained else "Eğitilmedi"
            acc = f"{self.app.model_accuracy:.1%}" if self.app.model_trained else "-"
            monitoring = "Açık" if self.app.monitoring else "Kapalı"
            speed = float(self.app.speed_var.get()) if hasattr(self.app, 'speed_var') else 1.0
            last_prob = self.app.prediction_history[-1] if self.app.prediction_history else 0.0
            idx = int(self.app.current_index)
            return (
                f"📊 Veri: {df_len} satır\n"
                f"🧠 Model: {trained} (Doğruluk: {acc})\n"
                f"🎮 İzleme: {monitoring} (index: {idx}/{df_len})\n"
                f"⏩ Hız: {speed:.1f}x\n"
                f"🤖 AI Olasılığı: {last_prob:.2f}"
            )
        except Exception:
            return "Durum okunamadı."



