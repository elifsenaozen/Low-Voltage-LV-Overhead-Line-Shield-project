#include "ACS712.h"

// 5A versiyonu, A0
ACS712 sensor(ACS712_05B, A0);

// Ayarlar
const int   BURST_SAMPLES = 200;   // her ölçümde kaç örnek alalım
const float DEADBAND      = 0.0003; // 0.3 mA (çok küçük bir ölü bölge)
const float EMA_ALPHA     = 0.15;   // yumuşatma (0.05–0.3 arası deneyebilirsin)

float emaI = 0.0;
bool  emaInit = false;

void setup() {
  Serial.begin(9600);
  delay(500);

  // 0A iken kalibre ET! (üzerinden akım geçmiyor olmalı)
  sensor.calibrate();
}

void loop() {
  // Hızlı örnekleme + ortalama
  double acc = 0.0;
  for (int i = 0; i < BURST_SAMPLES; i++) {
    acc += sensor.getCurrentDC(); // kütüphanenin ofset/ölçek düzeltmesiyle okuma
    delayMicroseconds(800);       // ~0.8 ms aralık (çok küçük bekleme)
  }
  float i_avg = acc / BURST_SAMPLES;

  // EMA ile yumuşatma
  if (!emaInit) {
    emaI = i_avg;
    emaInit = true;
  } else {
    emaI = (1.0 - EMA_ALPHA) * emaI + EMA_ALPHA * i_avg;
  }

  // Çok küçük değerleri sıfırla (ölü bölge)
  float I = (fabs(emaI) < DEADBAND) ? 0.0f : emaI;

  // Yazdır
  Serial.print("DC Current: ");
  Serial.print(I, 4);          // 4-5 basamak yeterli; 6 gereksiz gürültü gösterir
  Serial.println(" A");

  delay(250); // daha akıcı çıktı
}