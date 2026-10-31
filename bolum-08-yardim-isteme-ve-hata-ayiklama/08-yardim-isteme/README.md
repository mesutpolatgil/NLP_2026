# NLP 2026 · Bölüm 08: Yardım İsteme ve Hata Ayıklama
### Hugging Face Course — Chapter 8: Asking for Help and Debugging

Bu çalışma, **Sivas Cumhuriyet Üniversitesi Doğal Dil İşleme Dersi (NLP 2026)** kapsamında Hugging Face LLM Kursu 8. Bölümüne yönelik hazırlanan resmi **5'li teslimat paketini** içermektedir.

---

## 📦 Zorunlu 5'li Teslimat Paketi ve Klasör Mimarisi

Ders standartlarına ve depo hiyerarşisine tam uyumlu olarak organize edilen modül yapısı aşağıdadır:

```
08-yardim-isteme/
├── README.md                                  # Kök proje tanıtım ve navigasyon belgesi
├── .gitignore                                 # Git dışlama kuralları (önbellek, os, taslaklar)
│
├── ceviri/                                    # 1. MODÜL: ÇEVİRİ
│   ├── README.md                              # Modül tanıtımı
│   └── 2026-guz-nlp08.md                      # Bölüm 8 Türkçe teknik çevirisi (1.969 satır)
│
├── sunum/                                     # 2. MODÜL: SUNUM
│   ├── README.md                              # Modül tanıtımı
│   ├── 2026-guz-nlp08.pptx                    # 11 slaytlık resmi akademik sunum
│   ├── SLAYT_ICERIGI.md                       # Slaytların tam metin ve kod dökümü
│   ├── preview.jpg                            # Sunum kapak önizlemesi
│   └── assets/                                # Vektörel diyagram ve grafik varlıkları
│
├── uygulama/                                  # 3. MODÜL: UYGULAMA (KOD LABI)
│   ├── README.md                              # Modül ve sürüm tanıtımı
│   └── v1-2026-nlp08/
│       ├── main.ipynb                         # Hata izi ve minimal sorun bildirim not defteri
│       ├── requirements.txt                   # Bağımlılık listesi
│       └── README.md                          # Uygulama çalıştırma yönergesi
│
├── soru-cevap/                                # 4. MODÜL: SORU-CEVAP
│   ├── README.md                              # Modül tanıtımı
│   └── 2026-guz-nlp08.md                      # Akademik test soruları ve çeldirici analizleri (✅)
│
└── red-team/                                  # 5. MODÜL: RED-TEAM & GÜVENLİK
    ├── README.md                              # Modül tanıtımı ve test senaryoları
    ├── 2026-guz-nlp08-red-team.md             # Kapsamlı statik güvenlik ve zafiyet raporu (G1-G6)
    ├── 2026-guz-nlp08-red-team.pptx           # 8 slaytlık Red-Team risk sunumu
    ├── preview.jpg                            # Red-team kapak önizlemesi
    └── assets/                                # Kapak görsel varlıkları
```

---

## 🎯 Modül Özetleri

### 1. 📖 [Çeviri Modülü](ceviri/README.md)
Hugging Face Course Chapter 8 içeriğinin eksiksiz Türkçe çevirisidir.
* Traceback'in anatomisi ve aşağıdan yukarıya doğru okunması.
* Hugging Face forumlarında etkili soru sorma yöntemleri.
* PyTorch ve TensorFlow eğitim hatlarında hata ayıklama adımları.
* GitHub üzerinde minimal ve yeniden üretilebilir (reproducible) issue yazımı.

### 2. 🖥️ [Sunum Modülü](sunum/README.md)
* **[2026-guz-nlp08.pptx](sunum/2026-guz-nlp08.pptx):** Sınıf içi ve akademik paylaşıma uygun, modern tipografi ve NLP 2026 temalı 11 slayt.
* **[SLAYT_ICERIGI.md](sunum/SLAYT_ICERIGI.md):** Office yazılımı gerekmeden slayt içeriklerini, karşılaştırma tablolarını ve adımları doğrudan Markdown formatında okuma imkânı.

### 3. 🧪 [Uygulama Modülü](uygulama/README.md)
* **[v1-2026-nlp08/main.ipynb](uygulama/v1-2026-nlp08/main.ipynb):** Kasıtlı bir etiketleme hatası (`KeyError`) üzerinden traceback çözümleme, assertion ile düzeltmeyi doğrulama ve standart issue şablonunu uygulama laboratuvarı.
* Standart bağımlılıklar: `transformers`, `torch`, `datasets`, `jupyter`, `ipykernel`.

### 4. 📝 [Soru-Cevap Modülü](soru-cevap/README.md)
* **[2026-guz-nlp08.md](soru-cevap/2026-guz-nlp08.md):** Vize/Final düzeyinde çoktan seçmeli akademik sorular, çeldirici analizleri ve detaylı gerekçeli doğru cevaplar (✅).

### 5. 🛡️ [Red-Team Modülü](red-team/README.md)
* **[2026-guz-nlp08-red-team.md](red-team/2026-guz-nlp08-red-team.md):** Kurs metnine yönelik statik güvenlik denetimi:
  * **G1–G6 Bulguları:** Güvenilmeyen model yükleme, typosquatting tehdidi, doğrulamasız Hub yüklemesi, forumlara traceback sızıntısı, güncel olmayan jeton anlatımı ve kopyala-yapıştır güveni.
  * **Teknik Güncellik:** `eval_strategy`, `processing_class`, `distutils` kaldırılması vb.
  * **Deney Planı:** Hata enjeksiyonu ve gözlem beklentileri.
* **[2026-guz-nlp08-red-team.pptx](red-team/2026-guz-nlp08-red-team.pptx):** Red-Team bulgularının 8 slaytlık yönetici özeti.

---

## 🔒 Güvenlik ve Gizlilik Beyanı
Bu depodaki tüm çalışmalar akademik ve sentetik verilerle yürütülmüştür. Hiçbir gerçek API anahtarı, erişim belirteci (token), şifre veya kişisel tanımlayıcı veri (PII) barındırmaz.
