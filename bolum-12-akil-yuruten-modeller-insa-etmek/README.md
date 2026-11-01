# NLP 2026 · Bölüm 12: Akıl Yürüten Modeller İnşa Etmek
### Hugging Face LLM Course · Chapter 12: Build Reasoning Models

Bu klasör, Hugging Face LLM Course 12. bölümü için hazırlanan teslim paketini içerir: pekiştirmeli öğrenme, DeepSeek-R1 makalesi, GRPO ve GRPO'nun TRL ile uygulanması.

## Klasör yapısı

```
bolum-12-akil-yuruten-modeller-insa-etmek/
└── 12-akil-yuruten-modeller/
    ├── ceviri/
    │   ├── 2026-guz-nlp12.md               # 12/1–12/7 sayfalarının Türkçe çevirisi
    │   └── 2026-guz-nlp12-resimler/        # Çeviride kullanılan kurs görselleri
    ├── sunum/
    │   └── 2026-guz-nlp12.pdf              # 67 slaytlık sunum
    ├── soru-cevap/
    │   └── 2026-guz-nlp12.md               # Kurs quizleri ve sunum değerlendirme soruları
    ├── red-team/
    │   ├── 2026-guz-nlp12.md               # GRPO ödül fonksiyonları zafiyet raporu (5 test)
    │   └── red_team_testleri.py            # Testlerin çalıştırılabilir kodu
    └── uygulama/
        └── v1-2026-nlp12/
            ├── README.md                   # Çalıştırma yönergesi
            ├── grpo_egitimi.py             # 12.5: GRPO + LoRA ile 50 token hedefli ince ayar
            ├── sonuclar.md                 # Eğitim öncesi / sonrası ölçüm sonuçları
            ├── requirements.txt
            ├── .gitignore
            ├── unsloth-12-6/               # 12.6: Colab'da çalıştırılmış Unsloth not defteri
            ├── ilk-deneme-cpu/             # 12.5 öncesi kısa CPU denemesi
            └── kurs-kodlari/               # 12.3b ve 12.4 kod blokları
                ├── ciktilar/               # Her kodun çalıştırma çıktısı (.txt)
                └── grafikler/              # grafikler.py ile üretilen grafikler
```

## Kaynak

[Hugging Face LLM Course, Chapter 12](https://huggingface.co/learn/llm-course/chapter12/1) · Apache License 2.0
