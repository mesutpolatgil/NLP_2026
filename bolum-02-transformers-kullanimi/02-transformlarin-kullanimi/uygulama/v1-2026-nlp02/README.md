> **Uyarı:** Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.

# Model İçi İşleyiş ve Tokenizer — Hugging Face Transformers

Bu depo, Hugging Face `transformers` kütüphanesinde pipeline soyutlamasının arka planındaki mimari adımları uygulamalı olarak gösteren bir Jupyter Notebook (`main.ipynb`) içerir. Çalışma üç ana temeli ele alır: **Tokenizer iç mekanizmaları**, **temel model ve sınıflandırma başlıkları (AutoModel)** ve **toplu işlem (batching, padding, attention mask)** süreçleri.

## İçerik

Notebook aşağıdaki bölümlerden oluşur:

| Bölüm | Başlık | Açıklama |
|-------|--------|----------|
| 1 | Ortam ve Kurulum | Gerekli paketlerin kurulumu ve donanım (CPU/GPU) yapılandırması |
| 2 | Tokenizer Temelleri | Metin parçalama (tokenize), ID eşleme, özel token'lar ve geri çözümleme (decode) |
| 3 | Model Mimarileri ve Çıktı Analizi | `AutoModel` (hidden states) ile `AutoModelForSequenceClassification` (logits) tensör akışı |
| 4 | Toplu İşleme (Batching) ve Maskeleme | Dinamik dolgulama (padding), kırpma (truncation) ve `attention_mask` mekanizması |
| 5 | Model Kaydetme ve Yeniden Yükleme | Model ağırlıklarının ve sözlük yapılandırmasının yerel diske aktarılması (`save_pretrained`) |

## Kullanılan Teknolojiler

- **Python**
- **PyTorch** — tensör işlemleri, logit manipülasyonu ve softmax hesaplamaları
- **Hugging Face Transformers** — model mimarileri ve `AutoTokenizer` / `AutoModel` API'si
- **Matplotlib & NumPy** — logits ve normalize edilmiş sınıf olasılıklarının görselleştirilmesi

## Kullanılan Modeller

- `distilbert-base-uncased-finetuned-sst-2-english` — duygu analizi mimarisi üzerinden model içi tensör akışı ve logits analizi

## Kurulum

Gerekli bağımlılıkları yükleyin:

```bash
pip install transformers torch accelerate matplotlib numpy
Kullanım
Notebook'u Jupyter, Google Colab veya VS Code üzerinden açıp hücreleri sırayla çalıştırmanız yeterlidir:

Bash
jupyter notebook main.ipynb
Not: Modeller ilk çalıştırmada Hugging Face Hub'dan indirilir, bu nedenle internet bağlantısı gerekir.

Bölüm Detayları
1. Tokenizer Mekanizması
Modelin sayısal dilini oluşturan alt adımlar adım adım yürütülür:

Tokenizasyon: Ham metnin alt kelime parçalarına (subwords) ayrılması.

Sayısal Eşleme: Belirteçlerin sözlük (vocab) indislerine (input_ids) çevrilmesi.

Decoding: Sayısal tensörlerin insan tarafından okunabilir metne dönüştürülmesi ve özel belirteçlerin ([CLS], [SEP]) yönetimi.

2. AutoModel vs. Görev Başlıkları
İki model türünün tensör çıktı boyutları ve mimari farkları karşılaştırılır:

Model Tipi	Çıktı Tensörü	Çıktı Boyutu	Kullanım Amacı
AutoModel	last_hidden_state	[Batch Size, Sequence Length, Hidden Size]	Özellik çıkarımı ve gömme vektörleri
AutoModelForSequenceClassification	logits	[Batch Size, Num Classes]	Doğrudan sınıf tahmini ve skorlama
Model çıktısı olan ham logits değerleri torch.softmax işleminden geçirilerek toplamı 1 olan gerçek olasılık değerlerine dönüştürülür.

3. Toplu İşleme ve Attention Mask
Farklı uzunluktaki girdiler paralel işlenirken karşılaşılan matris boyut eşitsizliği çözülür:

Padding: Kısa dizilere modelin maksimum uzunluğuna veya batch uzunluğuna kadar dolgu token'ı eklenmesi.

Attention Mask: Modelin öz-dikkat (self-attention) mekanizmasında dolgu token'larını yok saymasını sağlayan ikili (0/1) maskeleme tensörü.

4. Yerel Kayıt Döngüsü
Eğitilen veya indirilen modellerin çevrimdışı kullanım için saklanması:

tokenizer.save_pretrained("./my_model")

model.save_pretrained("./my_model")

Bu işlem hem model katman ağırlıklarını (model.safetensors) hem de model konfigürasyonunu (config.json, vocab.txt) hedeflenen klasöre yazar.

Çıktı Örneği
Logits değerlerinin olasılık skorlarına dönüştürülmesi:

Plaintext
Girdi Metni: "This course is useful."
Ham Logits: [[-3.12, 3.45]]
Softmax Sonrası Olasılıklar: [Negatif: 0.0014, Pozitif: 0.9986]
Tahmin Edilen Sınıf: POSITIVE (%99.86)
Lisans
Bu çalışma eğitim amaçlıdır. Kullanılan önceden eğitilmiş model ağırlıkları Hugging Face Hub üzerindeki orijinal Apache 2.0 lisansına tabidir.