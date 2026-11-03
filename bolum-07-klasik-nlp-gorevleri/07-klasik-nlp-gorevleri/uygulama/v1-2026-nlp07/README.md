# Bölüm 7 Gerçek Pretrained NLP Uygulamaları

**Takım:** NLP07 · 2026 Güz

| Kişi | Rol |
|---|---|
| Buğrahan Budakoğlu | Dokümantasyon Yöneticisi |
| Berk Talha Aslan | Sistem Mimarı |
| Umut Türker | Sınav Komiseri |
| Mert Ali Alkan | QA & Adli Bilişim Uzmanı |

Uygulama altı klasik NLP görevini gerçek pretrained modellerle gösterir.
Sıfırdan başlatılmış mini model veya ezberletilmiş dört satırlık eğitim verisi
ana gösterimde kullanılmaz. Bu koşum çıkarımdır; fine-tuning yapıldığı iddia edilmez.
Model örnekleri İngilizce, çeviri yönü İngilizce-Almancadır; model kartlarının
dil sınırları korunur. Türkçe bir ürün için ayrıca uygun checkpoint ve değerlendirme gerekir.

## Dosyalar

| Dosya | İşlev |
|---|---|
| `main.ipynb` | Bağımsız, Colab uyumlu, kodların içinde göründüğü açıklamalı notebook |
| `main.py` | Aynı görevlerin terminalden çalıştırılan karşılığı |
| `deney_terminali.py` | Mert'in gerçek kontrol/bozma/yeniden test çıktıları |
| `requirements.txt` | Kullanılan API ailesini koruyan sürüm aralıkları |
| `requirements-tested.txt` | Gerçekte doğrulanan ortamın sürümleri |
| `sonuclar.md` | Okunabilir gerçek koşum raporu; uzun JSON yerine tablo ve seçilmiş çıktı |

## Modeller

| Görev | Checkpoint | Gösterilen işlem |
|---|---|---|
| NER | `dslim/bert-base-NER` | Token sınıflandırma ve varlık birleştirme |
| MLM | `distilbert/distilbert-base-uncased` | Maske için top-k tokenlar ve skorlar |
| Çeviri | `google-t5/t5-small` | Görev öneki, `text_target`, `generate` |
| Özetleme | `google-t5/t5-small` | Kaynak belge, gerçek özet, enjeksiyon karşılaştırması |
| Causal LM | `distilbert/distilgpt2` | Gerçek otoregresif devam ve kurgu bilgi deneyi |
| Extractive QA | `distilbert/distilbert-base-cased-distilled-squad` | Cevap aralığı, çelişkili bağlam ve taşan pencereler |
| Ek sohbet güvenliği | `Qwen/Qwen2.5-1.5B-Instruct` | Gerçek chat template ile system/user rol deneyi |

## Kurulum

Python 3.12 önerilir. Diğer Python/platform sürümleri bu çalışmada ayrıca
test edilmedi. İlk çalıştırma yaklaşık 1.5 GB temel model ağırlığı indirir;
ek sohbet modeli yaklaşık 3 GB daha gerektirir. Yerelde CPU ve yeterli RAM gerekir.
GPU zorunlu değildir. Model ağırlıklarını ve sanal ortamı Git'e eklemeyin.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --task ner --output ner-sonuc.json
.\.venv\Scripts\python.exe main.py --task qa --output qa-sonuc.json
```

Linux/macOS: `python3 -m venv .venv`, Python yolu `.venv/bin/python`.
Tamamını çalıştırmak için `python main.py --task all --output sonuclar.json`.
Bu seçenek ek sohbet modelini de içerir. Tek görev seçenekleri:
`ner`, `mlm`, `seq2seq`, `causal`, `qa`, `chat`, `loss`.

Notebook: Jupyter ortamında `main.ipynb` dosyasını açın veya Colab'a yükleyin.
İlk hücre paketleri kurar. Notebook başka bir `.py` dosyasını içe aktarmadan
çalışır. Kaynak kodlar, tokenlar, varlık tabloları ve MLM skor grafiği görünürdür.

## Yerel görev deneylerini yeniden üretme

```powershell
python deney_terminali.py --case qa --output qa-kanit.json
python deney_terminali.py --case summary --output ozet-kanit.json
python deney_terminali.py --case causal --output causal-kanit.json
python deney_terminali.py --case chat --output sohbet-kanit.json
python deney_terminali.py --case loss --output loss-kanit.json
```

Her komut modeli gerçekten çalıştırır; eski çıktı metnini tekrar ekrana basmaz.
JSON'da `main.py` dosyasının SHA256 değeri kaydedilir. Bu komutlar yerel
Hugging Face görev modellerini çalıştırır.

Üç sohbet modeliyle yapılan yeni karşılaştırmalı red-team deneyleri ayrı,
bağımsız [Colab notebookunda](../../red-team/2026-guz-nlp07.ipynb) bulunur.
Kodlar ve gerçek API çıktıları notebook içinde kayıtlıdır; kayıtlı kipte anahtar
gerekmez. Yeni API çağrıları için `CANLI_CALISTIR` kutusunu açın; bu kip kendi
Groq API anahtarınızı ve internet erişimini gerektirir. Senaryolar ve bulgular [red-team rehberinde](../../red-team/2026-guz-nlp07.md)
tek dosyadan anlatılabilir.

## Önbellekten ve çevrimdışı çalışma

Varsayılan yükleme model kimliğini Hub'dan alır. Sınıf internetine bağımlı
kalmamak için modelleri önceden indirin. İsteğe bağlı `NLP07_MODEL_DIR`
ortam değişkeni, örneğin `dslim--bert-base-NER` alt klasörlerini içeren
yerel model dizinini gösterebilir. Sohbet modeli için `NLP07_CHAT_MODEL_DIR`
ayrı yerel snapshot yolu kabul eder. Bunlar kurulum kolaylığıdır;
başka bilgisayardaki kullanıcıya özgü yollar kaynakta sabitlenmez.
Sabit checkpoint revision bilgileri gerçek sonuç kaydında yer alır.

## Kapsam ve yorum

NER için görev başlığı eğitilmiş checkpoint kullanılır. Bir base BERT'in
adını NER pipelineına vermek, başlığın eğitilmiş olduğunu kanıtlamaz.
QA bağlam zehirleme ile sohbet modelinde rol güvenliği farklı deneylerdir.
Etkisiz saldırılar da aynen raporlanır. Uydurulmuş kişiyle alınan devam
tarihsel bilgi olarak sunulmaz. Özetin bilgi atlaması ile kaynağa aykırı
iddia üretmesi ayrı incelenir. Model skorları doğruluk garantisi değildir.

Bu uygulamada bağımsız benchmark, BLEU/ROUGE ve entity-level F1 ölçümü
yapılmadı. Kaynak kursun uzun ince ayar ve sıfırdan ön eğitim kodları
çeviride bulunur; buradaki pretrained çıkarım onların tamamının yeniden üretimi değildir.

## Kaynaklar

### Sunum dosyaları

[PowerPoint sunumu](../../sunum/2026-guz-nlp07.pptx) · [PDF sunumu](../../sunum/2026-guz-nlp07.pdf)

108 sayfalık akış: Buğrahan 1–22, Berk 23–61, Umut 62–88, Mert 89–108.
Kaynak Bölüm 7’nin dokuz başlığı ve 13 resmî quiz sorusu sırasıyla işlenir.
Diğer 15 özgün soru, soru-cevap dosyasında korunur; ana sunumda tekrarlanmaz.
Mert’in deneyleri, kaynak ders tamamlandıktan sonra ayrı ekip uygulaması olarak
sunumda ve kayıtlı çıktılı Colab notebookunda bulunur.

Sunumdaki kaynak eğitim kodları ve kaynak sonuçları dersin Türkçe anlatımıdır.
Bu yerel uygulama hazır modellerle çıkarım yapar; kaynaktaki uzun eğitimlerin
bu revizyonda yeniden çalıştırıldığı iddia edilmez.

### Teknik kaynaklar

[HF Bölüm 7](https://huggingface.co/learn/llm-course/en/chapter7/1) ·
[NER](https://huggingface.co/dslim/bert-base-NER) ·
[DistilBERT MLM](https://huggingface.co/distilbert/distilbert-base-uncased) ·
[T5-small](https://huggingface.co/google-t5/t5-small) ·
[DistilGPT-2](https://huggingface.co/distilbert/distilgpt2) ·
[SQuAD QA](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad) ·
[Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
