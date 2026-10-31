# Bölüm 7 Yerel NLP Uygulaması

**Takım:** NLP07 · **Ekip:** Buğrahan Budakoğlu, Berk Talha Aslan, Umut Türker, Mert Ali Alkan

Bu uygulama altı NLP görevinin girdi, hedef, eğitim ve çıktı akışını gerçek
Hugging Face Transformer sınıflarıyla gösterir. 1 katmanlı, 32 boyutlu küçük
modeller rastgele başlatılır; 2-4 insan yazımı örnekle 100 adım eğitilir.
Önceden eğitilmiş model veya veri kümesi indirilmez. Kurulumdan sonra çevrimdışı
çalışır, GPU ve Hugging Face hesabı gerekmez; Hub'a yükleme yapmaz.

## Dosyalar

| Dosya | İşlev |
|---|---|
| `main.py` | Altı model eğitimi, çıkarım ve beş hata deneyi |
| `main.ipynb` | Aynı modülü kullanan, yerel çıktıları kaydedilmiş ders akışı |
| `requirements.txt` | Doğrulanan doğrudan bağımlılıklar |
| `sonuclar.md` | Gerçek terminal koşumu, ortam ve sonuç tablosu |

## Kurulum ve çalıştırma

Python 3.12 önerilir; bu koşum Python 3.12.14 / Windows CPU ile yapıldı.
Uygulama klasöründe terminal açın:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --task all --steps 100
.\.venv\Scripts\python.exe main.py --task red-team
```

Linux/macOS: sanal ortam için `python3 -m venv .venv`, Python yolu olarak
`.venv/bin/python` kullanın. PyTorch tekerleğinin platformunuza uygun olması gerekir.
Notebook için ayrıca `python -m pip install jupyter` kurup `python -m jupyter notebook main.ipynb`
çalıştırın. Jupyter'de bu sanal ortamın kernelini seçin. Temel teslim testi
`python main.py` komutudur; Jupyter temel betik için zorunlu değildir.

## Kaynak sırasıyla uygulama

| Bölüm | Görev | Model sınıfı | Hedef |
|---|---|---|---|
| 7.2 | NER | `BertForTokenClassification` | Token etiketi; özel konumlar -100 |
| 7.3 | MLM | `BertForMaskedLM` | Yalnız maskeli şehir tokenı |
| 7.4 | Çeviri | `BartForConditionalGeneration` | Eşleşen Türkçe cümle |
| 7.5 | Özetleme | Ayrı `BartForConditionalGeneration` | Kısa referans cümlesi |
| 7.6 | Causal LM | `GPT2LMHeadModel` | İçeride bir token kaydırılmış hedef |
| 7.7 | Extractive QA | `BertForQuestionAnswering` | Başlangıç/bitiş tokenı |

Örnek: `python main.py --task qa --steps 100`. Diğer seçenekler: `ner`, `mlm`,
`translation`, `summarization`, `causal`, `red-team`, `all`.
`--output sonuc.json` verilirse terminaldeki JSON ayrıca belirtilen dosyaya yazılır.

## Gerçek yerel sonuçlar

Seed 7, CPU, 2 iş parçacığı; her model 100 adım. Bütün görevlerde ileri geçiş,
geri yayılım, optimizer adımı ve değerlendirme/çıkarım çalıştırıldı.

| Görev | Parametre | İlk kayıp | Son kayıp |
|---|---:|---:|---:|
| Token sınıflandırma | 14,371 | 1.072433 | 0.000263 |
| Maskeli dil modelleme | 15,439 | 3.849792 | 0.002862 |
| Çeviri | 31,328 | 3.814235 | 0.001912 |
| Özetleme | 31,328 | 3.784526 | 0.001683 |
| Nedensel dil modelleme | 18,368 | 3.871957 | 0.232638 |
| Soru yanıtlama | 14,338 | 2.508127 | 0.000136 |

Kaybın düşmesi uygulamanın öğrenme akışını gösterir. Bu tablo aynı eğitim
örneklerinde hesaplanmıştır; test kümesi, gerçek Türkçe başarısı veya
pretrained modelle karşılaştırma değildir. Exact match/token accuracy alanları
da yalnız eğitim örnekleri içindir. BLEU, ROUGE ve entity-level F1 hesaplanmadı.

## Hata ve düzeltme deneyleri

1. `-100` yerine gerçek `O=0`: kayba katılan hedef sayısı 2, 4, 2.
2. QA bağlam kırpma: cevap ilk pencerede yok; stride ile 8 pencerenin 2'sinde var.
3. Gerçek WordPiece: `Ankaralı` iki parçaya ayrılır; `word_ids` ile etiket hizalanır.
4. İnsan yazımı bozuk Python: `SyntaxError`; iki nokta eklenince ayrıştırma geçer.
5. Olumsuzluğu silinen cümlede sözcük örtüşmesi 0.833333 kalır; anlam tersine döner.

Kontroller başarısızsa betik hata koduyla çıkar. Deney ayrıntıları
[red-team raporundadır](../../red-team/2026-guz-nlp07.md).

## Ders örneğinden farkı ve sınırlar

Kaynak kurs büyük veri üzerinde pretrained modellerle ince ayar gösterir.
Bu yerel laboratuvar aynı hedef yapılarını düşük kaynakla öğretmek için
tasarlanmıştır; bütün kurs eğitimlerinin yeniden üretimi değildir. Ana tokenizer
yerel WordLevel'dır; gerçek alt kelime deneyi ayrıca WordPiece ile yapılır.
MLM'de maske konumu sabittir; dinamik maskeleme genellemesi iddia edilmez.
Veri lisansı sorunu yaratmamak için örnekler insan yazımıdır.

Veri bölme, bağımsız test, entity F1, BLEU/ROUGE, model kartı ve pretrained
karşılaştırması sonraki gerçek proje aşamasıdır. Kaynak çevirisindeki TensorFlow
ve uzun eğitim kodları bu uygulamanın çalıştırıldığı anlamına gelmez.

## Kaynaklar

[HF LLM Course Bölüm 7](https://huggingface.co/learn/llm-course/en/chapter7/1) ·
[BERT](https://huggingface.co/docs/transformers/model_doc/bert) ·
[BART](https://huggingface.co/docs/transformers/model_doc/bart) ·
[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2) ·
[Tokenizers](https://huggingface.co/docs/tokenizers)
