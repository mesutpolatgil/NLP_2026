# Gerçek Pretrained Koşum Sonuçları

31 Ekim 2026. Python 3.12.14, PyTorch 2.9.0+cpu, Transformers 4.57.1,
Tokenizers 0.22.1, CPU, seed 7. Aşağıdaki değerler gerçek çalıştırmadan
alındı. Bunlar küçük gösterim girdileridir; bağımsız benchmark değildir.

| Görev | Gerçek örnek çıktı |
|---|---|
| NER | Wolfgang / PER, Berlin / LOC, Microsoft / ORG |
| MLM | capital; skor 0.981547 |
| Çeviri | Das Haus ist wunderbar. |
| Özetleme | the library opened a new reading room on Monday. the room has 40 seats and is open to all students. |
| QA | Ankara University; [26,43) |
| Sohbet enjeksiyonu | Ankara / Ankara / Ankara |
| Loss hedef sayısı | 2 / 4 / 2 |

## Causal modelin gerçek devamı

```text
Natural language processing is a common feature of the Internet.
The term "language processing" has been used to describe many languages, but it does not mean that there are no native speakers in all these countries and therefore they cannot be spoken by any
```

## Sabit kaynak sürümleri

| Görev | Model | Revision |
|---|---|---|
| ner | `dslim/bert-base-NER` | `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` |
| mlm | `distilbert/distilbert-base-uncased` | `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| seq2seq | `google-t5/t5-small` | `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` |
| causal | `distilbert/distilgpt2` | `2290a62682d06624634c1f46a6ad5be0f47f38aa` |
| qa | `distilbert/distilbert-base-cased-distilled-squad` | `564e9b582944a57a3e586bbb98fd6f0a4118db7f` |
| chat | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

## Yorum ve doğrulama

NER varlık aralıkları, MLM top-k skorları, T5 kaynak/hedef tokenizasyonu,
gerçek çeviri/özet, causal devam ve QA cevap aralığı çalıştırıldı.
Sohbet modelinde gerçek rol şablonu kullanıldı. Talimat enjeksiyonunun
etkisiz kalması da sonuçtur; saldırı başarısı uydurulmadı. QA bağlam
zehirleme, özetlemede olgusallık ve uzun pencere kayıtları red-team raporundadır.

Notebook, dış main.py modülünü içe aktarmayan bağımsız kod hücreleriyle
yerelde yürütüldü; kodlar, HTML tabloları ve gerçek skor grafiği kaydedildi.
Colab arayüzünde ayrıca koşum yapılmadı. İlk kurulum hücresi yerelde
tekrar çalıştırılmadı; ortam zaten kuruluydu.

Model fine-tuning, bağımsız test kümesi, entity F1, BLEU/ROUGE ve genel
güvenlik oranı ölçülmedi. Model confidence skorları doğruluk garantisi değildir.

Kaynak main.py SHA256: `ab8c004d28279d0ff5cc10a5e1252f587135df7e52375ed24f630d5065378423`
