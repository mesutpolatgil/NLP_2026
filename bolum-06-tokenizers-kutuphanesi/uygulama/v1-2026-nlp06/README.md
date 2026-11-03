> **Uyarı:** Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.

# Bölüm 6 Uygulaması: 🤗 Tokenizers ile Türkçe Tokenizer İnşası

Bu klasördeki `main.ipynb`, Hugging Face LLM Course **Bölüm 6 (The 🤗 Tokenizers library)** içeriğinin tamamını **Türkçe bir derlem** üzerinde uygular. Kursun giriş kısmındaki temel fikir şudur: Bir dil için eğitilmiş tokenizer başka bir dilde verimsiz çalışır. Bu fikri Türkçe gibi eklemeli (agglutinative) bir dille sınıyoruz.

## İçerik

| # | Kısım | Kursta karşılığı | Ne yapılıyor? |
|---|---|---|---|
| 1 | Kurulum | – | Paketlerin kurulumu, sürüm kontrolü |
| 2 | Türkçe derlem ve üreteç | 6.2 | 7.803 Türkçe cümlenin indirilmesi, 1.000'lik yığınlar veren `get_training_corpus()` |
| 3 | Normalizasyon ve ön-tokenizasyon | 6.5 | `BertNormalizer` ile Türkçe normalizer karşılaştırması; BERT, GPT-2 ve SentencePiece ön-tokenizer'ları |
| 4 | Sıfırdan WordPiece | 6.7, 6.9 | Model → normalizer → ön-tokenizer → trainer → post-processor → decoder |
| 5 | Sıfırdan BPE | 6.6, 6.9 | Bayt düzeyinde BPE (GPT-2 tarzı), emoji testi |
| 6 | Sıfırdan Unigram | 6.8, 6.9 | Metaspace + Unigram (XLNet / SentencePiece tarzı) |
| 7 | Karşılaştırma | 6.5 | Doğurganlık (kelime başına token), `[UNK]` oranı, uzun kelime bölünmesi, grafik |
| 8 | Hızlı tokenizer'ın süper güçleri | 6.3 | `offsets`, `word_ids`, `word_to_chars`, `char_to_token` |
| 9 | 🤗 Transformers'a sarmalama | 6.4, 6.9 | `PreTrainedTokenizerFast`, padding, `stride` ile örtüşmeli parçalama, kaydetme ve yeniden yükleme |
| 10 | (Opsiyonel) GPT-2'den yeniden eğitim | 6.2 | `train_new_from_iterator()` ile İngilizce GPT-2 tokenizer'ını Türkçe derlemde yeniden eğitme |

## Öne çıkan sonuçlar

Aynı Türkçe derlemde, aynı kelime dağarcığı boyutuyla (8.000) eğitilen üç tokenizer'ın Türkçe test kümesindeki sonuçları:

| Algoritma | Doğurganlık (token / kelime) | `[UNK]` oranı | "Çekoslovakyalılaştıramadıklarımızdanmışsınız" |
|---|---|---|---|
| WordPiece | 1,87 | %0,00 | 14 parça |
| BPE (bayt düzeyinde) | 1,99 | %0,00 | 13 parça |
| Unigram | 1,97 | %0,00 | 14 parça |

Üç algoritma da Türkçe ekleri (`-larımız`, `-dan`, `-mış`, `-sınız`) ayrı parçalar olarak öğrenir:

```
öğretmenlerimizden → WordPiece: ['öğretmen', '##lerimiz', '##den']
                     Unigram  : ['▁öğretmen', 'lerimiz', 'den']
```

Standart Unicode küçük harf kuralının Türkçe hatası ve düzeltmesi:

```
Orijinal         : IŞIK ve İSTANBUL'da
BertNormalizer   : isik ve istanbul'da    ← "ışık" yanlış, aksanlar silinmiş
NFKC + Lowercase : işik ve i̇stanbul'da    ← "İ" iki karaktere (i + U+0307) dönüşmüş
Türkçe normalizer: ışık ve istanbul'da    ← doğru
```

> Bu tokenizer'ların nasıl kırılabileceği (İngilizce tokenizer'ın Türkçede çökmesi, harf çakışmaları, görünmez karakter saldırıları, offset kayması) `../../red-team/` klasöründeki rapor ve notebook'ta ölçülmüştür.

## Kullanılan Teknolojiler

- **Python 3.9+**
- **🤗 Tokenizers:** WordPiece, BPE ve Unigram modelleri, normalizer'lar, ön-tokenizer'lar, trainer'lar
- **🤗 Transformers:** `PreTrainedTokenizerFast`, `AutoTokenizer`
- **Matplotlib:** karşılaştırma grafiği

## Veri

[UD Turkish-BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN) ağaç bankasının eğitim (7.803 cümle) ve test (979 cümle) kümeleri kullanılmıştır. Lisansı **CC BY-SA 4.0**'dır. Dosyalar ilk çalıştırmada GitHub'dan `veri/` klasörüne indirilir. Hugging Face hesabı ya da `datasets` kütüphanesi gerekmez.

## Kurulum ve Çalıştırma

### Google Colab
Notebook'u Colab'a yükleyip **Çalışma Zamanı → Tümünü Çalıştır** seçmeniz yeterlidir. İlk hücre gerekli paketleri kurar.

### Lokal (Jupyter)

```bash
python -m venv venv
# Windows: venv\Scripts\activate    |   Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook main.ipynb
```

- 1–9. kısımlar internet olmadan da çalışır (yalnızca ilk çalıştırmada derlem indirilir).
- 10. kısım, GPT-2 tokenizer'ını Hugging Face Hub'dan indirir. Hub'a erişim yoksa hücre hata vermeden atlanır.
- Notebook'un tamamı normal bir dizüstü bilgisayarda yaklaşık **15 saniyede** çalışır; GPU gerekmez.

## Test

Notebook lokal ortamda `jupyter nbconvert --execute` ile baştan sona çalıştırılarak test edilmiştir (Python 3.13, tokenizers 0.23, transformers 5.21). Hiçbir hücre hata vermemektedir.

## Çıktılar

Notebook çalıştırıldığında şunlar oluşturulur:

- `veri/` – indirilen derlem dosyaları
- `turkce-wordpiece-tokenizer/` – 🤗 Transformers ile yeniden yüklenebilen Türkçe WordPiece tokenizer (`tokenizer.json`, `tokenizer_config.json`)
