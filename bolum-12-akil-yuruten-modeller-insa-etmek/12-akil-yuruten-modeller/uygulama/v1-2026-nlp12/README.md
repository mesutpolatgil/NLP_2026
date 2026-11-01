# GRPO ile 50 Token Hedefli İnce Ayar

Bu klasör, Hugging Face LLM Course Bölüm 12.5'teki GRPO uygulamasının yerel GPU ortamında çalıştırılmış sürümünü içerir.

## Amaç

Önceden eğitilmiş `SmolLM2-135M-Instruct` modelinin cevaplarını GRPO ve LoRA kullanarak yaklaşık 50 token uzunluğuna yönlendirmek.

Bu çalışma cevapların doğruluğunu ya da akıl yürütme becerisini ölçmez. Başarı ölçütü, cevap uzunluğunun 50 token hedefine uzaklığıdır.

## Kullanılan bileşenler

| Bileşen | Değer |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Veri kümesi | `mlabonne/smoltldr` |
| Eğitim örneği | 2.000 |
| Değerlendirme | Eğitimde kullanılmamış 10 doğrulama örneği |
| İnce ayar | LoRA |
| Eğitim yöntemi | GRPO |
| Hedef | 50 token |
| Girdi üst sınırı | 512 token |
| Cevap üst sınırı | 96 token |
| Eğitim | 1 epoch, 1.000 adım |

## Ödül fonksiyonu

```text
Ödül = -|cevap token sayısı - 50|
```

Ödülün sıfıra yaklaşması, üretilen cevabın 50 token hedefine yaklaştığını gösterir.

## Kurulum

Python 3.10 veya üzeri ve CUDA destekli bir NVIDIA GPU önerilir.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

Veri kümesini, girdi uzunluklarını ve ilk iki girdiyi incelemek için:

```bash
python grpo_egitimi.py
```

Eğitimi başlatmak için:

```bash
python grpo_egitimi.py --train
```

İlk çalıştırmada model ve veri kümesi Hugging Face üzerinden indirilir. Eğitim çıktıları `calisma_ciktilari/` klasörüne kaydedilir. Bu klasör model ağırlıkları ve checkpoint içerdiği için Git'e eklenmez.

## Sonuç

Sonuçların ayrıntısı `sonuclar.md` dosyasındadır. Eğitim sonrası 50 token hedefine ortalama uzaklık 43,2 token'dan 9,5 token'a düşmüştür.

## 12.6: Unsloth ile GRPO (Google Colab)

`unsloth-12-6/gemma3_1b_grpo_colab.ipynb`, kursun 12.6 sayfasında bağlantısı verilen Unsloth not defterinin Google Colab'da (Tesla T4 GPU) çalıştırılmış hâlidir. Bütün hücre çıktıları dosyanın içinde kayıtlıdır.

| Bileşen | Değer |
|---|---|
| Model | `unsloth/gemma-3-1b-it` + LoRA (r = 8) |
| Veri kümesi | `openai/gsm8k` (7.473 matematik problemi) |
| Ödül fonksiyonları | `match_format_exactly`, `match_format_approximately`, `check_answer`, `check_numbers` |
| Eğitim | 50 adım, her soru için 4 üretim |

Not: Unsloth'un güncel not defteri, kurs sayfasındaki koddan farklıdır (50 adım, `<start_working_out>` / `<SOLUTION>` etiketleri ve farklı ödül fonksiyonları). Çalıştırılan sürüm budur.

**Sonuçlar (eğitim günlüğünden):**

| Ölçüm | İlk 10 adım ort. | Son 10 adım ort. |
|---|---:|---:|
| Toplam ödül | 0,19 | 0,48 |
| Ortalama cevap uzunluğu | 507 token | 346 token |
| Üretim sınırına takılan cevap oranı | %35 | %10 |

- `check_numbers` ödülü 50 adımın 33'ünde pozitif oldu, yani model bazı sorularda doğru sayıyı buldu.
- `match_format_exactly` ve `check_answer` 50 adım boyunca hiç ödül vermedi: model istenen formatı (`<end_working_out>` ve `</SOLUTION>` dahil) tam olarak hiç tutturamadı.
- KL değeri hep 0'dır, çünkü bu sürümde varsayılan `beta = 0.0`.
- Kurs sayfası iyileşmenin 150–200 adımdan sonra görülebileceğini söyler. 50 adım bu açıdan kısa bir denemedir.

## İlk deneme (CPU)

`ilk-deneme-cpu/` klasörü, asıl eğitimden önce 14 Ekim 2026'da CPU üzerinde yapılan kısa denemeyi içerir: 12 adım, 32 eğitim örneği, 8 test örneği. Bu kısa eğitimde hedefe ortalama uzaklık yalnızca 42,0'dan 41,6 token'a indi. Sonuç, GPU'da 1.000 adımlık asıl eğitime (`grpo_egitimi.py`, `sonuclar.md`) geçilmesinin nedenini gösterir. Ayrıntılar `ilk-deneme-cpu/SONUC_RAPORU.md` dosyasındadır. Eğitilmiş LoRA ağırlıkları boyutu nedeniyle eklenmemiştir.

## Kurs kodları (12.3b ve 12.4)

`kurs-kodlari/` klasörü, kursun 12.3b ve 12.4 sayfalarındaki kod örneklerini ve bunların çalıştırma çıktılarını içerir. Bu kodlar GPU gerektirmez.

| Dosya | Sayfa | Ne yapar |
|---|---|---|
| `s3b_1_avantaj.py` | 12.3b | Grup ödüllerinden avantaj hesaplama |
| `s3b_2_teori_ornegi.py` | 12.3b | Teorideki 8 cevaplık avantaj örneği |
| `s3b_3_kirpma.py` | 12.3b | Olasılık oranı ve kırpma (ε = 0,2) |
| `s3b_4_politika_guncelleme.py` | 12.3b | Kırpılmış hedef ve KL cezası ile politika kaybı |
| `s4_1_odul_fonksiyonlari.py` | 12.4 | Uzunluk, problem ve format tabanlı ödül fonksiyonları |
| `s4_2_grpoconfig.py` | 12.4 | `GRPOConfig` parametre kontrolü |
| `s4_3_minimal_trainer.py` | 12.4 | Minimal `GRPOTrainer` örneği (Hugging Face Hub erişimi ve GPU gerekir) |
| `grafikler.py` | 12.3b–12.4 | `grafikler/` klasöründeki grafikleri üretir |

```bash
pip install torch trl matplotlib numpy
cd kurs-kodlari
python s3b_1_avantaj.py
python grafikler.py
```

Çıktılar `kurs-kodlari/ciktilar/` klasöründedir.

`s4_3_minimal_trainer.py` dışındaki tüm dosyalar çalıştırıldı ve çıktıları kontrol edildi. `s4_3` Hugging Face Hub'dan model ve veri indirdiği için burada çalıştırılmadı; aynı veri kümesiyle tam GRPO eğitimi `grpo_egitimi.py` ile yapıldı.

## Kaynak

Hugging Face LLM Course, Bölüm 12.5: [Practical Exercise: Fine-tune a model with GRPO](https://huggingface.co/learn/llm-course/chapter12/5)
