# Bölüm 12.5 — Küçük modelle GRPO uygulaması

Bu uygulama, SmolLM2-135M-Instruct modelini GRPO + LoRA ile 50 token uzunluğunda çıktı üretmeye yönlendirir. Kaynaktaki karakter/token karışıklığı düzeltilmiştir.

## Dosyalar

- **GRPO_12_5_Uygulama.ipynb**: açıklamalı Jupyter / Google Colab defteri.
- **train_grpo.py**: CPU veya GPU üzerinde eğitim, önce/sonra ölçüm ve kayıt.
- **modeli_dene.py**: kaydedilen adaptörü yeniden yükleyerek metin üretme.
- **requirements.txt**: uygulamanın temel bağımlılık sürümleri.
- Eğitim tamamlandıysa **sonuclar/** içinde adaptör, ham yanıtlar, eğitim günlüğü ve ölçümler bulunur.

## Google Colab

1. Colab'da **Dosya → Not defteri yükle** üzerinden ipynb dosyasını yükleyin.
2. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU** seçin.
3. Hücreleri sırayla çalıştırın. GPU ile varsayılan deneme 60 adımdır.
4. Son hücre, adaptörü ve sonuçları ZIP olarak indirir.

Defterin kod hücreleri sözdizimi açısından kontrol edilmiştir. Colab oturumu bu bilgisayardan çalıştırılmamıştır.

## Yerel kullanım

Python 3.12 veya 3.13 ve uygun PyTorch kurulumu gerekir. Ayrı bir sanal ortam önerilir.

    python -m pip install -r requirements.txt

PyTorch ayrıca kurulmalıdır. CPU için:

    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

GPU kurulumu için [resmî PyTorch kurulum seçicisini](https://pytorch.org/get-started/locally/) kullanın.

Kısa CPU denemesi:

    python train_grpo.py --device cpu --steps 12 --train-samples 32 --eval-samples 8

GPU denemesi:

    python train_grpo.py --device cuda --steps 60 --train-samples 256 --eval-samples 24

Kaydedilen modeli deneme:

    python modeli_dene.py --prompt "Summarize briefly: A cat slept by the window."

Bu çalışma alanında CPU ortamı, çalışma klasörünün work/grpo_cpu_env dizinindedir. Global Python paketleri değiştirilmeden ayrı sanal ortama eğitim bağımlılıkları eklenmiştir.

## Ne ölçülüyor?

Ödül = −|üretilen token sayısı − 50|. En iyi ödül sıfırdır. Özel tokenlar sayılmaz. Eğitim ve test istemleri ayrıdır; önce/sonra karşılaştırmasında aynı örnekleme tohumları kullanılır. Üretim üst sınırı 96 yeni tokendır; bu sınıra ulaşma oranı da raporlanır.

Bu alıştırma uzunluk kontrolünü öğretir. Daha iyi özet, daha doğru bilgi veya daha iyi akıl yürütme elde edildiğini tek başına göstermez. Küçük model ve kısa eğitim nedeniyle sonuçlar değişken olabilir.

İstemler özgün İngilizce veri kümesinden alınır. Türkçe açıklamalar, İngilizce model/veriyle yapılan deneyi anlatır. Veri kümesi kamuya açık Reddit metinleri içerir.

W&B'ye giriş yapılmaz, uzaktaki bir hizmete eğitim kaydı gönderilmez ve model Hub'a yayımlanmaz. İlk çalıştırmada model/veri indirmek için internet gerekir. Tekrar çalıştırma aynı sonuclar klasöründeki ölçümleri ve adaptörü yeniler.

## Kaynaklar

- [Hugging Face — Ders 12.5](https://huggingface.co/learn/llm-course/chapter12/5)
- [TRL 0.16.1 — GRPOTrainer](https://huggingface.co/docs/trl/v0.16.1/grpo_trainer)
- [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [mlabonne/smoltldr veri kümesi](https://huggingface.co/datasets/mlabonne/smoltldr)


## Bu paketin gerçek çalıştırma durumu

CPU üzerinde 12 GRPO adımı tamamlandı; adaptör yeniden yüklendi ve metin üretildi. 8 ayrı test isteminde ortalama mutlak uzunluk hatası 42'den 41,625'e düştü. Bu küçük fark, 50 token hedefinin öğrenildiği anlamına gelmez. Ayrıntılar SONUC_RAPORU.md dosyasındadır.
