Uyarı: Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.

# Bölüm 9: Demo Oluşturma ve Paylaşma (Building and Sharing Demos)

**Kaynak Bağlantısı:** [Hugging Face Course - Building and Sharing Demos](https://huggingface.co/learn/nlp-course/chapter9/1)
**Takım Bilgisi:** RotaNLP

---

## Giriş

> Not: Kod örnekleri, Hugging Face kursunun bu bölümünde kullanılan Gradio API yaklaşımını yansıtır. Gradio sürümüne göre bazı parametrelerin kullanımı değişebilir.

Hugging Face LLM Kursunun bu bölümünde, Doğal Dil İşleme (NLP - Natural Language Processing) modellerinin ötesine geçerek makine öğrenimi uygulamalarının etkileşimli demolarını oluşturma ve dünyayla paylaşma süreçleri incelenmektedir. Kursun ilk bölümlerinde 🤗 Transformers, 🤗 Datasets ve 🤗 Tokenizers kütüphaneleri ile model eğitme ve ince ayar (fine-tuning) süreçleri ele alınmıştır. 9. Bölüm ise geliştirilen modelleri Gradio kütüphanesi vasıtasıyla kullanıcı dostu grafik arayüzlerine (GUI - Graphical User Interface) dönüştürmeyi ve Hugging Face Spaces üzerinde kalıcı olarak barındırmayı hedefler.

---

## 1. İlk Demo Sürümünü Oluşturma

Gradio, makine öğrenimi modelleri, veri bilimsel fonksiyonlar ve API'ler için hızlıca etkileşimli web arayüzleri oluşturmayı sağlayan açık kaynaklı bir Python paketidir.

### Kurulum

Gradio kütüphanesini yerel Python ortamınıza, Jupyter Notebook veya Google Colab üzerine kurmak için `pip` paket yöneticisi kullanılır:

```bash
pip install gradio
```

### Temel Merhaba Dünya Örneği
Gradio sözdizimine aşina olmak için basit bir metin işleme fonksiyonu tanımlanabilir ve gr.Interface yapısı ile arayüze dönüştürülebilir:

```python
import gradio as gr

def greet(name):
    return "Merhaba " + name

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

### Kodun İncelemesi:

greet(name): Kullanıcıdan name string parametresini alan ve metnin önüne "Merhaba " ekleyerek döndüren temel Python fonksiyonudur.

gr.Interface: Tahmin fonksiyonunu (fn), giriş bileşen türünü (inputs) ve çıkış bileşen türünü (outputs) tanımlayan ana sınıftır.

demo.launch(): Yerel bir web sunucusu başlatır. Kod bir Jupyter/Colab not defterinde çalıştırılıyorsa arayüz doğrudan hücre altında, bir Python betiğinden çalıştırılıyorsa varsayılan olarak http://localhost:7860 adresinde açılır.

### Arayüz Bileşenlerini Özelleştirme
Gradio, fonksiyon parametre adlarından (örneğin name) otomatik etiketler türetir. Metin kutularını özelleştirmek ve gelişmiş parametreler eklemek için dize kısayolları (string shortcuts) yerine bileşen sınıfları örneklendirilir:

```python
import gradio as gr

def greet(name):
    return "Merhaba " + name

# Textbox sınıfı örneklendiriliyor
textbox = gr.Textbox(label="Adınızı buraya yazın:", placeholder="John Doe", lines=2)

gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()
```

## 2. Model Tahminlerini Arayüze Dahil Etme
Gradio fonksiyonları, yapay zeka modellerini çağırarak gerçek zamanlı çıkarım (inference) yapabilir. 🤗 Transformers kütüphanesinin pipeline() fonksiyonu ile GPT-2 metin üretimi modeli arayüze bağlanabilir:

```python
from transformers import pipeline
import gradio as gr

# Metin üretimi hattını yükleme
model = pipeline("text-generation")

def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
```

## 3. Interface Sınıfını Anlama
gr.Interface sınıfı, makine öğrenimi modellerini hızlıca sarmalamak için 3 zorunlu parametre gerektirir:

`gr.Interface(fn, inputs, outputs)`

| Parametre | Açıklama |
|---|---|
| `fn` | Gradio arayüzü tarafından sarmalanan tahmin/işlem fonksiyonudur. |
| `inputs` | Giriş bileşeni türü veya bileşenlerinin listesidir (ör. `"text"`, `"image"`, `"mic"`, `gr.Textbox()`). |
| `outputs` | Çıkış bileşeni türü veya bileşenlerinin listesidir (ör. `"text"`, `"image"`, `"label"`). |

### Sesli İletişim Örneği
Sistem, ses verilerini almak, işlemek ve ses olarak geri döndürmek için gr.Audio bileşenini destekler. Aşağıdaki örnekte, kullanıcının mikrofondan kaydettiği ses verisi bir NumPy dizisi (numpy.ndarray) olarak alınır ve ters çevrilerek oynatılabilir bir ses bileşeni olarak döndürülür:

```python
import numpy as np
import gradio as gr

def reverse_audio(audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

mic = gr.Audio(source="microphone", type="numpy", label="Burada konuşun...")
gr.Interface(fn=reverse_audio, inputs=mic, outputs="audio").launch()
```

### Birden Fazla Giriş ve Çıktı İşleme
Karmaşık fonksiyonlarda, birden fazla girdiyi ve çıktıyı işlemek için bileşenler liste yapısı içinde iletilir. Giriş bileşenlerinin sırası fonksiyon argümanlarına, çıkış bileşenlerinin sırası ise fonksiyonun dönüş değerlerine (return değerlerine) birebir eşlenir:

```python
import numpy as np
import gradio as gr

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def generate_tone(note, octave, duration):
    sr = 48000
    a4_freq = 440
    tones_from_a4 = 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    sound = np.linspace(0, duration, duration * sr)
    sound = (20000 * np.sin(sound * (2 * np.pi * frequency))).astype(np.int16)
    return sr, sound

gr.Interface(
    fn=generate_tone,
    inputs=[
        gr.Dropdown(notes, type="index"),
        gr.Slider(minimum=4, maximum=6, step=1),
        gr.Number(value=1, label="Saniye cinsinden süre"),
    ],
    outputs="audio",
).launch()
```

### Otomatik Konuşma Tanıma (ASR) Örneği
Aşağıdaki örnekte, ses dosyasını veya mikrofondan gelen kaydı metne dönüştüren otomatik konuşma tanıma (automatic-speech-recognition) modeli sarmalanmaktadır:

```python
from transformers import pipeline
import gradio as gr

model = pipeline("automatic-speech-recognition")

def transcribe_audio(audio):
    transcription = model(audio)["text"]
    return transcription

gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(source="upload", type="filepath", optional=True),
    outputs="text",
).launch()
```

## 4. Demoları Özelleştirme ve Başkalarıyla Paylaşma
### Arayüz Bilgilerini ve Görünümünü Mükemmelleştirme
gr.Interface sınıfı, arayüzü zenginleştirmek için ek parametreler sunar:

title: Arayüzün en üstünde görünen başlık metni.

description: Giriş/çıkış alanlarının üstünde yer alan metin, Markdown veya HTML açıklaması.

article: Arayüzün en altında yer alan genişletilmiş makale veya dipnot alanı.

theme: Görsel tema seçeneği ("default", "huggingface", "grass", "peach", "dark-peach" vb.).

examples: Kullanıcıların tıklayıp hızlıca deneyebileceği örnek veri matrisi.

live: True olarak ayarlandığında, kullanıcı girdisi her değiştiğinde fonksiyonu buton tıklaması beklemeden anlık çalıştırır.

```python
import gradio as gr

title = "Rick'e bir soru sor"
description = """
```

Bot, Rick and Morty diyaloglarına dayalı soruları yanıtlamak üzere eğitildi. Rick'e her şeyi sorabilirsiniz!
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""
article = "Bu demoya temel oluşturan [orijinal Rick and Morty Bot'a](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) göz atın."

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[["Ne yapıyorsun?"], ["Zaman yolculuğuyla nereye gidelim?"]],
).launch()
### Geçici Bağlantılar İle Paylaşma (share=True)
Yerel makinenizde çalışan bir demoyu dış dünyaya geçici olarak açmak için launch() metoduna share=True parametresi verilir:

```python
gr.Interface(fn=classify_image, inputs="image", outputs="label").launch(share=True)
```

Bu işlem, local sunucuya tünellenen ve 72 saat boyunca geçerli olan XXXXX.gradio.app şeklinde bir bağlantı oluşturur. İşlem kullanıcının kendi cihazında gerçekleşir ve Gradio sunucularında veri depolanmaz.

### Çizim Tanıma Uygulaması Örneği (PyTorch Integration)
Aşağıdaki eksiksiz örnekte, önceden eğitilmiş bir PyTorch Modeli (pytorch_model.bin) ve sınıf etiketleri (class_names.txt) kullanılarak canlı bir çizim tanıma arayüzü kurulmaktadır:

```python
from pathlib import Path
import torch
from torch import nn
import gradio as gr

LABELS = Path("class_names.txt").read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)

state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        values, indices = torch.topk(probabilities, 5)
        return {LABELS[i]: v.item() for i, v in zip(indices, values)}

arayuz = gr.Interface(
    fn=predict,
    inputs="sketchpad",
    outputs="label",
    theme="grass",
    title="Çizim Tanıma",
    description="Bir nesne çizin, algoritma gerçek zamanlı olarak tahmin etsin!",
    article="<p style='text-align: center'>Çizim Tanıma | Demo Model</p>",
    live=True,
)

arayuz.launch(share=True)
```

## 5. Hugging Face Hub ve Spaces Entegrasyonları
Gradio, Hugging Face ekosistemiyle doğrudan entegre çalışır. gr.Interface.load() metodu sayesinde modelleri bellek kaygısı gütmeden Çıkarım API'si (Inference API) üzerinden çağırabilir veya mevcut Spaces uygulamalarını yerel ortama yükleyebilirsiniz.

### Hub Üzerinden Model Yükleme
Hugging Face Hub'da barındırılan büyük modeller (örneğin EleutherAI/gpt-j-6B), bilgisayarın bellek sınırlarını zorlamadan huggingface/ veya model/ ön ekiyle sarmalanabilir:

```python
import gradio as gr

title = "GPT-J-6B"
description = "GPT-J 6B için Gradio Demosu."
article = "<p style='text-align: center'><a href='https://github.com/kingoflolz/mesh-transformer-jax' target='_blank'>GPT-J-6B Repo</a></p>"

gr.Interface.load(
    "huggingface/EleutherAI/gpt-j-6B",
    inputs=gr.Textbox(lines=5, label="Metin Girin"),
    title=title,
    description=description,
    article=article,
).launch()
```

### Spaces Üzerinden Alan Yükleme ve Özelleştirme
Hugging Face Spaces üzerinde çalışan başka bir alanı spaces/ ön eki ile doğrudan çağırıp parametrelerini (örneğin girdi türünü webcam yaparak) değiştirebilirsiniz:

```python
import gradio as gr

gr.Interface.load(
    "spaces/abidlabs/remove-bg",
    inputs="webcam",
    title="Web kameranızın arka planını kaldırın!",
).launch()
```

## 6. Gelişmiş Interface Özellikleri
### Oturum Durumu (Session State) İle Veri Kalıcılığı
Gradio varsayılan olarak her istekte durumu sıfırlar. Ancak bir Sohbet Botu (Chatbot) geliştirirken geçmiş etkileşimleri saklamak için oturum durumu (state) kullanılır.

Oturum Durumu Kullanım Adımları:

Tahmin fonksiyonuna durumu temsil eden ek bir parametre (ör. history) verilir.

Fonksiyonun dönüş değerine güncellenmiş durum eklenir.

inputs ve outputs listelerine "state" veya gr.State() bileşeni eklenir.

```python
import random
import gradio as gr

def chat(message, history):
    history = history or []
    if message.startswith("Kaç tane"):
        response = random.randint(1, 10)
    elif message.startswith("Nasılsın"):
        response = random.choice(["Harika", "İyi", "Tamam", "Kötü"])
    elif message.startswith("Nerede"):
        response = random.choice(["Burada", "Orada", "Bir yerde"])
    else:
        response = "Bilmiyorum"

    history.append((message, str(response)))
    return history, history

iface = gr.Interface(
    fn=chat,
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    allow_flagging="never",
)

iface.launch()
```

### Yorumlanabilirlik (Interpretation) İle Model Tahminlerini Anlama
Makine öğrenimi modellerinin kara kutu yapısını kırmak ve girdinin hangi bölümlerinin çıktıya daha çok etki ettiğini göstermek için interpretation parametresi kullanılır.

interpretation="default": Girdi üzerindeki hassasiyet gradyanlarını hesaplar.

interpretation="shap": Shapley değerlerine dayalı yorumlama sağlar.

```python
import requests
import tensorflow as tf
import gradio as gr

# Modeli yükleme
inception_net = tf.keras.applications.MobileNetV2()

# ImageNet etiketlerini indirme
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}

image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)

gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    interpretation="default",
    title="Gradyan Görüntü Sınıflandırması + Yorumlama Örneği",
).launch()
```

## 7. Gradio Bloklarına Giriş (gr.Blocks)
gr.Interface yüksek seviyeli ve hızlı bir API sunarken, karmaşık yerleşim düzenleri ve çok adımlı veri akışları için düşük seviyeli gr.Blocks API'si kullanılır.

### Interface ve Blocks Karşılaştırması
Özellik	gr.Interface	gr.Blocks
Seviye
Yüksek Seviye (High-level)

Düşük Seviye (Low-level)

Esneklik
Sınırlı, standart düzen

Tam kontrol (Flexbox düzeni, sekmeler, sütunlar)

Veri Akışı
Tek adımlı (Girdi
→ Fonksiyon
→ Çıktı)

Çok adımlı, dinamik ve zincirleme olaylar

### Temel Blocks Yapısı
gr.Blocks, Python with bağlam yöneticisi (context manager) altında bileşenlerin ve olayların (events) tanımlanması esasına dayanır:

```python
import gradio as gr

def flip_text(x):
    return x[::-1]

demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
        # Metni Ters Çevir!
```

        Aşağıya yazmaya başlayın, çıktıyı göreceksiniz.
        """
    )
    input_text = gr.Textbox(placeholder="Bu metni çevir")
    output_text = gr.Textbox()

    input_text.change(fn=flip_text, inputs=input_text, outputs=output_text)

demo.launch()
### Düzen Tasarımı: Sütunlar, Satırlar ve Sekmeler
gr.Row(), gr.Column(), gr.Tabs() ve gr.TabItem() yapıları kullanılarak web arayüzünün mimarisi özelleştirilebilir:

```python
import numpy as np
import gradio as gr

demo = gr.Blocks()

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with demo:
    gr.Markdown("Bu demo ile metin veya resim dosyalarını çevirin.")
    with gr.Tabs():
        with gr.TabItem("Metni Çevir"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")

        with gr.TabItem("Resmi Çevir"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Çevir")

    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()
```

### Çok Adımlı Demolar ve Olay Bağlama (Event Binding)
Bir bileşenin çıktısı, başka bir bileşenin girdisi yapılarak zincirleme (multi-step) yapılar kurgulanabilir. Örneğin, konuşmayı metne çeviren ASR modelinin çıktısı, duygu analizi modeline aktarılabilir:

```python
from transformers import pipeline
import gradio as gr

asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(text):
    return classifier(text)[0]["label"]

demo = gr.Blocks()

with demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Konuşmayı Tanı")
    b2 = gr.Button("Duygu Analizi Sınıflandır")

    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()
```

### Bileşen Özelliklerini Dinamik Güncelleme (update)
Fonksiyondan bir değer yerine gr.update() yöntemi döndürülerek bileşenlerin görünürlüğü, satır sayısı veya seçenekleri kullanıcı etkileşimine göre değiştirilebilir:

```python
import gradio as gr

def change_textbox(choice):
    if choice == "short":
        return gr.Textbox.update(lines=2, visible=True)
    elif choice == "long":
        return gr.Textbox.update(lines=8, visible=True)
    else:
        return gr.Textbox.update(visible=False)

with gr.Blocks() as demo:
    radio = gr.Radio(
```

        ["short", "long", "none"],
        label="Ne tür bir deneme yazmak istersiniz?"
    )
    text = gr.Textbox(lines=2, interactive=True)

    radio.change(fn=change_textbox, inputs=radio, outputs=text)

demo.launch()
### Bölüm Sonu Özeti
Bu bölümde öne çıkan temel kazanımlar şunlardır:

Gradio Interface API'si: Birkaç satır kodla tahmin fonksiyonlarının etrafında arayüzler oluşturulmuş, girdi ve çıktı bileşenleri özelleştirilmiştir.

Esnek Paylaşım: share=True ile geçici bağlantılar oluşturulmuş ve Hugging Face Spaces altyapısıyla ücretsiz, kalıcı barındırma mekanizmaları incelenmiştir.

Ekosistem Entegrasyonu: Hugging Face Hub üzerindeki hazır modeller ve Spaces uygulamaları Interface.load() yöntemiyle projeye dahil edilmiştir.

Gelişmiş Arayüz Teknikleri: Oturum durumu (state) ile sohbet botu geçmişleri saklanmış, interpretation ile model kararları açıklanabilir kılınmıştır.

Gradio Blocks mimarisi: Sayfa düzeni (Row, Column, Tabs), olay tetikleyicileri ve dinamik bileşen güncellemeleri (update) üzerinde tam kontrol sağlanarak çok adımlı yapay zeka sistemleri geliştirilmiştir.
