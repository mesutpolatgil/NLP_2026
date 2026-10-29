> **Uyarı:** Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.

# Bölüm 1: Quiz ve Sınav Soruları

**Hazırlayan:** Sıla Taşan (Sınav Komiseri & Ölçme Değerlendirme) · **Kaynak:** [Hugging Face LLM Course, Bölüm 1](https://huggingface.co/learn/llm-course/chapter1/1)

## İçindekiler

1. [Hızlı Quiz (1/7)](#hizli-quiz)
2. [Sertifikasyon Sınavı (1/11)](#sertifikasyon)
3. [Özgün Mimari Soruları](#ozgun)

---

<a id="hizli-quiz"></a>

## 1. Hızlı Quiz (1/7)

**Soru 1:** Hub'ı keşfedin ve `roberta-large-mnli` kontrol noktasını (checkpoint) arayın. Bu model hangi görevi yerine getirmektedir?
* **A)** Özetleme (Summarization)
* **B)** Metin sınıflandırma (Text classification)
* **C)** Metin üretimi (Text generation)
* **Doğru Cevap:** **B**

**Soru 2:** Aşağıdaki kod ne döndürecektir? (`pipeline("ner", grouped_entities=True)`)
* **A)** "Pozitif" veya "negatif" etiketleriyle bu cümle için sınıflandırma skorları döndürecektir.
* **B)** Bu cümleyi tamamlayan üretilmiş bir metin döndürecektir.
* **C)** Kişileri, kurumları veya yerleri temsil eden kelimeleri döndürecektir.
* **Doğru Cevap:** **C**

**Soru 3:** Bu kod örneğindeki `...` (boşluk) yerine ne gelmelidir?

`filler = pipeline("fill-mask", model="bert-base-cased")`

`result = filler("...")`

* **A)** `This <mask> has been waiting for you.`
* **B)** `This [MASK] has been waiting for you.`
* **C)** `This man has been waiting for you.`
* **Doğru Cevap:** **B**

**Soru 4:** Zero-shot-classification kodu neden hata verecektir?

`classifier = pipeline("zero-shot-classification")`

`result = classifier("This is a course about the Transformers library")`

* **A)** Bu pipeline, metni sınıflandırmak için etiketlerin (labels) verilmesini gerektirir.
* **B)** Bu pipeline sadece bir değil, birkaç cümle gerektirir.
* **C)** Hugging Face Transformers kütüphanesi her zamanki gibi bozuktur.
* **D)** Bu pipeline daha uzun girdiler gerektirir; bu girdi çok kısadır.
* **Doğru Cevap:** **A**

**Soru 5:** "Transfer learning" (Transfer öğrenme) ne demektir?
* **A)** Önceden eğitilmiş bir modelin bilgisini, onu aynı veri seti üzerinde eğiterek yeni bir modele aktarmaktır.
* **B)** Önceden eğitilmiş bir modelin bilgisini, ikinci modeli ilk modelin ağırlıklarıyla (weights) başlatarak yeni bir modele aktarmaktır.
* **C)** İkinci modeli, ilk modelle aynı mimaride inşa ederek modelin bilgisini aktarmaktır.
* **Doğru Cevap:** **B**

**Soru 6:** Doğru mu yanlış mı? Bir dil modelinin (language model) ön eğitimi (pretraining) için genellikle etiketlere ihtiyacı yoktur.
* **A)** Doğru (True)
* **B)** Yanlış (False)
* **Doğru Cevap:** **A** 

**Soru 7:** "Model", "mimari" (architecture) ve "ağırlıklar" (weights) terimlerini en iyi açıklayan cümleyi seçin.
* **A)** Eğer model bir bina olsaydı, mimari onun projesi, ağırlıklar ise içinde yaşayan insanlar olurdu.
* **B)** Mimari bir modeli oluşturmak için bir harita, ağırlıklar ise haritada temsil edilen şehirlerdir.
* **C)** Bir mimari, modeli oluşturmak için art arda gelen matematiksel fonksiyonlardır ve ağırlıklar ise bu fonksiyonların parametreleridir.
* **Doğru Cevap:** **C**

**Soru 8:** Prompt'ları (komutları) üretilmiş metinlerle (generated text) tamamlamak için bu model türlerinden hangisini kullanırsınız?
* **A)** Encoder (Kodlayıcı) modeli
* **B)** Decoder (Çözücü) modeli
* **C)** Sequence-to-sequence (Diziden-diziye) modeli
* **Doğru Cevap:** **B**

**Soru 9:** Metinleri özetlemek (summarization) için bu model türlerinden hangisini kullanırsınız?
* **A)** Encoder (Kodlayıcı) modeli
* **B)** Decoder (Çözücü) modeli
* **C)** Sequence-to-sequence (Diziden-diziye) modeli
* **Doğru Cevap:** **C**

**Soru 10:** Metin girdilerini belirli etiketlere göre sınıflandırmak (text classification) için bu model türlerinden hangisini kullanırsınız?
* **A)** Encoder (Kodlayıcı) modeli
* **B)** Decoder (Çözücü) modeli
* **C)** Sequence-to-sequence (Diziden-diziye) modeli
* **Doğru Cevap:** **A** 

**Soru 11:** Bir modelde gözlemlenen önyargının (bias) olası kaynakları neler olabilir?
* **A)** Model, önceden eğitilmiş (pretrained) bir modelin ince ayar (fine-tuned) yapılmış versiyonudur ve önyargıyı ondan almıştır.
* **B)** Modelin üzerinde eğitildiği veri seti önyargılıdır.
* **C)** Modelin optimize ettiği metrik önyargılıdır.
* **Doğru Cevap:** **A, B ve C seçeneklerinin tümü.**

---

<a id="sertifikasyon"></a>

## 2. Sertifikasyon Sınavı (1/11)

**Soru 1:** Çeviri veya özetleme gibi diziden-diziye (sequence-to-sequence) görevler için genellikle hangi tür Transformer mimarisi en uygundur?
- **A)** Sadece Kodlayıcı (Encoder-only, örn. BERT)
- **B)** Sadece Çözücü (Decoder-only, örn. GPT)
- **C)** Kodlayıcı-Çözücü (Encoder-decoder, örn. T5, BART)
- **D)** Tekrarlayan Sinir Ağı (Recurrent Neural Network - RNN)

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: C**

**Detaylı Açıklama:**

- **A)** Sadece kodlayıcı (Encoder-only) modeller, metin üretmek için değil; cümlenin tamamını anlayıp sınıflandırma gibi görevler yapmak için uygundur.
- **B)** Sadece çözücü (Decoder-only) modeller, çeviri gibi çift yönlü bağlam gerektiren işlemlerden ziyade, serbest metin üretimi için tasarlanmıştır.
- **C)** Kodlayıcı-Çözücü (Encoder-decoder) modeller, bir girdi dizisini alıp (anlayıp) tamamen yeni bir çıktı dizisi üretmeyi gerektiren çeviri ve özetleme görevleri için en uygun mimaridir.
- **D)** Tekrarlayan Sinir Ağları (RNN), bir Transformer mimarisi türü değildir; eski nesil sıralı işleme ağlarıdır.

</details>
<br>

**Soru 2:** Hangi Hugging Face pipeline'ı, bu belirli etiketler üzerinde önceden ince ayar (fine-tuning) gerektirmeden, çıkarım (inference) anında sağlanan rastgele etiketleri kullanarak metin sınıflandırmaya olanak tanır?
- **A)** text-classification
- **B)** sentiment-analysis
- **C)** zero-shot-classification
- **D)** token-classification

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: C**

**Detaylı Açıklama:**

- **A)** Text-classification pipeline'ı, modelin eğitim aşamasında gördüğü sabit etiketlere ihtiyaç duyar.
- **B)** Sentiment-analysis genellikle sadece "olumlu/olumsuz" gibi önceden belirlenmiş duygu etiketleriyle çalışır.
- **C)** Zero-shot-classification, modelin önceden hiç eğitilmediği, kullanıcının çıkarım anında belirlediği yepyeni etiket kategorileriyle metni anında sınıflandırmasını sağlayan özel bir mekanizmadır.
- **D)** Token-classification, metnin bütününü değil, kelimeleri tek tek sınıflandırmak (örn: varlık ismi tanıma - NER) için kullanılır.

</details>
<br>

**Soru 3:** Transformer modelleri için yaygın olarak kullanılan üç ana mimari kategori hangileridir?
- **A)** Evrişimli Sinir Ağları (CNN), Tekrarlayan Sinir Ağları (RNN), Dikkat Ağları (Attention Networks)
- **B)** Küçük ölçekli, Orta ölçekli, Büyük ölçekli
- **C)** Sadece Kodlayıcı (Encoder-only), Sadece Çözücü (Decoder-only), Kodlayıcı-Çözücü (Encoder-decoder / Sequence-to-sequence)
- **D)** Üretici Çekişmeli Ağlar (GAN), Varyasyonel Otoenkoderler (VAE), Difüzyon Modelleri (Diffusion Models)

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: C**

**Detaylı Açıklama:**

- **A)** CNN ve RNN, Transformer tabanlı olmayan farklı sinir ağı mimarileridir.
- **B)** Bu sınıflandırma, modellerin mimarisini değil, yalnızca parametre boyutlarını ifade eder.
- **C)** Transformer mimarisi temelde metni anlayan (Encoder), metin üreten (Decoder) ve her ikisini birleştiren (Encoder-decoder) olmak üzere üç ana türe ayrılır.
- **D)** GAN, VAE ve Difüzyon modelleri genellikle doğal dil işleme için değil, görsel/ses üretimi için kullanılan mimarilerdir.

</details>
<br>

**Soru 4:** Büyük Dil Modeli (LLM) çıkarımında (inference), 'sıcaklık' (temperature) örnekleme parametresi temel olarak neyi kontrol eder?
- **A)** Üretilecek maksimum token sayısını.
- **B)** Göz önünde bulundurulan alternatif dizi sayısını (beam width).
- **C)** Çıktının rastgeleliğini/yaratıcılığını (düşük değerler daha odaklı/belirlenimci, yüksek değerler daha rastgeledir).
- **D)** Çıktıda halihazırda görünmüş olan token'lara uygulanan cezayı.

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: C**

**Detaylı Açıklama:**

- **A)** Maksimum token sayısı `max_new_tokens` veya `max_length` parametreleriyle kontrol edilir.
- **B)** Alternatif dizi sayısı `num_beams` parametresiyle (Beam Search) kontrol edilir.
- **C)** Sıcaklık (temperature) değeri Softmax fonksiyonundaki olasılık dağılımını manipüle ederek modelin daha çeşitli/yaratıcı veya daha sabit/garantici cevaplar vermesini sağlar.
- **D)** Bu işlem `repetition_penalty` (tekrarlama cezası) parametresiyle yapılır.

</details>
<br>

**Soru 5:** LLM çıkarımı sırasında kullanılan Key-Value (KV) Cache optimizasyonunun temel amacı nedir?
- **A)** Başlangıç komutunu işlemenin (prefill aşaması) hesaplama maliyetini azaltmak.
- **B)** Ağırlıklarını sıkıştırarak modelin bellek ayak izini azaltmak.
- **C)** Önceki adımlardaki ara dikkat (attention) hesaplamalarını depolayıp yeniden kullanarak token üretimini (decode aşaması) hızlandırmak.
- **D)** Girdi bağlamının dikkat edilecek en alakalı kısımlarını otomatik olarak seçmek.

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: C**

**Detaylı Açıklama:**

- **A)** Prefill aşaması paralelizasyon ile hızlandırılır, KV Cache'in asıl faydası bu aşamadan sonra başlar.
- **B)** Ağırlıkları sıkıştırma işlemi "Quantization" (Nicemleme) optimizasyonunun görevidir.
- **C)** KV Cache, modelin her yeni token üretirken geçmiş tokenların matris değerlerini baştan hesaplamasını önler. Bu değerler önbellekte (cache) tutularak çok ciddi bir hız ve verimlilik artışı elde edilir.
- **D)** Bu doğrudan Attention mekanizmasının kendi matematiksel görevidir.

</details>
<br>

**Soru 6:** Bir Hugging Face `pipeline()` fonksiyonu, girdi metnini çıkarım için işlerken tipik olarak hangi üç ana üst düzey adımı gerçekleştirir?
- **A)** Tokenizasyon, Model Eğitimi, Metin Üretimi
- **B)** Ön işleme (Preprocessing), Model Çıkarımı (Model Inference), Son işleme (Postprocessing)
- **C)** Veri Yükleme, Model İnce Ayarı, Kontrol Noktası Kaydetme
- **D)** Girdi Gömme, Dikkat Hesaplama, Çıktı Çözme

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: B**

**Detaylı Açıklama:**

- **A ve C)** Pipeline fonksiyonu model eğitimi (training) veya ince ayarı (fine-tuning) yapmak için değil, halihazırda eğitilmiş modeli kullanmak için tasarlanmıştır.
- **B)** Pipeline sırasıyla metni sayılara çevirir (Preprocessing/Tokenization), bu sayıları modele verip tahmin alır (Inference) ve modelin ürettiği ham çıktıları insanın okuyabileceği metin formatına geri döndürür (Postprocessing).
- **D)** Bunlar model mimarisinin içindeki düşük seviyeli matematiksel hesaplama adımlarıdır.

</details>
<br>

**Soru 7:** Aşağıdakilerden hangisi Büyük Dil Modellerinin (LLM'ler) önemli bir sınırlaması olarak kabul edilir?
- **A)** Çeviri veya özetleme gibi görevleri yerine getirememesi.
- **B)** Akıcı ve dilbilgisi açısından doğru metin üretmede zorluk çekmesi.
- **C)** Kendinden emin bir şekilde yanlış veya anlamsız bilgiler üretme potansiyeli (halüsinasyonlar).
- **D)** Sınırlı uygulanabilirlik, yalnızca temel anahtar kelime tespiti için yararlı olması.

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: C**

**Detaylı Açıklama:**

- **A ve B)** Büyük Dil Modelleri çeviri, özetleme ve son derece akıcı, dilbilgisi kurallarına uygun metin üretme konusunda çok başarılıdır.
- **C)** Modeller gerçek dünyayı doğrudan algılamazlar, istatistiksel token olasılıkları üretirler. Bu durum, tamamen yanlış veya uydurma bir bilgiyi son derece mantıklı ve doğruymuş gibi sunmalarına (halüsinasyon) yol açan en büyük sınırlamadır.
- **D)** LLM'ler anahtar kelime tespitinin çok ötesinde mantıksal yürütme ve çoklu görev yeteneklerine sahiptir.

</details>
<br>

**Soru 8:** Daha geniş bir alan olan Doğal Dil İşleme (NLP) ile Büyük Dil Modelleri (LLM'ler) arasındaki temel fark nedir?
- **A)** NLP metin üretimine odaklanırken, LLM'ler metin sınıflandırmasına odaklanır.
- **B)** NLP daha geniş bir alandır, LLM'ler ise boyutları ve genel yetenekleriyle bilinen güçlü bir alt kümesidir.
- **C)** LLM'ler tüm uygulamalar için göreve özel eğitim gerektirirken, NLP modelleri gerektirmez.
- **D)** NLP modelleri genellikle daha büyüktür ve LLM'lerden daha fazla veri gerektirir.

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: B**

**Detaylı Açıklama:**

- **A)** Her ikisi de hem metin üretimi hem de sınıflandırma yapabilir.
- **B)** NLP (Doğal Dil İşleme), dilbilimi ve yapay zekayı birleştiren devasa bir bilimsel şemsiye alandır. LLM'ler (Büyük Dil Modelleri), derin öğrenme teknikleriyle oluşturulmuş ve bu geniş alanın sadece bir parçası olan devasa boyutlu spesifik ağ mimarileridir.
- **C)** Tam tersine, LLM'ler zero-shot (sıfır atış) yetenekleri sayesinde göreve özel eğitim olmadan da birçok görevi başarabilir.
- **D)** LLM'ler (Large Language Models) adından da anlaşılacağı üzere geleneksel NLP modellerinden çok daha büyük parametrelere sahiptir ve daha fazla veri gerektirir.

</details>
<br>

**Soru 9:** Bir komut aldıktan sonra metin üretmek için Büyük Dil Modeli (LLM) çıkarım sürecinde (inference process) tipik olarak yer alan iki farklı aşama nelerdir?
- **A)** Model Derleme ve Nicemleme
- **B)** Ön Doldurma / Prefill (komutu işleme) ve Çözme / Decode (sıralı olarak token üretme)
- **C)** Tokenizasyon ve Gömme (Embedding) Araması
- **D)** Dikkat (Attention) Hesaplama ve Softmax Uygulaması

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: B**

**Detaylı Açıklama:**

- **A)** Bu işlemler modelin çalışma anı (inference) değil, optimizasyon aşamasıyla ilgilidir.
- **B)** Metin üretim süreci iki aşamalıdır: Model önce kullanıcının girdiği istemin tamamını aynı anda işleyerek bağlamı anlar (Prefill aşaması). Ardından yanıtı verirken kelimeleri oto-regresif olarak tek tek ardışık şekilde oluşturmaya başlar (Decode aşaması).
- **C ve D)** Bu seçenekler, prefill ve decode aşamalarının içinde gerçekleşen mikro işlemlerdir, ana süreç aşamaları değildir.

</details>
<br>

**Soru 10:** Girdideki token'ları maskelemeyi ve modeli orijinal token'ları tahmin etmesi için eğitmeyi içeren, genellikle BERT gibi kodlayıcı (encoder) modelleriyle ilişkilendirilen ön eğitim hedefi hangisidir?
- **A)** Nedensel Dil Modelleme (Causal Language Modeling - CLM)
- **B)** Diziden-Diziye Çeviri (Sequence-to-Sequence Translation)
- **C)** Sonraki Cümle Tahmini (Next Sentence Prediction - NSP)
- **D)** Maskeli Dil Modelleme (Masked Language Modeling - MLM)

<details>
<summary>Cevap ve Açıklamalar</summary>

**Doğru Cevap: D**

**Detaylı Açıklama:**

- **A)** CLM, maskeleme yapmadan yalnızca bir sonraki kelimeyi tahmin etmeye dayanan (genellikle GPT gibi Decoder modellerin kullandığı) yöntemdir.
- **B)** Seq2Seq, çeviri görevleri için uygulanan bir hedeftir.
- **C)** NSP, BERT'in kullandığı diğer bir hedef olsa da maskelemeyi değil, iki cümlenin ardışık olup olmadığını tahmin etmeyi içerir.
- **D)** Masked Language Modeling (MLM), cümlenin içindeki rastgele kelimelerin kapatılarak (maskelenerek) modelin o boşluğa gelecek doğru kelimeyi sağdan ve soldan gelen çift yönlü bağlama bakarak bulmasını sağlayan BERT ön eğitim tekniğidir.

</details>
<br>

---

<a id="ozgun"></a>

## 3. Özgün Mimari Soruları

Bu sınavda bir seçenek listesinden doğru cevabı seçmeniz istenecektir. Denetimli ince ayarın (supervised fine-tuning) temelleri konusunda bilginiz ölçülecektir.

**Soru 1:** Transformer mimarisindeki Self-Attention mekanizmasında, girdi kelimelerinin birbirleriyle olan ilişkisini hesaplamak için üç farklı vektör kullanılır: Query ($Q$), Key ($K$) ve Value ($V$). Geleneksel Seq2Seq modellerine kıyasla, bu üçlü matris çarpım yapısının (Scaled Dot-Product Attention) modele sağladığı en büyük mimari avantaj nedir ve matris çarpımı sonucunda elde edilen "Attention Score" neyi ifade eder?

<details>
<summary>Doğru Cevap</summary>

Self-Attention mekanizması, geleneksel modellerin aksine metni kelime kelime değil **paralel** işleyerek donanımsal hız ve uzun metinlerde bağlamı kaybetmeme avantajı sağlar. Elde edilen "Attention Score" ise kısaca, bir kelimenin cümledeki diğer kelimelerle ne kadar anlamsal bağı olduğunu gösteren bir ağırlık dağılımıdır. 

**Detaylar:**
- **Mimari Avantaj:** Geleneksel Seq2Seq modelleri (RNN, LSTM) veriyi ardışık (sıralı) işlemek zorundadır; bu da uzun dizilimlerde bağlamın unutulmasına yol açar. Self-Attention mekanizması ise her bir token'ın cümledeki diğer tüm token'larla olan ilişkisini **paralel olarak tek seferde** matris çarpımlarıyla hesaplar. Bu yapı, hem donanım (GPU/TPU) seviyesinde yüksek işlem hızı sağlar hem de uzun mesafeli bağımlılıkları (long-term dependencies) veri kaybı olmaksızın yakalama imkanı sunar.
- **Attention Score'un Anlamı:** $Q$ (Query) ve $K^T$ (Key matrisinin transpozu) çarpıldıktan sonra sonuçların Softmax fonksiyonundan geçirilmesiyle elde edilen skordur. Bu skor, incelenen kelimenin (Query) cümledeki diğer kelimelerden (Key) ne oranda bilgi (Value) alması gerektiğini belirten **matematiksel bir olasılık dağılımını** ifade eder.

</details>
<br>

**Soru 2:** BERT gibi Encoder tabanlı modeller "Bi-directional" (çift yönlü) Attention kullanırken, GPT gibi Decoder tabanlı modeller "Masked" (maskeli) Attention kullanır. Masked Attention mekanizmasında maskeleme işlemi matematiksel olarak nasıl gerçekleştirilir ve bu işlemin mimarideki temel amacı nedir?

<details>
<summary>Doğru Cevap</summary>

Masked Attention, modelin metin üretirken henüz yazılmamış olan "gelecekteki" kelimeleri görerek kopya çekmesini (veri sızıntısını) engellemek için kullanılan bir mekanizmadır. Bu engelleme işlemi, matematiksel olarak gelecekteki kelimelerin dikkat (attention) ağırlıklarının sıfırlanmasıyla yapılır.

**Detaylar:**
- **Matematiksel Gerçekleştirim:** Masked Attention uygulamasında, matris çarpımı ($Q \times K^T$) sonucunda ortaya çıkan skor matrisinin üst üçgen kısmı (henüz üretilmemiş olan, gelecekteki token'lara denk gelen kısımlar) $-\infty$ (eksi sonsuz) veya çok büyük negatif değerlerle değiştirilir (maskelenir). Bu matris Softmax işlemine sokulduğunda, $e^{-\infty}$ matematiksel olarak $0$'a yaklaşacağından, bu gelecek kelimelerin dikkat (attention) ağırlığı tamamen sıfırlanmış olur.
- **Temel Amaç:** Bu işlemin mimarideki yegane amacı, modelin otoregresif olarak (kelime kelime) metin üretirken gelecekteki kelimeleri görüp hile yapmasını (**veri sızıntısı / data leakage**) kesin bir şekilde önlemektir. Böylelikle model, bir sonraki kelimeyi tahmin ederken yalnızca kendisinden önceki (solundaki ve mevcut) bağlama odaklanmaya zorlanır.

</details>
<br>

**Soru 3:** Modelin tek bir Attention mekanizması yerine Multi-Head Attention (Çoklu Dikkat Başlıkları) kullanması modelin öğrenme kapasitesini nasıl etkiler? Farklı "Head"lerin aynı anda çalışması ne tür bir bilginin yakalanmasını sağlar?

<details>
<summary>Doğru Cevap</summary>

Multi-Head Attention, modelin aynı cümleyi eşzamanlı olarak farklı "uzmanlık" açılarından (örneğin dilbilgisi, duygu, anlamsal referanslar vb.) incelemesini sağlar. Bu sayede modelin tek bir noktaya odaklanıp diğer anlamları kaçırma riski ortadan kalkar ve öğrenme kapasitesi büyük ölçüde artar.

**Detaylar:**
- **Öğrenme Kapasitesine Etkisi:** Multi-Head Attention, tüm dikkat ağırlıklarının tek bir anlamsal yöne odaklanması kısıtını aşar. Girdi vektörleri, farklı ve daha düşük boyutlu **alt uzaylara (subspaces)** yansıtılarak (project edilerek) paralel olarak incelenir. Bu yaklaşım, modelin temsil gücünü (representation power) zenginleştirir ve öğrenme kapasitesini katlayarak artırır.
- **Yakalanan Bilgi Türleri:** Farklı "Head"ler (başlıklar) birbirinden bağımsız çalışarak cümlenin farklı anlamsal ve sözdizimsel özelliklerini öğrenir. Örneğin; bir başlık yalnızca özne-yüklem uyumuna bakarken, başka bir başlık zamirlerin hangi isimleri işaret ettiğini (coreference) takip edebilir, bir diğeri ise metnin duygusal tonuna odaklanabilir. Bu paralellik sayesinde insan dilinin çok boyutlu ve karmaşık yapısı, matematiksel olarak çok daha isabetli bir biçimde modellenmiş olur.

</details>
