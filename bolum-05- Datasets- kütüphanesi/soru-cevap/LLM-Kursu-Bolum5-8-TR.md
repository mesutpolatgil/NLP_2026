<!-- Kaynak: https://huggingface.co/learn/llm-course/chapter5/8 — Hugging Face LLM Course, Apache-2.0 lisansı (github.com/huggingface/course). Orijinal İngilizce metinden Türkçeye çevrilmiş; cevaplar, notlar ve "Bizim Sorularımız" bölümü sonradan eklenmiştir. -->

# Bölüm sonu testi

Bu bölümde pek çok konuyu ele aldık! Tüm ayrıntıları kavrayamadıysanız endişelenmeyin; sonraki bölümler işlerin arka planda nasıl yürüdüğünü anlamanıza yardımcı olacak.

Yine de devam etmeden önce bu bölümde öğrendiklerinizi test edelim.

### 1. 🤗 Datasets içindeki load_dataset() fonksiyonu, aşağıdaki konumlardan hangisinden veri seti yüklemenize olanak tanır?

A) Yerel olarak, örneğin dizüstü bilgisayarınızdan ✅

B) Hugging Face Hub ✅

C) Uzak bir sunucu ✅

Cevap: A, B ve C ✅

**📝 Not:**
`load_dataset()` veri nerede olursa olsun aynı arayüzle çalışır.

Örneğin:
`load_dataset("csv", data_files="veri.csv")` → yerel dosya
`load_dataset("squad")` → Hub
`load_dataset("json", data_files="https://.../veri.json.gz")` → uzak sunucu

A doğru: Yerel dosyalar `data_files` ile verilir.
B doğru: Hub'daki veri setleri yalnızca adıyla yüklenir.
C doğru: `data_files` argümanına doğrudan URL verilebilir.

### 2. GLUE görevlerinden birini aşağıdaki gibi yüklediğinizi varsayalım:

```python
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")
```

Aşağıdaki komutlardan hangisi `dataset` içinden 50 elemanlık rastgele bir örneklem üretir?

A) `dataset.sample(50)`

B) `dataset.shuffle().select(range(50))` ✅

C) `dataset.select(range(50)).shuffle()`

**📝 Not:**
Önce karıştır, sonra seç: `shuffle()` → `select(range(50))`

A yanlış: `Dataset` nesnesinin `sample()` diye bir metodu yoktur.
C yanlış: Önce ilk 50 örnek seçilir, sonra sadece bu 50 örnek karıştırılır; örneklem rastgele değildir.

### 3. Ev hayvanlarıyla ilgili pets_dataset adında bir veri setiniz olduğunu ve bu veri setinde her hayvanın adını belirten bir name sütunu bulunduğunu varsayalım. Aşağıdaki yaklaşımlardan hangisi, veri setini adı “L” harfiyle başlayan tüm hayvanlar için filtrelemenizi sağlar?

A) `pets_dataset.filter(lambda x : x['name'].startswith('L'))` ✅

B) `pets_dataset.filter(lambda x['name'].startswith('L'))`

C) `def filter_names(x): return x['name'].startswith('L')` gibi bir fonksiyon oluşturup `pets_dataset.filter(filter_names)` komutunu çalıştırmak. ✅

Cevap: A ve C ✅

**📝 Not:**
`filter()`, her örnek için `True`/`False` döndüren bir fonksiyon bekler. Bu fonksiyon lambda da olabilir, normal (`def`) fonksiyon da.

A doğru: Lambda, örneği `x` olarak alır ve koşulu döndürür.
B yanlış: `lambda x['name']` geçerli bir Python sözdizimi değildir; lambda parametresi bir değişken adı olmalıdır.
C doğru: Adlandırılmış fonksiyon, lambda ile aynı işi görür.

### 4. Bellek eşleme (memory mapping) nedir?

A) CPU ile GPU RAM'i arasında bir eşleme

B) RAM ile dosya sistemi depolaması arasında bir eşleme ✅

C) 🤗 Datasets önbelleğindeki iki dosya arasında bir eşleme

**📝 Not:**
Disk (Arrow dosyası) ↔ RAM: Dosya belleğe kopyalanmaz; gerektiğinde ilgili kısmı doğrudan diskten okunur.

A yanlış: Bellek eşleme CPU–GPU veri aktarımıyla ilgili değildir.
C yanlış: İki dosyayı birbirine eşlemez; diskteki dosyayı belleğe eşler.

### 5. Aşağıdakilerden hangileri bellek eşlemenin başlıca faydalarıdır?

A) Bellekle eşlenmiş dosyalara erişmek, diske okuma ya da yazma yapmaktan daha hızlıdır. ✅

B) Uygulamalar, son derece büyük bir dosyanın tamamını önce RAM'e okumak zorunda kalmadan verinin belirli bölümlerine erişebilir. ✅

C) Daha az enerji tüketir, böylece pilinizin ömrü uzar.

Cevap: A ve B ✅

**📝 Not:**
Bellek eşleme sayesinde onlarca GB'lık bir veri seti, RAM'de çok az yer kaplayarak hızlı bir şekilde gezilebilir.

A doğru: Klasik okuma/yazma çağrılarına göre erişim daha hızlıdır.
B doğru: Sadece ihtiyaç duyulan kısım okunur; dosyanın tamamı RAM'e yüklenmez.
C yanlış: Bellek eşlemenin amacı enerji tasarrufu değildir.

### 6. Aşağıdaki kod neden hata verir?

```python
from datasets import load_dataset

dataset = load_dataset("allocine", streaming=True, split="train")
dataset[0]
```

A) RAM'e sığmayacak kadar büyük bir veri setini akış (stream) ile almaya çalışıyor.

B) Bir `IterableDataset` nesnesine (indeksle) erişmeye çalışıyor. ✅

C) `allocine` veri setinin `train` bölümü (split) yok.

**📝 Not:**
`streaming=True` → `IterableDataset`. Bu nesne indekslenemez, yalnızca üzerinde gezinilebilir.

Örneğin:
`next(iter(dataset))` veya `dataset.take(1)`

A yanlış: Akışın amacı zaten RAM'e sığmayan veriyle çalışmaktır; bu bir hata nedeni değildir.
C yanlış: `allocine` veri setinde `train` bölümü vardır.

### 7. Aşağıdakilerden hangileri bir veri seti kartı (dataset card) oluşturmanın başlıca faydalarıdır?

A) Veri setinin amaçlanan kullanımı ve desteklenen görevleri hakkında bilgi sunar; böylece topluluktaki diğer kişiler onu kullanma konusunda bilinçli bir karar verebilir. ✅

B) Bir derlemde (corpus) bulunan önyargılara dikkat çekilmesine yardımcı olur. ✅

C) Topluluktaki diğer kişilerin veri setimi kullanma olasılığını artırır. ✅

Cevap: A, B ve C ✅

**📝 Not:**
Veri seti kartı, veri setinin "kullanım kılavuzudur": nereden geldiğini, ne için uygun olduğunu ve sınırlılıklarını anlatır.

A doğru: Kullanıcılar veri setinin işlerine uyup uymadığını kart sayesinde anlar.
B doğru: Bilinen önyargı ve kısıtların belgelenmesi, bu konulara dikkat çeker.
C doğru: İyi belgelenmiş veri setleri daha çok tercih edilir.

### 8. Anlamsal arama (semantic search) nedir?

A) Bir sorgudaki kelimeler ile bir derlemdeki belgeler arasında birebir eşleşmeleri aramanın bir yolu

B) Bir sorgunun bağlamsal anlamını kavrayarak eşleşen belgeleri aramanın bir yolu ✅

C) Arama doğruluğunu artırmanın bir yolu

**📝 Not:**
Sorgu → Embedding → Benzer embedding'e sahip belgeler

Anlamsal arama kelimelere değil, anlama bakar. Aynı kelimeleri içermeyen ama aynı şeyi anlatan belgeleri de bulabilir.

A yanlış: Bu, anahtar kelime (lexical) aramasının tanımıdır.
C yanlış: Doğruluğu artırabilir ama bu bir sonuçtur; anlamsal aramanın tanımı değildir.

### 9. Asimetrik anlamsal aramada genellikle şunlar bulunur:

A) Kısa bir sorgu ve bu sorguyu yanıtlayan daha uzun bir paragraf ✅

B) Yaklaşık aynı uzunlukta sorgular ve paragraflar

C) Uzun bir sorgu ve bu sorguyu yanıtlayan daha kısa bir paragraf

**📝 Not:**
Asimetrik: kısa soru ↔ uzun cevap metni. Bölümdeki GitHub issue arama motoru buna bir örnektir.

B yanlış: Benzer uzunluktaki metinlerin karşılaştırılması simetrik aramadır.
C yanlış: Tipik kullanımın tersidir; sorgular genelde kısa olur.

### 10. 🤗 Datasets'i konuşma işleme gibi başka alanlarda kullanılacak verileri yüklemek için kullanabilir miyim?

A) Hayır

B) Evet ✅

**📝 Not:**
🤗 Datasets yalnızca metinle sınırlı değildir; ses (audio) ve görüntü (image) veri setlerini de destekler.

A yanlış: Kütüphane, konuşma ve görüntü işleme veri setleriyle de çalışır.

## Bizim Sorularımız

Aşağıdaki sorular kursun orijinal içeriği değildir; Bölüm 5'in ünitelerini (5.1–5.7) pekiştirmek için hazırlanmış orta düzey ek sorulardır.

### 5.1 Giriş

**Soru 1**
Bölüm 5'in temel amacı nedir? Hangi konu bu bölümün kapsamı dışında kalır?

**Cevap**
Bölüm 5, 🤗 Datasets ile veri setlerini yükleme, temizleme, dönüştürme, büyük veriyle çalışma, kendi veri setini oluşturup paylaşma ve anlamsal arama yapma konularını ele alır. Bir dil modelini sıfırdan ön-eğitmek (pretraining) ise bu bölümün kapsamı dışındadır.

**Soru 2**
Bölüm 5'te kendi veri setimizi oluştururken veriler hangi kaynaktan toplanır ve bu veri seti daha sonra ne için kullanılır?

**Cevap**
Veriler bir GitHub deposunun (🤗 Datasets deposu) issue'larından ve yorumlarından toplanır. Oluşturulan veri seti, bölümün sonunda bu issue'lar üzerinde çalışan bir anlamsal arama motoru kurmak için kullanılır.

**Soru 3**
Bölümün sonunda kurulan anlamsal arama motorunda benzerlik araması hangi kütüphane ile yapılır ve bu kütüphane ne işe yarar?

**Cevap**
FAISS (Facebook AI Similarity Search) kullanılır. FAISS, çok sayıda embedding vektörü arasında, sorgu vektörüne en yakın olanları hızlı ve verimli biçimde bulmayı sağlayan bir indeksleme ve arama kütüphanesidir.

### 5.2 Veri setim Hub'da değilse ne olur?

**Soru 4**
JSON dosyanızdaki tüm örnekler en üst düzeydeki `"data"` anahtarının altında iç içe duruyorsa `load_dataset("json", ...)` çağrısına hangi argümanı eklersiniz?

**Cevap**
`field="data"` argümanı eklenir. Bu argüman, asıl örneklerin JSON içinde hangi anahtarın altında bulunduğunu belirtir.

Örneğin:
`load_dataset("json", data_files="SQuAD_it-train.json", field="data")`

**Soru 5**
`train.json` ve `test.json` dosyalarını tek çağrıda yükleyip `train` ve `test` bölümlerine sahip bir `DatasetDict` nasıl elde edilir?

**Cevap**
`data_files` argümanına, bölüm adlarını dosyalara eşleyen bir sözlük verilir:
`load_dataset("json", data_files={"train": "train.json", "test": "test.json"})`
Sonuç, `train` ve `test` anahtarlarına sahip bir `DatasetDict` olur.

**Soru 6**
Elinizde `.json.gz` gibi sıkıştırılmış bir dosya ya da bu dosyanın uzak bir URL'si var. 🤗 Datasets bu durumda nasıl davranır?

**Cevap**
Dosyayı elle açmanıza gerek yoktur. Yerel yol ya da URL doğrudan `data_files` argümanına verilir; 🤗 Datasets gzip, zip, tar gibi sıkıştırmaları otomatik olarak açar, URL ise önce indirilir.

### 5.3 Dilimleme ve parçalama zamanı

**Soru 7**
`Dataset.map()` fonksiyonunda `batched=True` kullanıldığında fonksiyonun aldığı ve döndürdüğü veri nasıl değişir? Bunun avantajı nedir?

**Cevap**
Fonksiyon tek bir örnek yerine, her sütun için değer listeleri içeren bir sözlük (bir grup örnek) alır ve yine listelerden oluşan bir sözlük döndürmelidir. Bu sayede işlem toplu yapılır ve özellikle hızlı (fast) tokenizer'larla çok daha hızlı çalışır.

**Soru 8**
Uzun metinleri `return_overflowing_tokens=True` ile birden fazla parçaya bölen bir fonksiyonu `map()` ile uyguladığınızda sütun uzunlukları uyuşmadığı için hata alıyorsunuz. Neden olur ve nasıl çözülür?

**Cevap**
Fonksiyon bir örnekten birden fazla satır ürettiği için yeni sütunların satır sayısı eski sütunlarınkiyle uyuşmaz. En yaygın çözüm, `map()` çağrısında `remove_columns=dataset.column_names` ile eski sütunları kaldırmaktır. Alternatif olarak eski sütunlar, yeni satır sayısına uyacak şekilde çoğaltılabilir.

**Soru 9**
`drug_dataset.set_format("pandas")` çalıştırıldıktan sonra `drug_dataset["train"][:]` ne döndürür? Veri seti eski biçimine nasıl döndürülür?

**Cevap**
Dilimleme bir pandas `DataFrame` döndürür; böylece pandas fonksiyonları doğrudan kullanılabilir. Eski (Arrow) biçimine dönmek için `drug_dataset.reset_format()` çağrılır.

### 5.4 Büyük veri? 🤗 Datasets imdada!

**Soru 10**
Pile'ın PubMed Abstracts alt kümesi gibi diskte onlarca GB tutan bir veri seti yüklendiğinde RAM kullanımı neden çok düşük kalır?

**Cevap**
Veri, Apache Arrow formatında diskte tutulur ve bellek eşleme (memory mapping) ile kullanılır. Dosyanın tamamı RAM'e kopyalanmaz; yalnızca ihtiyaç duyulan kısımlar doğrudan diskten okunur.

**Soru 11**
Akış (streaming) modunda yüklenen bir `IterableDataset` üzerinde `shuffle(buffer_size=10_000, seed=42)` çağrıldığında karıştırma nasıl yapılır?

**Cevap**
Veri setinin tamamı indirilip karıştırılmaz. Örnekler 10.000 elemanlık bir tampona (buffer) alınır ve sıradaki örnek bu tampondan rastgele seçilir. Yani karıştırma yaklaşık (yerel) olarak yapılır; tampon büyüdükçe karıştırma kalitesi artar.

**Soru 12**
Akış modundaki bir veri setinin ilk 1.000 örneği doğrulama kümesi, geri kalanı eğitim kümesi olacak şekilde nasıl ayrılır?

**Cevap**
`IterableDataset` indekslenemediği için dilimleme (`dataset[:1000]`) kullanılamaz. Bunun yerine:
`valid = dataset.take(1000)` ve `train = dataset.skip(1000)`
kullanılır.

### 5.5 Kendi veri setinizi oluşturmak

**Soru 13**
GitHub REST API'den issue çekerken kişisel erişim token'ı (`GITHUB_TOKEN`) kullanmanın temel nedeni nedir?

**Cevap**
Kimliği doğrulanmamış istekler saatte çok az sayıyla (60) sınırlıdır. Token ile yapılan isteklerde bu limit saatte yaklaşık 5.000'e çıkar; böylece binlerce issue'ya sahip depolar da çekilebilir.

**Soru 14**
GitHub'ın issues uç noktası pull request'leri de döndürür. Kursta issue'lar ile pull request'ler nasıl ayırt edilir?

**Cevap**
Pull request kayıtlarında `pull_request` alanı doludur, normal issue'larda boştur. Bu alana bakan bir fonksiyon `map()` ile uygulanır ve veri setine `is_pull_request` adlı yeni bir sütun eklenir.

**Soru 15**
Oluşturduğunuz veri setini `push_to_hub()` ile Hugging Face Hub'a yüklemeden önce hangi adım gereklidir? Neden?

**Cevap**
Hugging Face hesabına giriş yapılmalıdır: terminalde `huggingface-cli login`, notebook'ta `notebook_login()`. Hub'a yazma işlemi kimlik doğrulama gerektirir; giriş yapılmadan yükleme reddedilir.

### 5.6 FAISS ile anlamsal arama

**Soru 16**
Kursta metin embedding'leri oluşturulurken kullanılan "CLS pooling" yöntemi nedir?

**Cevap**
Modelin son gizli katmanındaki ilk token'ın, yani `[CLS]` token'ının vektörünün tüm metnin temsili olarak alınmasıdır:
`model_output.last_hidden_state[:, 0]`

**Soru 17**
Her issue'nun birden fazla yorum içeren `comments` sütunu, her satırda tek bir yorum olacak şekilde nasıl açılır? Bu neden yapılır?

**Cevap**
Veri seti pandas biçimine getirilir ve `DataFrame.explode("comments")` kullanılır; ardından `Dataset.from_pandas()` ile tekrar veri setine dönülür. Böylece her yorum ayrı bir satır olur ve her biri için ayrı bir embedding hesaplanabilir.

**Soru 18**
`get_nearest_examples("embeddings", question_embedding, k=5)` çağrısının çalışabilmesi için önce ne yapılmalıdır? Bu çağrı ne döndürür?

**Cevap**
Önce `add_faiss_index(column="embeddings")` ile embedding sütunu üzerinde bir FAISS indeksi oluşturulmalıdır. Çağrı, sorguya en benzer 5 örneği ve bunların benzerlik skorlarını döndürür.

### 5.7 🤗 Datasets, tamam!

**Soru 19**
İşlenmiş bir veri setini diske kaydetmek ve daha sonra en hızlı şekilde geri yüklemek için hangi metot çifti kullanılır?

**Cevap**
`save_to_disk()` ile kaydedilir, `load_from_disk()` ile geri yüklenir. Veri Arrow formatında saklandığı için yeniden yükleme JSON veya CSV'ye göre çok daha hızlıdır.

**Soru 20**
Bölüm 5'te öğrenilen tipik bir veri hazırlama akışını sırasıyla yazınız.

**Cevap**
Yükle (`load_dataset`) → Temizle/Dönüştür (`filter`, `map`) → Böl (`train_test_split`) → Kaydet (`save_to_disk`) veya paylaş (`push_to_hub`)

**Soru 21**
Birden fazla akış (streaming) veri seti tek bir veri setinde nasıl birleştirilir? Bu birleştirme nasıl çalışır?

**Cevap**
`interleave_datasets()` fonksiyonu kullanılır. Bu fonksiyon örnekleri kaynak veri setlerinden sırayla (dönüşümlü) alır; sonuç da yine bir `IterableDataset` olur.

[GitHub'da güncelle](https://github.com/huggingface/course/blob/main/chapters/en/chapter5/8.mdx)
