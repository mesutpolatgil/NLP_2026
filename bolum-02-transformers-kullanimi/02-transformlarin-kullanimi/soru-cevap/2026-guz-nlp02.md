> **Uyarı:** Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.


### 1. Dil modelleme pipeline'ının sırası nedir?

**A)** Model, metni işler ve ham tahminler döndürür. Daha sonra tokenizer bu tahminleri anlamlandırır ve gerektiğinde tekrar metne dönüştürür.

**B)** Önce tokenizer metni işler ve ID'ler döndürür. Model bu ID'leri işler ve bir tahmin çıktısı üretir; bu çıktı bir metin olabilir.

**C)** Tokenizer metni işler ve ID'ler döndürür. Model bu ID'leri işler ve bir tahmin çıktısı üretir. Daha sonra tokenizer, bu tahminleri tekrar bir metne dönüştürmek için kullanılabilir. ✅

**📝 Not:**  
`Text → Tokenizer → IDs → Model → Prediction`

**A yanlış:** Model doğrudan ham metinle başlamaz.  
**B eksik:** Çıktının tekrar metne çevrilmesini belirtmez.

---

### 2. Temel Transformer modelinin ürettiği tensor kaç boyutludur ve bu boyutlar nelerdir?

**A)** 2: Sequence length (dizi uzunluğu) ve batch size (batch boyutu)

**B)** 2: Sequence length (dizi uzunluğu) ve hidden size (gizli katman boyutu)

**C)** 3: Sequence length (dizi uzunluğu), batch size (batch boyutu) ve hidden size (gizli katman boyutu) ✅

**📝 Not:**  
Transformer çıktısı 3 boyutludur:  
`(Batch Size, Sequence Length, Hidden Size)`

- **Batch Size:** Aynı anda işlenen örnek sayısı.
- **Sequence Length:** Bir girdideki token sayısı.
- **Hidden Size:** Her tokenın model tarafından oluşturulan temsilinin boyutu.

**A yanlış:** Hidden size eksik.  
**B yanlış:** Batch size eksik.

---

### 3. Aşağıdakilerden hangisi subword tokenization'a bir örnektir?

**A)** WordPiece

**B)** Character-based Tokenization (Karakter tabanlı tokenization)

**C)** Splitting on whitespace and punctuation (Boşluk ve noktalama işaretlerine göre ayırma)

**D)** BPE

**E)** Unigram

**F)** Yukarıdakilerin hiçbiri

**Cevap:** **A, D ve E** ✅

**📝 Not:**  
Subword tokenization, kelimeleri daha küçük anlamlı parçalara ayırır. **WordPiece, BPE ve Unigram** subword tokenization yöntemleridir.

WordPiece = Kelimeleri daha küçük alt kelime parçalarına (subword) ayırır.
Character-based Tokenization = Kelimeleri tek tek karakterlere ayırır.
Whitespace & Punctuation = Metni boşluk ve noktalama işaretlerine göre böler.
BPE (Byte Pair Encoding) = Kelimeleri sık kullanılan alt kelime parçalarına ayırır.
Unigram = Kelimeleri oluşturabilecek uygun alt kelime parçalarını seçerek böler.
Subword Tokenization = Kelimeleri tamamen tek parça almak yerine daha küçük anlamlı parçalara böler

---

### 4. Model head nedir?

**A)** Temel Transformer ağının, tensorleri doğru katmanlara yönlendiren bir bileşenidir.

**B)** Self-attention mekanizması olarak da bilinir; bir tokenın temsilini dizideki diğer tokenlara göre uyarlar.

**C)** Transformer'ın tahminlerini göreve özel bir çıktıya dönüştürmek için kullanılan, genellikle bir veya birkaç katmandan oluşan ek bir bileşendir. ✅

**📝 Not:**  
Model head, temel Transformer'ın üzerine eklenir ve çıktıyı belirli bir göreve uygun hale getirir.

Örneğin:  
`Transformer → Classification Head → Positive / Negative`

**A yanlış:** Model head'in görevi tensorleri katmanlara yönlendirmek değildir.  
**B yanlış:** Self-attention, Transformer'ın bir mekanizmasıdır; model head değildir.

---

### 5. AutoModel nedir?

**A)** Verileriniz üzerinde otomatik olarak eğitim yapan bir modeldir.

**B)** Checkpoint'e göre doğru mimariyi döndüren bir nesnedir. ✅

**C)** Girdilerinde kullanılan dili otomatik olarak tespit ederek doğru ağırlıkları yükleyen bir modeldir.

**📝 Not:**  
AutoModel, verilen **checkpoint'e bakarak uygun model mimarisini seçer ve yükler.**

Örneğin:  
`"bert-base-cased" → BERT mimarisi`

**A yanlış:** AutoModel model eğitmez; önceden eğitilmiş modeli yükler.  
**C yanlış:** Girdinin dilini algılayıp model seçmez; checkpoint zaten belirtilmiştir.

---

### 6. Farklı uzunluktaki dizileri birlikte batch'lerken hangi tekniklerin farkında olunmalıdır?

**A)** Kırpma (Truncating) ✅

**B)** Tensor döndürme (Returning tensors)

**C)** Padding (Doldurma) ✅

**D)** Attention masking (Dikkat maskesi) ✅

**📝 Not:**  
Farklı uzunluktaki dizileri aynı batch'te işlemek için **padding**, fazla uzun dizileri sınırlamak için **truncation**, padding tokenlarının dikkate alınmaması için **attention mask** kullanılır.

**A doğru:** Uzun dizileri belirlenen maksimum uzunluğa indirir.  
**B yanlış:** Tensor formatında çıktı almakla ilgilidir; farklı uzunluktaki dizileri yönetmez.  
**C doğru:** Kısa dizileri aynı uzunluğa getirmek için kullanılır.  
**D doğru:** Padding tokenlarının model tarafından dikkate alınmasını engeller.

---

### 7. Bir sequence classification modelinin çıktısı olan logits değerlerine SoftMax fonksiyonunu uygulamanın amacı nedir?

**A)** Logits değerlerini daha güvenilir hale getirmek için yumuşatır.

**B)** Anlaşılabilir olmaları için bir alt ve üst sınır uygular.

**C)** Çıktı değerlerinin toplamı 1 olur ve böylece olasılıksal olarak yorumlanabilir. ✅

**📝 Not:**  
**Logits**, modelin sınıflar için ürettiği ham değerlerdir. SoftMax bu değerleri **olasılıklara** dönüştürür ve çıktıların toplamı **1** olur.

Örneğin:  
`Positive: 0.8 | Negative: 0.2`

**A yanlış:** SoftMax'ın amacı yalnızca değerleri "yumuşatmak" değildir.  
**B yanlış:** Temel amaç alt-üst sınır belirlemek değil, değerleri olasılık olarak yorumlanabilir hale getirmektir.

---

### 8. Tokenizer API'sinin büyük bölümü hangi method etrafında şekillenmiştir?

**A)** `encode`, çünkü metni ID'lere ve ID'leri tahminlere dönüştürebilir.

**B)** Tokenizer nesnesini doğrudan çağırmak. ✅

**C)** `pad`

**D)** `tokenize`

**📝 Not:**  
Tokenizer API'sinin merkezinde **`__call__` methodu** bulunur. Tokenizer'ı doğrudan çağırarak metni modele uygun şekilde işleyebiliriz.

Örneğin:  
`tokenizer("Hello!", return_tensors="pt")`

**A yanlış:** `encode` metni ID'lere dönüştürebilir ancak API'nin merkezindeki method değildir.  
**C yanlış:** `pad` yalnızca padding işlemi için kullanılır.  
**D yanlış:** `tokenize` metni tokenlara ayırır ancak API'nin temel methodu değildir.

---

### 9. Bu kod örneğinde `result` değişkeni ne içerir?

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
```

**A)** Her bir stringin bir token olduğu bir string listesi. ✅

**B)** Bir ID listesi.

**C)** Tüm tokenları içeren bir string.

📝 Not:
tokenize() metni tokenlara ayırır ve sonucu bir liste olarak döndürür.

Örneğin:
"Hello!" → ["Hello", "!"]

A doğru: Sonuç, her elemanı bir token olan string listesidir.
B yanlış: ID'lere dönüştürme işlemi tokenize() tarafından yapılmaz.
C yanlış: Sonuç tek bir string değil, tokenlardan oluşan bir listedir.

---

### 10. Aşağıdaki kodda yanlış olan bir şey var nedir?

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
```

**A)** Hayır, kod doğru görünüyor.

**B)** Tokenizer ve model her zaman aynı checkpoint'ten olmalıdır. ✅

**C)** Her girdi bir batch olduğu için tokenizer ile padding ve truncation yapmak iyi bir uygulamadır.

📝 Not:
Tokenizer ve modelin uyumlu checkpointlerden olması gerekir. Burada tokenizer BERT, model ise GPT-2 olduğu için uyumsuzdur.

BERT Tokenizer → ❌ → GPT-2 Model

A yanlış: Farklı checkpoint kullanıldığı için kodda uyumsuzluk vardır. C yanlış: Padding ve truncation her durumda zorunlu değildir; asıl problem tokenizer-model uyumsuzluğudur

# End-of-Chapter Quiz – Bizim Sorularımız

## Soru 1

*Farklı uzunluktaki metinler aynı batch içinde verilirken neden padding yapılır? Bu işlemde attention_mask ne işe yarar?*

### Cevap

Modeller verileri tensör olarak paralel işlediğinden, batch içindeki tüm dizilerin aynı boyutta olması gerekir; kısa diziler padding ile tamamlanır. attention_mask ise modele hangi token'ların gerçek veri (1), hangilerinin dolgu (0) olduğunu bildirir; modelin boş dolguları dikkate almasını engeller.
---

## Soru 2

*AutoModel ile AutoModelForSequenceClassification arasındaki temel fark nedir?*

### Cevap

AutoModel: Temel Transformer gövdesini yükler ve gizli durumları (hidden states) çıktı olarak verir.

AutoModelForSequenceClassification: Temel modelin üzerine sınıflandırma katmanı (head) eklenmiş halidir; doğrudan sınıf skorlarını (logits) üretir.

---

## Soru 3

*Metin üretiminde temperature parametresi neyi kontrol eder? Değeri düşürmek çıktıyı nasıl etkiler?*


### Cevap

temperature, modelin token seçimindeki rastlantısallığı ve yaratıcılığı kontrol eder. Değer düştükçe (0'a yaklaştıkça) üretim daha kararlı, tahmin edilebilir ve kuralcı hale gelir; değer yükseldikçe çeşitlilik ve rastlantısallık artar.

