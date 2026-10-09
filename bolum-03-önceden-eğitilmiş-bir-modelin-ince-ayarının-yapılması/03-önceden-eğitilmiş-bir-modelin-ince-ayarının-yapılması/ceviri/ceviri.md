Uyarı: Bu içerik, SCÜ Şarkışla UBYO Doğal Dil İşleme dersi kapsamında tamamen eğitim amaçlı çevrilmiş ve derlenmiştir. Orijinal dokümantasyon kaynakları (Hugging Face LLM Course, Transformers, Datasets, Tokenizers kütüphaneleri) kendi orijinal lisanslarına (Apache 2.0, MIT, CC-BY 4.0) tabidir. Bu çalışmanın hiçbir ticari amacı yoktur.

# Bölüm 3 — İnce Ayar: Veri Hazırlama, Eğitim ve Değerlendirme

## İçindekiler

- [Giriş](#giriş)
- [Veriyi işleme](#veriyi-işleme)
- [Trainer API ile modelin ince ayarı](#trainer-api-ile-modelin-ince-ayar-)
- [Özel eğitim döngüleri](#özel-eğitim-döngüleri)
- [Öğrenme eğrilerini anlama](#öğrenme-eğrilerini-anlama)
- [Bölümün özeti](#ince-ayar-tamam)
- [Bölüm sonu sertifikası](#bölüm-sonu-sertifikası)

## Giriş

Bölüm 2’de tahmin üretmek için tokenizer’ların ve önceden eğitilmiş modellerin nasıl kullanılacağını inceledik. Peki, önceden eğitilmiş bir modeli belirli bir görevi çözmek üzere ince ayardan geçirmek istersek ne yapmalıyız? Bu bölümün konusu budur. Bölüm sonunda şunları öğrenmiş olacaksınız:

- Hub’daki büyük bir veri kümesini Datasets kütüphanesinin güncel özellikleriyle hazırlamayı,
- Trainer API’nin üst düzey arayüzünü ve güncel iyi uygulamaları kullanarak model ince ayarı yapmayı,
- Optimizasyon teknikleriyle özel bir eğitim döngüsü uygulamayı,
- Accelerate kütüphanesiyle farklı donanım düzenlerinde dağıtık eğitimi kolayca çalıştırmayı,
- En iyi performans için güncel ince ayar uygulamalarını kullanmayı.

**Temel kaynaklar:** Başlamadan önce veri işleme hakkındaki 🤗 Datasets belgelerini gözden geçirmek faydalı olabilir.

Bu bölüm, Transformers dışındaki bazı Hugging Face kütüphanelerini de tanıtacaktır. Datasets, Tokenizers, Accelerate ve Evaluate gibi kütüphanelerin modelleri daha verimli ve etkili biçimde eğitmeye nasıl yardımcı olduğunu göreceğiz.

Bölümün ana kısımları farklı konulara odaklanır:

- **Bölüm 2:** Modern veri ön işleme teknikleri ve verimli veri kümesi yönetimi.
- **Bölüm 3:** Trainer API ve güncel özellikleri.
- **Bölüm 4:** Eğitim döngülerini sıfırdan uygulama ve Accelerate ile dağıtık eğitim.

Bölüm sonunda, kendi veri kümeleriniz üzerinde hem üst düzey API’lerle hem de özel eğitim döngüleriyle, güncel iyi uygulamaları izleyerek model ince ayarı yapabileceksiniz.

> **Bu bölümde ne geliştireceksiniz?** Bir BERT modelini metin sınıflandırması için ince ayardan geçirecek ve teknikleri kendi veri kümelerinize ve görevlerinize uyarlamayı öğreneceksiniz.

Bu bölüm yalnızca PyTorch’a odaklanır. PyTorch, modern derin öğrenme araştırmalarında ve üretim ortamlarında standart çerçeve hâline gelmiştir. Hugging Face ekosisteminin güncel API’lerini ve iyi uygulamalarını kullanacağız.

Eğittiğiniz modelleri Hugging Face Hub’a yüklemek için bir Hugging Face hesabına ihtiyacınız vardır.

## Veriyi işleme

Önceki bölümdeki örneği sürdürerek, tek bir yığın (batch) üzerinde bir dizi sınıflandırıcısını şöyle eğitebiliriz:

```python
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Önceki bölümdekiyle aynı
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# Bu kısım yeni
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

Yalnızca iki cümleyle eğitim yapmak iyi sonuç vermez. Daha iyi sonuçlar için daha büyük bir veri kümesi hazırlamak gerekir.

Bu bölümde örnek olarak MRPC (Microsoft Research Paraphrase Corpus) veri kümesini kullanacağız. William B. Dolan ve Chris Brockett’in makalesinde tanıtılan veri kümesi, cümle çiftlerinden oluşur. Her çifte, cümlelerin aynı anlama gelip gelmediğini belirten bir etiket atanmıştır. Veri kümesi 5.801 çift içerir. Küçük olduğu için üzerinde deney yapmak kolaydır.

### Hub’dan veri kümesi yükleme

Hub yalnızca modelleri barındırmaz; farklı dillerde pek çok veri kümesi de içerir. Bu bölümü tamamladıktan sonra yeni bir veri kümesini yükleyip işlemeyi deneyebilirsiniz. Şimdilik GLUE karşılaştırma ölçütünün 10 veri kümesinden biri olan MRPC’ye odaklanalım. GLUE, makine öğrenmesi modellerinin 10 metin sınıflandırma görevi üzerindeki başarısını ölçmek için kullanılan akademik bir karşılaştırma ölçütüdür.

Datasets kütüphanesi, Hub’dan veri kümesi indirip önbelleğe almak için basit bir komut sunar:

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

Sonuç bir `DatasetDict` nesnesidir. Eğitim, doğrulama ve test kümelerini; her biri `sentence1`, `sentence2`, `label` ve `idx` sütunlarını içerir:

```text
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

`DatasetDict`, eğitim kümesinde 3.668, doğrulama kümesinde 408 ve test kümesinde 1.725 cümle çifti barındırır. Komut, verileri varsayılan olarak `~/.cache/huggingface/datasets` dizinine indirip önbelleğe alır. `HF_HOME` ortam değişkenini ayarlayarak önbellek klasörünü değiştirebilirsiniz.

Sözlükte olduğu gibi indeksleme yaparak ham veri kümesindeki bir cümle çiftine erişebiliriz:

```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

Örnek çıktı:

```python
{
    'idx': 0,
    'label': 1,
    'sentence1': 'Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.',
    'sentence2': 'Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.'
}
```

Etiketler zaten tamsayıdır; bu nedenle ön işleme gerektirmez. Hangi tamsayının hangi etikete karşılık geldiğini sütun özelliklerinden görebiliriz:

```python
raw_train_dataset.features
```

```python
{
    'sentence1': Value(dtype='string', id=None),
    'sentence2': Value(dtype='string', id=None),
    'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
    'idx': Value(dtype='int32', id=None)
}
```

Arka planda `label`, `ClassLabel` türündedir ve tamsayı–etiket eşlemesi adlar alanında saklanır: `0`, `not_equivalent` (eşdeğer değil); `1`, `equivalent` (eşdeğer) anlamına gelir.

> **Deneyin:** Eğitim kümesinin 15. ve doğrulama kümesinin 87. öğesine bakın. Etiketleri nedir?

### Veri kümesini ön işleme

Modelin anlayabileceği biçime getirmek için metni sayılara dönüştürmemiz gerekir. Bunu tokenizer ile yaparız. Tokenizer’a tek cümle ya da cümle listesi verebiliriz; her cümle çiftinin ilk ve ikinci cümlelerini ayrı ayrı tokenize edebiliriz:

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```

> **Derinlemesine inceleme:** Daha gelişmiş tokenizasyon teknikleri ve farklı tokenizer’ların çalışma biçimi için 🤗 Tokenizers belgelerine ve yemek kitabındaki tokenizasyon kılavuzuna bakın.

İki diziyi modele ayrı ayrı verip cümlelerin eş anlamlı olup olmadığını tahmin edemeyiz. Dizileri çift olarak ele almalı ve BERT’in beklediği biçimde ön işlemeliyiz. Tokenizer, iki diziyi birlikte alabilir:

```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

```python
{
    'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

`input_ids` ve `attention_mask` anahtarlarını Bölüm 2’de ele almıştık. Buradaki `token_type_ids`, girdinin hangi bölümünün birinci, hangisinin ikinci cümle olduğunu gösterir.

> **Deneyin:** Eğitim kümesinin 15. öğesini alın. İki cümleyi ayrı ayrı ve çift olarak tokenize edin. Sonuçlar arasındaki fark nedir?

`input_ids` içindeki kimlikleri tekrar sözcüklere dönüştürürsek:

```python
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
```

şu sonucu elde ederiz:

```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]',
 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
```

İki cümle verildiğinde model girdiyi `[CLS] cümle1 [SEP] cümle2 [SEP]` biçiminde bekler. `token_type_ids` ile eşleştirildiğinde ilk cümle ve ayraçlarının `0`, ikinci cümle ve ayraçlarının `1` kimliğini aldığı görülür.

Başka bir kontrol noktası (checkpoint) seçerseniz, tokenize edilmiş girdide `token_type_ids` bulunmayabilir. Örneğin DistilBERT kullanıldığında bu alan döndürülmez. Bu kimlikler yalnızca modelin ön eğitim sırasında öğrendiği ve kullanabildiği durumlarda döndürülür.

BERT, ön eğitimde token türü kimliklerini kullanır. Bölüm 1’de ele aldığımız maskeli dil modelleme amacına ek olarak, cümle çiftleri arasındaki ilişkiyi modelleyen **sonraki cümle tahmini** (next sentence prediction) amacı da vardır. Model, rastgele maskelenmiş token’lar içeren bir cümle çifti alır ve ikinci cümlenin birincisini izleyip izlemediğini tahmin eder. Görevin kolay olmaması için örneklerin yarısında cümleler özgün belgede art arda gelir, diğer yarısında ise farklı belgelerden seçilir.

Tokenizer ve model için aynı checkpoint’i kullandığınız sürece `token_type_ids` olup olmadığını genellikle ayrıca yönetmeniz gerekmez: tokenizer, modelin beklediği alanları sağlar.

Tokenizer’ın bir cümle çiftini nasıl işlediğini gördük. Şimdi tüm veri kümesini, ilk cümleleri ve ardından ikinci cümleleri liste olarak vererek tokenize edebiliriz. Dolgu (padding) ve kesme (truncation) seçenekleri de kullanılabilir:

```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

Bu yöntem `input_ids`, `attention_mask` ve `token_type_ids` alanlarını içeren, değerleri liste listeleri olan bir sözlük döndürür. Tüm veri kümesini tokenizasyon sırasında bellekte tutacak kadar RAM yoksa yöntem çalışmaz. Datasets kütüphanesinin veri kümeleri diskte Apache Arrow dosyaları olarak tutulur; belleğe yalnızca istenen örnekler alınır.

Veriyi veri kümesi biçiminde tutmak için `Dataset.map()` yöntemini kullanabiliriz. Bu yöntem, tokenizasyondan başka ön işleme adımlarına da esneklik sağlar. `map()`, tanımladığımız işlevi veri kümesinin her öğesine uygular:

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

İşlev, veri kümesindeki bir öğe gibi bir sözlük alır ve `input_ids`, `attention_mask` ile `token_type_ids` anahtarlarını içeren yeni bir sözlük döndürür. Örnek sözlük birden çok örnek içerdiğinde de çalışır; çünkü tokenizer cümle çiftleri listelerini işleyebilir. Böylece `map()` çağrısında `batched=True` kullanabilir ve tokenizasyonu önemli ölçüde hızlandırabiliriz. Tokenizer, 🤗 Tokenizers kütüphanesindeki Rust ile yazılmış bir tokenizer üzerine kuruludur. Çok sayıda girdiyi bir arada verdiğimizde hızından en iyi şekilde yararlanırız.

Şimdilik tokenizasyon işlevine `padding` parametresi eklemedik. Tüm örnekleri veri kümesindeki en uzun örneğe göre doldurmak verimli değildir. Örnekleri yığın hâline getirirken, her yığında yalnızca o yığının en uzun örneğine kadar doldurmak daha iyidir. Uzunlukları çok değişen girdilerde bu yaklaşım zaman ve işlem gücü tasarrufu sağlar.

> **Performans ipuçları:** Verimli veri işleme için 🤗 Datasets performans kılavuzuna bakın.

`batched=True`, işlevin örnekleri tek tek değil, aynı anda birden çok örnek üzerinde çalışmasını sağlar. Bu da ön işlemeyi hızlandırır:

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

Datasets kütüphanesi, ön işleme işlevinin döndürdüğü sözlükteki her anahtar için veri kümesine yeni bir alan ekler:

```text
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```

`map()` çağrısına `num_proc` argümanı vererek ön işleme sırasında çoklu işlem de kullanabilirsiniz. Bu örnekte bunu yapmıyoruz; Tokenizers kütüphanesi örnekleri daha hızlı tokenize etmek için zaten birden fazla iş parçacığı kullanıyor. Bu kütüphaneyi temel almayan hızlı olmayan bir tokenizer kullanıyorsanız, çoklu işlem hız kazandırabilir.

`tokenize_function`, `input_ids`, `attention_mask` ve `token_type_ids` anahtarlarını döndürdüğünden bu üç alan veri kümesinin tüm bölümlerine eklenir. Ön işleme işlevi, mevcut bir alan için yeni değer döndürerek onu değiştirmek üzere de kullanılabilir.

Son olarak, örnekleri bir araya getirirken tümünü yığındaki en uzun öğenin uzunluğuna tamamlamamız gerekir. Bu tekniğe **dinamik dolgu** denir.

### Dinamik dolgu

Bir yığındaki örnekleri bir araya getiren işleve **collate function** (birleştirme işlevi) denir. Bu işlevi `DataLoader` oluştururken argüman olarak verebilirsiniz. Varsayılan işlev örnekleri PyTorch tensörlerine dönüştürüp birleştirir. Girdilerimizin uzunlukları farklı olduğu için varsayılan yöntem burada uygun değildir. Dolguyu erteleyip yalnızca gerektiğinde, her yığında uyguluyoruz; böylece gereksiz dolgu token’ları azalır ve eğitim hızlanır. TPU kullanırken bu yaklaşım sorun çıkarabilir: TPU’lar ek dolgu gerektirse bile sabit şekilleri tercih eder.

> **Optimizasyon kılavuzu:** Dolgu stratejileri ve TPU kullanımı dâhil eğitim performansını iyileştirme hakkında 🤗 Transformers performans belgelerine bakın.

Datasets örneklerini bir yığına eklerken gereken miktarda dolgu uygulayan bir birleştirme işlevi tanımlamamız gerekir. Transformers kütüphanesindeki `DataCollatorWithPadding`, kullanılacak dolgu token’ını ve modelin girdileri soldan mı sağdan mı doldurmayı beklediğini tokenizer’dan öğrenir:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Eğitim kümesinden birkaç örnek alıp yığın oluşturalım. `idx`, `sentence1` ve `sentence2` sütunları gerekmeyecekleri ve metin oldukları için çıkarılır; dizelerin tensöre dönüştürülmesi mümkün değildir. Ardından yığındaki girdilerin uzunluklarına bakalım:

```python
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
```

```text
[50, 59, 47, 67, 59, 50, 62, 32]
```

Örneklerin uzunluğu 32 ile 67 arasında değişir. Dinamik dolgu, bu yığındaki örneklerin tümünün en uzun örnek olan 67 token’a tamamlanması demektir. Dinamik dolgu olmasaydı örneklerin tümünü veri kümesindeki en uzun diziye ya da modelin kabul edebileceği azami uzunluğa kadar doldurmak gerekirdi. Birleştirme işlevinin dinamik dolguyu doğru uyguladığını doğrulayalım:

```python
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

```text
{
    'attention_mask': torch.Size([8, 67]),
    'input_ids': torch.Size([8, 67]),
    'token_type_ids': torch.Size([8, 67]),
    'labels': torch.Size([8])
}
```

Ham metinden modelin işleyebileceği yığınlara geçtik; artık ince ayar yapmaya hazırız.

> **Deneyin:** Ön işleme adımlarını GLUE SST-2 veri kümesine uygulayın. Bu veri kümesi cümle çiftleri yerine tek cümlelerden oluşur; diğer adımlar benzer olmalıdır. Daha zor bir çalışma olarak, GLUE görevlerinden herhangi biriyle çalışabilecek bir ön işleme işlevi yazın.
>
> **Ek alıştırma:** 🤗 Transformers örneklerindeki uygulamalı çalışmalara göz atın.

Datasets kütüphanesinin güncel iyi uygulamalarıyla verileri ön işledik. Sırada, Hugging Face ekosisteminin güncel özellikleri ve optimizasyonlarıyla modelimizi eğitmek için modern Trainer API var.

### Bölüm değerlendirmesi

# Hugging Face Datasets & Tokenization Quiz

## 1. Dataset.map() fonksiyonunu batched=True ile kullanmanın temel avantajı nedir?

- A) Daha az bellek kullanır.
- **B) Birden fazla örneği aynı anda işler, tokenization'ı çok daha hızlı yapar.**
- C) Padding işlemini otomatik halleder.
- D) Veriyi PyTorch tensörlerine dönüştürür.

**Cevap: B**

---

## 2. Neden veri setindeki tüm dizileri maksimum uzunluğa kadar padding yapmak yerine dinamik padding kullanıyoruz?

- A) Dinamik padding model mimarisi tarafından zorunludur.
- **B) Sadece her batch içindeki maksimum uzunluğa kadar padding yaparak hesaplama yükünü azaltır.**
- C) Model doğruluğunu artırır.
- D) DataCollatorWithPadding kullanılırken zorunludur.

**Cevap: B**

---

## 3. BERT tokenization'da token_type_ids alanı neyi temsil eder?

- A) Her token'ın dizi içindeki konumunu.
- **B) Cümle çiftleri işlenirken her token'ın hangi cümleye ait olduğunu.**
- C) Her token için attention mask'i.
- D) Her token'ın kelime dağarcığındaki (vocabulary) ID'sini.

**Cevap: B**

---

## 4. load_dataset('glue', 'mrpc') ile bir veri seti yüklenirken ikinci argüman neyi belirtir?

- A) Yüklenecek veri setinin sürümünü.
- **B) GLUE benchmark'ı içindeki belirli görevi veya alt kümeyi.**
- C) Veri setinin split'ini (train/validation/test).
- D) Verinin döndürüleceği formatı.

**Cevap: B**

---

## 5. Eğitimden önce 'sentence1' ve 'sentence2' gibi sütunları kaldırmanın amacı nedir?

- A) Eğitim sırasında bellek tasarrufu sağlamak.
- **B) Model bu ham metin sütunlarını beklemez ve hata verir.**
- C) Bu sütunlar değerlendirme için gerekli değildir.
- D) Eğitim hızını önemli ölçüde artırır.

**Cevap: B**
## Trainer API ile modelin ince ayarı

Transformers, sunduğu önceden eğitilmiş modelleri kendi veri kümeniz üzerinde güncel iyi uygulamalarla ince ayardan geçirmek için `Trainer` sınıfını sağlar. Önceki bölümdeki veri ön işleme tamamlandıktan sonra `Trainer`’ı tanımlamak için birkaç adım kalır. En zor kısım muhtemelen `Trainer.train()` çalıştırılacak ortamı hazırlamaktır; CPU üzerinde eğitim çok yavaş ilerler. GPU’nuz yoksa Google Colab üzerinden ücretsiz GPU veya TPU erişimi edinebilirsiniz.

> **Eğitim kaynakları:** Eğitime başlamadan önce 🤗 Transformers eğitim kılavuzuna göz atın ve ince ayar yemek kitabındaki uygulamalı örnekleri inceleyin.

Aşağıdaki kod örneklerinde, önceki bölümdeki kodu çalıştırmış olduğunuz varsayılır. Gerekli adımların özeti şöyledir:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Eğitim

`Trainer`’ı tanımlamadan önce, eğitim ve değerlendirme sırasında kullanılacak hiperparametreleri içeren `TrainingArguments` sınıfını tanımlarız. Sağlamanız gereken tek zorunlu argüman, eğitilmiş modelin ve süreç boyunca alınan kontrol noktalarının kaydedileceği dizindir. Temel bir ince ayar için diğer varsayılan değerler genellikle yeterlidir:

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

Eğitim sırasında modelinizi otomatik olarak Hub’a yüklemek istiyorsanız `TrainingArguments` içine `push_to_hub=True` ekleyin. Bu konuyu Bölüm 4’te ele alacağız.

> **Gelişmiş yapılandırma:** Kullanılabilir tüm eğitim argümanları ve optimizasyon stratejileri için `TrainingArguments` belgelerine ve eğitim yapılandırması yemek kitabına bakın.

İkinci adım, modeli tanımlamaktır. Önceki bölümde olduğu gibi iki etiketli `AutoModelForSequenceClassification` sınıfını kullanacağız:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)
```

Önceden eğitilmiş modeli oluşturduğunuzda Bölüm 2’dekinden farklı olarak bir uyarı alırsınız. BERT, cümle çiftlerini sınıflandırma görevi üzerinde önceden eğitilmemiştir. Bu nedenle ön eğitim başlığı kaldırılır ve dizi sınıflandırmasına uygun yeni bir başlık eklenir. Uyarılar, bazı ağırlıkların (kaldırılan ön eğitim başlığına ait olanların) kullanılmadığını, yeni başlığa ait bazı ağırlıkların ise rastgele başlatıldığını bildirir. Modeli eğitmeniz gerektiğini belirten mesaj, şimdi yapacağımız işi anlatır.

Model hazır olduğunda, şimdiye kadar oluşturduğumuz nesneleri `Trainer`’a veririz: model, `training_args`, eğitim ve doğrulama veri kümeleri, `data_collator` ve `processing_class`. Daha yeni eklenen `processing_class` parametresi, `Trainer`’a hangi tokenizer’ın kullanılacağını bildirir:

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)
```

`processing_class` olarak tokenizer verdiğinizde, `Trainer`’ın varsayılan veri birleştirme işlevi `DataCollatorWithPadding` olur. Bu durumda `data_collator=data_collator` satırını atlayabilirsiniz. Veri işleme hattındaki bu önemli adımı göstermek için örnekte tuttuk.

> **Daha fazla bilgi:** `Trainer` sınıfı ve parametreleri için Trainer API belgelerine ve eğitim yemek kitabındaki gelişmiş kullanım örüntülerine bakın.

Veri kümesi üzerinde ince ayar yapmak için `Trainer`’ın `train()` yöntemini çağırmamız yeterlidir:

```python
trainer.train()
```

Bu, ince ayarı başlatır (GPU üzerinde birkaç dakika sürmesi beklenir) ve her 500 adımda eğitim kaybını bildirir. Ancak modelin ne kadar iyi ya da kötü performans gösterdiğini söylemez. Bunun iki nedeni vardır:

1. `TrainingArguments` içindeki `eval_strategy` değerini `"steps"` (belirlenen `eval_steps` aralığında) ya da `"epoch"` (her dönemin sonunda) yaparak eğitim sırasında değerlendirme yapılmasını istemedik.
2. Değerlendirme sırasında ölçüt hesaplayacak bir `compute_metrics()` işlevi vermedik. Bu işlev olmadan değerlendirme yalnızca kaybı yazdırır; bu sayı tek başına yorumlamak için çok sezgisel değildir.

### Değerlendirme

Yararlı bir `compute_metrics()` işlevinin nasıl oluşturulacağına bakalım. İşlev, `predictions` ve `label_ids` alanları olan adlandırılmış bir demet (`EvalPrediction`) alır. Ölçüt adlarını anahtar, değerlerini kayan noktalı sayı olarak içeren bir sözlük döndürür.

Modelden tahmin almak için `Trainer.predict()` komutunu kullanabiliriz:

```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

```text
(408, 2) (408,)
```

`predict()` yöntemi üç alanı olan başka bir adlandırılmış demet döndürür: `predictions`, `label_ids` ve `metrics`. `metrics` alanında veri kümesindeki kayıp ile tahmin süresine ait ölçümler (toplam ve ortalama süre) bulunur. `compute_metrics()` işlevini tamamlayıp `Trainer`’a verdiğimizde, bu alana işlevin döndürdüğü ölçütler de eklenir.

`predictions`, 408 × 2 biçiminde iki boyutlu bir dizidir; 408, veri kümesindeki öğe sayısıdır. Değerler, `predict()` ile verilen her öğe için logit’lerdir. Bölüm 2’de gördüğümüz gibi, Transformer modelleri logit döndürür. Bunları etiketlerle karşılaştırılabilir tahminlere çevirmek için ikinci eksendeki en yüksek değerin indeksini seçeriz:

```python
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```

Artık `preds` değerlerini etiketlerle karşılaştırabiliriz. `compute_metrics()` işlevinde Evaluate kütüphanesinin ölçütlerinden yararlanacağız. MRPC veri kümesiyle ilişkili ölçütleri, veri kümesini yüklediğimiz gibi bu kez `evaluate.load()` ile yükleyebiliriz. Dönen nesnenin `compute()` yöntemi ölçüt hesaplamasını yapar:

```python
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

```text
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}
```

Farklı değerlendirme ölçütleri ve stratejileri için 🤗 Evaluate belgelerine bakın.

Model başlığının rastgele başlatılması ölçütleri değiştirebileceğinden, aldığınız sonuçlar birebir aynı olmayabilir. Buradaki model doğrulama kümesinde %85,78 doğruluk ve %89,97 F1 puanı elde etmiştir. GLUE karşılaştırmasında MRPC sonuçları için bu iki ölçüt kullanılır. BERT makalesindeki tabloda temel model için F1 puanı 88,9 olarak raporlanmıştır. Makaledeki model büyük/küçük harf duyarsız (`uncased`), burada kullandığımız model ise duyarlı (`cased`) olduğundan sonuç farkı açıklanabilir.

Bütün adımları birleştirerek `compute_metrics()` işlevini şöyle tanımlarız:

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

Her dönemin sonunda ölçütleri raporlayacak biçimde yeni bir `Trainer` oluşturabiliriz:

```python
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
```

Yeni bir `TrainingArguments` nesnesi oluşturup `eval_strategy` değerini `"epoch"` yaptığımıza ve yeni bir model kurduğumuza dikkat edin. Aksi durumda daha önce eğittiğimiz modeli eğitmeye devam etmiş olurduk. Yeni eğitim çalışmasını başlatmak için:

```python
trainer.train()
```

Bu kez eğitim kaybına ek olarak her dönemin sonunda doğrulama kaybı ve ölçütler de raporlanır. Model başlığının rastgele başlatılması nedeniyle doğruluk ve F1 değerleri değişebilir; yine de sonuçların yaklaşık olarak aynı aralıkta olması beklenir.

### Gelişmiş eğitim özellikleri

`Trainer`, modern derin öğrenme iyi uygulamalarını erişilebilir kılan pek çok yerleşik özelliğe sahiptir.

**Karma duyarlıklı eğitim:** Daha hızlı eğitim ve daha düşük bellek kullanımı için `fp16=True` kullanın:

```python
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    fp16=True,  # Karma duyarlığı etkinleştir
)
```

**Gradyan biriktirme:** GPU belleği sınırlı olduğunda, etkili yığın boyutunu büyütmek için:

```python
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Etkili yığın boyutu = 4 * 4 = 16
)
```

**Öğrenme oranı zamanlayıcısı:** `Trainer` varsayılan olarak doğrusal azalma kullanır; bunu özelleştirebilirsiniz:

```python
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  # Farklı zamanlayıcıları deneyin
)
```

> **Performans optimizasyonu:** Dağıtık eğitim, bellek optimizasyonu ve donanıma özgü iyileştirmeler dâhil gelişmiş teknikler için 🤗 Transformers performans kılavuzuna bakın.

`Trainer`, birden fazla GPU veya TPU ile ek yapılandırma olmadan çalışabilir ve dağıtık eğitim için birçok seçenek sunar. Bunların tamamını Bölüm 10’da ele alacağız.

Böylece `Trainer` API ile ince ayara girişimizi tamamladık. Yaygın NLP görevlerinin çoğunda bu yaklaşımın uygulama örnekleri Bölüm 7’de verilecektir. Şimdi aynı işlemi yalın bir PyTorch eğitim döngüsüyle yapalım.

> **Daha fazla örnek:** 🤗 Transformers not defterlerinin kapsamlı koleksiyonuna göz atın.

### Bölüm değerlendirmesi

# Hugging Face Trainer Quiz

## 1. `Trainer` içindeki `processing_class` parametresinin amacı nedir?

- A) Kullanılacak model mimarisini belirtir.
- **B) Verinin işlenmesi için hangi tokenizer'ın kullanılacağını `Trainer`'a bildirir.**
- C) Eğitim yığın boyutunu belirler.
- D) Değerlendirme sıklığını ayarlar.

**Cevap: B**

---

## 2. Eğitim sırasında değerlendirmenin ne sıklıkta yapılacağını hangi `TrainingArguments` parametresi belirler?

- A) `eval_frequency`
- **B) `eval_strategy`**
- C) `evaluation_steps`
- D) `do_eval`

**Cevap: B**

---

## 3. `TrainingArguments` içinde `fp16=True` neyi etkinleştirir?

- A) Daha hızlı eğitim için 16 bit tamsayı duyarlığını.
- **B) Daha hızlı eğitim ve daha az bellek kullanımı sağlayan, 16 bit kayan noktalı sayılarla karma duyarlıklı eğitimi.**
- C) Tam olarak 16 dönem eğitim yapmayı.
- D) Dağıtık eğitim için 16 GPU kullanmayı.

**Cevap: B**

---

## 4. `Trainer` içindeki `compute_metrics` işlevi ne yapar?

- A) Eğitim sırasında kaybı hesaplar.
- **B) Logit'leri tahmine dönüştürür ve doğruluk ile F1 gibi değerlendirme ölçütlerini hesaplar.**
- C) Kullanılacak iyileştiriciyi seçer.
- D) Eğitim verisini ön işler.

**Cevap: B**

---

## 5. `Trainer`'a `eval_dataset` vermezseniz ne olur?

- A) Eğitim bir hatayla durur.
- B) `Trainer`, eğitim verisini otomatik olarak değerlendirme için böler.
- **C) Eğitim sürer; ancak eğitim sırasında değerlendirme ölçütleri elde edilmez.**
- D) Model değerlendirme için eğitim verisini kullanır.

**Cevap: C**

---

## 6. Gradyan biriktirme nedir ve nasıl etkinleştirilir?

- A) Gradyanları diske kaydeder; `save_gradients=True` ile etkinleştirilir.
- **B) Güncelleme yapmadan önce birden fazla yığındaki gradyanları biriktirir; `gradient_accumulation_steps` ile etkinleştirilir.**
- C) Gradyan hesaplamasını hızlandırır; `fp16` ile otomatik etkinleşir.
- D) Gradyan taşmasını önler; `gradient_clipping=True` ile etkinleştirilir.

**Cevap: B**

**Temel çıkarımlar:**

- Trainer API, eğitim sürecinin karmaşıklığının çoğunu yöneten üst düzey bir arayüz sunar.
- Verilerin doğru işlenmesi için tokenizer’ı `processing_class` ile belirtin.
- `TrainingArguments`; öğrenme oranı, yığın boyutu, değerlendirme stratejisi ve optimizasyonlar dâhil eğitimin tüm yönlerini yönetir.
- `compute_metrics`, yalnızca eğitim kaybının ötesinde özel değerlendirme ölçütleri ekler.
- Karma duyarlık (`fp16=True`) ve gradyan biriktirme gibi özellikler eğitim verimliliğini önemli ölçüde artırabilir.

## Özel eğitim döngüleri

Şimdi `Trainer` sınıfını kullanmadan, önceki bölümdeki sonuçları nasıl elde edeceğimizi göreceğiz. Modern PyTorch iyi uygulamalarını izleyerek eğitim döngüsünü sıfırdan uygulayacağız. Bölüm 2’deki veri işleme adımlarını tamamladığınız varsayılıyor. Gerekenlerin özeti:

> **Sıfırdan eğitim:** Bu bölüm önceki içeriklerin üzerine kuruludur. PyTorch eğitim döngüleri ve iyi uygulamalar hakkında ayrıntılı bilgi için 🤗 Transformers eğitim belgelerine ve özel eğitim yemek kitabına bakın.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Eğitime hazırlık

Eğitim döngüsünü yazmadan önce birkaç nesne tanımlamalıyız. İlk olarak, yığınlar üzerinde dolaşmak için kullanacağımız veri yükleyicileri (`DataLoader`) gerekir. Bunları oluşturmadan önce, `Trainer`’ın otomatik yaptığı bazı işlemleri elle tamamlamak üzere tokenize edilmiş veri kümelerine son bir ön işleme uygulamalıyız:

- Modelin beklemediği sütunları (örneğin `sentence1` ve `sentence2`) kaldırmak.
- `label` sütununun adını `labels` olarak değiştirmek; model argümanın bu adla verilmesini bekler.
- Veri kümelerinin liste yerine PyTorch tensörleri döndüreceği biçimi ayarlamak.

`tokenized_datasets` bu işlemlerin her biri için bir yöntem sunar:

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```

Sonuçta yalnızca modelin kabul edeceği sütunların kaldığını doğrulayabiliriz:

```text
["attention_mask", "input_ids", "labels", "token_type_ids"]
```

Artık veri yükleyicileri kolayca tanımlayabiliriz:

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator,
)
```

Veri işlemede hata olmadığını hızlıca kontrol etmek için bir yığını inceleyebiliriz:

```python
for batch in train_dataloader:
    break

{k: v.shape for k, v in batch.items()}
```

```text
{
    'attention_mask': torch.Size([8, 65]),
    'input_ids': torch.Size([8, 65]),
    'labels': torch.Size([8]),
    'token_type_ids': torch.Size([8, 65])
}
```

Eğitim veri yükleyicisinde `shuffle=True` kullandığımız ve yığın içindeki en uzun örneğe kadar dolgu yaptığımız için sizdeki gerçek boyutlar biraz farklı olabilir.

Veri ön işleme tamamlandıktan sonra modele geçebiliriz. Modeli önceki bölümdekiyle aynı şekilde oluştururuz:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)
```

Eğitimin sorunsuz ilerleyeceğinden emin olmak için yığını modele verelim:

```python
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

```text
tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```

Etiketler verildiğinde tüm Transformers modelleri kaybı döndürür. Ayrıca yığındaki her girdi için iki logit elde ederiz; bu örnekte tensörün boyutu 8 × 2’dir.

Eğitim döngüsünü yazmaya yaklaştık. Geriye bir iyileştirici ve öğrenme oranı zamanlayıcısı kaldı. `Trainer`’ın yaptıklarını elle tekrarladığımız için varsayılan değerleri kullanacağız. `Trainer`, ağırlık azalması düzenlileştirmesinde bir değişiklik içeren Adam çeşidi olan AdamW’yi kullanır (Ilya Loshchilov ve Frank Hutter’ın “Decoupled Weight Decay Regularization” makalesine bakın):

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

> **Modern optimizasyon ipuçları:** Daha iyi performans için şunları deneyebilirsiniz:
>
> - Ağırlık azalmalı AdamW: `AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)`
> - Bellek verimliliği sağlayan 8 bit Adam: `bitsandbytes` kullanımı.
> - Farklı öğrenme oranları: Büyük modellerde daha düşük oranlar (`1e-5`–`3e-5`) çoğu zaman daha iyi çalışır.
>
> **Optimizasyon kaynakları:** İyileştiriciler ve eğitim stratejileri için 🤗 Transformers optimizasyon kılavuzuna bakın.

Varsayılan öğrenme oranı zamanlayıcısı, en yüksek değerden (`5e-5`) sıfıra doğru doğrusal azalmadır. Zamanlayıcıyı tanımlamak için toplam eğitim adımını bilmeliyiz. Bu sayı, çalıştırılacak dönem sayısı ile eğitim veri yükleyicisindeki yığın sayısının (uzunluğunun) çarpımıdır. `Trainer` varsayılan olarak üç dönem kullandığından biz de öyle yapacağız:

```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print(num_training_steps)
```

```text
1377
```

### Eğitim döngüsü

Son bir adım: erişimimiz varsa GPU kullanmak isteriz. CPU’da eğitim birkaç dakika yerine saatler sürebilir. Modeli ve yığınları taşıyacağımız aygıtı tanımlayalım:

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```

```text
device(type='cuda')
```

Artık eğitime hazırız. Eğitimin ne zaman biteceğini takip edebilmek için `tqdm` kütüphanesiyle eğitim adımlarının üzerinde bir ilerleme çubuğu gösterelim:

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

> **Modern eğitim optimizasyonları:** Eğitim döngünüzü daha verimli hâle getirmek için şunları değerlendirin:
>
> - **Gradyan kırpma:** `optimizer.step()` öncesinde `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` ekleyin.
> - **Karma duyarlık:** Daha hızlı eğitim için `torch.cuda.amp.autocast()` ve `GradScaler` kullanın.
> - **Gradyan biriktirme:** Daha büyük yığınları benzetmek için birden fazla yığın boyunca gradyanları biriktirin.
> - **Kontrol noktası alma:** Kesinti durumunda eğitime devam edebilmek için model kontrol noktalarını düzenli kaydedin.
>
> **Uygulama kılavuzu:** Bu optimizasyonlara ilişkin ayrıntılı örnekleri 🤗 Transformers verimli eğitim kılavuzunda ve iyileştirici seçeneklerinde bulabilirsiniz.

Eğitim döngüsünün temelinin girişteki döngüye benzediğini görebilirsiniz. Herhangi bir raporlama istemediğimiz için bu döngü model performansı hakkında bilgi vermez. Bunun için değerlendirme döngüsü eklemeliyiz.

### Değerlendirme döngüsü

Daha önce olduğu gibi Evaluate kütüphanesinin sağladığı bir ölçütü kullanacağız. `metric.compute()` yöntemini gördük; ölçütler, tahmin döngüsünde ilerlerken `add_batch()` yöntemiyle yığınları biriktirebilir. Bütün yığınlar eklendikten sonra nihai sonucu `metric.compute()` ile alabiliriz:

> **Değerlendirme iyi uygulamaları:** Daha kapsamlı değerlendirme stratejileri ve ölçütler için 🤗 Evaluate belgelerine ve değerlendirme yemek kitabına bakın.

```python
import evaluate

metric = evaluate.load("glue", "mrpc")

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

```text
{'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}
```

Model başlığının başlatılmasındaki rastlantısallık ve verilerin karıştırılması sonuçları biraz değiştirebilir; ancak sonuçların yaklaşık olarak aynı aralıkta olması beklenir.

> **Deneyin:** Önceki eğitim döngüsünü, modelinizi SST-2 veri kümesinde ince ayardan geçirecek şekilde değiştirin.

### Eğitim döngünüzü Accelerate ile güçlendirin

Daha önce tanımladığımız eğitim döngüsü tek bir CPU veya GPU’da çalışır. 🤗 Accelerate kütüphanesiyle birkaç küçük değişiklik yaparak birden fazla GPU ya da TPU üzerinde dağıtık eğitimi etkinleştirebiliriz. Accelerate dağıtık eğitim, karma duyarlık ve aygıta yerleştirme karmaşıklığını otomatik olarak yönetir. Eğitim ve doğrulama veri yükleyicilerinin oluşturulmasından başlayarak elle yazılmış eğitim döngüsü şöyledir:

> **Accelerate’i derinlemesine inceleyin:** Dağıtık eğitim, karma duyarlık ve donanım optimizasyonu için 🤗 Accelerate belgelerine ve Transformers belgelerindeki uygulamalı örneklere bakın.

```python
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader,
    eval_dataloader,
    model,
    optimizer,
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Eklenecek ilk satır Accelerate içe aktarma satırıdır. İkinci satır, ortamı inceleyip uygun dağıtık yapılandırmayı başlatan `Accelerator` nesnesini oluşturur. Accelerate aygıt yerleşimini sizin için yaptığı için modeli aygıta taşıyan satırları kaldırabilirsiniz. İsterseniz `device` yerine `accelerator.device` kullanacak şekilde değiştirebilirsiniz.

İşin büyük kısmı veri yükleyicilerini, modeli ve iyileştiriciyi `accelerator.prepare()` yöntemine verdiğimiz satırda yapılır. Bu yöntem, dağıtık eğitimin doğru çalışmasını sağlamak için nesneleri uygun kapsayıcılarla sarar. Geriye kalan değişiklikler, yığını aygıta taşıyan satırı kaldırmak (veya `accelerator.device` kullanmak) ve `loss.backward()` yerine `accelerator.backward(loss)` yazmaktır.

> Bulut TPU’ların hızından yararlanmak için tokenizer’ın `padding="max_length"` ve `max_length` argümanlarıyla örnekleri sabit uzunluğa doldurmanız önerilir.

Kopyalayıp deneyebilmeniz için Accelerate ile tam eğitim döngüsü:

```python
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader,
    eval_dataloader,
    model,
    optimizer,
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Bu kodu `train.py` betiğine koyduğunuzda betik her tür dağıtık yapılandırmada çalıştırılabilir. Dağıtık ortamınızda denemek için önce aşağıdaki komutu çalıştırın. Soruları yanıtlamanız istenir ve cevaplarınız komutun kullanacağı bir yapılandırma dosyasına yazılır:

```bash
accelerate config
```

Dağıtık eğitimi başlatmak için:

```bash
accelerate launch train.py
```

Örneğin Colab’da TPU’ları denemek için not defterinde çalışacaksanız kodu `training_function()` içine koyun ve son hücrede şunu çalıştırın:

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

🤗 Accelerate deposunda başka örnekler bulabilirsiniz.

> **Dağıtık eğitim:** Çoklu GPU ve çok düğümlü eğitim hakkında kapsamlı bilgi için 🤗 Transformers dağıtık eğitim kılavuzunu ve ölçeklendirme yemek kitabını inceleyin.

### Sonraki adımlar ve iyi uygulamalar

Eğitimi sıfırdan uygulamayı öğrendiğinize göre, üretim kullanımı için şu noktaları göz önünde bulundurun:

- **Model değerlendirmesi:** Yalnızca doğruluğa değil, birden fazla ölçüte bakın. Kapsamlı değerlendirme için Evaluate kütüphanesini kullanın.
- **Hiperparametre ayarı:** Hiperparametreleri sistematik olarak optimize etmek için Optuna veya Ray Tune gibi kütüphaneleri değerlendirin.
- **Model izleme:** Eğitim boyunca eğitim ölçütlerini, öğrenme eğrilerini ve doğrulama performansını takip edin.
- **Model paylaşımı:** Eğitilen modelinizi topluluğun kullanımına açmak için Hugging Face Hub’da paylaşın.
- **Verimlilik:** Büyük modellerde gradyan kontrol noktaları, parametre verimli ince ayar (LoRA, AdaLoRA) veya kuantizasyon yöntemlerini değerlendirin.

Özel eğitim döngüleriyle ince ayara yönelik bu ayrıntılı inceleme burada sona eriyor. Eğitim sürecini bütünüyle denetlemeniz veya Trainer API’nin sunduklarının ötesinde özel eğitim mantığı uygulamanız gerektiğinde bu beceriler işinize yarayacaktır.

### Bölüm değerlendirmesi

**1\.** Adam ve AdamW iyileştiricileri arasındaki temel fark nedir?

- A) AdamW farklı bir öğrenme oranı zamanlayıcısı kullanır.
- B) AdamW, ayrıştırılmış ağırlık azalması düzenlileştirmesi içerir.
- C) AdamW yalnızca Transformer modellerinde çalışır.
- D) AdamW, Adam'den daha az bellek gerektirir.

**Cevap: B**

**2\.** Eğitim döngüsündeki işlemlerin doğru sırası nedir?

- A) İleri geçiş → Geri geçiş → İyileştirici adımı → Gradyanları sıfırlama.
- B) İleri geçiş → Geri geçiş → İyileştirici adımı → Zamanlayıcı adımı → Gradyanları sıfırlama.
- C) Gradyanları sıfırlama → İleri geçiş → İyileştirici adımı → Geri geçiş.
- D) İleri geçiş → Gradyanları sıfırlama → Geri geçiş → İyileştirici adımı.

**Cevap: B**

**3\.** Accelerate kütüphanesi öncelikle hangi konuda yardımcı olur?

- A) İleri geçişi optimize ederek modelleri hızlandırır.
- B) En iyi hiperparametreleri otomatik seçer.
- C) Çok az kod değişikliğiyle birden fazla GPU/TPU üzerinde dağıtık eğitimi mümkün kılar.
- D) Modelleri TensorFlow gibi farklı çerçevelere dönüştürür.

**Cevap: C**

**5\.** Eğitim döngüsünde yığınları neden aygıta taşırız?

- A) Eğitimi hızlandırmak için.
- B) Hesaplama için model ve verinin aynı aygıtta (CPU/GPU) bulunması gerektiğinden.
- C) Bellek tasarrufu için.
- D) DataLoader bunu zorunlu kıldığı için.

**Cevap: B**

**7\.** Değerlendirmeden önce `model.eval()` ne yapar?

- A) Model parametrelerini güncellenemeyecek şekilde dondurur.
- B) Dropout ve batch normalization gibi katmanların çıkarım sırasındaki davranışını etkinleştirir.
- C) Değerlendirme ölçütleri için gradyan hesaplamasını açar.
- D) Değerlendirme ölçütlerini otomatik hesaplar.

**Cevap: B**

**9\.** Değerlendirme sırasında `torch.no_grad()` ne işe yarar?

- A) Modelin tahmin üretmesini engeller.
- B) Gradyan takibini kapatarak bellek kullanımını ve hesaplama süresini azaltır.
- C) Model için değerlendirme kipini etkinleştirir.
- D) Sonuçların her çalıştırmada aynı olmasını sağlar.

**Cevap: B**

**11\.** Eğitim döngünüzde Accelerate kullandığınızda ne değişir?

- A) Eğitim döngüsünü baştan yazmanız gerekir.
- B) Temel nesneleri `accelerator.prepare()` ile sarar ve `loss.backward()` yerine `accelerator.backward()` kullanırsınız.
- C) Kodda GPU sayısını belirtmeniz gerekir.
- D) Farklı bir iyileştirici ve zamanlayıcı kullanmanız gerekir.

**Cevap: B**
 
**Temel çıkarımlar:**

- Elle yazılan eğitim döngüleri tam denetim sağlar; ancak doğru işlem sırasını anlamayı gerektirir: ileri geçiş → geri geçiş → iyileştirici adımı → zamanlayıcı adımı → gradyanları sıfırlama.
- Transformer modelleri için ağırlık azalmalı AdamW önerilir.
- Doğru davranış ve verimlilik için değerlendirmede `model.eval()` ve `torch.no_grad()` kullanın.
- Accelerate, az kod değişikliğiyle dağıtık eğitimi erişilebilir kılar.
- Tensörleri CPU/GPU’ya taşımak gibi aygıt yönetimi PyTorch hesaplamaları için önemlidir.
- Karma duyarlık, gradyan biriktirme ve gradyan kırpma gibi modern teknikler eğitim verimliliğini önemli ölçüde artırabilir.

## Öğrenme eğrilerini anlama

Artık hem Trainer API’siyle hem de özel eğitim döngüleriyle ince ayar yapmayı öğrendiniz. Sonuçları yorumlamak da önemlidir. Öğrenme eğrileri, eğitim sırasında model performansını değerlendirmeye ve sorunlar performansı düşürmeden önce olası problemleri belirlemeye yardımcı olur.

Bu bölümde doğruluk ve kayıp eğrilerini okumayı ve yorumlamayı, farklı eğri biçimlerinin model davranışı hakkında neler anlattığını ve yaygın eğitim sorunlarını nasıl gidereceğimizi inceleyeceğiz.

### Öğrenme eğrileri nedir?

Öğrenme eğrileri, modelin eğitim süresince performans ölçütlerinin nasıl değiştiğini gösteren çizimlerdir. İzlenecek en önemli iki eğri şunlardır:

- **Kayıp eğrileri:** Eğitim adımları veya dönemleri boyunca model kaybının nasıl değiştiğini gösterir.
- **Doğruluk eğrileri:** Eğitim adımları veya dönemleri boyunca doğru tahminlerin yüzdesini gösterir.

Bu eğriler modelin etkili biçimde öğrenip öğrenmediğini anlamamıza ve performansı iyileştirmek için ayarlamalar yapmamıza yardımcı olur. Transformers’ta ölçütler her yığın için ayrı ayrı hesaplanıp diske kaydedilebilir. Eğrileri görselleştirmek ve model performansını zamanla izlemek için Weights & Biases gibi kütüphaneler kullanılabilir.

### Kayıp eğrileri

Kayıp eğrisi, modelin hatasının zaman içinde nasıl azaldığını gösterir. Başarılı bir eğitimde genellikle aşağıdaki örneğe benzer bir eğri görülür:

![Eğitim ve doğrulama kaybının sağlıklı biçimde azalması](./1.png)

- **Başlangıç kaybı yüksek:** Model henüz optimize edilmediği için ilk tahminler zayıftır.
- **Kayıp azalır:** Eğitim ilerledikçe kaybın genellikle düşmesi beklenir.
- **Yakınsama:** Sonunda kayıp düşük bir değerde dengelenir; bu, modelin verideki örüntüleri öğrendiğine işaret eder.

Önceki bölümlerde olduğu gibi, ölçütleri Trainer API ile izleyip bir panoda görselleştirebiliriz. Aşağıdaki örnek, Weights & Biases ile bunun nasıl yapılacağını gösterir:

```python
# Trainer ile eğitim sırasında kaybı takip etme örneği
from transformers import Trainer, TrainingArguments
import wandb

# Deney takibi için Weights & Biases'i başlat
wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,  # Ölçütleri her 10 adımda kaydet
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="wandb",  # Kayıtları Weights & Biases'e gönder
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Eğit ve ölçütleri otomatik olarak kaydet
trainer.train()
```

### Doğruluk eğrileri

Doğruluk eğrisi, doğru tahminlerin yüzdesini zaman içinde gösterir. Model öğrendikçe doğruluk eğrilerinin genellikle yükselmesi beklenir. Doğruluk eğrisi, kayıp eğrisinden daha basamaklı görünebilir.

- **Düşük başlar:** Model örüntüleri henüz öğrenmediğinden başlangıç doğruluğunun düşük olması beklenir.
- **Eğitimle artar:** Model örüntüleri öğrenebiliyorsa doğruluk genellikle yükselir.
- **Düz platolar görülebilir:** Tahminler gerçek etiketlere yaklaşsa bile doğruluk artışı düzgün değil, basamaklar hâlinde olabilir.

Doğruluk eğrileri neden basamaklıdır? Kayıp sürekli bir değerdir; doğruluk ise ayrık tahminleri gerçek etiketlerle karşılaştırarak hesaplanır. Modelin güvenindeki küçük artışlar son tahmini değiştirmeyebilir. Bu nedenle bir karar eşiği aşılana kadar doğruluk sabit kalır.

### Yakınsama

Model performansının dengelenmesi ve kayıp ile doğruluk eğrilerinin yataylaşması **yakınsama** olarak adlandırılır. Bu, modelin veri örüntülerini öğrendiğini ve kullanılmaya hazır olduğunu gösterebilir. Amaç, modelin her eğitimde kararlı bir performansa yakınsamasıdır.

![Kayıp eğrisinde yakınsama bölgesi](./3.png)

Model yakınsadığında yeni veriler üzerinde tahmin yapabilir, performansını anlamak için değerlendirme ölçütlerine başvurabiliriz.

### Öğrenme eğrisi örüntülerini yorumlama

Eğrilerin biçimleri, modelin eğitimi hakkında farklı bilgiler verir. En yaygın örüntülere ve anlamlarına bakalım.

#### Sağlıklı öğrenme eğrileri

Düzgün ilerleyen bir eğitimde aşağıdakine benzer eğriler görülür:

![Sağlıklı eğitim ve doğrulama kaybı ile doğruluk eğrileri](./4.png)
Çizimde solda kayıp, sağda doğruluk eğrisi gösterilir. Kayıp başlangıçta yüksektir; zamanla azalması modelin iyileştiğine işaret eder. Kayıp, tahmin edilen çıktı ile gerçek çıktı arasındaki hatayı temsil ettiğinden azalması genellikle tahminlerin iyileştiğini gösterir.

Doğruluk eğrisi başlangıçta düşük olup eğitim ilerledikçe yükselir. Doğruluk, doğru sınıflandırılan örneklerin oranıdır; eğrinin yükselmesi modelin daha fazla doğru tahmin yaptığı anlamına gelir.

İki eğri arasındaki belirgin fark, düzgünlük ve doğruluk eğrisindeki platolardır. Kayıp düzgün biçimde düşerken doğruluk sürekli artmak yerine basamaklar hâlinde sıçrayabilir. Modelin çıktısı hedefe yaklaşsa bile son tahmin hâlâ yanlışsa kayıp iyileşebilir. Doğruluk ise tahmin doğru karar sınırını geçtiğinde yükselir.

Örneğin, kedi (0) ve köpek (1) ayrımı yapan ikili sınıflandırıcı, köpek görseli için 0,3 tahmin etsin. Bu, 0’a yuvarlanacağı için yanlış sınıflandırmadır. Bir sonraki adımda tahmin 0,4 olursa yine yanlıştır. Ancak 0,4 gerçek değer olan 1’e 0,3’ten daha yakın olduğundan kayıp azalır; doğruluk değişmez ve eğride plato görülür. Tahmin 0,5’i aşıp 1’e yuvarlandığında doğruluk yükselir.

Sağlıklı eğrilerin özellikleri:

- **Kaybın düzenli düşmesi:** Eğitim ve doğrulama kayıpları istikrarlı biçimde azalır.
- **Eğitim ve doğrulama performanslarının yakın olması:** İki kümenin ölçütleri arasındaki fark küçüktür.
- **Yakınsama:** Eğriler yataylaşır; model örüntüleri öğrenmiştir.

#### Uygulamalı örnekler

Öğrenme eğrilerini eğitim sırasında izlemenin bazı yollarına ve eğrilerde görülebilecek örüntülere bakalım.

**Eğitim sırasında** (`trainer.train()` çağrısından sonra) şu göstergeleri izleyebilirsiniz:

1. **Kaybın yakınsaması:** Kayıp hâlâ düşüyor mu, yoksa plato mu yaptı?
2. **Aşırı uyum belirtileri:** Eğitim kaybı düşerken doğrulama kaybı yükselmeye başladı mı?
3. **Öğrenme oranı:** Oran çok yüksekse eğriler fazla oynak, çok düşükse fazla düz mü?
4. **Kararlılık:** Soruna işaret edebilecek ani sıçramalar veya düşüşler var mı?

**Eğitim tamamlandıktan sonra** tüm eğrileri inceleyerek model performansını değerlendirebilirsiniz:

1. **Nihai performans:** Model kabul edilebilir performans düzeyine ulaştı mı?
2. **Verimlilik:** Aynı performans daha az dönemle elde edilebilir miydi?
3. **Genelleme:** Eğitim ve doğrulama performansları birbirine ne kadar yakın?
4. **Eğilimler:** Eğitime devam etmek performansı muhtemelen iyileştirir mi?

> **W&B panosu özellikleri:** Weights & Biases öğrenme eğrilerinin etkileşimli grafiklerini otomatik oluşturur. Birden fazla çalışmayı yan yana karşılaştırabilir, özel ölçütler ve görselleştirmeler ekleyebilir, olağan dışı davranışlar için uyarılar kurabilir ve sonuçları ekibinizle paylaşabilirsiniz. Ayrıntılar için Weights & Biases belgelerine bakın.

### Aşırı uyum (overfitting)

Aşırı uyum, model eğitim verisini gereğinden fazla öğrendiğinde ve doğrulama kümesinin temsil ettiği farklı verilere genelleme yapamadığında ortaya çıkar.

![Eğitim kaybı düşerken doğrulama kaybının yükselmesi: aşırı uyum](./2.png)
Belirtileri:

- Eğitim kaybı düşmeye devam ederken doğrulama kaybı yükselir veya plato yapar.
- Eğitim ve doğrulama doğruluğu arasında büyük fark oluşur.
- Eğitim doğruluğu, doğrulama doğruluğundan çok daha yüksektir.

Aşırı uyuma karşı çözümler:

- **Düzenlileştirme:** Dropout, ağırlık azalması veya başka düzenlileştirme teknikleri ekleyin.
- **Erken durdurma:** Doğrulama performansı iyileşmeyi bıraktığında eğitimi durdurun.
- **Veri artırma:** Eğitim verisinin çeşitliliğini artırın.
- **Model karmaşıklığını azaltma:** Daha küçük bir model veya daha az parametre kullanın.

![Aşırı uyumda eğitim ve doğrulama kayıplarının ayrışması](./5.png)
Aşağıdaki örnekte aşırı uyumu önlemek için erken durdurma kullanıyoruz. `early_stopping_patience` değerini 3 yapıyoruz; doğrulama kaybı art arda üç değerlendirme döneminde iyileşmezse eğitim durdurulur.

```python
# Erken durdurmayla aşırı uyumu yakalama örneği
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=10,  # Yüksek ayarla; gerekirse erken durdurma bitirecek
)

# Aşırı uyumu önlemek için erken durdurma ekle
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

### Eksik uyum (underfitting)

Eksik uyum, model veri içindeki temel örüntüleri yakalayamayacak kadar basit olduğunda ortaya çıkar. Nedenleri arasında modelin küçük veya kapasitesinin yetersiz olması, öğrenme oranının çok düşük olması, veri kümesinin küçük ya da temsil gücünün zayıf olması ve modelin uygun biçimde düzenlileştirilmemesi sayılabilir.

Belirtileri:

- Eğitim ve doğrulama kaybı yüksek kalır.
- Model performansı eğitimin erken aşamasında plato yapar.
- Eğitim doğruluğu beklenenden düşüktür.

Eksik uyuma karşı çözümler:

- **Model kapasitesini artırın:** Daha büyük bir model veya daha fazla parametre kullanın.
- **Daha uzun eğitin:** Dönem sayısını artırın.
- **Öğrenme oranını ayarlayın:** Farklı öğrenme oranları deneyin.
- **Veri kalitesini denetleyin:** Verilerin doğru ön işlendiğinden emin olun.

Aşağıdaki örnekte modelin örüntüleri öğrenip öğrenemeyeceğini görmek için dönem sayısını artırıyoruz:

```diff
 from transformers import TrainingArguments
 
 training_args = TrainingArguments(
     output_dir="./results",
-    num_train_epochs=5,
+    num_train_epochs=10,
 )
```

![Eğitim ve doğrulama performansının düşük kaldığı eksik uyum örneği](./6.png)
### Kararsız öğrenme eğrileri

Kararsız öğrenme eğrileri, modelin etkili biçimde öğrenmediğini gösterir. Olası nedenler:

- Öğrenme oranı çok yüksektir; model en uygun parametreleri aşarak ilerler.
- Yığın boyutu çok küçüktür; öğrenme yavaş veya gürültülü ilerler.
- Model uygun biçimde düzenlileştirilmemiştir ve eğitim verisine aşırı uyum gösterebilir.
- Veri kümesi doğru ön işlenmemiştir; model gürültüden öğreniyor olabilir.

Belirtileri:

- Kayıp veya doğruluk sık sık dalgalanır.
- Eğrilerde yüksek değişkenlik ya da kararsızlık görülür.
- Performans belirgin bir eğilim olmadan salınır.

Aşağıdaki grafikler eğitim ve doğrulama eğrilerindeki kararsız davranışı gösterir:

![Dalgalı eğitim ve doğrulama kayıpları](./7.png)
**Belirtileri:**

- Kayıp veya doğrulukta sık dalgalanmalar görülür.
- Eğrilerde yüksek değişkenlik veya kararsızlık vardır.
- Performans, belirgin bir eğilim olmadan dalgalanır.

Eğitim ve doğrulama eğrilerinin ikisi de kararsız davranış gösterir.
![Kararsız ve yüksek değişkenlik gösteren öğrenme eğrileri](./8.png)
Kararsız eğrileri düzeltmek için:

- **Öğrenme oranını düşürün:** Daha kararlı adımlar için adım boyutunu azaltın.
- **Yığın boyutunu artırın:** Daha büyük yığınlar daha kararlı gradyanlar sağlar.
- **Gradyanları kırpın:** Gradyanların aşırı büyümesini önleyin.
- **Veri ön işlemeyi iyileştirin:** Veri kalitesinin tutarlı olduğundan emin olun.

Aşağıdaki örnekte öğrenme oranını düşürüp yığın boyutunu artırıyoruz:

```diff
 from transformers import TrainingArguments
 
 training_args = TrainingArguments(
     output_dir="./results",
-    learning_rate=1e-5,
+    learning_rate=1e-4,
-    per_device_train_batch_size=16,
+    per_device_train_batch_size=32,
 )
```

> **Kaynak notu:** Kaynak PDF, bu örneğin açıklamasında öğrenme oranını düşürmeyi öneriyor; ancak kod farkında değer `1e-5`’ten `1e-4`’e yükseltilmiş. Kod, kaynağa sadık kalınarak korunmuştur; uygulamadan önce bu tutarsızlığı kontrol edin.

### Temel çıkarımlar

Öğrenme eğrilerini anlamak, etkili bir makine öğrenmesi uygulayıcısı olmak için önemlidir. Bu görsel araçlar eğitim ilerlemesi hakkında hızlı geri bildirim verir; eğitimi durdurma, hiperparametreleri ayarlama veya başka yaklaşımlar deneme konusunda bilinçli kararlar alınmasını sağlar. Pratik yaptıkça sağlıklı eğrileri tanımak ve sorunları gidermek kolaylaşır.

- Öğrenme eğrileri modelin eğitim sürecini anlamak için temel araçlardır.
- Kayıp ve doğruluk eğrilerinin ikisini de izleyin; farklı özelliklere sahip olduklarını unutmayın.
- Aşırı uyumda eğitim ve doğrulama performansları birbirinden ayrılır.
- Eksik uyumda hem eğitim hem doğrulama verisindeki performans zayıftır.
- Weights & Biases gibi araçlar öğrenme eğrilerini izlemeyi ve incelemeyi kolaylaştırır.
- Erken durdurma ve uygun düzenlileştirme yaygın eğitim sorunlarının çoğunu giderebilir.

**Sonraki adım:** Kendi ince ayar deneylerinizde öğrenme eğrilerini inceleyin. Farklı hiperparametreler deneyip eğrilerin nasıl değiştiğini gözlemleyin. Eğitim ilerlemesini yorumlama sezgisini geliştirmenin en iyi yolu uygulamadır.

### Bölüm değerlendirmesi

**1\.** Eğitim kaybı azalırken doğrulama kaybı yükselmeye başlarsa bu genellikle ne anlama gelir?

- A) Model başarılı biçimde öğreniyor ve iyileşmeye devam edecek.
- B) Model eğitim verisine aşırı uyum gösteriyor.
- C) Öğrenme oranı çok düşük.
- D) Veri kümesi çok küçük.

**Cevap: B**

**2\.** Doğruluk eğrileri neden düzgün artışlar yerine basamaklı ya da plato biçiminde görünür?

- A) Doğruluk hesabında hata vardır.
- B) Doğruluk ayrık bir ölçüttür; yalnızca tahminler karar sınırını geçtiğinde değişir.
- C) Model etkili biçimde öğrenmiyordur.
- D) Yığın boyutu çok küçüktür.

**Cevap: B**

**3\.** Çok dalgalı ve kararsız öğrenme eğrileri gözlemlendiğinde en iyi yaklaşım nedir?

- A) Yakınsamayı hızlandırmak için öğrenme oranını artırmak.
- B) Öğrenme oranını düşürmek ve gerekirse yığın boyutunu artırmak.
- C) Model iyileşmeyeceği için eğitimi hemen durdurmak.
- D) Tamamen farklı bir model mimarisine geçmek.

**Cevap: B**

**4\.** Erken durdurmayı ne zaman değerlendirmelisiniz?

- A) Her zaman; her türlü aşırı uyumu önler.
- B) Doğrulama performansı iyileşmeyi bıraktığında veya kötüleşmeye başladığında.
- C) Yalnızca eğitim kaybı hızla düşmeye devam ediyorsa.
- D) Modelin tam potansiyeline ulaşmasını engellediği için hiçbir zaman.

**Cevap: B**

**5\.** Modelinizin eksik uyum gösterdiğine ne işaret eder?

- A) Eğitim doğruluğunun doğrulama doğruluğundan çok yüksek olması.
- B) Hem eğitim hem doğrulama performansının düşük olması ve erken plato yapması.
- C) Öğrenme eğrilerinin dalgalanma olmadan çok düzgün olması.
- D) Doğrulama kaybının eğitim kaybından daha hızlı düşmesi.

**Cevap: B**

## İnce ayar tamam

Bu bölümde, ilk iki bölümde öğrendiğiniz modellere ve tokenizer’lara ek olarak, kendi verinizde güncel iyi uygulamalarla ince ayar yapmayı öğrendiniz. Bu bölümde:

- Hub’daki veri kümelerini ve modern veri işleme tekniklerini öğrendiniz.
- Dinamik dolgu ve veri birleştiricileri dâhil, veri kümelerini verimli yüklemeyi ve ön işlemeyi gördünüz.
- Üst düzey Trainer API ile güncel özellikleri kullanarak ince ayar ve değerlendirme yaptınız.
- PyTorch ile sıfırdan eksiksiz bir özel eğitim döngüsü kurdunuz.
- Accelerate ile kodun birden fazla GPU veya TPU’da çalışmasını sağladınız.
- Karma duyarlıklı eğitim ve gradyan biriktirme gibi modern optimizasyon tekniklerini uyguladınız.

Tebrikler! Transformer modellerinde ince ayarın temellerini öğrendiniz. Artık gerçek dünya makine öğrenmesi projelerine hazırsınız.

**Öğrenmeye devam edin:** Bilginizi derinleştirmek için 🤗 Transformers’ın belirli NLP görevlerine yönelik kılavuzlarını ve kapsamlı not defteri örneklerini inceleyin.

**Sonraki adımlar:**

- Öğrendiğiniz tekniklerle kendi veri kümenizde ince ayar yapmayı deneyin.
- Hugging Face Hub’daki farklı model mimarileriyle deneyler yapın.
- Projelerinizi paylaşmak ve destek almak için Hugging Face topluluğuna katılın.

Transformer yolculuğunuz yeni başlıyor. Bir sonraki bölümde modellerinizi ve tokenizer’larınızı toplulukla paylaşmayı ve büyüyen önceden eğitilmiş model ekosistemine katkıda bulunmayı inceleyeceğiz.

Veri ön işleme, eğitim yapılandırması, değerlendirme ve optimizasyon becerileri her makine öğrenmesi projesinin temelidir. Metin sınıflandırma, adlandırılmış varlık tanıma, soru yanıtlama veya başka bir NLP görevi üzerinde çalışırken bu teknikler işinize yarayacaktır.

**Başarı için ipuçları:**

- Özel eğitim döngülerine geçmeden önce Trainer API ile sağlam bir başlangıç modeli oluşturun.
- Daha iyi bir başlangıç noktası için görevinize yakın önceden eğitilmiş modelleri Hub’da arayın.
- Eğitiminizi uygun değerlendirme ölçütleriyle izleyin ve kontrol noktalarını kaydetmeyi unutmayın.
- Sonuç ve geri bildirim almak için modellerinizi ve veri kümelerinizi toplulukla paylaşın.

## Bölüm sonu sertifikası

Kursu tamamladığınız için tebrikler! Önceden eğitilmiş modelleri ince ayardan geçirmeyi, öğrenme eğrilerini anlamayı ve modelleri toplulukla paylaşmayı öğrendiniz. Şimdi bilginizi sınamak ve sertifikanızı almak için kısa sınava geçebilirsiniz.

Sınava katılmak için:

1. Hugging Face hesabınızda oturum açın.
2. Sınav sorularını yanıtlayın.
3. Yanıtlarınızı gönderin.
