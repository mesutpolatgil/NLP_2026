# Fine-Tuning a Pretrained Model — Sorular ve Cevaplar

> Hugging Face Fine-Tuning bölümü soru çözümleri: Bölüm Soruları (1–23), Bölüm Sonu Quiz (1–10) ve Özgün Sorular (1–3).

## İçindekiler

- [Bölüm Soruları](#bölüm-soruları)
- [Bölüm Sonu Quiz](#bölüm-sonu-quiz)
- [Özgün Sorular](#özgün-sorular)

---

# Bölüm Soruları

## Soru 1
**`Dataset.map()` fonksiyonunu `batched=True` argümanı ile kullanmanın temel avantajı nedir?**

- A) Daha az bellek kullanır.
- B) Aynı anda birden fazla örneği işleyerek tokenizasyon (parçalama) işlemini çok daha hızlı hale getirir.
- C) Dolgu (padding) işlemini sizin yerinize otomatik olarak halleder.
- D) Veriyi PyTorch tensörlerine dönüştürür.

**✅ Doğru Cevap: B**

`batched=True` kullandığımızda veri seti tek tek değil, toplu gruplar (batch) halinde işlenir. Özellikle Hugging Face'in Tokenizer'ları arka planda Rust ile yazıldığı için aynı anda birden fazla veriyi (multi-threading) çok hızlı işleyebilirler. Bu da süreci inanılmaz hızlandırır.

---

## Soru 2
**Veri setindeki tüm dizileri (sequences) veri setinin maksimum uzunluğuna kadar doldurmak (padding) yerine neden "dinamik dolgu" (dynamic padding) kullanıyoruz?**

- A) Dinamik dolgu, model mimarisi tarafından zorunlu kılınmıştır.
- B) Yalnızca o anki yığındaki (batch) maksimum uzunluğa kadar dolgu yaparak hesaplama yükünü (computational overhead) azaltır.
- C) Modelin doğruluğunu (accuracy) artırır.
- D) `DataCollatorWithPadding` kullanıldığında gereklidir.

**✅ Doğru Cevap: B**

Tüm veri setini en uzun cümleye göre doldurursak, kısa cümleler için tonla gereksiz `[PAD]` token'ı hesaplamak zorunda kalırız. Dinamik dolgu, sadece o an eğitim için alınan küçük gruptaki (batch) en uzun cümleyi baz alır. Bu da bilgisayarı gereksiz işlemlerden kurtarır ve eğitimi hızlandırır.

---

## Soru 3
**BERT tokenizasyonunda `token_type_ids` alanı neyi temsil eder?**

- A) Dizideki her bir token'ın konumunu (position).
- B) Cümle çiftleri işlenirken her bir token'ın hangi cümleye ait olduğunu.
- C) Her bir token için dikkat maskesini (attention mask).
- D) Her bir token'ın kelime dağarcığındaki (vocabulary) ID'sini.

**✅ Doğru Cevap: B**

BERT gibi bazı modeller, aynı anda iki cümleyi alıp aralarındaki ilişkiyi ölçmek için (örneğin bu iki cümle aynı anlama mı geliyor diye) eğitilmiştir. `token_type_ids` modeli yönlendirir; 0'lar birinci cümlenin token'larını, 1'ler ise ikinci cümlenin token'larını gösterir.

---

## Soru 4
**`load_dataset('glue', 'mrpc')` komutuyla bir veri seti yüklenirken, ikinci argüman (`'mrpc'`) neyi belirtir?**

- A) Yüklenecek veri setinin sürümü.
- B) GLUE kıyaslama testi (benchmark) içindeki belirli görev veya alt küme.
- C) Veri setinin bölümlendirilmesi (eğitim/doğrulama/test).
- D) Verilerin döndürüleceği format.

**✅ Doğru Cevap: B**

GLUE (General Language Understanding Evaluation), içinde 10 farklı NLP görevi barındıran dev bir veri setidir. `mrpc` (Microsoft Research Paraphrase Corpus), GLUE şemsiyesi altındaki alt görevlerden sadece birisidir (iki cümlenin birbirinin özeti/eşanlamlısı olup olmadığını kontrol eden görev).

---

## Soru 5
**Eğitim öncesinde `sentence1` ve `sentence2` gibi sütunları kaldırmanın amacı nedir?**

- A) Eğitim sırasında bellek tasarrufu sağlamak.
- B) Model bu ham metin (raw text) sütunlarını beklemez ve hata verir.
- C) Bu sütunlara değerlendirme (evaluation) için ihtiyaç duyulmaz.
- D) Eğitim hızını önemli ölçüde artırır.

**✅ Doğru Cevap: B**

Modeli eğitmeye başlamadan önce tokenizer kullanarak metinlerimizi modelin anlayacağı sayılara (token'lara) çeviriyoruz. PyTorch tabanlı Hugging Face modelleri, eğitilirken (forward pass aşamasında) kendilerine sadece `input_ids`, `attention_mask`, `labels` gibi sayısal tensörlerin gelmesini bekleyecek şekilde programlanmıştır.

Veri setinin içinde `sentence1` veya `sentence2` gibi doğrudan ham metin (string) barındıran sütunları bırakırsak, model bu sütunlarla ne yapacağını bilemez ve hata (error) fırlatır. Bu yüzden tokenization bittikten sonra asıl metin sütunlarını veri setinden kaldırıp yola sadece modelin anlayacağı sayısal değerlerle devam etmeliyiz.

---

## Soru 6
**`Trainer`'daki `processing_class` parametresinin amacı nedir?**

- A) Hangi model mimarisinin kullanılacağını belirtir.
- B) Veriyi işlerken Trainer'a hangi tokenizer'ı kullanması gerektiğini söyler.
- C) Eğitim için yığın boyutunu (batch size) belirler.
- D) Değerlendirme (evaluation) sıklığını kontrol eder.

**✅ Doğru Cevap: B**

Trainer sınıfı her şeyi otomatik hallederken veriyi nasıl tokenize edeceğini de bilmek ister. `processing_class=tokenizer` diyerek modelin metinleri nasıl parçalayacağını Trainer'a vermiş oluruz.

---

## Soru 7
**Hangi `TrainingArguments` parametresi eğitim sırasında değerlendirmenin (evaluation) ne sıklıkla yapılacağını kontrol eder?**

- A) `eval_frequency`
- B) `eval_strategy`
- C) `evaluation_steps`
- D) `do_eval`

**✅ Doğru Cevap: B**

Modeli eğitirken körü körüne gitmemek için `eval_strategy` (eskiden `evaluation_strategy` olarak da geçiyordu) kullanırız. Örneğin bunu `"epoch"` yaparsak, her eğitim turunun (epoch) sonunda modelin performansı değerlendirilir.

---

## Soru 8
**`TrainingArguments` içindeki `fp16=True` ne işe yarar?**

- A) Daha hızlı eğitim için 16-bit tam sayı (integer) hassasiyeti.
- B) Daha hızlı eğitim ve az bellek kullanımı için 16-bit kayan noktalı (floating-point) sayılarla karma hassasiyetli (mixed precision) eğitim.
- C) Tam olarak 16 epoch boyunca eğitim.
- D) Dağıtılmış eğitim (distributed training) için 16 GPU kullanmak.

**✅ Doğru Cevap: B**

Normalde yapay zeka modelleri 32-bit (fp32) sayılarla eğitilir. `fp16=True` dediğimizde, model kritik olmayan hesaplamaları 16-bit'e düşürerek GPU belleğinde büyük yer açar ve eğitimi hızlandırır. Buna **Mixed Precision (Karma Hassasiyet)** denir.

---

## Soru 9
**`Trainer` içindeki `compute_metrics` fonksiyonunun rolü nedir?**

- A) Eğitim sırasında loss (kayıp) değerini hesaplar.
- B) Logit'leri (ham model çıktılarını) tahminlere dönüştürür ve Accuracy (Doğruluk) ile F1 gibi değerlendirme metriklerini hesaplar.
- C) Hangi optimizer'ın kullanılacağını belirler.
- D) Eğitim verisine ön işleme yapar.

**✅ Doğru Cevap: B**

Trainer kendi başına sadece "Loss" (hata) değerini gösterir; bu da modelin ne kadar iyi olduğunu anlamak için net bir metrik değildir. `compute_metrics` fonksiyonu yazarak modelin ham tahmin skorlarını alıp "Yüzde kaç bildi?" (Accuracy) ya da F1 skoru gibi net başarı metriklerine çeviririz.

---

## Soru 10
**`Trainer`'a bir `eval_dataset` (değerlendirme veri seti) sağlamazsanız ne olur?**

- A) Eğitim hata vererek çöker.
- B) Trainer, eğitim verisini değerlendirme için otomatik olarak böler.
- C) Eğitim sırasında değerlendirme metriklerini göremezsiniz, ancak eğitim çalışmaya devam eder.
- D) Model, değerlendirme için eğitim verisini kullanır.

**✅ Doğru Cevap: C**

Trainer çok esnektir. Eğer sadece eğitmek istiyorsanız çalışır, ancak eğitim sırasında modelin yeni verilerde (validation set) nasıl performans gösterdiğini (Accuracy, F1 vs.) göremezsiniz. Kör bir uçuş olur; modelin ezberleyip ezberlemediğini (overfitting) anlamak zorlaşır.

---

## Soru 11
**Gradyan biriktirme (gradient accumulation) nedir ve nasıl etkinleştirilir?**

- A) Gradyanları diske kaydeder, `save_gradients=True` ile açılır.
- B) Modeli güncellemeden önce birkaç batch (yığın) boyunca gradyanları biriktirir, `gradient_accumulation_steps` ile açılır.
- C) Gradyan hesaplamasını hızlandırır, `fp16` ile otomatik açılır.
- D) Gradyan patlamasını engeller, `gradient_clipping=True` ile açılır.

**✅ Doğru Cevap: B**

Çok büyük modelleri eğitirken GPU belleği yetersiz kalıp "Out of Memory" hatası verebilir. Yüksek batch size kullanamadığımız için küçük batch'ler kullanırız; ancak `gradient_accumulation_steps` (örneğin 4 veya 8) diyerek, ağırlıkları her adımda güncellemek yerine bu küçük adımların gradyanlarını biriktirip sanki büyük bir batch kullanmışız gibi topluca güncelleriz. Düşük donanımla büyük model eğitmenin en önemli taktiklerinden biridir.

---

## Soru 12
**Adam ve AdamW optimizer'ları (eniyileyicileri) arasındaki temel fark nedir?**

- A) AdamW farklı bir öğrenme oranı (learning rate) planlayıcısı kullanır.
- B) AdamW, ayrıştırılmış ağırlık azaltma (decoupled weight decay) düzenlemesini (regularization) içerir.
- C) AdamW yalnızca transformer modelleriyle çalışır.
- D) AdamW, Adam'dan daha az bellek gerektirir.

**✅ Doğru Cevap: B**

AdamW'deki "W", "Weight Decay" (ağırlık azaltma) anlamına gelir. Normal Adam algoritması, ağırlık güncellemeleri sırasında weight decay'i gradyanla karıştırıyordu; AdamW bu düzenleme işlemini matematiksel olarak ayırarak (decoupled) overfitting'i çok daha başarılı şekilde engeller.

---

## Soru 13
**Bir eğitim döngüsünde (training loop) işlemlerin doğru sırası nedir?**

- A) İleri yayılım (Forward pass) → Geri yayılım (Backward pass) → Optimizer adımı → Gradyanları sıfırlama (Zero gradients)
- B) İleri yayılım → Geri yayılım → Optimizer adımı → Scheduler adımı → Gradyanları sıfırlama
- C) Gradyanları sıfırlama → İleri yayılım → Optimizer adımı → Geri yayılım
- D) İleri yayılım → Gradyanları sıfırlama → Geri yayılım → Optimizer adımı

**✅ Doğru Cevap: B**

PyTorch ile döngüyü yazarken sıralama hayati önem taşır:

1. Modeli veriden geçirip hatayı buluruz (forward).
2. Hatayı geriye yayarız (`loss.backward()`).
3. Optimizer ağırlıkları günceller (`optimizer.step()`).
4. Scheduler öğrenme oranını ayarlar (`lr_scheduler.step()`).
5. Bir sonraki adıma temiz başlamak için gradyanları sıfırlarız (`optimizer.zero_grad()`).

---

## Soru 14
**Accelerate kütüphanesi temel olarak neye yardımcı olur?**

- A) İleri yayılımı (forward pass) optimize ederek modellerinizin daha hızlı eğitilmesini sağlar.
- B) En iyi hiperparametreleri otomatik olarak seçer.
- C) Minimum kod değişikliği ile birden fazla GPU/TPU üzerinde dağıtılmış eğitimi (distributed training) mümkün kılar.
- D) Modelleri TensorFlow gibi farklı framework'lere dönüştürür.

**✅ Doğru Cevap: C**

Tek bir GPU ile devasa bir dil modelini eğitmek haftalar sürebilir. Accelerate sayesinde, yazdığımız standart kodu sadece birkaç küçük değişiklikle birden fazla GPU veya TPU üzerinde çalışacak şekilde dağıtabiliriz.

---

## Soru 15
**Bir eğitim döngüsünde yığınları (batches) neden cihaza (device) taşırız?**

- A) Eğitimi daha hızlı hale getirmek için.
- B) Çünkü hesaplama yapılabilmesi için modelin ve verilerin aynı cihazda (CPU veya GPU) bulunması zorunludur.
- C) Bellek tasarrufu sağlamak için.
- D) DataLoader tarafından zorunlu kılındığı için.

**✅ Doğru Cevap: B**

Model GPU'daysa veriler de GPU'da olmalıdır. PyTorch'un hesaplama yapabilmesi için her batch'in modelle aynı işlem birimine taşınması (`v.to(device)`) gerekir. Aksi halde sistem hata verir.

---

## Soru 16
**Değerlendirme (evaluation) öncesinde `model.eval()` ne işe yarar?**

- A) Model parametrelerini güncellenmemeleri için dondurur.
- B) Çıkarım (inference) işlemi için dropout ve batch normalization gibi katmanların davranışını değiştirir.
- C) Değerlendirme metrikleri için gradyan hesaplamasını etkinleştirir.
- D) Değerlendirme metriklerini otomatik olarak hesaplar.

**✅ Doğru Cevap: B**

Eğitim sırasında model aşırı ezberlemesin diye "dropout" tekniğiyle bazı nöronları rastgele kapatırız. Değerlendirmede ise modelin tam kapasite çalışmasını isteriz. `model.eval()` komutu rastgelelikleri (dropout vb.) kapatır ve modeli çıkarım davranışına geçirir.

---

## Soru 17
**Değerlendirme (evaluation) sırasında `torch.no_grad()` fonksiyonunun amacı nedir?**

- A) Modelin tahmin yapmasını engellemek.
- B) Gradyan takibini devre dışı bırakarak bellekten tasarruf etmek ve hesaplamayı hızlandırmak.
- C) Model için değerlendirme modunu etkinleştirmek.
- D) Çalıştırmalar arasında tutarlı sonuçlar elde edilmesini sağlamak.

**✅ Doğru Cevap: B**

Değerlendirme döngüsünde model yeni bir şey öğrenmez, sadece var olan bilgisiyle tahmin yapar. Bu yüzden gradyanları hafızada tutmasına gerek yoktur. `with torch.no_grad():` bloğu PyTorch'a gradyan hesaplamayacağını söyler; bellek tasarrufu sağlar ve süreci hızlandırır.

---

## Soru 18
**Eğitim döngünüzde Accelerate kullandığınızda ne değişir?**

- A) Tüm eğitim döngünüzü sıfırdan yeniden yazmanız gerekir.
- B) Önemli nesneleri `accelerator.prepare()` ile sararsınız (paketlersiniz) ve `loss.backward()` yerine `accelerator.backward(loss)` kullanırsınız.
- C) Kodunuzda GPU sayısını belirtmeniz gerekir.
- D) Farklı bir optimizer ve scheduler kullanmanız gerekir.

**✅ Doğru Cevap: B**

Accelerate mevcut kodunuzu bozmaz. Model, optimizer ve dataloader'ları `accelerator.prepare()` fonksiyonuna verip paketleriz. Değiştirmemiz gereken tek yer geriye yayılımdır: `loss.backward()` yerine `accelerator.backward(loss)` yazarız; kütüphane geri kalan donanım paylaştırmasını otomatik halleder.

---

## Soru 19
**Eğitim kaybı (training loss) düşerken doğrulama kaybı (validation loss) artmaya başladığında bu tipik olarak ne anlama gelir?**

- A) Model başarılı bir şekilde öğreniyor ve gelişmeye devam edecek.
- B) Model eğitim verisini ezberliyor (overfitting).
- C) Öğrenme oranı (learning rate) çok düşük.
- D) Veri seti çok küçük.

**✅ Doğru Cevap: B**

Training loss düşmeye devam ederken validation loss'un artması overfitting'in en net belirtisidir. Model eğitim verisine o kadar odaklanmıştır ki genel mantığı öğrenmek yerine cevapları ezberlemeye başlamıştır; bu yüzden daha önce görmediği validation verisinde hata oranı artar.

---

## Soru 20
**Doğruluk (accuracy) eğrileri neden pürüzsüz bir artış yerine genellikle "basamaklı" (steppy) veya plato benzeri bir yapı gösterir?**

- A) Doğruluk hesaplamasında bir hata vardır.
- B) Doğruluk, yalnızca tahminler karar sınırlarını aştığında değişen ayrık (kesikli) bir metriktir.
- C) Model etkili bir şekilde öğrenemiyordur.
- D) Yığın boyutu (batch size) çok küçüktür.

**✅ Doğru Cevap: B**

Loss değeri çok hassastır; model doğru cevaba milimetrik yaklaştığında bile pürüzsüzce düşer. Accuracy ise böyle değildir. Örneğin doğru cevabı 1 olan bir veri için modelin tahmini 0.3'ten 0.4'e çıksa bile ikisi de 0'a yuvarlandığı için accuracy değişmez. Accuracy ancak tahmin doğru sınıfa geçtiğinde (örneğin 0.6 ile 1'e) basamak şeklinde sıçrar.

---

## Soru 21
**Dengesiz, aşırı dalgalanan (erratic) öğrenme eğrileri gözlemlediğinizde en iyi yaklaşım nedir?**

- A) Yakınsamayı (convergence) hızlandırmak için öğrenme oranını artırmak.
- B) Öğrenme oranını (learning rate) düşürmek ve muhtemelen yığın boyutunu (batch size) artırmak.
- C) Model iyileşmeyeceği için eğitimi derhal durdurmak.
- D) Tamamen farklı bir model mimarisine geçmek.

**✅ Doğru Cevap: B**

Eğriler net bir yön olmadan sürekli inip çıkıyorsa model çok büyük adımlar atıp doğru noktayı kaçırıyor demektir. Çözüm, learning rate'i düşürerek daha küçük/temkinli adımlar atmasını sağlamak ve batch size'ı artırarak genellemeyi iyileştirmektir.

---

## Soru 22
**Erken durdurmayı (early stopping) ne zaman kullanmayı düşünmelisiniz?**

- A) Her zaman, çünkü her türlü ezberlemeyi (overfitting) engeller.
- B) Doğrulama (validation) performansı iyileşmeyi bıraktığında veya kötüleşmeye başladığında.
- C) Yalnızca eğitim kaybı (training loss) hala hızla düşüyorken.
- D) Hiçbir zaman, çünkü modelin tam potansiyeline ulaşmasını engeller.

**✅ Doğru Cevap: B**

Modeli sonsuza kadar eğitmek iyi bir fikir değildir. Validation performansı belirli bir noktadan sonra artmıyorsa veya kötüye gitmeye başladıysa (overfitting başladıysa), eğitimi en iyi ağırlıklarda kesmek en mantıklı yöntemdir.

---

## Soru 23
**Modelinizin yetersiz öğrendiğini (underfitting) ne gösterir?**

- A) Eğitim doğruluğu, doğrulama doğruluğundan çok daha yüksektir.
- B) Hem eğitim hem de doğrulama performansı kötüdür ve erkenden düzleşir (plato yapar).
- C) Öğrenme eğrileri hiçbir dalgalanma olmadan çok pürüzsüzdür.
- D) Doğrulama kaybı, eğitim kaybından daha hızlı düşüyordur.

**✅ Doğru Cevap: B**

Underfitting durumunda model çok basittir veya yeterince eğitilmemiştir; verideki örüntüyü yakalayamaz. Sonuç olarak ne training ne de validation setinde başarılı olur; her ikisinin de loss değeri yüksek kalır.

---

# Bölüm Sonu Quiz

## Soru 1
**Doğruluk (accuracy) öğrenme eğrileri neden genellikle pürüzsüz olan kayıp (loss) eğrilerine kıyasla "basamaklı" veya plato gibi görünür?**

- A) Doğruluk, kayıptan daha az güvenilir bir metriktir.
- B) Doğruluk, yalnızca bir tahmin karar sınırını aştığında iyileşen kesikli (ayrık) bir metrikken, kayıp sürekli (kesintisiz) bir değerdir.
- C) Bu yalnızca öğrenme oranı çok düşük olduğunda gerçekleşir.
- D) Değerlendirme kodunda bir hata (bug) olduğunu gösterir.

**✅ Doğru Cevap: B**

Accuracy zamanla artar ancak düzgün bir çizgi değil, basamaklar halinde yükselir; çünkü accuracy ancak tahmin doğru etikete geçtiğinde değişir. Örneğin doğru cevabı 1 olan bir veri için model önce 0,3, sonra 0,4 tahmin ettiğinde ikisi de 0'a (yanlış sınıfa) yuvarlandığı için accuracy aynı kalır ve grafikte düz bir basamak oluşur. Ancak 0,4 değeri doğru cevaba daha yakın olduğu için loss bu sırada pürüzsüz şekilde düşmeye devam eder.

---

## Soru 2
**BERT benzeri modeller bağlamında, bir cümle çifti tokenize edilirken `token_type_ids`'nin temel rolü nedir?**

- A) Hangi token'ların dolgu (padding) olduğunu belirtmek.
- B) Kelime dağarcığındaki (vocabulary) her bir token için benzersiz bir kimlik (ID) sağlamak.
- C) Her bir token'ın hangi cümleye ait olduğunu ayırt etmek.
- D) `[CLS]` ve `[SEP]` gibi özel token'ların konumlarını işaretlemek.

**✅ Doğru Cevap: C**

BERT gibi modeller ön eğitim (pre-training) aşamasında "Sonraki Cümle Tahmini" (Next Sentence Prediction) görevini öğrenmiştir. Modelin "İkinci cümle, birincinin devamı mı?" sorusunu cevaplayabilmesi için hangi kelimenin birinci, hangisinin ikinci cümleden geldiğini bilmesi gerekir.

---

## Soru 3
**`TrainingArguments` kullanarak bir `Trainer` yapılandırırken, sağlamanız gereken (zorunlu) tek argüman nedir?**

- A) Öğrenme oranı (learning rate).
- B) Epoch sayısı.
- C) Eğitilen modelin ve kontrol noktalarının (checkpoints) kaydedileceği bir klasör (`output_dir`).
- D) Değerlendirme stratejisi (evaluation strategy).

**✅ Doğru Cevap: C**

`TrainingArguments`, eğitimin tüm ayarlarını tek yerde toplayan sınıftır. Öğrenme oranından epoch sayısına kadar hemen her şeyin varsayılan (default) bir değeri vardır. Girmek zorunda olduğunuz tek şey, modelin kaydedileceği klasörün adıdır (`output_dir`).

---

## Soru 4
**Aşağıdaki öğrenme eğrisi, eğitim (training) ve doğrulama (validation) kaybını göstermektedir. En olası sorun nedir?**

![Soru 4 öğrenme eğrisi](./soru-4.jpeg)

- A) Yetersiz öğrenme (Underfitting)
- B) Yüksek öğrenme oranı (High learning rate)
- C) Model yeterince uzun süre eğitilmemiş (The model has not trained long enough)
- D) Ezberleme (Overfitting)

**✅ Doğru Cevap: D**

Mavi çizgi (training) sürekli aşağı giderken turuncu çizgi (validation) bir noktadan sonra tekrar yukarı tırmanıyor. Eğitim kaybı düşerken doğrulama kaybının artması overfitting'in en net belirtisidir. Çözüm, turuncu çizginin büküldüğü yerde eğitimi **erken durdurmak (early stopping)** olacaktır.

---

## Soru 5
**`TrainingArguments` içinde `fp16=True` ayarını yapmak neyi etkinleştirir?**

- A) Tam olarak 16 epoch boyunca eğitim yapılmasını.
- B) Tüm hesaplamalar için 16-bit tam sayı (integer) kullanılmasını.
- C) Uyumlu GPU'larda eğitimi hızlandırabilen ve bellek kullanımını azaltabilen karma hassasiyetli (mixed-precision) eğitimi.
- D) Yığın boyutunu (batch size) 16 olarak ayarlamayı.

**✅ Doğru Cevap: C**

Standart eğitimler 32-bit (fp32) kayan noktalı sayılarla yapılır ve GPU'da çok yer kaplar. `fp16=True` ile Trainer'a "Kritik olmayan bazı işlemleri 16-bit olarak hesapla, ancak ana ağırlıkları 32-bit tut" demiş oluruz. Buna **Mixed-Precision Training** denir. Eğitim kalitesinden ödün vermeden GPU'da büyük bir boş alan yaratıp eğitimi hızlandırır.

---

## Soru 6
**Aşağıdaki öğrenme eğrilerine dayanarak, modeli iyileştirmek için bir sonraki mantıklı adım ne olmalıdır?**

![Soru 6 öğrenme eğrisi](./soru-6.jpeg)

- A) Dropout ve weight decay (ağırlık azaltma) eklemek.
- B) Daha büyük bir model kullanmak veya daha fazla epoch boyunca eğitmek.
- C) Öğrenme oranını (learning rate) düşürmek.
- D) Erken durdurma (early stopping) kullanmak.

**✅ Doğru Cevap: A**

Training loss çok güzel düşmüş ve sıfıra yaklaşmış; yani model eğitim verisini çok iyi öğrenmiş. Ancak validation loss yüksekte düzleşmiş ve arada büyük bir boşluk var. Validation çizgisi yukarı kıvrılmadığı için klasik overfitting (dolayısıyla early stopping) denemez; ancak modelin eğitim verisine fazla adapte olduğu çok belli. Bu durumda **regularization** (dropout, weight decay) eklemek mantıklı adımdır.

---

## Soru 7
**Tokenizasyon işlemi için `Dataset.map()` fonksiyonunda `batched=True` kullanmanın temel performans faydası nedir?**

- A) Aynı anda birden fazla örneği işleyerek, hızlı tokenizer'ların (fast tokenizers) hızından faydalanır.
- B) Tüm örnekleri (samples) otomatik olarak aynı uzunluğa kadar doldurur (padding yapar).
- C) Örnekleri tek tek işleyerek bellek kullanımını azaltır.
- D) Veri setinin sırasının (order) korunmasını garanti eder.

**✅ Doğru Cevap: A**

`batched=True` ile veriler tek tek değil, toplu gruplar halinde gönderilir. Hugging Face'in arkada kullandığı Rust tabanlı tokenizer'lar birden fazla veriyi aynı anda (multi-threading) işleyebilecek şekilde tasarlanmıştır. Bu da ön işleme (preprocessing) süresini çok kısaltır.

---

## Soru 8
**Trainer API ile eğitim sırasında doğruluğu (accuracy) izlemek (monitör etmek) istiyorsunuz. Trainer'a sağlamanız gereken iki şey nedir?**

- A) `eval_dataset` ve bir `compute_metrics` fonksiyonu.
- B) `train_dataset` ve bir `data_collator`.
- C) `model` ve `eval_strategy='steps'` içeren `training_args`.
- D) `eval_dataset` ve `push_to_hub=True` içeren `training_args`.

**✅ Doğru Cevap: A**

Trainer varsayılan olarak sadece "Loss" değerini hesaplar. Eğitim sırasında net bir başarı yüzdesi görmek için iki şey gerekir:

1. Modelin kendini test edebileceği, eğitimde hiç görmediği bir kontrol verisi: `eval_dataset`.
2. Ham skorları (logit'leri) "Doğruluk: %85" gibi anlaşılır bir formata çeviren `compute_metrics` fonksiyonu.

---

## Soru 9
**Bir PyTorch eğitim döngüsü için `DataLoader` oluşturmadan önce, tokenize edilmiş veri seti için bu adımlardan hangisi tipik olarak GEREKLİ DEĞİLDİR?**

- A) Modelin kabul etmediği ham metin gibi sütunları kaldırmak.
- B) `label` sütununu `labels` olarak yeniden adlandırmak.
- C) `set_format('torch')` kullanarak veri seti formatını `'torch'` olarak ayarlamak.
- D) Tüm örnekleri (samples) modelin maksimum uzunluğuna kadar manuel olarak doldurmak (padding yapmak).

**✅ Doğru Cevap: D**

Trainer API'yi bırakıp saf PyTorch ile döngü yazarken şu üç işlemi elle yapmamız gerekir:

1. Ham metin sütunlarını silmek (`remove_columns`).
2. `label` ismini `labels` olarak değiştirmek (`rename_column`).
3. Veriyi tensör formatına çevirmek (`set_format("torch")`).

Manuel padding (D) gerekli değildir; çünkü padding işini `DataLoader`'a verilen `data_collator` (`DataCollatorWithPadding`) dinamik olarak halleder.

---

## Soru 10
**`DataCollatorWithPadding` sınıfının temel işlevi nedir?**

- A) Eğitimden önce tüm veri setini tokenize etmek.
- B) Veri setindeki tüm örneklere kesme (truncation) uygulamak.
- C) Her yığındaki (batch) örnekleri, yalnızca o yığının içindeki maksimum uzunluğa kadar dinamik olarak doldurmak (pad yapmak).
- D) 32 veya 64 gibi sabit boyutlu yığınlar (batches) oluşturmak.

**✅ Doğru Cevap: C**

Modeller tensör matematiğiyle çalıştığı için bir batch içindeki cümlelerin aynı uzunlukta olması gerekir. `DataCollatorWithPadding`, her batch'te en uzun cümleye bakar (örneğin 67) ve sadece o gruptaki diğer cümleleri bu uzunluğa tamamlar. Buna **dinamik padding** denir; gereksiz hesaplamayı önler ve eğitimi hızlandırır.

---

# Özgün Sorular

## Soru 1
**Hugging Face ekosisteminde, ham metin verilerini (cümleleri) modelin işleyebileceği sayısal dizilere dönüştüren temel araç aşağıdakilerden hangisidir?**

- A) Trainer
- B) Tokenizer
- C) DataLoader
- D) Optimizer

**✅ Cevap: B**

Modeller doğrudan metinleri okuyamaz, sayı matrisleriyle çalışır. Tokenizer, metni kelime veya alt kelime parçalarına bölerek sayısal kimliklere (ID) çeviren ilk ve en temel araçtır.

---

## Soru 2
**Trainer API kullanırken, modelin performansını (örneğin Accuracy veya F1 skorunu) her eğitim turunun sonunda otomatik olarak ölçmek için `TrainingArguments` içinde hangi ayarı yapmalıyız?**

- A) `fp16=True`
- B) `batched=True`
- C) `eval_strategy="epoch"`
- D) `learning_rate=5e-5`

**✅ Cevap: C**

Bu ayar, her epoch (tüm verinin modelden bir kez geçmesi) bittiğinde modelin ayrılan doğrulama (validation) seti üzerinde kendini test etmesini sağlar.

---

## Soru 3
**PyTorch ile kendi özel eğitim döngünüzü (training loop) yazarken kullandığınız `loss.backward()` komutu ne işe yarar?**

- A) Modelin ağırlıklarını günceller.
- B) Eski gradyanları sıfırlar.
- C) Hesaplanan hatayı geriye doğru yayarak gradyanları (hataların yönünü) bulur.
- D) Verileri ekran kartına (GPU) taşır.

**✅ Cevap: C**

Model ileri yayılımda hata miktarını (loss) bulduktan sonra, bu komutla hatanın kaynağı geriye dönük hesaplanır. Ağırlıkları asıl güncelleyen komut ise hemen ardından çalışan `optimizer.step()` komutudur.
