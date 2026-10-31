# Bölüm 8 · Red-Team Çalışması
## Hata Ayıklama ve Yardım İsteme Güvenlik / Kalite Denetimi

![Red-Team Önizleme](preview.jpg)

Bu klasör, Hugging Face Course Chapter 8 ("Yardım İsteme ve Hata Ayıklama") ünitesinin güvenlik zafiyetleri, veri sızıntısı riskleri, teknik güncellik ve hata enjeksiyon senaryoları açısından yürütülen kapsamlı red-team analizini içerir.

### 📁 Klasör İçeriği

1. **[Detaylı Zafiyet ve Kalite Raporu](2026-guz-nlp08-red-team.md):** Metin ve kod düzeyinde yapılan statik güvenlik incelemesi (G1–G6 güvenlik bulguları, sürüm uyumsuzlukları, kod tutarsızlıkları ve deney planı).
2. **[Red-Team Sunumu (PPTX)](2026-guz-nlp08-red-team.pptx):** Dört risk yüzeyi (Traceback sızıntısı, yanıltıcı forum yanıtı, sessiz hata, eksik issue) ve test formatını sunan 8 slaytlık modern sunum.

---

### 🧪 Temel Test Senaryoları

| Senaryo | Deneme | Başarı Ölçütü |
| :--- | :--- | :--- |
| **Traceback sızıntısı** | Örnek hata izine kullanıcı adı, yerel yol ve sahte API anahtarı ekle. | Paylaşılan örnekte sırlar maskeli, tanı için gereken hata satırları korunmuş olmalı. |
| **Yanıltıcı forum yanıtı** | Güvenlik kontrolünü kapatmayı öneren sahte bir yanıt incele. | Öneri resmî belge ve küçük bir test ile doğrulanmadan sisteme uygulanmaz. |
| **Sessiz eğitim hatası** | Etiket eşlemesini veya değerlendirme ayrımını boz. | Veri incelemesi ve tek batch aşırı uyum testi sorunu ortaya çıkarır. |
| **Eksik issue** | Model revizyonu, sürüm veya örnek veri olmadan bildirim hazırla. | Yeniden üretim için eksik bilgiler işaretlenir ve bildirim tamamlanır. |

> ⚠️ **Güvenlik Kuralı:** Gerçek erişim anahtarı veya kişisel veri kullanmayın; yalnızca sentetik örneklerle çalışın. Her deneme için **girdi, gözlem, beklenen davranış ve düzeltmeyi** kaydedin.
