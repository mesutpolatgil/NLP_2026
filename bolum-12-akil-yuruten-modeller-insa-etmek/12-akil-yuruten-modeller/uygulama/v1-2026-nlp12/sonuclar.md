# Deney Sonuçları

## Değerlendirme düzeni

Eğitimden önce ve sonra, eğitimde kullanılmamış doğrulama bölümündeki aynı 10 örnekte ölçüm yapıldı. Her iki ölçümde de cevap üretim üst sınırı 96 token olarak kullanıldı. Hedef uzunluk 50 tokendir.

## Genel sonuçlar

| Ölçüm | Eğitim öncesi | Eğitim sonrası |
|---|---:|---:|
| Ortalama cevap uzunluğu | 86,4 token | 55,5 token |
| 50 token hedefine ortalama uzaklık | 43,2 token | 9,5 token |
| 96 token sınırına ulaşan cevap | 8/10 | 1/10 |

Ortalama hata 33,7 token, yani yaklaşık %78 azaldı.

## Örnek bazında sonuçlar

| Örnek | Önce token | Sonra token | Önce hata | Sonra hata |
|---:|---:|---:|---:|---:|
| 1 | 96 | 48 | 46 | 2 |
| 2 | 96 | 96 | 46 | 46 |
| 3 | 80 | 55 | 30 | 5 |
| 4 | 96 | 50 | 46 | 0 |
| 5 | 16 | 50 | 34 | 0 |
| 6 | 96 | 43 | 46 | 7 |
| 7 | 96 | 51 | 46 | 1 |
| 8 | 96 | 39 | 46 | 11 |
| 9 | 96 | 65 | 46 | 15 |
| 10 | 96 | 58 | 46 | 8 |

10 örneğin 9'unda hata azaldı. İki örnek eğitim sonrasında tam 50 token üretti. Bir örnek 96 token üretim sınırında kaldı.

## Yorum

GRPO ve LoRA eğitimi, cevap uzunluğunu 50 token hedefine belirgin biçimde yaklaştırdı. Bu sonuç yalnızca cevap uzunluğu davranışını gösterir; cevapların doğruluğu veya akıl yürütme kalitesi ayrıca ölçülmemiştir.
