# Doğal Dil İşleme (NLP) - Açık Kaynak Sistem Mimarisi

Bu depo, 4. sınıf Doğal Dil İşleme dersinin laboratuvar merkezi, yerelleştirilmiş dokümantasyon arşivi ve sızma testi (red-teaming) alanıdır. Dersin ana omurgası Hugging Face LLM Course üzerinden yürütülmektedir. kaynak:https://huggingface.co/learn/llm-course

## 📌 İş Akışı ve Pull Request (PR) Kuralları

Bu deponun `main` dalına (branch) doğrudan kod veya doküman göndermek (Push) kesinlikle yasaktır[cite: 2]. Tüm süreç, yazılım sektöründeki standart "Code Review" (Kod İnceleme) mantığıyla yönetilecektir[cite: 2]:

1. **Dal Aç (Branching):** Kendinize atanan Hugging Face bölümü için yeni bir dal oluşturun (Örn: `git checkout -b takim-1-transformer`)[cite: 2].
2. **Geliştirme (.md / .qmd):** Orijinal metnin çevirilerini, yerelleştirilmiş "Cheat Sheet" özetlerinizi ve Python hata ayıklama (forensics) senaryolarınızı bu dalda hazırlayın. Sıkıştırılmış dosya (zip) veya Word belgesi kabul edilmez[cite: 2].
3. **Lokal Test:** Kodlarınızı ve dokümanınızı GitHub'a göndermeden önce mutlaka kendi bilgisayarınızda derleyerek test edin.
4. **Onay İste (Pull Request):** İşiniz bittiğinde `main` dalına bir PR açın[cite: 2]. Hata veren veya tahtada mimari olarak savunulamayan kodlar PR aşamasında doğrudan reddedilecektir.

## 👥 Takım Rolleri ve Depo Katkısı

Sınıftaki 4 kişilik Yapay Zeka Ar-Ge Takımları, bu repoyu aşağıdaki görev dağılımına göre besleyecektir:

* **Öğrenci 1 (Dokümantasyon Yöneticisi):** Hugging Face orijinal metnini çevirir ve kendi anladığı yerelleştirilmiş teknik rehber ile birlikte PR atar.
* **Öğrenci 2 (Sistem Mimarı):** Repodaki dokümanı kullanarak algoritmanın temelini ve icat edilme nedenini tahtada sınıfa sezgisel olarak açıklar.
* **Öğrenci 3 (QA & Adli Bilişim Uzmanı):** Repodaki kodu kasten bozarak veya modelin sınırlarını zorlayarak sistemin zafiyetini (hallucination, prompt injection) tahtada canlı olarak test eder.
* **Öğrenci 4 (Sınav Komiseri):** Ünite sonu quizlerini akademik değerlendirme formatına getirerek bölüm dokümanının sonuna ekler ve sınıfı test eder.

## 📂 Klasör Mimarisi

- `/bolum_01_transformer_modellerine_giris/`
- `/bolum_02_huggingface_kutuphaneleri/`
- `/bolum_03_ince_ayar_finetuning/`
- `README.md`

## ⚠️ Değerlendirme ve Zihinsel Derleme (Mental Execution)

Laboratuvar hazırlık aşamasında yapay zeka (LLM) araçlarını bir asistan olarak kullanmak serbesttir. Ancak takımınız tahtaya çıktığında, yazılan kodların ve manipüle edilen mimarinin arka planındaki mental süreci eksiksiz savunmak zorundadır[cite: 3]. Kodun çalışma zamanı (runtime) mantığını ve bellek yapısını açıklayamazsanız tesliminiz reddedilir[cite: 3]. Kodun sahibi yapay zeka değil, bizzat siz olmalısınız.
