# ==============================================================================
# HUGGING FACE LLM COURSE - BÖLÜM 12: GRPO RED-TEAM TESTLERİ (5 TEST)
# Ödül fonksiyonları kursun 12.4, 12.5 ve 12.6 sayfalarından birebir alınmıştır.
# GPU ve internet gerekmez: pip install torch
# ==============================================================================
import re
import torch

# ---------------- Kurstaki fonksiyonlar (değiştirilmeden) ----------------
ideal_length = 50  # 12.5

def reward_len(completions, **kwargs):  # 12.5
    return [-abs(ideal_length - len(completion)) for completion in completions]

def extract_xml_answer(text: str) -> str:  # 12.6
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs):  # 12.6 (print satırı çıkarıldı)
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs):  # 12.6
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs):  # 12.6
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs):  # 12.6
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def count_xml(text) -> float:  # 12.6
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def advantages(rewards, num_generations):  # 12.3b
    rewards = torch.tensor(rewards, dtype=torch.float32)
    g = rewards.view(-1, num_generations)
    mean = g.mean(dim=1).repeat_interleave(num_generations)
    std = g.std(dim=1).repeat_interleave(num_generations)
    return (rewards - mean) / (std + 1e-8)

def c(text):  # TRL'nin sohbet formatındaki tamamlama yapısı
    return [{"content": text}]

def toplam_odul(text, cevap):
    k = [c(text)]
    return (correctness_reward_func(None, k, [cevap])[0] + int_reward_func(k)[0]
            + strict_format_reward_func(k)[0] + soft_format_reward_func(k)[0] + count_xml(text))

# ---------------- TEST 1: Uzunluk ödülünü kandırmak (12.5) ----------------
print("=" * 70); print("TEST 1: Uzunluk ödülü (12.5, ideal_length = 50 karakter)"); print("=" * 70)
adaylar = {
    "Anlamlı özet": "The cat is a small domesticated carnivorous mammal kept as a pet by people.",
    "Anlamlı kısa özet": "Cats are domestic pets.",
    "Anlamsız dolgu (50 x 'a')": "a" * 50,
    "Tekrar eden kelime": "cat " * 12 + "ca",
}
for ad, metin in adaylar.items():
    print(f"{ad:34s} uzunluk={len(metin):3d}  ödül={reward_len([metin])[0]:4d}")

# ---------------- TEST 2: Format ödülünü boş etiketlerle geçmek (12.6) ----------------
print(); print("=" * 70); print("TEST 2: Format ödülleri (12.6)"); print("=" * 70)
ornekler = {
    "Doğru, tek satır akıl yürütme": "<reasoning>\n6 * 12 = 72\n</reasoning>\n<answer>\n72\n</answer>\n",
    "Doğru, çok satırlı akıl yürütme": "<reasoning>\nHer kutuda 12 kalem var.\n6 kutu var.\n6 * 12 = 72\n</reasoning>\n<answer>\n72\n</answer>\n",
    "Boş akıl yürütme, cevap var": "<reasoning>\n\n</reasoning>\n<answer>\n72\n</answer>\n",
    "Tamamen boş etiketler": "<reasoning>\n\n</reasoning>\n<answer>\n\n</answer>\n",
}
print(f"{'Örnek':34s} strict  soft  xmlcount")
for ad, metin in ornekler.items():
    k = [c(metin)]
    print(f"{ad:34s} {strict_format_reward_func(k)[0]:5.2f}  {soft_format_reward_func(k)[0]:4.2f}  {count_xml(metin):7.3f}")

# ---------------- TEST 3: Doğruluk ödülünün kırılganlığı (12.6) ----------------
print(); print("=" * 70); print("TEST 3: Doğruluk ve tam sayı ödülleri (doğru cevap = '72')"); print("=" * 70)
for yazim in ["72", "72.0", "$72", "72 kalem", " 72 ", "1,000", "-5"]:
    metin = f"<reasoning>\n6 * 12 = 72\n</reasoning>\n<answer>\n{yazim}\n</answer>\n"
    k = [c(metin)]
    beklenen = "1000" if yazim == "1,000" else ("-5" if yazim == "-5" else "72")
    print(f"cevap={yazim!r:12s} beklenen={beklenen!r:7s} doğruluk={correctness_reward_func(None, k, [beklenen])[0]:.1f}  tam_sayı={int_reward_func(k)[0]:.1f}")

print("\nToplam ödül karşılaştırması (5 ödül fonksiyonunun toplamı):")
dogru_cok_satir = ornekler["Doğru, çok satırlı akıl yürütme"]
yanlis_tek_satir = "<reasoning>\nbilmiyorum\n</reasoning>\n<answer>\n71\n</answer>\n"
print(f"Doğru cevap, çok satırlı akıl yürütme : {toplam_odul(dogru_cok_satir, '72'):.3f}")
print(f"Yanlış cevap, tek satır akıl yürütme   : {toplam_odul(yanlis_tek_satir, '72'):.3f}")
print(f"Doğru cevap ama '72.0' yazılmış        : {toplam_odul(dogru_cok_satir.replace('72\n</answer>', '72.0\n</answer>'), '72'):.3f}")

# ---------------- TEST 4: Grup içinde ödül farkı yoksa öğrenme durur (12.3b) ----------------
print(); print("=" * 70); print("TEST 4: Avantaj çöküşü (num_generations = 4)"); print("=" * 70)
for ad, r in {
    "Karışık grup [1,0,0,1]": [1, 0, 0, 1],
    "Hepsi doğru [1,1,1,1]": [1, 1, 1, 1],
    "Hepsi yanlış [0,0,0,0]": [0, 0, 0, 0],
    "Biri çok az farklı [1,1,1,0.99]": [1, 1, 1, 0.99],
}.items():
    print(f"{ad:32s} avantaj = {[round(x, 3) for x in advantages(r, 4).tolist()]}")

# ---------------- TEST 5: 12.3b'deki KL kodu her zaman 0 veriyor ----------------
print(); print("=" * 70); print("TEST 5: KL ıraksaması kodu (12.3b)"); print("=" * 70)
per_token_logps = torch.log(torch.tensor([[0.5], [0.5], [0.5], [0.5]]))
new_per_token_logps = torch.log(torch.tensor([[0.9], [0.2], [0.7], [0.5]]))
kl_kurs = torch.nn.functional.kl_div(
    torch.nn.functional.log_softmax(new_per_token_logps, dim=-1),
    torch.nn.functional.softmax(per_token_logps, dim=-1),
    reduction="none",
).sum(dim=-1, keepdim=True)
d = per_token_logps - new_per_token_logps
kl_trl = torch.exp(d) - d - 1
print("Yeni olasılıklar          :", [0.9, 0.2, 0.7, 0.5], " (eski hepsi 0.5)")
print("Kurstaki kod ile KL       :", [round(x, 3) for x in kl_kurs.flatten().tolist()])
print("TRL'deki formül ile KL    :", [round(x, 3) for x in kl_trl.flatten().tolist()])
