# Sayfa 3b - "3. Politikayı Güncellemek" kodu.
# Kurstaki kodda new_per_token_logps, per_token_logps, self.cliprange, self.beta tanımlı değil
# (gerçekte modelden gelir). Burada çalışabilmesi için ÖRNEK değerler verildi:
#   eski politika olasılığı her yanıt için 0.5, yeni politika olasılıkları aşağıda (sayfadaki 0.7, 0.9, 0.2 örnekleri dahil).
import torch
import torch.nn.functional as F

advantages = torch.tensor([0.8659, -0.8660, -0.8660, 0.8659, -0.8660, -0.8660, 0.8659, 0.8659]).unsqueeze(1)
old_p = torch.full((8, 1), 0.5)
new_p = torch.tensor([[0.7], [0.4], [0.2], [0.9], [0.5], [0.45], [0.55], [0.6]])
per_token_logps = old_p.log()       # eski politika log-olasılıkları  (8, 1)
new_per_token_logps = new_p.log()   # yeni politika log-olasılıkları  (8, 1)
cliprange, beta = 0.2, 0.04         # epsilon = 0.2 ; beta = 0.04 (DeepSeekMath)

# --- kurstaki kod ---
ratio = torch.exp(new_per_token_logps - per_token_logps)  # (8, 1)

eps = cliprange
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
pg_loss_max = torch.max(pg_losses1, pg_losses2)

per_token_kl = F.kl_div(
    F.log_softmax(new_per_token_logps, dim=-1),
    F.softmax(per_token_logps, dim=-1),
    reduction="none",
).sum(dim=-1, keepdim=True)

per_token_loss = pg_loss_max + beta * per_token_kl
# --- kurstaki kod sonu ---

torch.set_printoptions(precision=4)
print("ratio        :", [round(x, 3) for x in ratio.squeeze().tolist()])
print("clamp(ratio) :", [round(x, 3) for x in torch.clamp(ratio, 0.8, 1.2).squeeze().tolist()])
print("pg_losses1   :", [round(x, 3) for x in pg_losses1.squeeze().tolist()])
print("pg_losses2   :", [round(x, 3) for x in pg_losses2.squeeze().tolist()])
print("pg_loss_max  :", [round(x, 3) for x in pg_loss_max.squeeze().tolist()])
print("per_token_kl :", [round(x, 3) for x in per_token_kl.squeeze().tolist()])
print("per_token_loss:", [round(x, 3) for x in per_token_loss.squeeze().tolist()])
print("Şekil:", tuple(per_token_loss.shape), " Ortalama kayıp:", round(per_token_loss.mean().item(), 4))

# Not: Son boyut 1 olduğu için softmax hep 1 verir, bu yüzden yukarıdaki per_token_kl her zaman 0 çıkar.
# Kursun bağlantı verdiği TRL GRPOTrainer kodu KL'yi şu tahminciyle hesaplar (referans = eski politika varsayıldı):
#   per_token_kl = exp(ref_logp - logp) - (ref_logp - logp) - 1
ref_per_token_logps = per_token_logps
trl_kl = torch.exp(ref_per_token_logps - new_per_token_logps) - (ref_per_token_logps - new_per_token_logps) - 1
trl_loss = pg_loss_max + beta * trl_kl
print()
print("TRL tarzı KL :", [round(x, 3) for x in trl_kl.squeeze().tolist()])
print("TRL tarzı kayıp:", [round(x, 3) for x in trl_loss.squeeze().tolist()])
print("Ortalama kayıp (TRL KL ile):", round(trl_loss.mean().item(), 4))
