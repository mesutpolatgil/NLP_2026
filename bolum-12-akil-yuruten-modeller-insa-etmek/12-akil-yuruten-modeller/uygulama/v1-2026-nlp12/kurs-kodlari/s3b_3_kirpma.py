# Sayfa 3b - Kırpma (clip) örnekleri, epsilon = 0.2
# Durum 1: yeni 0.9 / eski 0.5 ; Durum 2: yeni 0.2 / eski 0.5 ; Çözümlü örnek: yeni 0.7 / eski 0.5
import torch
eps = 0.2
for ad, yeni, eski in [("Durum 1", 0.9, 0.5), ("Durum 2", 0.2, 0.5), ("Çözümlü örnek", 0.7, 0.5)]:
    ratio = torch.tensor(yeni / eski)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    print(f"{ad:14s}: oran = {yeni}/{eski} = {ratio:.1f}  ->  clip sonrası {clipped:.1f}")
