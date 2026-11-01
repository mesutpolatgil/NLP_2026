# Sayfa 3b - teori kısmındaki örnek: 8 yanıt, 4 doğru (1) 4 yanlış (0)
# Sayfadaki tablo: mean = 0.5, std = 0.53, A = +0.94 / -0.94
import torch
r = torch.tensor([1, 0, 0, 1, 1, 0, 0, 1], dtype=torch.float32)   # 4 doğru, 4 yanlış
mean, std = r.mean(), r.std()
print(f"mean(r) = {mean:.2f}")
print(f"std(r)  = {std:.2f}")
print(f"A (doğru)  = (1 - {mean:.2f}) / {std:.2f} = {(1-mean)/std:.2f}")
print(f"A (yanlış) = (0 - {mean:.2f}) / {std:.2f} = {(0-mean)/std:.2f}")
