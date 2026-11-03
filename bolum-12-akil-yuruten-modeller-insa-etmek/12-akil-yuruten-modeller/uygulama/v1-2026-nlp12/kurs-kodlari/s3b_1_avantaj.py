# Sayfa 3b - "2. Ödülleri Hesaplamak" kodu (kurstaki kodun aynısı, sadece print satırları eklendi)
import torch

# Şekil: (B * G,) = (8,) çünkü düzleştirdiğimiz 4'er üretimlik 2 grubumuz var
rewards = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1], dtype=torch.float32)
num_generations = 4

rewards_grouped = rewards.view(-1, num_generations)
mean_grouped_rewards = rewards_grouped.mean(dim=1)
std_grouped_rewards = rewards_grouped.std(dim=1)
print("Grouped Rewards:", rewards_grouped)
print("Mean per group:", mean_grouped_rewards)
print("Std per group:", std_grouped_rewards)

mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
print("Broadcasted Mean:", mean_grouped_rewards)
print("Broadcasted Std:", std_grouped_rewards)

advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
print("Advantages:", advantages)

advantages = advantages.unsqueeze(1)
print("Advantages (unsqueeze):", advantages)
print("Şekil:", tuple(advantages.shape))
