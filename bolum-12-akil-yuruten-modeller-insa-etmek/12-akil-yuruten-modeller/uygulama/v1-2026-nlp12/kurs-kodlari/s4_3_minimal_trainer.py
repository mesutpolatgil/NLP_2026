# Sayfa 4 - Minimal GRPOTrainer örneği (Hub'dan veri kümesi ve model indirmesi gerekir)
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

dataset = load_dataset("mlabonne/smoltldr", split="train")  # 5. sayfadaki alıştırmanın veri kümesi
print(dataset)
