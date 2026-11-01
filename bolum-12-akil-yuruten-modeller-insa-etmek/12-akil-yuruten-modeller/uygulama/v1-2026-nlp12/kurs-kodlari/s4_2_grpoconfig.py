# Sayfa 4 - GRPOConfig örneği. Kurstaki yazımda "num_generation" geçiyor;
# TRL'deki parametrenin adı "num_generations". İkisini de deniyoruz.
from trl import GRPOConfig, __version__
print("TRL sürümü:", __version__)

try:
    GRPOConfig(output_dir="output", num_train_epochs=3, num_generation=4, per_device_train_batch_size=4)
except TypeError as e:
    print("num_generation=4  ->  HATA:", e)

args = GRPOConfig(
    output_dir="output",
    num_train_epochs=3,
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    logging_steps=10,
    use_vllm=False,  # vLLM GPU ister; burada CPU olduğu için kapalı
    report_to="none",
    use_cpu=True, bf16=False,  # bu ortamda GPU yok
)
print("num_generations=4 ->  OK")
for k in ["num_generations", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "beta", "epsilon", "max_completion_length", "temperature"]:
    print(f"  {k:28} = {getattr(args, k)}")
