# Sayfa 4 - Ödül fonksiyonları (kurstaki kodun aynısı) + örnek tamamlamalarla çalıştırma.
# Örnek tamamlamalar kursun gömülü Marimo not defterlerinden (huggingface/notebooks, course/en/chapter13) alındı.
import re


def reward_length(completions, **kwargs):
    return [float(len(completion)) for completion in completions]


def reward_format(completions, **kwargs):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c) else 0.0 for c in completions]


def reward_len(completions, **kwargs):
    ideal_length = 20
    return [-abs(ideal_length - len(completion)) for completion in completions]


def extract_final_answer(completion):
    # grpo_math not defterindeki gibi: tamamlama doğrudan sayı
    return completion


def problem_reward(completions, answers, **kwargs):
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        try:
            answer = extract_final_answer(completion)
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)
        except:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()
            if len(think_content) > 20 and len(answer_content) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


uzunluk = [
    "Short",
    "Medium length text",
    "This is about twenty chars",
    "This is a slightly longer completion",
    "This is a much longer completion with more words",
]
print("=== reward_length ve reward_len (ideal_length = 20) ===")
for c, a, b in zip(uzunluk, reward_length(uzunluk), reward_len(uzunluk)):
    print(f"{c!r:52} uzunluk={len(c):3d}  reward_length={a:5.1f}  reward_len={b:4d}")

print("\n=== problem_reward ===")
problems = ["What is 5 + 7?", "Calculate 12 * 6", "What is 100 / 4?", "Solve for x: 3x = 15", "What is the square root of 81?"]
answers = [12, 72, 25, 5, 9]
model = [12, 92, 15, 0, 9]
for p, a, m, r in zip(problems, answers, model, problem_reward(model, answers)):
    print(f"{p:32} doğru={a:3d}  model={m:3d}  ödül={r}")

print("\n=== reward_format (katı ^...$) ve format_reward (içerik kontrollü) ===")
fmt = [
    "<think>Let me solve this step by step</think><answer>42</answer>",
    "<think>Short</think><answer>42</answer>",
    "<think>Let me solve this step by step</think>\n<answer>42</answer>",
    "The answer is 15 without any special format",
    "<think>I should use a loop</think><code>while True: pass</code>",
]
for c, a, b in zip(fmt, reward_format(fmt), format_reward(fmt)):
    print(f"{c!r:72} reward_format={a}  format_reward={b}")
