# Slayt grafikleri: 3b ve 4. sayfadaki kodların çıktılarından üretildi.
import os, re, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grafikler") + os.sep
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
T1, T2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#cfcec9",
    "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7", "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.titlecolor": T1, "axes.titlesize": 16})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name, dpi=200); plt.close(fig); print("kaydedildi:", name)

# 1) Avantaj değerleri (3b, ödül [1,0,0,1 | 0,0,1,1])
import torch
r = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1.0]); g = r.view(-1, 4)
A = ((r - g.mean(1).repeat_interleave(4)) / (g.std(1).repeat_interleave(4) + 1e-8)).numpy()
fig, ax = plt.subplots(figsize=(10, 5))
cols = [BLUE if a > 0 else ORANGE for a in A]
labels = [f"Yanıt {i+1}\nödül={int(v)}" for i, v in enumerate(r)]
ax.bar(range(8), A, color=cols, width=0.6, edgecolor="white", linewidth=2)
for i, a in enumerate(A):
    ax.text(i, a + (0.06 if a > 0 else -0.06), f"{a:+.2f}", ha="center", va="bottom" if a > 0 else "top", color=T1, fontsize=12)
ax.axhline(0, color=T2, lw=1); ax.axvline(3.5, color=GRAY, lw=1, ls="--")
ax.text(1.5, 1.25, "Grup 1  (ort. 0.5, std 0.577)", ha="center", color=T2); ax.text(5.5, 1.25, "Grup 2  (ort. 0.5, std 0.577)", ha="center", color=T2)
ax.set_xticks(range(8), labels, fontsize=11); ax.set_ylim(-1.35, 1.45); ax.set_ylabel("Avantaj  $A_i$")
ax.set_title("Avantaj: doğru yanıtlar (+) pekiştirilir, yanlışlar (−) caydırılır", loc="left")
save(fig, "3b_avantaj_grafik.png")

# 2) Kırpma fonksiyonu, epsilon = 0.2
x = np.linspace(0, 2.2, 300); eps = 0.2
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axhspan(1 - eps, 1 + eps, color="#cde2fb", alpha=0.6, lw=0)
ax.plot(x, x, color=GRAY, lw=2, ls="--", label="kırpılmamış oran")
ax.plot(x, np.clip(x, 1 - eps, 1 + eps), color=BLUE, lw=3, label="clip(oran, 0.8, 1.2)")
for ad, xv, dx, dy in [("Durum 1: 0.9/0.5 = 1.8 → 1.2", 1.8, -0.15, 0.32), ("Çözümlü örnek: 0.7/0.5 = 1.4 → 1.2", 1.4, -0.75, 0.55), ("Durum 2: 0.2/0.5 = 0.4 → 0.8", 0.4, 0.1, -0.45)]:
    yv = float(np.clip(xv, 0.8, 1.2))
    ax.plot([xv, xv], [xv, yv], color=ORANGE, lw=1.5, ls=":")
    ax.scatter([xv], [yv], s=80, color=ORANGE, zorder=5, edgecolor="white", linewidth=2)
    ax.annotate(ad, (xv, yv), (xv + dx, yv + dy), color=T1, fontsize=12, arrowprops=dict(arrowstyle="-", color=T2, lw=1))
ax.text(0.05, 1.03, "izin verilen aralık [0.8, 1.2]", color="#1c5cab", fontsize=12)
ax.set_xlabel("olasılık oranı  π_yeni / π_eski"); ax.set_ylabel("kullanılan oran"); ax.set_xlim(0, 2.2); ax.set_ylim(0, 2.2)
ax.legend(loc="upper left", frameon=False); ax.set_title("Kırpma fonksiyonu (ε = 0.2)", loc="left")
save(fig, "3b_kirpma_grafik.png")

# 3) Kırpılmış hedef: min(oran·A, clip(oran)·A), A = +0.94 ve A = −0.94
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
for ax, Av, c, ttl in [(axs[0], 0.94, BLUE, "Doğru yanıt (A = +0.94)"), (axs[1], -0.94, ORANGE, "Yanlış yanıt (A = −0.94)")]:
    ax.plot(x, x * Av, color=GRAY, lw=2, ls="--", label="oran · A")
    ax.plot(x, np.minimum(x * Av, np.clip(x, 0.8, 1.2) * Av), color=c, lw=3, label="min(oran·A, clip(oran)·A)")
    ax.axvspan(0.8, 1.2, color="#ecebe7", lw=0); ax.axhline(0, color=T2, lw=1)
    ax.set_title(ttl, loc="left"); ax.set_xlabel("olasılık oranı"); ax.legend(frameon=False, fontsize=11, loc="upper left" if Av > 0 else "lower left")
axs[0].set_ylabel("hedef (büyütülmek istenen)")
fig.suptitle("GRPO hedefindeki min + clip: model bir adımda en fazla %20 değişebilir", x=0.01, ha="left", fontsize=16, color=T1)
save(fig, "3b_hedef_min_clip_grafik.png")

# 4) Sayfa 4 ödül fonksiyonları
from s4_1_odul_fonksiyonlari import reward_len, problem_reward, format_reward, reward_format
comp = ["Short", "Medium length text", "This is about twenty chars", "This is a slightly longer completion", "This is a much longer completion with more words"]
rl = reward_len(comp)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(5), rl, color=BLUE, width=0.6)
for i, v in enumerate(rl): ax.text(i, v - 0.6, f"{v}", ha="center", va="top", color=T1)
ax.set_xticks(range(5), [f"\"{c[:18]}{'…' if len(c)>18 else ''}\"\n{len(c)} karakter" for c in comp], fontsize=10)
ax.axhline(0, color=T2, lw=1); ax.set_ylim(min(rl) - 4, 2); ax.set_ylabel("ödül")
ax.set_title("reward_len (ideal_length = 20): 20'ye en yakın olan en az ceza alır", loc="left")
save(fig, "4_reward_len_grafik.png")

probs = ["5 + 7", "12 * 6", "100 / 4", "3x = 15", "√81"]; ans = [12, 72, 25, 5, 9]; mod = [12, 92, 15, 0, 9]
pr = problem_reward(mod, ans)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.bar(range(5), pr, color=[AQUA if v == 1 else ORANGE for v in pr], width=0.6)
for i, v in enumerate(pr): ax.text(i, v + 0.03, f"model: {mod[i]}\ndoğru: {ans[i]}", ha="center", va="bottom", color=T1, fontsize=11)
ax.set_xticks(range(5), probs); ax.set_ylim(0, 1.5); ax.set_yticks([0, 1]); ax.set_ylabel("ödül")
ax.set_title("problem_reward: doğru cevap 1, yanlış cevap 0", loc="left")
save(fig, "4_problem_reward_grafik.png")

fmt = ["<think>Let me solve this step by step</think><answer>42</answer>", "<think>Short</think><answer>42</answer>",
       "<think>Let me solve this step by step</think>\n<answer>42</answer>", "The answer is 15 without any special format",
       "<think>I should use a loop</think><code>while True: pass</code>"]
short = ["think (uzun) + answer", "think (kısa) + answer", "think + alt satır + answer", "etiketsiz düz metin", "think + code"]
fr, rf = format_reward(fmt), reward_format(fmt)
fig, ax = plt.subplots(figsize=(11, 5)); w = 0.36; xs = np.arange(5)
ax.bar(xs - w/2 - 0.01, rf, w, color=GRAY, label="reward_format (katı: ^...$)")
ax.bar(xs + w/2 + 0.01, fr, w, color=BLUE, label="format_reward (içerik > 20 karakter)")
for i in range(5):
    ax.text(i - w/2, rf[i] + 0.03, f"{rf[i]:.1f}", ha="center", color=T1, fontsize=11); ax.text(i + w/2, fr[i] + 0.03, f"{fr[i]:.1f}", ha="center", color=T1, fontsize=11)
ax.set_xticks(xs, short, fontsize=11); ax.set_ylim(0, 1.35); ax.set_ylabel("ödül"); ax.legend(frameon=False, loc="upper right")
ax.set_title("Format ödülleri: <think>…</think><answer>…</answer>", loc="left")
save(fig, "4_format_reward_grafik.png")
