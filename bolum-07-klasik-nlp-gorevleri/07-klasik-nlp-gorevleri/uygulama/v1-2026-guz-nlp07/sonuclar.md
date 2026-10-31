# Gerçek Yerel Çalıştırma Kaydı

31 Ekim 2026. Terminal komutu: `python main.py --task all --steps 100`.
Notebook'un 8 kod hücresi ayrıca Python hücre yürütücüsüyle sırayla çalıştırıldı;
çıktılar bu çalıştırmalardan kaydedildi. Jupyter arayüzünde etkileşimli koşum yapılmadı.
Altı görev ve beş hata deneyinin JSON sonuçları terminalle tam eşleşti.

Standart hata akışındaki `loss_type=None` bildirimi GPT-2 için varsayılan
ForCausalLMLoss seçimini belirtir; süreç çıkış kodu 0'dır.
GPU kullanılmadı, ağ erişimi kapalıydı. Veriler main.py içinde insan yazımıdır.

## Kaynak parmak izleri

- `main.py` SHA256: `d6ecbf76b95a63c01f505e1aa2f9359ff77a4f519389a127dd8e29782dc609a0`
- `main.ipynb` SHA256: `a7f5af8cb034e4ae24cf74e19a919e7e5be1092b31d40768444c53ae42a34180`
- `requirements.txt` SHA256: `e84bca91a536aef54b678ecc02607905b0c8fe8761cfa01b4159537d53f003d7`

## Terminal çıktısı

```json
{
  "environment": {
    "python": "3.12.14",
    "torch": "2.9.0+cpu",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "device": "cpu",
    "seed": 7,
    "threads": 2
  },
  "scope": "İnsan yazımı çok küçük veri, rastgele başlatılmış gerçek Transformer modelleri. Eğitim örneği kontrolü, benchmark değildir.",
  "tasks": {
    "ner": {
      "initial_loss": 1.072433,
      "final_loss": 0.000263,
      "steps": 100,
      "parameters": 14371,
      "evaluation": "Aynı küçük eğitim verisinde kontrol; genelleme metriği değildir.",
      "tokens": [
        "[CLS]",
        "Ayşe",
        "Ankara",
        "şehrine",
        "gitti",
        ".",
        "[SEP]"
      ],
      "labels": [
        -100,
        1,
        2,
        0,
        0,
        0,
        -100
      ],
      "predictions": [
        0,
        1,
        2,
        0,
        0,
        0,
        0
      ],
      "train_token_accuracy": 1.0,
      "label_map": {
        "0": "O",
        "1": "B-PER",
        "2": "B-LOC"
      }
    },
    "mlm": {
      "initial_loss": 3.849792,
      "final_loss": 0.002862,
      "steps": 100,
      "parameters": 15439,
      "evaluation": "Aynı küçük eğitim verisinde kontrol; genelleme metriği değildir.",
      "masked_inputs": [
        "[CLS] Ayşe [MASK] şehrine gitti . [SEP]",
        "[CLS] Mert [MASK] şehrine gitti . [SEP]",
        "[CLS] Umut [MASK] şehrine gitti . [SEP]",
        "[CLS] Berk [MASK] şehrine gitti . [SEP]"
      ],
      "targets": [
        "Ankara",
        "Sivas",
        "İzmir",
        "Ankara"
      ],
      "predictions": [
        "Ankara",
        "Sivas",
        "İzmir",
        "Ankara"
      ],
      "note": "Bu dört eğitim örneğinde kişi adları şehir hedefini ayırt ettirir; yeni kişilere genelleme ölçülmedi."
    },
    "translation": {
      "initial_loss": 3.814235,
      "final_loss": 0.001912,
      "steps": 100,
      "parameters": 31328,
      "evaluation": "Aynı küçük eğitim verisinde kontrol; genelleme metriği değildir.",
      "sources": [
        "I love books .",
        "The weather is cold .",
        "Do not ship 12 boxes to {CITY} ."
      ],
      "references": [
        "Ben kitapları seviyorum .",
        "Hava soğuk .",
        "{CITY} şehrine 12 kutu gönderme ."
      ],
      "generated": [
        "Ben kitapları seviyorum .",
        "Hava soğuk .",
        "{CITY} şehrine 12 kutu gönderme ."
      ],
      "train_exact_match": 1.0,
      "metric_note": "Bu exact match küçük eğitim çiftlerindedir; BLEU/ROUGE hesaplanmadı."
    },
    "summarization": {
      "initial_loss": 3.784526,
      "final_loss": 0.001683,
      "steps": 100,
      "parameters": 31328,
      "evaluation": "Aynı küçük eğitim verisinde kontrol; genelleme metriği değildir.",
      "sources": [
        "Ayşe Ankara şehrine gitti . Toplantı yarın yapılacak .",
        "Mert Sivas şehrine gitti . Toplantı yarın yapılacak ."
      ],
      "references": [
        "Ayşe Ankara şehrine gitti .",
        "Mert Sivas şehrine gitti ."
      ],
      "generated": [
        "Ayşe Ankara şehrine gitti .",
        "Mert Sivas şehrine gitti ."
      ],
      "train_exact_match": 1.0,
      "metric_note": "Bu exact match küçük eğitim çiftlerindedir; BLEU/ROUGE hesaplanmadı."
    },
    "causal": {
      "initial_loss": 3.871957,
      "final_loss": 0.232638,
      "steps": 100,
      "parameters": 18368,
      "evaluation": "Aynı küçük eğitim verisinde kontrol; genelleme metriği değildir.",
      "prompt": "Mert",
      "generated": "Mert Sivas şehrine gitti .",
      "note": "Küçük sıfırdan model; metin devamı çalıştırıldı, kod üretim modeli değildir."
    },
    "qa": {
      "initial_loss": 2.508127,
      "final_loss": 0.000136,
      "steps": 100,
      "parameters": 14338,
      "evaluation": "Aynı küçük eğitim verisinde kontrol; genelleme metriği değildir.",
      "questions": [
        "Ayşe nereye gitti ?",
        "Mert nereye gitti ?",
        "Umut nereye gitti ?",
        "Berk nereye gitti ?"
      ],
      "references": [
        "Ankara",
        "Sivas",
        "İzmir",
        "Ankara"
      ],
      "predictions": [
        {
          "answer": "Ankara",
          "start": 5,
          "end": 11
        },
        {
          "answer": "Sivas",
          "start": 5,
          "end": 10
        },
        {
          "answer": "İzmir",
          "start": 5,
          "end": 10
        },
        {
          "answer": "Ankara",
          "start": 5,
          "end": 11
        }
      ],
      "train_exact_match": 1.0
    }
  },
  "red_team": {
    "RT01_loss_mask": {
      "active_targets": [
        2,
        4,
        2
      ],
      "gradient_positions": [
        [
          false,
          true,
          false,
          true,
          false
        ],
        [
          true,
          true,
          false,
          true,
          true
        ],
        [
          false,
          true,
          false,
          true,
          false
        ]
      ],
      "losses": [
        1.098612,
        1.098612,
        1.098612
      ],
      "conclusion": "O=0 gerçek hedeftir; -100 yok sayılır. Ortalama kayıp artmak zorunda değildir."
    },
    "RT02_qa_window": {
      "max_length": 16,
      "stride": 4,
      "cropped_has_answer": false,
      "window_count": 8,
      "answer_in_windows": [
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true
      ],
      "character_span": [
        215,
        220
      ],
      "conclusion": "Gerçek fast tokenizer ile cevap ilk pencerede kayboldu, taşan pencerelerle bulundu; model doğruluğu ölçülmedi."
    },
    "RT03_subword_alignment": {
      "tokens": [
        "[CLS]",
        "Ayşe",
        "Ankara",
        "##lı",
        "[SEP]"
      ],
      "word_ids": [
        null,
        0,
        1,
        1,
        null
      ],
      "naive_labels": [
        1,
        2
      ],
      "naive_length": 2,
      "token_length": 5,
      "corrected": [
        -100,
        1,
        2,
        -100,
        -100
      ]
    },
    "RT04_generated_code": {
      "broken": "SyntaxError",
      "fixed": "parse_ok",
      "executed": false,
      "note": "İnsan yazımı kod; hiçbir üretilmiş kod yürütülmedi."
    },
    "RT05_meaning": {
      "reference": "{CITY} şehrine 12 kutu gönderme .",
      "changed": "{CITY} şehrine 12 kutu gönder .",
      "reference_word_overlap": 0.833333,
      "fixed": "{CITY} şehrine 12 kutu gönderme .",
      "note": "Basit küme örtüşmesi; BLEU/ROUGE değildir. İnsan yazımı karşı örnek anlam kontrolünün gereğini gösterir."
    }
  }
}
```
