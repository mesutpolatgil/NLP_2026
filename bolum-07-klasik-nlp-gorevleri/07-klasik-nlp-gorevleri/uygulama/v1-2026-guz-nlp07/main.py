"""Bölüm 7: küçük, yerel Hugging Face modelleri ve yeniden üretilebilir hata deneyleri.

Veriler insan yazımıdır. Modeller rastgele başlatılır; bir pretrained modelin
kalitesini, Türkçe başarısını veya gerçek dünya genellemesini ölçmez.
"""
import argparse
import ast
import json
import platform
from pathlib import Path
import random

import torch
import transformers
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig, BertForTokenClassification, BertForMaskedLM,
    BertForQuestionAnswering, GPT2Config, GPT2LMHeadModel,
    BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast,
)

SEED = 7
TEXTS = [
    'Ayşe Ankara şehrine gitti .', 'Mert Sivas şehrine gitti .',
    'Umut İzmir şehrine gitti .', 'Berk Ankara şehrine gitti .',
    'Ayşe nereye gitti ?', 'Mert nereye gitti ?', 'Umut nereye gitti ?',
    'Berk nereye gitti ?', 'I love books .', 'Ben kitapları seviyorum .',
    'The weather is cold .', 'Hava soğuk .',
    'Ayşe Ankara şehrine gitti . Toplantı yarın yapılacak .',
    'Ayşe Ankara şehrine gitti .',
    'Do not ship 12 boxes to {CITY} .',
    '{CITY} şehrine 12 kutu gönderme .',
    '{CITY} şehrine 12 kutu gönder .',
    'bilgi kitap toplantı yarın yapılacak şehir nerede',
]

def setup():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    vocab = {t: i for i, t in enumerate(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])}
    for word in sorted(set(' '.join(TEXTS).split())):
        vocab[word] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                  unk_token='[UNK]', cls_token='[CLS]',
                                  sep_token='[SEP]', mask_token='[MASK]',
                                  bos_token='[CLS]', eos_token='[SEP]')

def bert_config(tok, **kwargs):
    return BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=64,
                      max_position_embeddings=128, hidden_dropout_prob=0.0,
                      attention_probs_dropout_prob=0.0, pad_token_id=0, **kwargs)

def fit(model, batch, steps):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Sonlu olmayan kayıp')
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad(): final = float(model(**batch).loss)
    if final >= losses[0]:
        raise AssertionError('Küçük eğitim kümesinde kayıp düşmedi')
    return {'initial_loss': round(losses[0], 6), 'final_loss': round(final, 6),
            'steps': steps, 'parameters': sum(p.numel() for p in model.parameters()),
            'evaluation': 'Aynı küçük eğitim verisinde kontrol; genelleme metriği değildir.'}

def ner(tok, steps):
    texts = TEXTS[:4]
    batch = tok(texts, padding=True, return_tensors='pt')
    labels = torch.full_like(batch['input_ids'], -100)
    for i in range(4): labels[i, 1:6] = torch.tensor([1, 2, 0, 0, 0])
    batch['labels'] = labels
    model = BertForTokenClassification(bert_config(tok, num_labels=3))
    result = fit(model, batch, steps)
    with torch.no_grad(): predicted = model(**batch).logits.argmax(-1)
    active = labels != -100
    result.update(tokens=tok.convert_ids_to_tokens(batch['input_ids'][0].tolist()),
                  labels=labels[0].tolist(), predictions=predicted[0].tolist(),
                  train_token_accuracy=round(float((predicted[active] == labels[active]).float().mean()), 6),
                  label_map={'0': 'O', '1': 'B-PER', '2': 'B-LOC'})
    return result

def mlm(tok, steps):
    batch = tok(TEXTS[:4], padding=True, return_tensors='pt')
    labels = torch.full_like(batch['input_ids'], -100)
    labels[:, 2] = batch['input_ids'][:, 2]
    batch['input_ids'][:, 2] = tok.mask_token_id
    batch['labels'] = labels
    model = BertForMaskedLM(bert_config(tok))
    result = fit(model, batch, steps)
    with torch.no_grad(): pred = model(**batch).logits[:, 2].argmax(-1)
    result.update(masked_inputs=[tok.decode(x) for x in batch['input_ids']],
                  targets=tok.convert_ids_to_tokens(labels[:, 2].tolist()),
                  predictions=tok.convert_ids_to_tokens(pred.tolist()),
                  note='Bu dört eğitim örneğinde kişi adları şehir hedefini ayırt ettirir; yeni kişilere genelleme ölçülmedi.')
    return result

def seq2seq(tok, steps, summary=False):
    if summary:
        sources = [TEXTS[12], 'Mert Sivas şehrine gitti . Toplantı yarın yapılacak .']
        targets = [TEXTS[13], TEXTS[1]]
    else:
        sources = [TEXTS[8], TEXTS[10], TEXTS[14]]
        targets = [TEXTS[9], TEXTS[11], TEXTS[15]]
    # Hedef yolu güncel text_target API'siyle hazırlanır.
    batch = tok(sources, text_target=targets, padding=True, return_tensors='pt')
    labels = batch['labels']
    labels[labels == tok.pad_token_id] = -100
    batch.pop('token_type_ids', None)
    config = BartConfig(vocab_size=len(tok), d_model=32, encoder_layers=1,
                        decoder_layers=1, encoder_attention_heads=2,
                        decoder_attention_heads=2, encoder_ffn_dim=64,
                        decoder_ffn_dim=64, max_position_embeddings=128,
                        pad_token_id=0, bos_token_id=2, eos_token_id=3,
                        decoder_start_token_id=2, dropout=0.0,
                        attention_dropout=0.0, activation_dropout=0.0,
                        forced_eos_token_id=3)
    model = BartForConditionalGeneration(config)
    result = fit(model, batch, steps)
    with torch.no_grad():
        generated = model.generate(input_ids=batch['input_ids'],
                                   attention_mask=batch['attention_mask'],
                                   max_new_tokens=20, do_sample=False)
    texts = tok.batch_decode(generated, skip_special_tokens=True)
    result.update(sources=sources, references=targets, generated=texts,
                  train_exact_match=sum(p == t for p, t in zip(texts, targets)) / len(targets),
                  metric_note='Bu exact match küçük eğitim çiftlerindedir; BLEU/ROUGE hesaplanmadı.')
    return result

def causal(tok, steps):
    batch = tok(TEXTS[:4], padding=True, return_tensors='pt')
    batch.pop('token_type_ids', None)
    labels = batch['input_ids'].clone()
    labels[batch['attention_mask'] == 0] = -100
    batch['labels'] = labels  # GPT2LMHeadModel kaydırmayı içeride yapar.
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_embd=32, n_layer=1,
                           n_head=2, n_positions=128, n_ctx=128, resid_pdrop=0.0,
                           embd_pdrop=0.0, attn_pdrop=0.0, pad_token_id=0,
                           bos_token_id=2, eos_token_id=3))
    result = fit(model, batch, steps)
    prompt = tok('Mert', add_special_tokens=False, return_tensors='pt')
    ids = torch.cat([torch.tensor([[2]]), prompt['input_ids']], dim=1)
    with torch.no_grad():
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=12, do_sample=False)
    result.update(prompt='Mert', generated=tok.decode(generated[0], skip_special_tokens=True),
                  note='Küçük sıfırdan model; metin devamı çalıştırıldı, kod üretim modeli değildir.')
    return result

def qa(tok, steps):
    contexts = TEXTS[:4]
    questions = TEXTS[4:8]
    answers = ['Ankara', 'Sivas', 'İzmir', 'Ankara']
    encoded = tok(questions, contexts, padding=True, return_offsets_mapping=True,
                  return_tensors='pt')
    offsets = encoded.pop('offset_mapping')
    starts, ends = [], []
    for i, answer in enumerate(answers):
        start = contexts[i].index(answer)
        end = start + len(answer)
        candidates = [j for j, (a, b) in enumerate(offsets[i].tolist())
                      if encoded.sequence_ids(i)[j] == 1 and a <= start < b]
        finish = [j for j, (a, b) in enumerate(offsets[i].tolist())
                  if encoded.sequence_ids(i)[j] == 1 and a < end <= b]
        starts.append(candidates[0]); ends.append(finish[0])
    batch = dict(encoded)
    batch.update(start_positions=torch.tensor(starts), end_positions=torch.tensor(ends))
    model = BertForQuestionAnswering(bert_config(tok))
    result = fit(model, batch, steps)
    with torch.no_grad(): output = model(**batch)
    predictions = []
    for i in range(4):
        valid = [j for j, sid in enumerate(encoded.sequence_ids(i)) if sid == 1]
        # Geçerli bağlam aralıklarını ortak başlangıç+bitiş skoruyla sırala.
        s, e = max(((s, e) for s in valid for e in valid if s <= e and e-s < 8),
                   key=lambda pair: float(output.start_logits[i, pair[0]] + output.end_logits[i, pair[1]]))
        a, b = int(offsets[i, s, 0]), int(offsets[i, e, 1])
        predictions.append({'answer': contexts[i][a:b], 'start': a, 'end': b})
    result.update(questions=questions, references=answers, predictions=predictions,
                  train_exact_match=sum(p['answer'] == a for p, a in zip(predictions, answers))/4)
    return result

def red_team(tok):
    records = {}
    labels = torch.tensor([-100, 1, -100, 2, -100])
    wrong = torch.tensor([0, 1, -100, 2, 0])
    gradients, counts, losses = [], [], []
    for target in [labels, wrong, labels]:
        logits = torch.zeros(5, 3, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100)
        loss.backward()
        counts.append(int((target != -100).sum()))
        gradients.append((logits.grad.abs().sum(-1) > 0).tolist())
        losses.append(round(float(loss.detach()), 6))
    assert counts == [2, 4, 2] and gradients[0] == gradients[2]
    records['RT01_loss_mask'] = dict(active_targets=counts, gradient_positions=gradients,
                                    losses=losses, conclusion='O=0 gerçek hedeftir; -100 yok sayılır. Ortalama kayıp artmak zorunda değildir.')
    context = 'bilgi ' * 35 + 'Mert Sivas şehrine gitti .'
    question, answer = 'Mert nereye gitti ?', 'Sivas'
    a = context.index(answer); b = a + len(answer)
    def visible(features, i=0):
        offsets = features['offset_mapping'][i] if isinstance(features['input_ids'][0], list) else features['offset_mapping']
        return any(sid == 1 and x <= a and y >= b for sid, (x, y) in zip(features.sequence_ids(i), offsets))
    cropped = tok(question, context, max_length=16, truncation='only_second', return_offsets_mapping=True)
    windows = tok(question, context, max_length=16, stride=4, truncation='only_second',
                  return_overflowing_tokens=True, return_offsets_mapping=True)
    mask = [visible(windows, i) for i in range(len(windows['input_ids']))]
    assert not visible(cropped) and any(mask)
    records['RT02_qa_window'] = dict(max_length=16, stride=4, cropped_has_answer=visible(cropped),
                                    window_count=len(mask), answer_in_windows=mask,
                                    character_span=[a, b], conclusion='Gerçek fast tokenizer ile cevap ilk pencerede kayboldu, taşan pencerelerle bulundu; model doğruluğu ölçülmedi.')
    # Gerçek WordPiece bölünmesinde sözcük/token hizası.
    wp = Tokenizer(models.WordPiece(vocab={'[UNK]': 0, '[CLS]': 1, '[SEP]': 2,
                                          'Ankara': 3, '##lı': 4, 'Ayşe': 5}, unk_token='[UNK]'))
    wp.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    wp.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',
                          special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    enc = wp.encode(['Ayşe', 'Ankaralı'], is_pretokenized=True)
    word_labels, aligned, previous = [1, 2], [], None
    for wid in enc.word_ids:
        aligned.append(-100 if wid is None or wid == previous else word_labels[wid])
        previous = wid
    assert enc.tokens == ['[CLS]', 'Ayşe', 'Ankara', '##lı', '[SEP]']
    assert aligned == [-100, 1, 2, -100, -100]
    records['RT03_subword_alignment'] = dict(tokens=enc.tokens, word_ids=enc.word_ids,
                naive_labels=word_labels, naive_length=len(word_labels), token_length=len(enc.ids), corrected=aligned)
    bad, fixed = 'def double(x)\n    return x * 2', 'def double(x):\n    return x * 2'
    try: ast.parse(bad)
    except SyntaxError as error: syntax = type(error).__name__
    else: raise AssertionError('Bozuk kod ayrıştırıldı')
    ast.parse(fixed)
    records['RT04_generated_code'] = dict(broken=syntax, fixed='parse_ok', executed=False,
                                         note='İnsan yazımı kod; hiçbir üretilmiş kod yürütülmedi.')
    # Olumsuzluk silinince sözcük örtüşmesi yüksek kalabilir.
    reference = TEXTS[15]
    changed = TEXTS[16]
    common = set(reference.split()) & set(changed.split())
    overlap = len(common) / len(set(reference.split()))
    assert 'gönderme' not in changed and overlap > 0.7
    records['RT05_meaning'] = dict(reference=reference, changed=changed,
                    reference_word_overlap=round(overlap, 6), fixed=reference,
                    note='Basit küme örtüşmesi; BLEU/ROUGE değildir. İnsan yazımı karşı örnek anlam kontrolünün gereğini gösterir.')
    return records

TASKS = {'ner': ner, 'mlm': mlm, 'translation': seq2seq,
         'summarization': lambda t, s: seq2seq(t, s, True), 'causal': causal, 'qa': qa}

def run(task='all', steps=100):
    tok = setup()
    result = {'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                             'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                             'device': 'cpu', 'seed': SEED, 'threads': 2},
              'scope': 'İnsan yazımı çok küçük veri, rastgele başlatılmış gerçek Transformer modelleri. Eğitim örneği kontrolü, benchmark değildir.',
              'tasks': {}}
    for name, func in TASKS.items():
        if task in ['all', name]:
            torch.manual_seed(SEED)
            result['tasks'][name] = func(tok, steps)
    if task in ['all', 'red-team']: result['red_team'] = red_team(tok)
    return result

def notebook_run(task='all', steps=100):
    """Notebook ile terminalin aynı kodu çalıştırmasını sağlar."""
    result = run(task, steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['all', 'red-team', *TASKS], default='all')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1: parser.error('--steps en az 1 olmalı')
    result = run(args.task, args.steps)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')

if __name__ == '__main__': main()
