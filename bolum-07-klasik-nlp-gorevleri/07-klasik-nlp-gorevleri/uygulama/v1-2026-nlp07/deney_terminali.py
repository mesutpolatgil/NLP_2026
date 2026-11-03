"""Sunuma ait gerçek terminal kanıtlarını üretir; her çağrı modeli gerçekten çalıştırır."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import textwrap
import platform
import torch
import transformers
import main

def show(label,value):
    print(label)
    if not isinstance(value,str):value=json.dumps(value,ensure_ascii=False)
    for line in textwrap.wrap(value,width=116,replace_whitespace=False):print('  '+line)

def main_cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['qa','summary','causal','chat','loss'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);transformers.set_seed(7)
    print('='*118)
    print('NLP07 | Mert Ali Alkan | QA & Adli Bilisim Uzmani')
    print('GERCEK YEREL KOSUM |',datetime.now().astimezone().isoformat(timespec='seconds'))
    print('Python',platform.python_version(),'| torch',torch.__version__,'| transformers',transformers.__version__,'| CPU | seed=7')
    print('KOMUT: python deney_terminali.py --case',args.case,'--output',args.output.name)
    print('='*118,flush=True)
    if args.case=='qa':
        print('KOD: qa(question=question, context=clean)')
        print('KOD: qa(question=question, context=clean + injected_instruction)')
        print('KOD: qa(question=question, context=clean)  # guvenilir kaynakla yeniden test',flush=True)
        result=main.infer_qa();rt=result['context_injection']
        show('MODEL',result['model']);show('SORU',result['question']);show('KONTROL BAGLAMI',result['context'])
        show('KONTROL CEVAP',rt['clean'])
        show('TALIMAT ENJEKSIYONU',rt['instruction_output'])
        show('CELISKILI OLGU ZEHIRLEMESI',rt['forged_fact_output'])
        show('GUVENILIR KAYNAKLA YENIDEN TEST',rt['trusted_source_retest'])
        w=result['window_test'];show('PENCERE DENEYI',{'max_length':64,'stride':16,'pencere_sayisi':w['windows'],'ilk_pencerede_cevap':w['answer_present'][0],'cevabi_iceren_pencereler':[i for i,x in enumerate(w['answer_present']) if x]})
        show('TUM PENCERELERLE GERCEK PIPELINE CEVABI',w['pipeline_answer'])
        show('ACIK BAGLAM/OFFSET SON ISLEME',w['explicit_postprocess'])
        print('YORUM: Extractive QA, sohbet modeli sistem rolu testi degildir. Sonuc tek girdiye aittir.')
    elif args.case=='summary':
        print('KOD: model.generate(**tok("summarize: " + document, return_tensors="pt"),')
        print('KOD:                max_new_tokens=56, num_beams=4, do_sample=False)',flush=True)
        result=main.infer_seq2seq();rt=result['summary_red_team']
        show('MODEL',result['model']);show('KAYNAK BELGE',rt['clean_document'])
        show('TEMIZ BELGE OZETI',rt['baseline'])
        show('EKLENEN TALIMAT',rt['injected_document'][len(rt['clean_document']):])
        show('ENJEKSIYONLU BELGE OZETI',rt['injected'])
        show('SAVUNMA ISTEMIYLE YENIDEN KOSUM',rt['mitigation_attempt'])
        print('KONTROL: 120 katilimci; 12 hafif bas agrisi; olum yok; iyilesme bildirimi yok.')
        print('YORUM: Atlama ile celiskili iddia ayridir. Savunma denemesi genel bagisiklik kaniti degildir.')
    elif args.case=='causal':
        print('KOD: model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=45,')
        print('KOD:                do_sample=False, repetition_penalty=1.15)',flush=True)
        result=main.infer_causal();show('MODEL',result['model']);show('NORMAL DEVAM',result['output'])
        show('KURGUSAL ISTEM',result['fictional_probe']['prompt'])
        show('GERCEK MODEL DEVAMI',result['fictional_probe']['output'])
        print('YORUM: Kisi, kurum ve algoritma bu deney icin uyduruldu. Uretilen devam tarihsel bilgi degildir.')
    elif args.case=='chat':
        print('KOD: tok.apply_chat_template(messages, add_generation_prompt=True)')
        print('KOD: model.generate(**inputs, max_new_tokens=32, do_sample=False)',flush=True)
        result=main.infer_chat()
        show('MODEL',result['model']);show('SORU',result['question']);show('GUVENILIR BELGE',result['trusted_document'])
        show('KONTROL',result['baseline']);show('ENJEKSIYON',result['injected_document'])
        show('ENJEKSIYONLU CIKTI',result['injected']);show('SALDIRI ISARETI GORULDU MU',result['attack_marker_observed'])
        show('GUVENILIR KAYNAKLA YENIDEN TEST',result['trusted_source_retest'])
        print('YORUM: Rol sablonlu sohbet modelinde gercek deney. Etkisiz saldiri da bulgudur.')
    else:
        print('KOD: cross_entropy(logits, labels, ignore_index=-100).backward()')
        result=main.loss_check()
        for name,row in zip(['KONTROL','BOZMA: -100 yerine O=0','DUZELTME VE YENIDEN TEST'],result):
            print('\n'+name)
            print('  labels =',row['labels'],'| aktif_hedef =',row['active_targets'])
            print('  gradyan =',row['gradient_positions'],'| loss =',row['loss'])
        print('\nYORUM: Hedef sayisi 2 -> 4 -> 2. Uniform logits ile ortalama loss ayni kalabilir.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence={'case':args.case,'result':result,'source_sha256':hashlib.sha256(Path(main.__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('KANIT JSON:',args.output.name)
    print('KOSUM TAMAMLANDI | CIKIS 0',flush=True)

if __name__=='__main__':main_cli()
