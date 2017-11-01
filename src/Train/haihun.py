import re
import numpy as np
import sys
import time
import argparse
from datetime import datetime as dt
from gensim.models import word2vec

from src.Model import FFNN
from config import INPUT_PATH, OUTPUT_PATH
from TNLineSentences import TNLineSentencesST as LS

import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

'''
'-', '—'の読みを予測するFFNNを設計した。

ラベルの対応
|ラベル|0|1|
|読み|なし|to|

現状のHaihun学習済みモデルは、
OUTPUT_PATH + Haihun-20171012-172355.model


'''

n_dim = 128

def main():
    parser = argparse.ArgumentParser(description='Train haihun')
    parser.add_argument('--model-path', '-m', default=None,
                        help='model path')
    args = parser.parse_args()
    
    todaydetail = dt.today()
    todaydetailf = todaydetail.strftime("%Y%m%d-%H%M%S")
    print('start at ' + todaydetailf)
    
    #
    start = time.time()
    #
    
    embed = word2vec.Word2Vec.load(OUTPUT_PATH+'embeddings.model')
    
    #
    mid = time.time()
    #
    
    model = FFNN.FFNN(n_dim*2, 2)
    if not args.model_path:
        sentences = LS(INPUT_PATH+'dataset/')
        Train(model, embed, sentences)
    else:
        serializers.load_npz(args.model_path, model)
        Test(model, embed)
    enddetail = dt.today()
    enddetailf = enddetail.strftime("%Y%m%d-%H:%M:%S")
    print('end at ' + enddetailf)
    
    #
    end = time.time()
    #
    
    print("Number of sentences for training:\t"+str(model.n_sentences))
    print("Reading time:\t"+str(mid-start))
    print("Training time:\t"+str(end-mid))
    print("=>successfully finished!")
    
def getV(p, sentence, embed):
    v = None
    flg = False
    if 0<p<len(sentence)-1:
        pre = sentence[p-1]
        nex = sentence[p+1]
        if not (pre in embed.wv and nex in embed.wv):
            print(sentence, p, " not in vocabulary.")
        else:
            v_pre=embed.wv[pre]
            v_nex=embed.wv[nex]
            v = np.concatenate([v_pre, v_nex])
            v = F.broadcast(v.astype(np.float32))
            flg = True
    elif p==0:
        nex = sentence[p+1]
        if not nex in embed.wv:
            print(sentence, p, " not in vocabulary.")
        else:
            v_pre = np.zeros(n_dim, dtype=np.float32)
            v_nex=embed.wv[nex]
            v = np.concatenate([v_pre, v_nex])
            v = F.broadcast(v.astype(np.float32))
            flg = True
    elif p==len(sentence)-1:
        pre = sentence[p-1]
        if not pre in embed.wv:
            print(sentence, p, " not in vocabulary.")
        else:
            v_pre=embed.wv[pre]
            v_nex = np.zeros(n_dim, dtype=np.float32)
            v = np.concatenate([v_pre, v_nex])
            v = F.broadcast(v.astype(np.float32))
            flg = True
    return v, flg
    
def Train(model, embed, sentences):
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    source=[]
    target=[]
    #stopper=0
    for s in sentences:
        # ToDo
        # 1. -と—が両方入っている場合がある
        # 2. -が連続で2回以上続く場合がある
        if '-' in s[0] or '—' in s[0]:
            for p, word in enumerate(s[0]):
                if word == '-' or word == '—':
                    v, flg = getV(p, s[0], embed)
                    if flg:
                        source.append(v)
                        if s[1][p] == 'sil' or s[1][p] == '<self>':
                            target.append(0)
                        elif s[1][p] == 'to':
                            target.append(1)
                        else:
                            assert False, '入力値:{0}'.format(s[1][p])

                        if len(source)==50:
                            #Train
                            target = F.broadcast(np.array(target, dtype=np.int32))
                            batch = len(source)
                            source = F.reshape(F.concat(source, axis=0), (batch, n_dim * 2))
                            model.zerograds()
                            loss = model(source, target)
                            print("loss: %6f" % loss.data)
                            loss.backward()
                            optimizer.update()

                            source=[]
                            target=[]
    model_name = 'Haihun-' + todaydetailf+'.model'
    serializers.save_npz(OUTPUT_PATH + model_name, model)
    print('=>save the model: '+model_name)
    
def Test(model, embed):
    out = open(OUTPUT_PATH + 'output_haihun.csv', "w", encoding='UTF8')
    out.write('"id","after"\n')
    
    test = open(INPUT_PATH + "en_test.csv", encoding='UTF8')
    line = test.readline().strip()
    
    base = open(OUTPUT_PATH + 'baseline4_en.csv', "r", encoding='UTF8')
    line_b = base.readline().strip()
    
    sentenceID = '0'
    sentence = []
    sentence_b = []
    while 1:
        line = test.readline().strip()
        line_b = base.readline().strip()
        if line == '':
            break
        #test
        pos = line.find(',')
        i1 = line[:pos]
        line = line[pos + 1:]

        pos = line.find(',')
        i2 = line[:pos]
        line = line[pos + 1:]

        line = line[1:-1]
        
        #baseline
        pos_b = line_b.find(',')
        line_b = line_b[pos_b + 1:]

        pos_b = line_b.find(',')
        line_b = line_b[pos_b + 1:]

        line_b = line_b[1:-1]
        
        if line.isdigit():
            if len(line)==4:
                line = "<YEAR>"
            elif len(line)==3:
                line = "<3_DIGIT>"
            elif len(line)==2:
                line = "<2_DIGIT>"
        
        if i1 == sentenceID:
            sentence.append(line)
            sentence_b.append(line_b)
        else:
            for p, word in enumerate(sentence):
                if '-' == word or '—' == word:
                    if word == '-' or word == '—':
                        v, flg = getV(p, sentence, embed)
                        if flg:
                            source = F.reshape(v, (1, n_dim * 2))
                            pred = F.argmax(model.fwd(source))
                            print("pred:", pred.data)
                            if pred.data == 1:
                                sentence_b[p] = 'to'
                            else:
                                sentence_b[p] = word
                out.write('"' + sentenceID + '_' + str(p) + '",')
                out.write('"' + sentence_b[p] + '"')
                out.write('\n')
            
            sentence = [line]
            sentence_b = [line_b]
            sentenceID = i1
    for p, word in enumerate(sentence):
        if '-' == word or '—' == word:
            if word == '-' or word == '—':
                v, flg = getV(p, sentence, embed)
                if flg:
                    source = F.reshape(v, (1, n_dim * 2))
                    pred = F.argmax(model.fwd(source))
                    print("pred:", pred.data)
                    if pred.data == 1:
                        sentence_b[p] = 'to'
                    else:
                        sentence_b[p] = word
        out.write('"' + sentenceID + '_' + str(p) + '",')
        out.write('"' + sentence_b[p] + '"')
        out.write('\n')
    out.close()
    test.close()
    base.close()
    
if __name__=='__main__':
    main()