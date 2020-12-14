from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import corpus_chrf
s1 = [['I', 'am', 'a', 'boy', 'dressed', 'in', 'white', 'shirt', 'black', 'shoe'], ['boy', 'with', 'black', 'hair', 'suit']]
s2 =[['I', 'am', 'boy', 'in', 'black', 'suit', 'grey', 'hair']]
ch_s1 = s1[0]
ch_s2 = ' '.join(s2[0])
ch_st_s1 = [' '.join(i) for i in s1]
print(corpus_bleu([s1],s2,weights = (1,0,0,0)))
print(corpus_chrf([ch_st_s1],[ch_s2]))