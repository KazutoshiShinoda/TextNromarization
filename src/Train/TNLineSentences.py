from gensim.models import word2vec

class TNLineSentences(word2vec.PathLineSentences):
    def __iter__(self):
        """iterate through the files"""
        self.n_sentences = 0
        for file_name in self.input_files:
            print('Reading file %s', file_name)
            source = open(file_name, encoding='UTF8')
            line = source.readline()
            sentence = []
            while 1:
                line = source.readline().strip()
                if line == '':
                    break
                line = line.replace(',NA,', ',"NA",')
                pos = line.find('","')
                text = line[pos + 2:]
                if text[:3] == '","':
                    #カンマは無視
                    continue
                text = text[1:-1]
                arr = text.split('","')
                if arr[0].isdigit():
                    if len(arr[0])==4:
                        arr[0] = "<YEAR>"
                    elif len(arr[0])==3:
                        arr[0] = "<3_DIGIT>"
                    elif len(arr[0])==2:
                        arr[0] = "<2_DIGIT>"
                if arr[0] != '<eos>':
                    sentence.append(arr[0])
                else:
                    self.n_sentences += 1
                    #print('Ex.:', sentence)
                    if self.n_sentences % 1000000 == 0:
                        print('Proccessed %i sentences.' % self.n_sentences)
                    i=0
                    while i < len(sentence):
                        yield sentence[i:i + self.max_sentence_length]
                        i += self.max_sentence_length
                    sentence = []
                    
class TNLineSentencesST(word2vec.PathLineSentences):
    def __iter__(self):
        """iterate through the files"""
        self.n_sentences = 0
        for file_name in self.input_files:
            print('Reading file %s', file_name)
            f = open(file_name, encoding='UTF8')
            line = f.readline()
            sentence = []
            answer = []
            while 1:
                line = f.readline().strip()
                if line == '':
                    break
                line = line.replace(',NA,', ',"NA",')
                pos = line.find('","')
                text = line[pos + 2:]
                if text[:3] == '","':
                    #カンマは無視
                    continue
                text = text[1:-1]
                arr = text.split('","')
                if arr[0].isdigit():
                    if len(arr[0])==4:
                        arr[0] = "<YEAR>"
                    elif len(arr[0])==3:
                        arr[0] = "<3_DIGIT>"
                    elif len(arr[0])==2:
                        arr[0] = "<2_DIGIT>"
                if arr[0] != '<eos>':
                    sentence.append(arr[0])
                    answer.append(arr[1])
                else:
                    self.n_sentences += 1
                    #print('Ex.:', sentence)
                    if self.n_sentences % 1000000 == 0:
                        print('Proccessed %i sentences.' % self.n_sentences)
                    i=0
                    while i < len(sentence):
                        yield (sentence[i:i + self.max_sentence_length],
                               answer[i:i + self.max_sentence_length])
                        i += self.max_sentence_length
                    sentence = []
                    answer=[]