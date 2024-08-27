import numpy as np
import pandas as pd

class BiGram:
    def __init__(self, word_id_path='word_id.csv', CMatrix_path='CMatrix.csv', add=0, k=1):
        # dictionary lecture
        data = pd.read_csv(word_id_path)
        self.unique_words = data['word'].values
        ids = data['id'].values
        self.word_id = {} # empty dictionary
        for i in range(len(self.unique_words)):
            self.word_id[str(self.unique_words[i])] = ids[i]

        # CMatrix lecture
        data = pd.read_csv(CMatrix_path)
        counts = data['counts'].values
        self.Cmatrix = np.zeros(shape=(len(counts),len(counts)), dtype=np.int32)
        for i in range(len(counts)):
            if isinstance(counts[i], str):
                #print(i,counts[i])
                word_ci = counts[i].split(',')
                for wi in word_ci: 
                    ind = int(wi.split(':')[0])
                    auxc = int(wi.split(':')[1])
                    self.Cmatrix[i,ind] = auxc
        
        # scalate the bigram counts
        self.Cmatrix = k*self.Cmatrix

        # add to smote probabilities
        self.Cmatrix += add
    
    def next_word(self, word):
        # get a random index considering the maximum count
        wi = self.word_id[word] # word id
        counts = self.Cmatrix[wi]
        auxi = np.random.randint(1,np.sum(counts)+1)

        # iterates over the possible range until the index selection
        c = 0 # counter
        i = 0 # iterator
        while c < auxi:
            c += counts[i]
            i += 1
        i = i -1 # the right index belongs to the previous range

        return self.unique_words[i]