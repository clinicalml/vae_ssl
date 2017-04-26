import numpy as np

class DataLoader:

    def __init__(self,data,batchsize=1,shuffle=True):
        self.data = data 
        self.N = len(self.data)
        self.batchsize = batchsize
        self.__i = 0
        self.__idx = np.arange(self.N)
        self.shuffle = shuffle

    def reset(self):
        self.__i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(self.N/float(self.batchsize)))

    def next(self):
        
        if self.__i == 0 and self.shuffle:
            self.__idx = np.random.permutation(self.__idx)

        if self.__i >= self.N:
            self.reset()
            raise StopIteration

        idx = self.__idx[self.__i:min(self.N,self.__i+self.batchsize)]
        minibatch = self.data[idx]
        self.__i += self.batchsize
        return minibatch

if __name__ == '__main__':
    data = np.arange(10)
    dataloader = DataLoader(data,batchsize=3,shuffle=False)
    print 'not shuffled'
    print "length = ",len(dataloader)
    for i,d in enumerate(dataloader):
        print 'batch %s:' % i,d
    dataloader = DataLoader(data,batchsize=3,shuffle=True)
    print 'shuffled'
    print "length = ",len(dataloader)
    for i,d in enumerate(dataloader):
        print 'batch %s:' % i,d
