import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PAD, BOS, EOS = 1, 2, 3

def base_tokenizer(text):
    '''
    a simple tokenizer to split on space and converts the sentence to list of words
    '''

    return [tok.strip() for tok in text.split(' ')]

def numericalize(text, stoi, device):
    '''
    convert the list of words to a list of corresponding indexes
    '''   
    #check text type
    if isinstance(text, list):
        pass
    else:
        ValueError("Text must be list of tokens")

    numericalized_text = []

    for token in text:
        if token in stoi.keys():
            numericalized_text.append(stoi[token])
        else:
            #out-of-vocab words are represented by UNK token index
            numericalized_text.append(stoi['<UNK>'])
            
    return torch.tensor(numericalized_text).to(device)

def padding_batch(batch, device='cpu'):
    '''
    padding input sentences when inference
    '''
    if isinstance(batch, list):
        len_list = torch.tensor([len(s) for s in batch]).to(device)
        # max_len = max(len_list)
        # pad_sentences = torch.tensor([s + [pad] * (max_len - len(s)) if len(s) < max_len else s for s in batch]).to(device)
        pad_sentences = pad_sequence(batch,batch_first=True, padding_value = PAD)        
        return (pad_sentences, len_list)
    else:
        raise TypeError('Check input data type')

#######################################################
#               Define Vocabulary Class
#######################################################

class Vocabulary:
  
    '''
    __init__ method is called by default as soon as an object of this class is initiated
    we use this method to initiate our vocab dictionaries
    '''
    def __init__(self, sentence_list, freq_threshold=5, max_vocab=32000, tgt=False):
        '''
        freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
        max_size : max source vocab size. Eg. if set to 10,000, we pick the top 10,000 most frequent words and discard others
        '''
        #initiate the index to token dict
        ## <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        ## <SOS> -> start token, added in front of each sentence to signify the start of sentence
        ## <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        ## <UNK> -> words which are not found in the vocab are replace by this token
        
        if tgt == True:
            print('Target vocab includes BOS and EOS token')
            self.itos = {0: '<UNK>', 1:'<PAD>', 2:'<BOS>', 3: '<EOS>'}
            self.idx = 4 #index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk
        else:
            self.itos = {0: '<UNK>', 1:'<PAD>'}
            self.idx = 2
        #initiate the token to index dict
        self.stoi = {k:j for j,k in self.itos.items()} 
        
        self.freq_threshold = freq_threshold
        self.max_size = max_vocab
        
        if isinstance(sentence_list, list):
            self.sentence_bucket = sentence_list
            self.build_vocab()
        else:
            raise TypeError("sentence bucket's type shoulde be list")
    
    '''
    __len__ is used by dataloader later to create batches
    '''
    def __len__(self):
        return len(self.itos)

    '''
    build the vocab: create a dictionary mapping of index to string (itos) and string to index (stoi)
    output ex. for stoi -> {'the':5, 'a':6, 'an':7}
    '''
    def build_vocab(self):
        #calculate the frequencies of each word first to remove the words with freq < freq_threshold
        frequencies = {}  #init the freq dict
        
        #calculate freq of words
        for sentence in self.sentence_bucket:
            for word in base_tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                    
                    
        #limit vocab by removing low freq words
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold} 
        
        #limit vocab to the max_size specified
        frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-self.idx]) # idx =4 for pad, start, end , unk
            
        #create vocab
        for word in frequencies.keys():
            self.stoi[word] = self.idx
            self.itos[self.idx] = word
            self.idx+=1
    
class NmtDataset(Dataset):
    '''Create a TranslationDataset given tokenized&bpe corpus.
    Initiating Variables
    path: Common prefix of paths to the data files for both languages.
    exts: A tuple containing the extension to path for each language.
    transform : If we want to add any augmentation
    freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
    max_length : max sequence length(dropping if src&trg len > max_length)
    max_vocab : max vocab size
    '''
    
    def __init__(self, nmt_dataset, src_vocab, tgt_vocab, transform=None):
        
        
        self.transform = transform      
        
        self.source_texts = nmt_dataset['src']
        self.target_texts = nmt_dataset['tgt']
        
        self.src_vocaburary = src_vocab
        self.tgt_vocaburary = tgt_vocab
        # print(self.tgt_vocaburary.stoi)

        
    def __len__(self):
        return len(self.source_texts)
    
    
    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index and 
        return its numericalize source and target values using the vocabulary objects we created in __init__
        '''
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]
        
        if self.transform is not None:
            source_text = self.transform(source_text)
            
        #numericalize texts ['<BOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [2,12,2,9,24,3] if tgt
        numerialized_source = []
        numerialized_source += numericalize(source_text, self.src_vocaburary.stoi)
    
        numerialized_target = [self.tgt_vocaburary.stoi["<BOS>"]]
        numerialized_target += numericalize(target_text, self.tgt_vocaburary.stoi)
        numerialized_target.append(self.tgt_vocaburary.stoi["<EOS>"])
        
        #convert the list to tensor and return
        return [torch.tensor(numerialized_source), torch.tensor(numerialized_target)]

#######################################################
#               Collate fn
#######################################################

class MyCollate:
    '''
    class to add padding to the batches
    collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class
    is used on single example
    '''
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch]
        src_len = torch.tensor([len(item[0]) for item in batch])
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=True, padding_value = self.pad_idx) 

        #get all target indexed sentences of the batch
        target = [item[1] for item in batch]
        trg_len = torch.tensor([len(item[1]) for item in batch])
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=True, padding_value = self.pad_idx)
        return [[source, src_len], [target, trg_len]]
    
    

#######################################################
#            Define Dataloader Functions
#######################################################
class NmtDataLoader():
    def __init__(self, train_path=None, valid_path=None, exts=('en', 'ko'), batch_size=128, max_length=255,
                 freq_threshold=5, max_vocab=32000, shared_vocab=False, 
                 num_workers=4, shuffle=True, pin_memory=True, device='-1'):

        if len(device.split(',')) > 1 and len(device) >= 1 :
            num_workers = len(device.split(',')) * num_workers
        else:
            pass
        print(f'Number of workers : {num_workers}')
        print(f'pin memory(data transfer speed is more faster) : {pin_memory}')
        
        if train_path is not None and valid_path is not None and exts is not None:
            try:
                self.train_set = self._get_corpus(train_path, exts, max_length)
                self.valid_set = self._get_corpus(valid_path, exts, max_length)
            except Exception as ex:
                print(ex)
                raise ValueError("Please check train&valid path, and extension tuple of src&tgt") 

            ##VOCAB class has been created above
            #Initialize source vocab object and build vocabulary
            if shared_vocab:
                self.src_vocab = Vocabulary(self.train_set['src'] + self.valid_set['src'], freq_threshold, max_vocab, tgt=False)
                self.tgt_vocab = Vocabulary(self.train_set['tgt'] + self.valid_set['tgt'], freq_threshold, max_vocab, tgt=True)
            else:
                self.src_vocab = Vocabulary(self.train_set['src'], freq_threshold, max_vocab, tgt=False)
                self.tgt_vocab = Vocabulary(self.train_set['tgt'], freq_threshold, max_vocab, tgt=True)

            print(f'SRC Vocab Size : {len(list(self.src_vocab.itos))}')
            print(f'TGT Vocab Size : {len(list(self.tgt_vocab.itos))}')
            self.train_dataset = NmtDataset(self.train_set, self.src_vocab, self.tgt_vocab)
            self.valid_dataset = NmtDataset(self.valid_set, self.src_vocab, self.tgt_vocab)
            
            # If we run a next(iter(data_loader)) we get an output of batch_size * (num_workers+1)
            # self.train_iter = self.get_train_loader(self.train_dataset, batch_size, num_workers, shuffle, pin_memory)
            self.train_iter = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, 
                                         num_workers=num_workers, pin_memory=pin_memory, #increase num_workers according to CPU 
                                         collate_fn=MyCollate(pad_idx=self.src_vocab.stoi['<PAD>'])) ##get pad_idx for collate fn, MyCollate class runs __call__ method by default
            self.valid_iter = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle, 
                                         num_workers=num_workers, pin_memory=pin_memory, 
                                         collate_fn=MyCollate(pad_idx=self.src_vocab.stoi['<PAD>']))     
        else:
            pass
    
    def _get_corpus(self, path, exts, max_length):
        #get source and target texts
        if not path.endswith('.'):
            path += '.'
            
        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        pair_corpus = {'src' : [], 'tgt' : []}
        
        num_lines = sum(1 for line in open(src_path,'r'))
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in tqdm(zip(src_file, trg_file), desc='Loading data...#', total=num_lines):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
                    continue
                if src_line != '' and trg_line != '':
                    pair_corpus['src'].append(src_line)
                    pair_corpus['tgt'].append(trg_line)

        return pair_corpus
    
    def load_vocab(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

if __name__ == '__main__':
    
    loader = NmtDataLoader(train_path='/home/user/transformer-nmt/data/sample.train', 
                           valid_path='/home/user/transformer-nmt/data/sample.valid',
                           exts=('en', 'ko'),
                           batch_size=128,
                           max_length=255)

    for batch in loader.train_iter:
        break
    print(batch[0])