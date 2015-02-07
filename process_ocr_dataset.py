import miscIO as mlio
import numpy as np
import os
from gzip import GzipFile as gfile

def load(dir_path,load_to_memory=False):
    """
    Loads the OCR letters dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    
    input_size=128
    targets = set(range(26))
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        tokens = line.split()
        seq_size = len(tokens)/(input_size+1)
        input = np.zeros((seq_size,input_size))
        target = -1*np.ones((seq_size),dtype=int)

        example = np.array([int(i) for i in tokens]).reshape((seq_size,-1))
        input[:] = example[:,:input_size]
        target[:] = example[:,input_size]
        return input,target

    train_file,valid_file,test_file = [os.path.join(dir_path, 'ocr_letters_' + ds + '.txt') for ds in ['train','valid','test']]
    # Get data
    
    ###miscIO is used here.
    ###
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [5502, 688, 687] 
    if load_to_memory:
        train = [ example for example in train ]
        valid = [ example for example in valid ]
        test = [ example for example in test ]
        
    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l,'targets':targets} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):

    print 'Splitting dataset into training/validation/test sets'
    file = gfile(os.path.join(dir_path,'letter.data.gz'))
    train_file,valid_file,test_file = [open(os.path.join(dir_path, 'ocr_letters_' + ds + '.txt'),'w') for ds in ['train','valid','test']]
    letters = 'abcdefghijklmnopqrstuvwxyz'

    all_data = []
    s = ''
    # Putting all data in memory
    for line in file:
        tokens = line.strip('\n').strip('\t').split('\t')            
        s += ' '.join(tokens[6:])
        target = letters.find(tokens[1])
        if target < 0:
            print 'Target ' + tokens[1] + ' not found!'
        s = s + ' ' + str(target)

        if int(tokens[2]) == -1: # new word starts next
            s = s + '\n'
            all_data += [s]
            s = ''
        else:
            s = s + ' '
            
    # Shuffle data
    import random
    random.seed(12345)
    perm = range(len(all_data))
    random.shuffle(perm)
    line_id = 0
    train_valid_split =  5502
    valid_test_split =  5502 + 688
    for i in perm:
        s = all_data[i]
        if line_id < train_valid_split:
            train_file.write(s)
        elif line_id < valid_test_split:
            valid_file.write(s)
        else:
            test_file.write(s)
        line_id += 1
    train_file.close()
    valid_file.close()
    test_file.close()
    print 'Done                     '
