#Import modules

import spacy
from tqdm.auto import tqdm
from spacy.tokens import DocBin
from ml_datasets import imdb
import matplotlib.pyplot as plt

################ Loading model and IMDB dataset ##############################################################################################

# load a medium sized english language model in spacy
nlp = spacy.load('en_core_web_md')
print(spacy.__version__)
print(nlp.pipe_names)

# load movie reviews as a tuple (text, label)
train_data, valid_data = imdb()
print(train_data[:2])
print(valid_data[:2])

########################## Functions #########################################################################################################

def make_docs(data):
    """
      this will take a list of texts and labels 
      and transform them in spacy documents
      
      data: list(tuple(text, label))
      
      returns: List(spacy.Doc.doc)  
    """
    
    docs = []
    # nlp.pipe([texts]) is way faster than running 
    # nlp(text) for each text
    # as_tuples allows us to pass in a tuple, 
    # the first one is treated as text
    # the second one will get returned as it is.
    
    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):
        
        # we need to set the (text)cat(egory) for each document
        if label == 'pos':
          doc.cats["categories"] = 1
        else:
          doc.cats["categories"] = 0
        
        # put them into a nice list
        docs.append(doc)
    
    return docs

def score(data,nlp):
  '''
      This function takes the labeled arguments and nlp objects and returns thresolds, precion, recall and all scores
      Args : 
            data - (list (text,label))
            nlp - (nlp object)
      Returns :
            thresolds - list of thresolds in range 0 to 1, stepsize 0.1
            precison - list of precisions for different thresolds
            recall - list of recalls for different thresolds
            all_result - list of dcitionaries containing tn,fp,fn,tp for different thresolds

  '''
  thresolds = [x / 10.0 for x in range(0, 11, 1)]
  precision = []
  recall = []
  result = {}
  all_result = []
  predicted_label = []
  true_label = []
  fpr = []
  for thresold in tqdm(thresolds, total = len(thresolds)):
    for doc, label in nlp.pipe(data, as_tuples=True):
      if doc.cats['categories']>thresold:
        predicted_label.append('pos')
        true_label.append(label)
      else:
        predicted_label.append('neg')
        true_label.append(label)
    
      for y_true,y_pred in list(zip(true_label,predicted_label)):
        if y_true == 'neg' and y_pred == 'neg':
          if 'tn' in result.keys():
            result['tn'] += 1
          else:
            result['tn'] = 1
        if y_true == 'neg' and y_pred == 'pos':
          if 'fp' in result.keys():
            result['fp'] += 1
          else:
            result['fp'] = 1
        if y_true == 'pos' and y_pred == 'neg':
          if 'fn' in result.keys():
            result['fn'] += 1
          else:
            result['fn'] = 1
        if y_true == 'pos' and y_pred == 'pos':
          if 'tp' in result.keys():
            result['tp'] += 1
          else:
            result['tp'] = 1
    all_result.append(result)

    if 'fp' not in result.keys():
      precision.append(1)
    else:
      precision.append(result['tp']/(result['tp']+result['fp']))
      
    if 'fn' not in result.keys():
      recall.append(1)
    else:
      recall.append(result['tp']/(result['tp']+result['fn']))

    if 'fp' not in result.keys():
      fpr.append(0)
    elif 'tn' not in result.keys():
      fpr.append(1)
    else:
      fpr.append(result['fp']/(result['fp']+result['tn']))
      
  return thresolds,precision,recall,fpr,all_result

############## Preparing train and validation data #####################################################################################################

num_training_texts = 10000

train_docs = make_docs(train_data[:num_training_texts])
print(train_docs[:2])

#save training data into spacyV3 format
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("/content/data/train.spacy")

# preparing validation data
num_validation_text = 2500
valid_docs = make_docs(valid_data[:num_validation_text])
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("./data/valid.spacy")

total_positives_training = len([train_docs[i].cats['categories'] for i in range(len(train_docs)) if train_docs[i].cats['categories'] == 1])
total_negatives_training = len([train_docs[i].cats['categories'] for i in range(len(train_docs)) if train_docs[i].cats['categories'] == 0])
total_postives_valid = len([valid_docs[i].cats['categories'] for i in range(len(valid_docs)) if valid_docs[i].cats['categories'] == 1])
total_negatives_valid = len([valid_docs[i].cats['categories'] for i in range(len(valid_docs)) if valid_docs[i].cats['categories'] == 0])
print("Total positives in training set :",total_positives_training)
print("Total negatives in training set :",total_negatives_training)
print("Total positives in validation set :",total_postives_valid)
print("Total negatives in validation set :",total_negatives_valid)

######################## Evaluating the model ####################################################################################################

nlp =  spacy.load("./output/model-best")
print(nlp.pipe_names)

num_validation_text = 1250
data = valid_data[:num_validation_text]
thresolds,precision,recall,fpr,scores= score(data,nlp)

test_docs = make_docs(data)
test_postives_valid = len([test_docs[i].cats['categories'] for i in range(len(test_docs)) if test_docs[i].cats['categories'] == 1])
test_negatives_valid = len([test_docs[i].cats['categories'] for i in range(len(test_docs)) if test_docs[i].cats['categories'] == 0])
test_postives_valid,test_negatives_valid

plt.figure(figsize =(8,4))
plt.subplot(1,2,1)
plt.plot(thresolds,precision)
plt.ylabel('Precision')
plt.subplot(1,2,2)
plt.plot(thresolds,recall)
plt.ylabel('Recall')
plt.tight_layout()

plt.figure(figsize = (8,4))
plt.plot(fpr,recall)
plt.xlabel('FPR')
plt.ylabel('TPR/Recall')