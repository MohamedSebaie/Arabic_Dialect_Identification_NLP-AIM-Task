
import re
import os
import glob
import time
import torch
import numpy
import random
import aranorm
import logging
import datetime
import joblib
import numpy as np
import pandas as pd
from numpy import pi
import torch.nn as nn
import preprocess_arabert
import pyarabic.araby as arb
from sklearn.utils import resample
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import f1_score,accuracy_score,recall_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertForSequenceClassification,BertTokenizer,AdamW,get_linear_schedule_with_warmup


keys_dictionary   = {0:"EG",1:"PL",2:"KW",3:"LY",4:"QA",5:"JO",6:"LB",7:"SA",8:"AE",9:"BH",
                     10:"OM",11:"SY",12:"DZ",13:"IQ",14:"SD",15:"MA",16:"YE",17:"TN"}

labels_dictionary = {"EG":0,"PL":1,"KW":2,"LY":3,"QA":4,"JO":5,"LB":6,"SA":7,"AE":8,"BH":9,
                     "OM":10,"SY":11,"DZ":12,"IQ":13,"SD":14,"MA":15,"YE":16,"TN":17}

keys_dictionary_full   = {0:"Egypt",1:"Palestine",2:"Kuwait",3:"Libya",4:"Qatar",5:"Jordan",6:"Lebanon",7:"Saudi Arabia",
						8:"United Arab Emirates",9:"Bahrain",10:"Oman",11:"Syria",12:"Algeria",13:"Iraq",14:"Sudan",15:"Morocco",
						16:"Yemen",17:"Tunisia"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# To control logging level for various modules used in the application:
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
#**************************************************************************************************************
def tweet_preprcessing(tweet):
    text_preprocessed = preprocess_arabert.preprocess(tweet, do_farasa_tokenization=True)
    preprocessed_tweet= aranorm.normalize_arabic_text(text_preprocessed)
    return preprocessed_tweet
#**************************************************************************************************************
#**************************************************************************************************************
def predictLinearSVC(tweet,path):
  print('Predicting dialect for tweet...')
  
  model= joblib.load(path)
  df = pd.DataFrame({'T':tweet})
  df['T'] = df['T'].apply(tweet_preprcessing)
  dialect = keys_dictionary_full.get(model.predict(df['T'])[0])
  return dialect
#**************************************************************************************************************
def flat_accuracy(logits, labels,device):
  pred_flat = np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
  labels_flat = labels.cpu().detach().numpy().flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)
#*****************************************************************************************************
#*****************************************************************************************************
def get_report(predictions,true_labels):
  pred = [item for sublist in predictions for item in sublist]

  true_label = [item for sublist in true_labels for item in sublist]

  prediction = []
  for i in range(len(pred)):
    prediction.append(np.argmax(pred[i], axis=0).flatten()[0])

  print(classification_report(true_label, prediction,target_names= list(labels_dictionary.keys())))
#**************************************************************************************************************
def get_preds(logits,device):
  preds = np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
  return preds.tolist()
#**************************************************************************************************************
def get_labels(labels,device):
  labels = labels.cpu().detach().numpy().flatten()
  return labels.tolist()
#**************************************************************************************************************
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
#**************************************************************************************************************
def bert_pre_processing(tweet,farasa=None):
  text_preprocessed = [preprocess_arabert.preprocess(txt, do_farasa_tokenization=True , farasa=farasa) for txt in tweet] 
  text_preprocessed=[aranorm.normalize_arabic_text(txt) for txt in text_preprocessed]
  return text_preprocessed
#**************************************************************************************************************
def read_csv(file_path,feature,label):
  
  '''
  read pre-porcessed file
  '''
  df=pd.read_csv(file_path)
  df.dropna(inplace=True)
  tweets = list(df[feature])
  labels = list(df[label])
  
  return (tweets,labels)
#**************************************************************************************************************
def create_bert_dataloader(train,valid=None,test_size=0.05,batch_size = 32,split_train=True,test=False):
  
  
  if split_train:
    tweets , labels = train
    train_tweets, valid_tweets,train_labels,  valid_labels = train_test_split(tweets, labels, test_size=test_size, random_state=42,stratify=labels)
  if test:
    train_tweets, train_labels = train
  else:
    train_tweets, train_labels = train
    valid_tweets , valid_labels =valid

  

  train_input,train_mask = ( [ input_ for input_,mask in train_tweets],
                            [mask for input_,mask in train_tweets]  )
  
  # return train_input,train_mask

  # transfrom lists to tensors
  # return train_input,train_mask,train_labels

  train_input,train_mask = [torch.cat(train_input, dim=0)
                            ,torch.cat(train_mask, dim=0) ]
  train_labels = torch.tensor(train_labels)

  train_dataset = TensorDataset(train_input,train_mask,train_labels)
  
  train_dataloader = DataLoader(
              train_dataset,  # The training samples.
              sampler = RandomSampler(train_dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )
  if not test:
    valid_input,valid_mask = ( [ input_ for input_,mask in valid_tweets],[mask for input_,mask in valid_tweets]  )
    valid_input,valid_mask = [torch.cat(valid_input, dim=0),torch.cat(valid_mask, dim=0) ]

    valid_labels           = torch.tensor(valid_labels)

    val_dataset            = TensorDataset(valid_input,valid_mask,valid_labels)

    validation_dataloader  = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
  
    return train_dataloader,validation_dataloader
  else:
    return train_dataloader
#**************************************************************************************************************
def Batchpredict(model,prediction_dataloader):
  # Prediction on test set

  print('Predicting labels for test sentences...')

  # Put model in evaluation mode
  model.eval()

  # Tracking variables 
  predictions , true_labels = [], []

  # Predict 
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs[0]
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().tolist()
    label_ids = b_labels.to('cpu').tolist()
    
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

  print('    DONE.')
  return predictions,true_labels
#**************************************************************************************************************

def predictDialect(tweet,path,modelSize="base"):
  print('Predicting dialect for tweet...')
  tweet=tweet_preprcessing(tweet)
  print(tweet)
  
  if modelSize=="base":
    BERTmodel_class  = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter",
                                                                     num_labels = 18,
                                                                     output_attentions = False, 
                                                                     output_hidden_states = False)
  else:
    BERTmodel_class  = BertForSequenceClassification.from_pretrained("aubmindlab/bert-large-arabertv02-twitter",num_labels = 18)

  BERTmodel_class.load_state_dict(torch.load(path))
  BERTmodel_class.eval()

  tokenizer= AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")

  encoded_dict = tokenizer.encode_plus(
                            tweet,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                      )

  input_ids=encoded_dict['input_ids']
  attention_mask=encoded_dict['attention_mask']

  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = BERTmodel_class(input_ids, token_type_ids=None, 
                      attention_mask=attention_mask)

  logits = outputs[0]
    
    # Move logits and labels to CPU
  logits = logits.detach().cpu().tolist()
  prediction = []
  for i in range(len(logits)):
    prediction.append(np.argmax(logits[i], axis=0).flatten()[0])
  print("is from:")
  return keys_dictionary_full.get(prediction[0])
#**************************************************************************************************************
def get_model(bertModel,path=None,freeze_bert=None,embedding=None):
  model = get_bert_classifier(bertModel,path)
  if freeze_bert:
    for param in model.bert.parameters():
      param.requires_grad = False
  return model
#**************************************************************************************************************
def get_bert_classifier(bertModel,path=None):
  # if no saved model in path create a new one
  if path==None:
    return BertForSequenceClassification.from_pretrained(
        bertModel, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 18, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )
  else:
    print("using fine tuned model ")
    return BertForSequenceClassification.from_pretrained(
        path, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 18, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )
#**************************************************************************************************************
def printModel_Parameters(model):  
  # Get all of the model's parameters as a list of tuples.
  params = list(model.named_parameters())

  print('The BERT model has {:} different named parameters.\n'.format(len(params)))

  print('==== Embedding Layer ====\n')

  for p in params[0:5]:
      print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

  print('\n==== First Transformer ====\n')

  for p in params[5:21]:
      print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

  print('\n==== Output Layer ====\n')

  for p in params[-4:]:
      print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#**************************************************************************************************************
def run_model(model,data_loader,train=False,optimizer=None,
              scheduler=None,device="cuda", loss_func=None):

  if train:
    model.train()
  else :
    model.eval()

  # Reset the total loss for this epoch.

  total_loss = 0
  total_accuracy=0
  total_f1 =0
  t0 = time.time()
  all_preds =[]
  all_labels =[]
  for step, batch in enumerate(data_loader):

          # Progress update every 40 batches.
          if step % 250 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
              
              # Report progress.
              if train:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(data_loader), elapsed))

          # Unpack this training batch from our dataloader. 

          # `batch` contains three pytorch tensors:
          #   [0]: input ids 
          #   [1]: attention masks
          #   [2]: labels 
          input_ids = batch[0].to(device)
          input_mask = batch[1].to(device)
          labels = batch[2].to(device)

          model.zero_grad()        

          
          outputs = model(input_ids, 
                              token_type_ids=None, 
                              attention_mask=input_mask, 
                              labels=labels)
         
          model_loss= outputs[0]
          logits= outputs[1]
          

          total_accuracy += flat_accuracy(logits, labels,device)
          
          
          all_preds += get_preds(logits,device)
          all_labels += get_labels(labels,device)
          if loss_func:
            loss=loss_func(logits.view(-1,21) 
                              ,labels.view(-1))
            total_loss +=loss 
            loss.backward()
            
          else:
            total_loss += model_loss  
            model_loss.backward()      
          
          


          # Perform a backward pass to calculate the gradients.
          

          # Clip the norm of the gradients to 1.0.
          # This is to help prevent the "exploding gradients" problem.
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

          if train:
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

  avg_loss = total_loss / len(data_loader)            
  avg_acc = total_accuracy / len(data_loader)
  avg_f1 = f1_score(all_labels,all_preds,average='macro')
  recall = recall_score(all_labels,all_preds,average='macro')

  return avg_loss,avg_acc,avg_f1,recall,all_preds,all_labels
#**************************************************************************************************************
################################################################################
def save_model(model,model_name,model_path,f1_score,accuracy):
  parent_dir=os.getcwd()
  os.chdir(model_path)
  folder_name = model_name
  all_files = os.listdir()
  if folder_name not in all_files:
    os.mkdir(str(folder_name))
  os.chdir(folder_name)
  torch.save(model,"best_validation "+str(f1_score))
  os.chdir(parent_dir)
################################################################################
def train(train_loader,valid_loader, epochs=4,learning_rate=2e-5,regularization = 0.01,eps=1e-8,
          model=None,device="cuda",loss_weights =None,loss_func=None ,save_path="save models",model_name=None):
  
  format_time(time.time()-time.time())
  
  
  model.to(device)
  
  optimizer = AdamW(model.parameters(),
                    lr = learning_rate, 
                    eps = eps)
  

  total_steps = len(train_loader) * epochs

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)



  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # We'll store a number of quantities such as training and validation loss, 
  # validation accuracy, and timings.
  training_stats = []

  # Measure the total training time for the whole run.
  total_t0 = time.time()

  # if loss func not specified use model 's own loss
  if loss_func == "weighted_CrossEntropy":
    print("using weighted_CrossEntropy loss")
    # loss_func=nn.CrossEntropyLoss(weight=loss_weights,size_average=False)
    loss_func=nn.CrossEntropyLoss()
  
  # best_valid_f1 = 0
  # best_valid_acc = 0
  # best_valid_preds =[]
  # For each epoch...
  for epoch_i in range(epochs):
      
      # ========================================
      #               Training
      # ========================================
      
      # Perform one full pass over the training set.

      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      # print('Training...')
      model.to(device)

      # Measure how long the training epoch takes.
      t0 = time.time()

      training_loss,training_acc,training_f1,training_recall,training_preds,training_labels=run_model(model,train_loader,True,optimizer,scheduler,device=device,loss_func=loss_func)
      
      training_time = format_time(time.time() - t0)
      
      print("  Average training loss: {:.6f}".format(training_loss))
      print("  Average training accuracy: {0:.4f}".format(training_acc))
      print("  Average training f1: {0:.4f}".format(training_f1))
      print("  Average training recall: {0:.4f}".format(training_recall))
      # print("  Total Training Time: {0:.4f}".format(training_time))
      print("-"*50)

      

      
      
      valid_loss,valid_acc,valid_f1,valid_recall,valid_preds,valid_labels = run_model(model,valid_loader,device=device,loss_func=loss_func)
      
      # best_valid_f1=valid_f1

      # if valid_acc > best_valid_acc :
      #   best_valid_acc=valid_acc
      #   best_valid_preds =valid_preds 
      #   path = save_path
      #   torch.save(model.cpu().state_dict(), path+'/model'+str(epoch_i + 1)+"_"+str(valid_acc)+".pth") # saving model
      #   model.cuda()

     

      print("  Average validation loss: {0:.4f}".format(valid_loss))
      print("  Average validation accuracy: {0:.4f}".format(valid_acc))
      print("  Average validation f1: {0:.4f}".format(valid_f1))
      print("  Average validation recall: {0:.4f}".format(valid_recall))
      # print("  Total Validation Time: {0:.4f}".format(validation_time))
      print("-"*50)
  #     training_stats.append(
  #       {
  #           'epoch': epoch_i + 1,
  #           'Training Loss': training_loss,
  #           'Valid. Loss': valid_loss,
  #           'Valid. Accur.': valid_acc,
  #           'Training Time': training_time,
  #           'Validation Time': validation_time
  #       }
  #   )
  # return training_stats           
#**************************************************************************************************************
# creating tokenizer class uses bert by default
class Tokenizer():
  def __init__(self,tokenizer):
    self.tokenizer = tokenizer
  def bert_tokenize_tweet(self,sent):
    encoded_dict = self.tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )
    
    return encoded_dict['input_ids'],encoded_dict['attention_mask']
    

  

  def bert_tokenize_data(self,text,labels):

    input_ids = []
    attention_masks = []

    for tweet in text:
      input_id,atten_mask=self.bert_tokenize_tweet(tweet)
      input_ids.append(input_id)
      attention_masks.append(atten_mask)

    tweets = list(zip(input_ids,attention_masks))

    data = (tweets,labels)
    

    return data
#**************************************************************************************************************