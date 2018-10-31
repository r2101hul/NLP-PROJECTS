# Natural Language Processing using NLTK

# Install NLTK - pip install nltk

# Tokenization of paragraphs/sentences
import nltk
import re
import heapq
import numpy as np

paragraph = """Banks are sending messages to every account holder asking them to upgrade their debit
     and credit cards. You too must have got one such message. But there are chances that you might have 
     ignored the message taking it for a spam. Check the message once again, it is not a spam. This one
     is a useful message from the bank you have deposited your money with. Now,  you need to replace your
     existing debit and credit cards with new ones, if you have not already done.But why do you need to
     take such a pain and replace the debit and credit cards? Legally speaking, the Reserve Bank of India
     (RBI) has directed the banks to do so. The older debit and credit cards will become useless after
     December 31.
    
    Banks have to follow the directions of the RBI. This directive from the RBI was necessitated
     in order protect you from some unscrupulous online predator. Your money must be secured with
     the banks. It is their responsibility. Debit and credit card piracy has been a major issue, 
     as has been found in online banking fraud cases. The new chip-based cards have been prescribed to 
     keep your money and transaction safe.The existing debit and credit cards are magnetic stripe-only 
     cards. Their cloning has become a major challenge to those responsible for safe monetary transaction. 
     The new cards are EMV chip-based.  EVM stands for Europay, Mastercard, Visa. The old magnetic
     stripe-only debit and credit cards are to be replaced by EMV chip-based ones by 31 December 2018.
    Replacement of existing debit and credit cards with the new EMV chip-based cards is free of cost. 
    Banks bear the cost for new cards. The EMV chip-based debit and credit cards are in use since January
     2016.The RBI had made it mandatory for banks to issue only EMV chip-based cards to new customers
     opening new accounts  or applying for new debit or credit cards after 31 January 2016."""
               
               
# Tokenize sentences the sentences
dataset = nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W',' ',dataset[i])
    dataset[i] = re.sub(r'\s+',' ',dataset[i])


# Creating word histogram
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# Select best 100 features
freq_words = heapq.nlargest(100,word2count,key=word2count.get)

# Converting sentences to vectors
X = []
for data in dataset:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
        
X = np.asarray(X)