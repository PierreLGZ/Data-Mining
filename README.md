# Data Mining

This project focuses on learning in an unbalanced data context with a specific application on fraud detection.

## About the data

The data on which you are going to work are real data, they come from a retailer as well as from some banking organizations (FNCI and Banque de France). Each line represents a transaction carried out by check in a store of the sign somewhere in France, they are not raw and several variables are already created variables, i.e. are resulting from the feature engineering, we have a set of 23 variables which have the following meaning:  

- ZIBZIN: identifier relative to the person, i.e. it is his bank identifier (relative to the checkbook in use). 
- IDAvisAuthorisAtionCheque: identifier of the current transaction. 
- Amount: amount of the transaction. 
- DateTransaction: date of the transaction. 
- CodeDecision : this is a variable which can take 4 values :  
    0 : the transaction has been accepted by the store.  
    1 : the transaction and therefore the customer is part of a white list (good payers). You will not find any in this database.  
    2 : the customer is part of a black list, his history indicates that he is a bad payer (outstanding payments or banking incidents in progress), his transaction is then automatically refused.  
    3 : customer who was stopped by the system in the past for a more or less justified reason.  
- CheckCPT1 : number of transactions made by the same bank identifier during the same day. 
- CheckCPT2: number of transactions made by the same bank identifier during the last three days. 
- CPT3Check: number of transactions carried out by the same bank identifier over the last seven days. 
- D2CB: duration of knowledge of the customer (by his bank identifier), in days. For legal reasons, this knowledge period cannot exceed two years. 
- ScoringFP1: score of abnormality of the basket relative to a first family of products (e.g.: foodstuffs). 
- ScoringFP2 : score of abnormality of the basket relative to a second family of products (ex : electronics). 
- ScoringFP3 : score of abnormality of the basket relative to a third family of products (ex : others). 
- RateImpNb_RB : unpaid rates recorded according to the region where the transaction takes place. 
- RateImpNB_CPM : rate of unpaid items relative to the store where the transaction took place. 
- ChequeNumberDiscrepancy: difference between the cheque numbers. 
- NbrMagasin3D: number of different stores visited in the last 3 days. 
- DiffDateTr1 : difference (in days) to the previous transaction. 
- DiffDateTr2: difference (in days) to the second last transaction. 
- DiffDateTr3 : difference (in days) to the last but one transaction. 
- CA3TRetMtt : amount of the last transactions + amount of the current transaction
current transaction. 
- CA3TR: amount of the last three transactions. 
- Time: time of the transaction. 
- FlagImpaye : acceptance (0) or refusal of the transaction (1). 
