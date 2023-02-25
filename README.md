## Our approach

We need to understand what causes a transaction to be 1. be flagged as fraudulent 'isFlaggedFraud' and 2. what are the characteristics of fraudulent transactions 'isFraud'

First step is to understand the different columns in our dataset as well as the the types of transactions.


## Transactions types

CASH-IN is the process of increasing the balance of account by paying in cash to a merchant.

CASH-OUT is the opposite process of CASH-IN, it means to withdraw cash from a merchant which decreases the balance of the (merchant)account.

DEBIT is similar process than CASH-OUT and involves sending the money from the mobile money service
to a bank account.

PAYMENT is the process of paying for goods or services to merchants which decreases the balance of the account and increases the balance of the receiver.

TRANSFER is the process of sending money to another user of the service through the mobile money platform.

## Type of accounts

Customer account (C)
Merchant account (M)

## Features

'oldbalanceOrg' is the amount on the originator account before the transaction

'newbalanceOrig' is the amount on the originator account after the transaction.

'nameOrig' is the name of the originator account (customer who started the transaction).

'nameDest' is the name of the destinator account (customer who is the recipient of the transaction).

'oldbalanceDest' is the amount on the destinator account before the transaction.

'newbalanceDest' is the amount on the destinator account after the transaction.

'amount' is the amount of the transaction in the local currency.

'step' maps a unit of time in the real world. In this dataset, 1 step is 1 hour of time. Total steps 743 (30 days simulation)

'isFlaggedFraud' are transactions that were flagged as fraud by the rule-based algorithm but that are yet to be reviewed and confirmed as fraudulent by the investigator.

'isFraud' are transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system. 'isFraud' transactions were flagged as fraudulent by the rule-based system and were then confirmed as fraudulent by the investigator.
