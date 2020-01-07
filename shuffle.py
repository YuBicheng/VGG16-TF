from random import shuffle

with open('train.txt', 'r') as f:
    train_list = f.readlines()
shuffle(train_list)
with open('shuffle.txt', 'w') as g:
    for l in train_list:
        g.write(l)
