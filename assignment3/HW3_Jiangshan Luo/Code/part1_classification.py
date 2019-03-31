
# coding: utf-8

# # Preparing the Data

# In[2]:


import glob

train_files = glob.glob('./data/cities_train/train/*.txt')
val_files = glob.glob('./data/cities_val/val/*.txt')


# In[3]:


import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicode_to_ascii('Ślusàrski'))


# In[4]:


# Build the category_lines dictionary, a list of names per language
category_lines_train = {}
category_lines_val = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, "r", encoding='utf-8', errors='ignore').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in train_files:
    category = filename.split('/')[-1].split('.')[0]
    category = filename.split('\\')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines_train[category] = lines

for filename in val_files:
    category = filename.split('/')[-1].split('.')[0]
    category = filename.split('\\')[-1].split('.')[0]
    lines = readLines(filename)
    category_lines_val[category] = lines

n_categories = len(all_categories)
print('n_categories =', n_categories)


# Now we have `category_lines`, a dictionary mapping each category (language) to a list of lines (names). We also kept track of `all_categories` (just a list of languages) and `n_categories` for later reference.

# In[5]:


print(category_lines_train['af'][:5])
print(category_lines_val['af'][:5])


# # Turning Names into Tensors
# 
# Now that we have all the names organized, we need to turn them into Tensors to make any use of them.
# 
# To represent a single letter, we use a "one-hot vector" of size `<1 x n_letters>`. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. `"b" = <0 1 0 0 0 ...>`.
# 
# To make a word we join a bunch of those into a 2D matrix `<line_length x 1 x n_letters>`.
# 
# That extra 1 dimension is because PyTorch assumes everything is in batches - we're just using a batch size of 1 here.

# In[6]:


import torch

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor


# In[7]:


print(letter_to_tensor('J'))


# In[8]:


print(line_to_tensor('Jones').size())


# # Creating the Network
# 
# Before autograd, creating a recurrent neural network in Torch involved cloning the parameters of a layer over several timesteps. The layers held hidden state and gradients which are now entirely handled by the graph itself. This means you can implement a RNN in a very "pure" way, as regular feed-forward layers.
# 
# This RNN module (mostly copied from [the PyTorch for Torch users tutorial](https://github.com/pytorch/tutorials/blob/master/Introduction%20to%20PyTorch%20for%20former%20Torchies.ipynb)) is just 2 linear layers which operate on an input and hidden state, with a LogSoftmax layer after the output.
# 
# ![](https://i.imgur.com/Z2xbySO.png)

# In[9]:


import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


# ## Manually testing the network
# 
# With our custom `RNN` class defined, we can create a new instance:

# In[10]:


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


# To run a step of this network we need to pass an input (in our case, the Tensor for the current letter) and a previous hidden state (which we initialize as zeros at first). We'll get back the output (probability of each language) and a next hidden state (which we keep for the next step).
# 
# Remember that PyTorch modules operate on Variables rather than straight up Tensors.

# In[11]:


input = Variable(letter_to_tensor('A'))
hidden = rnn.init_hidden()

output, next_hidden = rnn(input, hidden)
print('output.size =', output.size())


# For the sake of efficiency we don't want to be creating a new Tensor for every step, so we will use `line_to_tensor` instead of `letter_to_tensor` and use slices. This could be further optimized by pre-computing batches of Tensors.

# In[12]:


input = Variable(line_to_tensor('Albert'))
hidden = rnn.init_hidden()

output, next_hidden = rnn(input[0], hidden)
print(output)


# As you can see the output is a `<1 x n_categories>` Tensor, where every item is the likelihood of that category (higher is more likely).

# # Preparing for Training
# 
# Before going into training we should make a few helper functions. The first is to interpret the output of the network, which we know to be a likelihood of each category. We can use `Tensor.topk` to get the index of the greatest value:

# In[13]:


def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

print(category_from_output(output))


# We will also want a quick way to get a training example (a name and its language):

# In[14]:


import random

def random_training_pair():                                                                                                               
    category = random.choice(all_categories)
    line = random.choice(category_lines_train[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_pair()
    print('category =', category, '/ line =', line)


# In[15]:


def random_val_pair():                                                                                                               
    category = random.choice(all_categories)
    line = random.choice(category_lines_val[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_val_pair()
    print('category =', category, '/ line =', line)


# # Training the Network
# 
# Now all it takes to train this network is show it a bunch of examples, have it make guesses, and tell it if it's wrong.
# 
# For the [loss function `nn.NLLLoss`](http://pytorch.org/docs/nn.html#nllloss) is appropriate, since the last layer of the RNN is `nn.LogSoftmax`.

# We will also create an "optimizer" which updates the parameters of our model according to its gradients. We will use the vanilla SGD algorithm with a low learning rate.

# In[16]:


criterion = nn.NLLLoss()

learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


# Each loop of training will:
# 
# * Create input and target tensors
# * Create a zeroed initial hidden state
# * Read each letter in and
#     * Keep hidden state for next letter
# * Compare final output to target
# * Back-propagate
# * Return the output and loss

# In[17]:


def train(category_tensor, line_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data

def validation(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    return loss.data


# Now we just have to run that with a bunch of examples. Since the `train` function returns both the output and loss we can print its guesses and also keep track of loss for plotting. Since there are 1000s of examples we print only every `print_every` time steps, and take an average of the loss.

# In[18]:


import time
import math

n_epochs = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss_train = 0
current_loss_val = 0
all_losses_train = []
all_losses_val = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()


# In[17]:


for epoch in range(1, n_epochs + 1):
    # Get a random training input and target
    category, line, category_tensor, line_tensor = random_training_pair()
    output, loss = train(category_tensor, line_tensor)
    current_loss_train += loss
    
    category, line, category_tensor, line_tensor = random_val_pair()
    loss = validation(category_tensor, line_tensor)
    current_loss_val += loss
    
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses_train.append(current_loss_train / plot_every)
        all_losses_val.append(current_loss_val / plot_every)
        current_loss_train = 0
        current_loss_val = 0


# # Plotting the Results
# 
# Plotting the historical loss from `all_losses` shows the network learning:

# In[19]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.plot(all_losses_train, color='skyblue', label='training loss')
plt.plot(all_losses_val, color='olive', label='validation loss')
plt.legend()


# # Evaluating the Results
# 
# To see how well the network performs on different categories, we will create a confusion matrix, indicating for every actual language (rows) which language the network guesses (columns). To calculate the confusion matrix a bunch of samples are run through the network with `evaluate()`, which is the same as `train()` minus the backprop.

# In[20]:


def compute_acc(matrix):
    micro_acc = torch.sum(matrix.diag()) / torch.sum(matrix)
    macro_acc = TP_pos = 0
    for row in matrix:
        macro_acc += (row[TP_pos] / row.sum())
        TP_pos += 1
    
    macro_acc /= len(matrix)
    return round(float(micro_acc), 8), round(float(macro_acc), 8)


# In[21]:


# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_val_pair()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

micro_acc, macro_acc = compute_acc(confusion)
print('micro_acc:', micro_acc, 'macro_acc:', macro_acc)

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()


# # Different RNN Configurations

# In[22]:


def RNN_config(n_hidden=128):
    rnn = RNN(n_letters, n_hidden, n_categories)
    
    return rnn

# Keep track of losses for plotting
def train_config(n_epochs=100000):
    print_every = 5000
    plot_every = 1000
    current_loss_train = 0
    current_loss_val = 0
    all_losses_train = []
    all_losses_val = []
    for epoch in range(1, n_epochs + 1):
        # Get a random training input and target
        category, line, category_tensor, line_tensor = random_training_pair()
        output, loss = train(category_tensor, line_tensor)
        current_loss_train += loss

        category, line, category_tensor, line_tensor = random_val_pair()
        loss = validation(category_tensor, line_tensor)
        current_loss_val += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses_train.append(current_loss_train / plot_every)
            all_losses_val.append(current_loss_val / plot_every)
            current_loss_train = 0
            current_loss_val = 0
    
    return all_losses_train, all_losses_val

def show_loss(all_losses_train, current_loss_val):
    plt.figure()
    plt.plot(all_losses_train, color='skyblue', label='training loss')
    plt.plot(all_losses_val, color='olive', label='validation loss')
    plt.legend()

def acc_score():
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Just return an output given a line
    def evaluate(line_tensor):
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_val_pair()
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    
    micro_acc, macro_acc = compute_acc(confusion)
    print('micro_acc:', micro_acc, 'macro_acc:', macro_acc)
    
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# ## learning_rate=0.0001

# In[22]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion = nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(), lr=0.0001)

print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## learning_rate=0.001

# In[23]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion = nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(), lr=0.001)

print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## learning_rate=0.0015

# In[52]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(), lr=0.0015)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=150000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## learning_rate=0.002

# In[40]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(), lr=0.002)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## learning_rate=0.003

# In[41]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(), lr=0.003)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## hidden layer size = 256

# In[42]:


print('n_hidden=256')
rnn = RNN_config(n_hidden=256)

criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(), lr=0.001)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## hidden layer size = 512

# In[51]:


print('n_hidden=512')
rnn = RNN_config(n_hidden=512)

criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(rnn.parameters(), lr=0.001)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=150000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## optimizer = RMSprop

# In[23]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion=nn.NLLLoss()
optimizer=torch.optim.RMSprop(rnn.parameters(), lr=0.001)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## optimizer = Adam

# In[49]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion=nn.NLLLoss()
optimizer=torch.optim.Adam(rnn.parameters(), lr=0.001)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()


# ## optimizer = ASGD

# In[50]:


print('n_hidden=128')
rnn = RNN_config(n_hidden=128)

criterion=nn.NLLLoss()
optimizer=torch.optim.ASGD(rnn.parameters(), lr=0.001)
print('all_losses_train, current_loss_val')
all_losses_train, all_losses_val = train_config(n_epochs=100000)
show_loss(all_losses_train, all_losses_val)

print('acc_score')
acc_score()

