from model.network import HybridNetwork
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser()

seq_length_rnn = 5   #Number of timesteps for prediction.
input_dim_rnn = 15   # Number of features
hidden_dim_rnn = 100
output_dim_rnn = 10   # Number of output
input_dim_fc = 6

batch_size = 128

print(torch.rand((seq_length_rnn, batch_size, input_dim_rnn)).shape)

X_rnn =  Variable(torch.rand((seq_length_rnn, batch_size, input_dim_rnn)))

print(torch.rand((batch_size,input_dim_fc)).shape)

X_fc =  Variable(torch.rand((batch_size,input_dim_fc)))

y  =   Variable(torch.randn((batch_size,1)))



learning_rate = 0.1

momentum = 0.9

model = HybridNetwork(input_dim_rnn,hidden_dim_rnn,output_dim_rnn,input_dim_fc)

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# print("Args: %s" % args)
#
# if args.cuda:
#     model.cuda()

loss_function=nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=momentum)

for epoch in range(300):
    model.zero_grad()
    model.hidden = model.init_hidden()

    tag_scores = model.forward(X_fc,X_rnn)
    loss = loss_function(tag_scores, y)
    print("in epoch %d loss is %s" % (epoch,loss))
    loss.backward()
    optimizer.step()

# for epoch in range(300):
#     for i in range(batch_size):
#         model.zero_grad()
#         model.hidden = model.init_hidden()
#         tag_scores = model.forward(X_fc[i,:],X_rnn[:,i,:])
#         loss = loss_function(tag_scores[i], y[i,:])
#         print("in epoch %d loss is %s" % (epoch,loss))
#         loss.backward()
#         optimizer.step()





