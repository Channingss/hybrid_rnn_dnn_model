import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from to
class HybridNetwork(nn.Module):

    def __init__(self,input_dim_rnn, hidden_dim_rnn, output_dim_rnn,input_dim_fc):
        super(HybridNetwork, self).__init__()

        self.hidden_dim_rnn = hidden_dim_rnn

        self.rnn= nn.LSTM(input_dim_rnn, hidden_dim_rnn,num_layers=3,dropout=0.2)

        self.hidden2tag  = nn.Linear(hidden_dim_rnn, output_dim_rnn, bias=False)

        self.hidden = self.init_hidden()

        self.fc = nn.Sequential(
            nn.Linear(input_dim_fc, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 50),
            # nn.BatchNorm1d(1,momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Linear(50, 50),
            # nn.BatchNorm1d(1, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
        )

        self.endfc= nn.Sequential(
            nn.Linear(50 + output_dim_rnn, 100),
            nn.Linear(100, 1),
            nn.ReLU(inplace=True),
        )

    def init_hidden(self):
        return (Variable(torch.zeros(3, 128, self.hidden_dim_rnn)),Variable(torch.zeros(3, 128, self.hidden_dim_rnn)))

    def forward(self,input_fc,input_rnn):

        # print(input_rnn.size())
        #
        # input_fc = input_fc.view(-1, input_fc.size()[0])
        #
        # input_rnn =input_rnn.view(input_rnn.size()[0], 1, input_rnn.size()[1])

        # output_rnn,self.hidden = self.rnn(input_rnn)

        output_rnn,self.hidden = self.rnn(input_rnn, self.hidden)

        print(self.hidden[0].size())
        print(self.hidden[1].size())

        output_rnn = self.hidden2tag(output_rnn[-1])

        output_fc = self.fc(input_fc)

        # print(output_fc.size())
        # print(output_rnn.size())

        input_endfc =torch.cat((output_rnn,output_fc),1)

        output = self.endfc(input_endfc)

        return output



