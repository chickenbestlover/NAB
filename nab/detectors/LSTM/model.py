import torch.nn as nn
import torch
from torch.autograd import Variable

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, features), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1).expand_as(x)
        std = x.std(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type='LSTM', input_size=1, output_size=1,embed_size=200, hidden_size=200, nlayers=2, dropout=0.5, layerNorm=True,resLearn = True):
        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(input_size, embed_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embed_size, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size)


        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.performLayerNorm = layerNorm
        self.norm1 = LayerNorm(self.hidden_size)
        self.norm2 = LayerNorm(self.hidden_size)
        self.resLearn = resLearn

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        #self.encoder.bias.data.fill_(0)
        #self.decoder.bias.data.fill_(0)

    def forward(self, input, hidden):
        #emb = self.drop(self.encoder(input))
        #print input.contiguous().view(input.size(0) * input.size(1), 1).size()

        emb = self.drop(self.encoder(input.contiguous().view(input.size(0)*input.size(1),1)))
        x = emb.clone()
        if self.performLayerNorm:
            #print emb.size()
            emb = self.norm1.forward(emb)
            #print emb.size()
            #print '.........'
        #print emb.size()
        #print emb.view(input.size(0), input.size(1), self.hidden_size).size()
        output, hidden = self.rnn(emb.view(input.size(0), input.size(1), self.hidden_size), hidden)
        #print output.size()
        #print hidden[0].size()
        output = self.drop(output)
        output = output.view(output.size(0)*output.size(1), output.size(2))

        if self.performLayerNorm:
            #print output.size()
            output = self.norm2.forward(output)
            #print output.size()
        if self.resLearn:
            output = output + x
        decoded = self.decoder(output)
        #decoded = decoded + x
        return decoded.view(input.size(0), input.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_())