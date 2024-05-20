import torch 
from torch.nn import Module, Linear, Softmax, NLLLoss


class CBOW (Module):
    def __init__(self, vocab_size : int, embedding_dim : int, window_size : int):
        super().__init__()
        self.N = embedding_dim 
        self.V = vocab_size
        self.C = window_size

        # input W shape (V,N)
        self.inputWeight= Linear(in_features= self.V, out_features= self.N, bias = False)  
        # output W shape (N,V)
        self.outputWeight= Linear(in_features=self.N, out_features= self.V, bias = False)
        self.softmax= Softmax(dim=0) 
        
        self.reset_parameters()
        return None

    def reset_parameters(self):
        self.inputWeight= Linear(in_features= self.V, out_features= self.N, bias = False)  
        self.outputWeight= Linear(in_features=self.N, out_features= self.V, bias = False)
        pass

    def forward(self,context_words : torch.Tensor): 
        assert context_words.shape[0] == self.C , "Window Size not matching"

        hidden = None
        for i in range(self.C):
            word = context_words[i]
            if isinstance(hidden, torch.Tensor): 
                hidden = hidden  + self.inputWeight(word)
            else:
                hidden = self.inputWeight(word)
        try : 
            hidden = hidden / self.C
        except : 
            assert hidden, "error while calculating hidden"

        out = self.outputWeight(hidden)
        return self.softmax(out)
    

    def backward(self, output : torch.Tensor, target : torch.Tensor):
        loss_fn = NLLLoss()
        loss = loss_fn(output, target)
        
        # call for grad attribute update
        loss.backward()
        return loss




if __name__ == "__main__":
    voc = 5
    win = 4
    dim = 2

    cbow = CBOW(vocab_size=voc, embedding_dim= dim, window_size=win)

    x = torch.rand(win,voc)
    print("Context Words :")
    __import__('pprint').pprint(x)

    out = cbow.forward(x)
    print("Target Word :")
    __import__('pprint').pprint(out)

