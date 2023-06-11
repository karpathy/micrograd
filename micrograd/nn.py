import random
from micrograd.engine import Value
import math
'''
possible future additions:
- training with multiple batches
- adding dropout
'''

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True, use_bias=False):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.use_bias=use_bias
        if self.use_bias:
            self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        if self.use_bias:
            act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        else:
            act = sum((wi*xi for wi,xi in zip(self.w, x)))
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Linear(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Linear layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts,use_bias=True):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i!=len(nouts)-1,use_bias=use_bias) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class CausalAttentionHead(Module):
    #TODO: figure out how to fit attention into this framework
    def __init__(n_embd,use_bias=False):
        self.n_embd=n_embd
        self.q=Linear(n_embd,n_embd,use_bias=use_bias)
        self.k=Linear(n_embd,n_embd,use_bias=use_bias)
        self.v=Linear(n_embd,n_embd,use_bias=use_bias)
    def __call__(self):
        pass
    def parameters(self):
        pass
    def __repr__(self):
        pass

class MultiHeadAttention(Module):
    #TODO
    pass


class LayerNorm(Module):
    #TODO
    pass

class Embedding(Module):
    def __init__(self,vocab_size,n_embd):
        self.vocab_size=vocab_size
        self.n_embd=n_embd
        #we embed each one-dimensional values to n_embd dimensions, and those are our trainable weights
        self.embeddings_list=[[Value(random.uniform(-1,1)) for _ in range(n_embd)] for _ in range(vocab_size)]
    def __call__(self,x):
        assert(0<=x and x<vocab_size)
        return self.embeddings_list[x]
    def parameters(self):
        return [p for embedding in self.embeddings_list for p in embedding]
    def __repr__(self):
        return f"Embedding of [{"|".join(emb) for emb in embeddings_list}]"

class GPT_MLP(Module):
    def __init__(self, n_embd,use_bias=False):
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, use_bias=use_bias,nonlin=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, use_bias=use_bias,nonlin=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_proj(x)
        return x
    def parameters(self):
        pass
    def __repr__(self):
        pass
         
class TransformerBlock(Module):
    def __init__(self, n_embd, m_head, use_bias=False):
        self.n_embd=n_embd
        self.ln_1 = LayerNorm(n_embd, use_bias=use_bias)
        self.attn = MultiHeadAttention(n_embd,n_head)
        self.ln_2 = LayerNorm(n_embd, use_bias=use_bias)
        self.mlp = GPT_MLP(n_embd, use_bias=use_bias)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    def parameters(self):
        pass
    def __repr__(self):
        pass

class GPT(Module):
    def __init__(self, n_layer, vocab_size, n_embd, n_head, use_bias=False):
        self.token_embedding=Embedding(vocab_size,n_embd)
        #not using positional embeddings! (https://twitter.com/a_kazemnejad/status/1664277559968927744)
        self.transformer_blocks=[TransformerBlock(n_embd,n_head,use_bias) for _ in range(n_layer)]
        self.layernorm_final=[]
    def 
class CrossEntropyLoss(Module):
    def __init__(self,logits,values,reduction="sum"):
        #only supporting unweighted cross-entropy loss
        self.logits=logits
        self.values=values
        self.num_classes=len(logits)
        self.reduction=reduction
        assert(self.reduction in ["mean","sum"])
    def __call__():
        #TODO: add normalization for numeric stability (should scale largest value to 1, but not sure how that affects backprop)
        #formula copied from https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        sum_exp=sum(l.exp() for l in logits)
        ls=[-math.log(logits[i].exp()/sum_exp)*values[i] for i in range self.num_classes]
        if self.reduction=="sum":
            return sum(ls)
        else:
            return sum(ls)/num_classes
    def parameters(self):
        return [l for l in self.logits]