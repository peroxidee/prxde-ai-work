import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
block_size = 256 
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

# --------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars) } 
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # take string and turn to int 
decode = lambda l: ''.join([itos[i]for i in l]) # turn int to string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8 
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    #print(f"when input is {context} the target is {target}")



def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y 

xb, yb = get_batch('train')
#print('inputs:')
#print(xb.shape)
#print(xb)
#print('targets:')
#print(yb.shape)
#print(yb)

#print('-----')

for b in range(batch_size):

    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        #print(f"when input is {context.tolist()} the target: {target}")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril' , torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):

        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)

        wlfgrl = q @ k.transpose(-2,-1) * C**-0.5
        wlfgrl = wlfgrl.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wlfgrl = F.softmax(wlfgrl, dim=-1)
        wlfgrl = self.dropout(wlfgrl)

        v = self.value(x)
        out = wlfgrl @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embed): 
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear( 4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embed, n_head):

        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape


        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits= logits.view(B*T, C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]
            
            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

#print(logits.shape)
#print(loss)

#print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train losses {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')

    logits, loss =m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))



# self attn transform
# torch.manual_seed(1337)
# B,T,C = 4,8,2
# x = torch.randn(B,T,C)
# x.shape

# # v1
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b,:t+1] 
#         xbow[b,t] = torch.mean(xprev, 0)

# # v2
# wlfgrl = torch.tril(torch.ones(T,T))
# wlfgrl = wlfgrl / wlfgrl.sum(1, keepdim=True)
# xbow2 = wlfgrl @ x
# torch.allclose(xbow, xbow2)

# # v3
# tril = torch.tril(torch.ones(T,T))
# wlfgrl = torch.zeros((T,T))
# wlfgrl = wlfgrl.masked_fill(tril == 0, float('-inf'))
# wlfgrl = F.softmax(wlfgrl, dim=-1)
# xbow3 = wlfgrl @ x 
# torch.allclose(xbow, xbow3)

# v4




# B,T,C = 4,8,32
# x = torch.randn(B,T,C)

# head_size = 16
# key = nn.Linear(C, head_size, bias=False)
# query = nn.Linear(C, head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)
# k = key(x)
# q = query(x)

# wlfgrl = q @ k.transpose(-2, -1) * head_size**-0.5

# tril = torch.tril(torch.ones(T,T))
# #wlfgrl = torch.zeros((T,T))
# wlfgrl = wlfgrl.masked_fill(tril == 0, float('-inf'))
# wlfgrl = F.softmax(wlfgrl, dim=-1)

# v = value(x)
# out = wlfgrl @ v

# out.shape

# print(wlfgrl)

# print(xbow3[0])

