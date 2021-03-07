import torch 
import torch.nn as nn 

class Attention(nn.Module):
    def __init__(self,embedding_sz,heads): 
        super(Attention, self).__init__()
        self.embedding_sz = embedding_sz   #size of input embeddings
        self.heads = heads
        self.head_dim = self.embedding_sz // self.heads
        if self.head_dim * self.heads != self.embedding_sz:
            raise Exception("Embedding Size not divisible by number of heads") 
        
        #keys queries and values 
        self.keys = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.fc = nn.Linear(self.head_dim * self.heads, self.embedding_sz) #fc for embedding 

    def forward(self,values,queries,keys,mask):
        N = queries.shape[0]
        #put them into self.heads number of pieces 
        values_len,query_len,keys_len = values.shape[1], queries.shape[1], keys.shape[1]
        values = values.reshape(N,values_len,self.heads,self.head_dim) # n x v x h x d
        queries = queries.reshape(N,query_len,self.heads,self.head_dim) #n x q x h x d
        keys = keys.reshape(N,keys_len,self.heads,self.head_dim) #n x k x h x d
        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys]) # n x h x q x k 

        if mask is not None:
            energy = energy.masked_fill(mask==0,-1e20)
        attention1 = torch.softmax(energy / ((self.embedding_sz) ** 0.5),dim=3)
        ##attention size : (n x h x q x k )
        ##values size : (n x v x h x d)
        attention = torch.einsum("nhqv,nvhd->nqhd",[attention1,values]).reshape(N,query_len,self.heads * self.head_dim)
        print(attention.shape)
        return self.fc(attention)


class TransformerBlock(nn.Module):
    def __init__(self,embedding_sz,heads,drop_prob,expansion):
        super(TransformerBlock,self).__init__()
        self.embedding_sz = embedding_sz
        self.norm1 = nn.LayerNorm(self.embedding_sz)
        self.norm2 = nn.LayerNorm(self.embedding_sz)
        self.attention = Attention(embedding_sz,heads) 
        self.drop = nn.Dropout(drop_prob)
        
        self.feed_forward = nn.Sequential(
                nn.Linear(embedding_sz,embedding_sz*expansion),
                nn.ReLU(),
                nn.Linear(expansion*embedding_sz,embedding_sz),
                )
         
    def forward(self,values,keys,queries,mask):
        attention = self.attention(values,queries,keys,mask)
        norm1 = self.drop(self.norm1(attention + queries))  
        forward = self.feed_forward(norm1)
        norm2 = self.drop(self.norm2(forward + norm1))
        return norm2

class Encoder(nn.Module):
    def __init__(self,vocab_sz, embedding_sz, num_layers,heads,expansion,drop_prob,max_length,device):
        super(Encoder,self).__init__()
        self.embedding_sz = embedding_sz
        self.word_embedding = nn.Embedding(vocab_sz,self.embedding_sz)
        self.pose_embedding = nn.Embedding(max_length,self.embedding_sz) 
        self.layers = nn.ModuleList([
                    TransformerBlock(self.embedding_sz,heads,drop_prob,expansion)
                    for _ in range(num_layers)
            ])
        self.dropout = nn.Dropout(drop_prob)
        self.device = device

    def forward(self,x,mask): 
        N, sequence_len = x.shape 
        positions = torch.arange(0,sequence_len).expand(N,sequence_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.pose_embedding(x))
        for layer in self.layers:
            #SELF ATTENTION ! Q = K = V
            out = layer(out,out,out,mask)
        return out 

class DecoderBlock(nn.Module):
    def __init__(self,heads,embedding_sz,expansion,drop_prob): 
        super(DecoderBlock,self).__init__()
        self.heads = heads
        self.embedding_sz = embedding_sz
        self.block = TransformerBlock(self.embedding_sz,heads,drop_prob,expansion) 
        self.dropout = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(self.embedding_sz)
        self.attention = Attention(self.embedding_sz,self.heads)
       
    def forward(self,x,values,keys,src_mask,trg_mask):
        attn = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attn + x))
        return self.block(values,keys,query,src_mask)



class Decoder(nn.Module):
    def __init__(self,vocab_sz,embedding_sz,num_layers,heads,expansion,drop_prob,max_length,device):
        super(Decoder,self).__init__()
        self.Block = DecoderBlock(heads,embedding_sz,expansion,drop_prob)
        self.layers = nn.ModuleList([
             self.Block for _ in range(num_layers)]) 
        self.fc = nn.Linear(embedding_sz,vocab_sz)
        self.drop = nn.Dropout(drop_prob)
        self.device = device
        self.dropout = nn.Dropout(drop_prob)
        self.embedding_sz = embedding_sz
        self.word_embedding = nn.Embedding(vocab_sz,self.embedding_sz)
        self.pose_embedding = nn.Embedding(max_length,self.embedding_sz)


    def forward(self,x,encoder_output,src_mask,trg_mask):
        N, sequence_len = x.shape 
        positions = torch.arange(0,sequence_len).expand(N,sequence_len).to(self.device)
        x =self.dropout(self.word_embedding(x) + self.pose_embedding(x))
        for layer in self.layers:
            x = layer(x,encoder_output,encoder_output,src_mask,trg_mask) 
        out = self.fc(x) 

        return out 


class Transformer(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,embedding_sz=512,num_layers=6,forward_expansion=4,heads=8,dropout=0,device="cpu",max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size,embedding_sz,num_layers,heads,forward_expansion,dropout,max_length,device)
        self.decoder = Decoder(trg_vocab_size,embedding_sz,num_layers,heads,forward_expansion,dropout,max_length,device)
        self.decoder = Decoder(trg_vocab_size,embedding_sz,num_layers,heads,forward_expansion,dropout,max_length,device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])










