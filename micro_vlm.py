import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.profiler

def preprocess_data(cfg, device):
    import datasets, cv2
    dataset = datasets.load_dataset(cfg.dataset.dataset_name)

    # cbuffer = CircularBuffer(cfg.dataset.buffer_size, cfg)
    dataset = dataset['validation']

    chars = sorted(list(set([item for row in dataset["question"] for item in row] + [item for row in dataset["multiple_choice_answer"] for item in row]))) ## Flatten to a long string
    cfg.vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode_txt = lambda s: [stoi[c] for c in s] # text encoder to tokens: 
    decode_txy = lambda l: ''.join([itos[i] for i in l]) # token decoder to text: 
    print("vocab_size:", cfg.vocab_size)
    print("example text encode:", encode_txt(dataset["question"][0])) 

    # dataset = dataset.add_column("text2", np.array(text))
    
    # dataset.set_format(type="torch", columns=["image", "question", "multiple_choice_answer", "text2"], device=device)

    if cfg.dataset.save_initial_dataset:
        dataset__ = {"image": [], "question": [], "multiple_choice_answer": []}
        for i in range(min(cfg.trim, dataset.num_rows)):
            item = dataset[i]
            if item["image"].mode == "RGB":
                dataset__["image"].append(cv2.resize(np.array(item["image"], dtype=np.uint8), (cfg.image_shape[0], cfg.image_shape[1])))
                dataset__["question"].append(item["question"])
                dataset__["multiple_choice_answer"].append(item["multiple_choice_answer"])
        # del dataset
        ## Convert lists to torch tensors
        dataset__["image"] = np.array(dataset__["image"], dtype=np.uint8)
        dataset__["question"] = np.array(dataset__["question"])
        dataset__["multiple_choice_answer"] = np.array(dataset__["multiple_choice_answer"])
        from datasets import Image, Dataset
        dataset = Dataset.from_dict(dataset__)

        new_features = dataset.features.copy()
        new_features["image"] = Image()
        dataset.cast(new_features)
        print('Features:', dataset.features)
        # dataset.save_to_disk("datasets/" + cfg.dataset.to_dataset_name_new + ".hf")
        # dataset.push_to_hub(cfg.dataset.to_dataset_name_new)

    images = []
    text = []
    lengths = []
    padding_ = " " * (int(cfg.phrase_length)+1) ## making up 64 for now.
    for i in range(min(len(dataset), cfg.trim)):
        if dataset[i]["image"].mode != "RGB":
            continue
        text_t = dataset[i]["question"]+dataset[i]["multiple_choice_answer"]
        lengths.append(min(max(cfg.max_block_size+2, len(text_t)), cfg.phrase_length+1)) ## plus two for padding and target token
        ## Pad the text to at least 64
        text_ = text_t[0:65] + padding_[len(text_t):cfg.phrase_length + 1]
        text.append(np.array(encode_txt(text_[:cfg.phrase_length + 1])))
        assert len(text[-1]) == cfg.phrase_length + 1
        images.append(cv2.resize(np.array(dataset[i]["image"], dtype=np.uint8), (cfg.image_shape[0], cfg.image_shape[1])))
        assert images[-1].shape == (cfg.image_shape[0], cfg.image_shape[1], cfg.image_shape[2])
    dataset_ = {"image": torch.tensor(np.array(images, dtype=np.uint8), dtype=torch.uint8, device=device),
                "text2": torch.tensor(np.array(text, dtype=np.int32), dtype=torch.long, device=device),
                "text_lengths": np.array(lengths, dtype=np.int32)
                }

    return dataset_, encode_txt, decode_txy

def get_batch(dataset, cfg):
    from PIL import Image
    import cv2
    ## Get a batch of data from the hugging face dataset dataset
    ix = np.random.randint(len(dataset["image"]), size=(cfg.batch_size))
    X_img = dataset['image'][ix]    
    
    ## Need to get a random section of each row in the txt data that is randomly cropped to max_block_size + 1
    r_ixd = np.random.randint(0, dataset['text_lengths'][ix] - (cfg.max_block_size + 1), size=(cfg.batch_size)) ## some text strings can be shorter than the block size
    txt = dataset['text2'][ix]  ## plus one for the target token
    txt = [txt[i][r_ixd[i]:r_ixd[i]+cfg.max_block_size+1] for i in range(len(txt))  ]  ## Grab random crops of text
    txt = torch.stack(txt, dim=0)
    ## The target token is the next token after the block
    Y = txt[:, -1]
    X_text = txt[:, 0:cfg.max_block_size]
        
    ## Convert images to float32
    X_img = X_img.float() / 255.0
    return X_text, X_img, Y


@torch.no_grad()
def estimate_loss(model, dataset, cfg, encode_txt):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X_text, X_img, Y = get_batch(dataset, cfg)
            logits, loss = model(X_text, X_img, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches

def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

## This is an encoder head (full attention)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout, cfg):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        context_length = cfg.max_block_size + (int(cfg.image_shape[0] / cfg.patch_size) ** 2) + 1
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        # [TODO]
        """
        [DEFAULT]
        # TODO: 
        ## Provide the block masking
        pass
        [/DEFAULT]
        """
        # if mask == None:
        #     mask = torch.ones((T, ), device=self.device) ## (1, T)
        # [/TODO]
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        ### Block masked attention
        # wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
        wei[:,64:] = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))[:,64:] # (B, T, T) The attention after the first image tokens should be causal (only attend to previous tokens)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout, cfg):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout, cfg=cfg) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), ## This is where the information may be stored.
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x,)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, cfg):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout, cfg=cfg)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class VLM(nn.Module):
  def __init__(self, cfg, mlp_ratio=4):
    super(VLM, self).__init__()
    # self._dataset = dataset
    self._cfg = cfg
    # [TODO]
    """
    [DEFAULT]
    # TODO: 
    ## Provide the logic for the GRP network

    # 4) Transformer encoder blocks

    # 5) Classification MLPk
    
    [/DEFAULT]
    """
    # self.patch_size = (self._cfg.image_shape[0] / self._cfg.n_patches, self._cfg.image_shape[1] / self._cfg.n_patches)
    #Positional embedding
    self.register_buffer('positional_embeddings', calc_positional_embeddings(1 +        
                      int(cfg.image_shape[0] / cfg.patch_size) ** 2 +
                      self._cfg.max_block_size, ## image
                      cfg.n_embd), 
                      persistent=False)

    self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
    self.class_tokens = nn.Parameter(torch.rand(1, cfg.n_embd))

    self.input_d = int(self._cfg.image_shape[2] * cfg.patch_size * cfg.patch_size)

    self.lin_map = nn.Linear(self.input_d, self._cfg.n_embd, bias=False) 
    self.lin_map_pose = nn.Linear(7, self._cfg.n_embd, bias=True) 
    self.lin_map_pose_m1 = nn.Linear(30, self._cfg.n_embd, bias=True) 

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(self._cfg.n_embd, self._cfg.n_head, dropout=self._cfg.dropout, cfg=cfg) for _ in range(self._cfg.n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self._cfg.n_embd, self._cfg.action_bins),  # Output size is action_bins 
    )
    self.mlp_m1 = nn.Sequential(
        nn.Linear(self._cfg.n_embd, 12),  # Output size is action_bins
    )
    # [/TODO]

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, X_text, X_images, targets):
    # Dividing images into patches
    n, c, h, w = X_images.shape
    # [TODO]
    """
    [DEFAULT]
    # TODO: 
    ## Provide the logic to produce the output and loss for the GRP
    
    # Map the vector corresponding to each patch to the hidden size dimension

    # Adding classification and goal_img tokens to the tokens

    # Adding positional embedding

    # Compute blocked masks

    # Transformer Blocks

    # Getting the classification token only

    # Compute output and loss

    [/DEFAULT]
    """
    # patches = get_patches_fast(images[:,:,:,:3]) ## Only use the first 3 channels of the image
    # patches_more = get_patches_fast(images[:,:,:,3:])
    # obs_patches = [get_patches_fast(images[:,:,:,3*i:3*(i+1)] for i in range(self._cfg.policy.obs_stacking))] ## Only use the first 3 channels of the image
    obs_patches = get_patches_fast(X_images, self._cfg) 
    if self._cfg.dataset.encode_with_t5:
        goals_e = X_text ## This is actually the embedding from the T5 model
        B, T, E = X_text.shape
        # T = 1
        # goals_e = torch.reshape(goals_e, (B, 1, E)) ## Reshape to match the embedding size
    else:
        goals_e = self.token_embedding_table(X_text)
        B, E = X_text.shape
        T = self._cfg.max_block_size
    
    # Running linear layer tokenization to get embeddings
    # Map the vector corresponding to each patch to the hidden size dimension
    out_obs = self.lin_map(obs_patches) ## List of tensors, one for each stacked observation
    
    out = torch.cat((out_obs, goals_e, self.class_tokens.expand(n, 1, -1)), dim=1)
    
    # Adding positional embedding
    out = out + self.positional_embeddings.repeat(n, 1, 1)
    
    # Transformer Blocks
    for block in self.blocks:
        out = block(out)

    # Getting the classification token only as the last 1 token(s)
    out = out[:, -1]
    logits = self.mlp(out)
        
    if targets is None:
        loss = None
    else:
        B, C = logits.shape
        # logits = logits.view(B*T, C)
        targets = targets.view(B)
        loss = F.cross_entropy(logits, targets)
    # [/TODO]
    return (logits, loss)
  
  def generate(self, idx, image, max_new_tokens, cfg):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -cfg.max_block_size:]
            # get the predictions
            logits, loss = self(idx_cond, image, None)
            # focus only on the last time step
            logits = logits # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
  

import hydra
from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="conf", config_name="grp-mini")
@hydra.main(config_path="./conf", config_name="micro-vlm-64pix")
def my_main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print ("cfg:", OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if (device == 'cuda') else "")
    cfg.device = device

    wandb = None
    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config= OmegaConf.to_container(cfg),
            name=cfg.experiment.name,
        )
        wandb.run.log_code(".")

    tokenizer = None
    # text_model = None
    if cfg.dataset.encode_with_t5: ## Load T5 model
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(cfg.dataset.t5_version)
        # text_model = T5ForConditionalGeneration.from_pretrained(cfg.dataset.t5_version)

    dataset, encode_txt, decode_txt = preprocess_data(cfg, device)
    
    model = VLM(cfg)
    model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    ## Print the amount of memory used by the model
    print("Memory used by the model:", torch.cuda.memory_allocated(device) / 1e6, "MB")
    ## Print the amount of memory used by the dataset cBuffer
    from pympler import asizeof
    # cBuffer.print_mem_footprint()


    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    import torch.optim.lr_scheduler as lr_scheduler
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)

    for iter in range(cfg.max_iters):

        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, cfg, encode_txt)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, memory {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
            if not cfg.testing:
                wandb.log({"train loss": losses['train'], "val loss": losses['val'],
                        "memory": torch.cuda.memory_allocated(device) / 1e6,
                        "buffer_size": asizeof.asizeof(dataset) / 1e6}, step=iter)
            xq, xi, yb = get_batch(dataset, cfg)
            print("Q: " + decode_txt(xq[0].tolist())) ## Print the question
            print("A: " + decode_txt(model.generate(xq[:1], xi[:1], max_new_tokens=cfg.max_block_size, cfg=cfg)[0].tolist())) ## Decode the responses learned so far

        if iter == cfg.max_iters - 1:
            path_ = "./microVLM.pth"
            torch.save(model, path_)
            print("Model saved to " + path_)

        # if iter % cfg.data_shuffel_interval == 0 and iter > 0:
        #     ## Update the dataset
        #     shared_queue.put('shuffle')

        xq, xi, yb = get_batch(dataset, cfg)

        # evaluate the loss
        logits, loss = model(xq, xi, yb)
        loss.backward()

        if (iter + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if not cfg.testing:
        wandb.finish()
    # shared_queue.put(None)
    # data_thread.join()

    return losses['val']
 
if __name__ == "__main__":
    results = my_main()
    print("results:", results)