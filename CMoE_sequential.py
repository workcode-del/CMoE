import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm
from CMoE_utils import construct_moe
from datautils import *
from sft_utils import simple_sft

DEV = torch.device('cuda:0')


@torch.no_grad()
def cmoe_ppl_eval(model, testloader, eval_set, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Don't move model parts to single device, let them stay on their distributed devices
    dtype = next(iter(model.parameters())).dtype
    inps = []
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # Store input from the first layer (which is already on the correct device)
            inps.append(inp.clone())
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    
    # Get the first layer's device
    first_layer_device = next(layers[0].parameters()).device
    layers[0] = Catcher(layers[0].to(first_layer_device))
    
    testenc = testloader.input_ids
    nsamples = testenc.shape[1] // model.seqlen
    nsamples = 64
    print('ppl evaluation samples:', nsamples)


    # Get input samples from all layers
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(first_layer_device)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    
    # Convert list to tensor on the first layer's device
    print(len(inps), nsamples, testenc.shape[1], model.seqlen)
    inps = torch.stack(inps)
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    # Process each layer, keeping it on its current device
    for i in tqdm(range(len(layers)), desc='Processing...'):
        layer = layers[i]
        layer_device = next(layer.parameters()).device
        
        # Move input to layer's device
        layer_inps = inps.to(layer_device)
        layer_outs = torch.zeros_like(layer_inps)
        
        for j in range(nsamples):
            layer_outs[j] = layer(layer_inps[j], 
                                attention_mask=attention_mask, 
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
        
        # Move output back to first layer's device
        outs = layer_outs.to(first_layer_device)
        
        # Swap inps and outs for next iteration
        inps, outs = outs, torch.zeros_like(outs)

    final_layer_device = next(layers[-1].parameters()).device
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(final_layer_device)
    model.lm_head = model.lm_head.to(final_layer_device)

    testenc = testenc.to(final_layer_device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states).contiguous()
        
        hidden_states = hidden_states.to(final_layer_device)

        lm_head_weight = model.lm_head.weight.to(final_layer_device)
        with torch.no_grad():
            lm_logits = torch.nn.functional.linear(hidden_states, lm_head_weight, None)
        lm_logits = lm_logits.squeeze(1)
 #       lm_logits = model.lm_head(hidden_states)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print("ppl: ", ppl.item())
    model.config.use_cache = use_cache

    return ppl.item()

def cmoe_sequential(model, dataloader, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = args.carve_bsz
    
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(DEV))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    moe_outs = torch.zeros_like(inps)
    
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print('Ready.')
    # model.cuda()
    # layers.cuda()

    inp = copy.deepcopy(inps[0])

    # MoE Carving
    carve_inp = copy.deepcopy(inp)
    for layer in tqdm(layers, desc = 'Carving MoE layers...'):
        moe_out = construct_moe(layer, 
            carve_inp, 
            attention_mask, 
            position_ids,
            position_embeddings,
            n_experts = args.nexperts,
            n_activated = args.nactivated,
            n_shared = args.nshared,
            args = args
        )
        carve_inp = moe_out
    
    tick_1 = time.time()

    print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, bsz = args.carve_bsz
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    tick_2 = time.time()
    
    # LoRa-based Supervised Fine-tuning
    for layer in layers:
            layer.mlp.cus_training = True

    model.cuda()
    model = simple_sft(model, args, epoch = args.epoch)

    for layer in layers:
        layer.mlp.cus_training = False

    model.eval()

    model.config.use_cache = use_cache
    
    return model, tick_1, tick_2, pre_ppl
