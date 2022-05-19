import logging
import torch
from time import time
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from random import randint
from tqdm import tqdm
from torch.optim import Adam


logging.getLogger().setLevel(logging.INFO)


def main(args):

    # instantiate model and tokenizer
    tok = AutoTokenizer.from_pretrained(args.pre_trained_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.pre_trained_name)

    # get device
    device = torch.device(args.device)

    # move model to right device
    model.to(device=device)

    do_backprop = args.mode == 'training'

    # instantiate simple optimizer
    if do_backprop:
        optim = Adam(model.parameters(), lr=1e-04)

    # create fake inputs (performance does not depend on the input tokens, just on the sequence length)
    input_ids = [[randint(0, tok.vocab_size - 1) for _ in range(args.sequence_length)]] * args.batch_size
    attention_mask = [[1] * args.sequence_length] * args.batch_size
    labels = [randint(0, 1)] * args.batch_size

    # create input dict
    inputs = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # transform inputs in tensors
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}

    logging.info("Input tensors size:")
    for k, v in inputs.items():
        logging.info(f" * {k}: {v.shape}")

    start_time = time()
    for _ in tqdm(range(args.steps), desc="Testing...", total=args.steps):
        # move inputs to correct device
        # cannot do it before because in a real-world scenario the data will be always different
        data = {k: v.to(device=device) for k, v in inputs.items()}

        if do_backprop:
            optim.zero_grad()
    
        if do_backprop:
            res = model(**data)
        else:
            with torch.no_grad():
                res = model(**data)
    
        if do_backprop:
            res.loss.backward()
            optim.step()

    logging.info(f"Model {args.pre_trained_name} took {(time() - start_time):.2f} seconds to do {args.steps} steps in {args.mode} with batch size {args.batch_size} on {args.device}.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pre_trained_name', type=str, default='bert-base-cased', help="Path or name of pretrained model.")
    parser.add_argument('--device', type=str, required=True, choices=('cpu', 'cuda', 'mps'), help="Which device to use.")
    parser.add_argument('--batch_size', type=int, default=32, required=False, help="Training or inference batch size")
    parser.add_argument('--mode', type=str, required=False, default='training', choices=('training', 'inference'), help="Training or just inference.")
    parser.add_argument('--steps', type=int, default=100, required=False, help="Number of step to train.")
    parser.add_argument('--sequence_length', type=int, default=128, required=False, help="Input sequence length.")
    args = parser.parse_args()
    main(args)
