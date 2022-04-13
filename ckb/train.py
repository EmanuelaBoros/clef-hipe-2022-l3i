# -*- coding: utf-8 -*-

from ckb import compose
from ckb import datasets
from ckb import evaluation
from ckb import losses
from ckb import models
from ckb import sampling
from ckb import scoring

from transformers import AutoTokenizer
from transformers import AutoModel

import torch
import argparse

_ = torch.manual_seed(42)

device = 'cuda' #  You should own a GPU, it is very slow with cpu.


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='embeddia',
                    choices=['embeddia'])
parser.add_argument('--model', type=str, default='stacked',
                    choices=['bert', 'stacked'])
parser.add_argument('--directory', type=str, default='caches')
parser.add_argument('--language', type=str, default='english')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-5)

# for elaborate preditions of multiple files
parser.add_argument('--dataset_dir', type=str)  # input directory files
parser.add_argument('--output_dir', type=str)  # output predistions directory
parser.add_argument('--extension', type=str, default='txt')  # output predistions directory
# for elaborate preditions of multiple files

parser.add_argument('--train_dataset', type=str)
parser.add_argument('--test_dataset', type=str)
parser.add_argument('--dev_dataset', type=str)

parser.add_argument('--pre_trained_model', type=str)

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--continue_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval or not.")
# in case of do_eval, load model from saved dir/best
parser.add_argument('--saved_model', type=str)


args = parser.parse_args()
directory = args.directory
pre_trained_model = args.pre_trained_model

with open(args.train_dataset, 'r') as f:
    train = f.readlines()
with open(args.test_dataset, 'r') as f:
    test = f.readlines()
with open(args.dev_dataset, 'r') as f:
    valid = f.readlines()

train = [(x.split('\t')[0], x.split('\t')[1], x.split('\t')[2].strip()) for x in train]
test = [(x.split('\t')[0], x.split('\t')[1], x.split('\t')[2].strip()) for x in test]
valid = [(x.split('\t')[0], x.split('\t')[1], x.split('\t')[2].strip()) for x in valid]
# Train, valid and test sets are a list of triples.
#train = [
#    ('My Favorite Carrot Cake Recipe', 'made_with', 'Brown Sugar'),
#    ('My Favorite Carrot Cake Recipe', 'made_with', 'Oil'),
#    ('My Favorite Carrot Cake Recipe', 'made_with', 'Applesauce'),
#    
#    ('Classic Cheesecake Recipe', 'made_with', 'Block cream cheese'),
#    ('Classic Cheesecake Recipe', 'made_with', 'Sugar'),
#    ('Classic Cheesecake Recipe', 'made_with', 'Sour cream'),
#]
#
#valid = [
#    ('My Favorite Carrot Cake Recipe', 'made_with', 'A bit of sugar'), 
#    ('Classic Cheesecake Recipe', 'made_with', 'Eggs')
#]
#
#test = [
#    ('My Favorite Strawberry Cake Recipe', 'made_with', 'Fresh Strawberry')
#]

print(train[:20])
# Initialize the dataset, batch size should be small to avoid RAM exceed. 
dataset = datasets.Dataset(
    batch_size = 1,
    train = train,
    valid = valid,
    test = test,
    seed = 42,
)

model = models.Transformer(
    model = AutoModel.from_pretrained(pre_trained_model),
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model),
    entities = dataset.entities,
    relations = dataset.relations,
    gamma = 9,
    scoring = scoring.TransE(),
    device = device,
)

model = model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.00005,
)
    
evaluation = evaluation.Evaluation(
    entities = dataset.entities,
    relations = dataset.relations,
    true_triples = dataset.train + dataset.valid + dataset.test,
    batch_size = 1,
    device = device,
)

# Number of negative samples to show to the model for each batch.
# Should be small to avoid memory error.
sampling = sampling.NegativeSampling(
    size = 1,
    entities = dataset.entities,
    relations = dataset.relations,
    train_triples = dataset.train,
)

pipeline = compose.Pipeline(
    epochs = 20,
    eval_every = 3, # Eval the model every {eval_every} epochs.
    early_stopping_rounds = 1, 
    device = device,
)

pipeline = pipeline.learn(
    model = model,
    dataset = dataset,
    evaluation = evaluation,
    sampling = sampling,
    optimizer = optimizer,
    loss = losses.Adversarial(alpha=0.5),
)
#    {'MRR': 0.3958, 'MR': 2.75, 'HITS@1': 0.0, 'HITS@3': 0.75, 'HITS@10': 1.0}
print(evaluation.eval(model = model, dataset = dataset.valid))
print(evaluation.eval(model = model, dataset = dataset.test))
import pdb;pdb.set_trace()


