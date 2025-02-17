# -*- coding: utf-8 -*-

from models.models import StackedTransformersCRF, BertCRF
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback, CheckPointCallback
from fastNLP import SpanFPreRecMetric, BucketSampler
from modules.pipe_main import DataReader
from modules.callbacks import EvaluateCallback
from modules.utils import set_rng_seed
from embeddings import CamembertEmbedding
from fastNLP.embeddings import StackEmbedding#, BertEmbedding
from fastNLP.embeddings import CNNCharEmbedding
from embeddings import BertEmbedding
from fastNLP.embeddings import StaticEmbedding, LSTMCharEmbedding, ElmoEmbedding
from modules.TransformerEmbedding import TransformerCharEmbed
import os
import torch
import argparse
from transformers import AdamW
from predictor import Predictor
import multiprocessing
set_rng_seed(rng_seed=2020)

from tqdm import tqdm
import time
from itertools import islice
from functools import partial

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='embeddia',
                    choices=['embeddia'])
parser.add_argument('--model', type=str, default='stacked',
                    choices=['bert', 'stacked'])
parser.add_argument('--directory', type=str, default='caches')
parser.add_argument('--language', type=str, default='english')
parser.add_argument('--n_heads', type=int, default=12)
parser.add_argument('--head_dims', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--attn_type', type=str, default='transformer')
parser.add_argument('--trans_dropout', type=float, default=0.45)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--after_norm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--warmup_steps', type=float, default=0.01)
parser.add_argument('--fc_dropout', type=float, default=0.4)
parser.add_argument('--pos_embed', type=str, default='sin')
parser.add_argument('--encoding_type', type=str, default='bioes')
parser.add_argument('--device', type=int, default=None)
parser.add_argument('--no_cpu', type=int, default=10)

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
dataset = args.dataset
n_heads = args.n_heads
head_dims = args.head_dims
num_layers = args.num_layers
attn_type = args.attn_type
trans_dropout = args.trans_dropout
batch_size = args.batch_size
lr = args.lr
pos_embed = args.pos_embed
warmup_steps = args.warmup_steps
after_norm = args.after_norm
fc_dropout = args.fc_dropout
no_cpu = args.no_cpu
normalize_embed = True

encoding_type = args.encoding_type
name = directory + '/bert_{}_{}_{}.pkl'.format(
    dataset, encoding_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)
output_dir = args.output_dir
if output_dir:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset_dir = args.dataset_dir
    # change if other extension
    files = [os.path.join(path, f) for path, directories, files in os.walk(dataset_dir) for f in files if f.endswith("."+args.extension)
        and not os.path.exists(os.path.join(path.replace(dataset_dir, output_dir), f))]
#    import pdb
#    pdb.set_trace()
paths = {'test': args.test_dataset,
         'train': args.train_dataset,
         'dev': args.dev_dataset}


def _foo(my_number):
    square = my_number * my_number
    time.sleep(1)
    return square


def sliding_window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


@cache_results(name, _refresh=False)
def load_data(paths, load_embed=True):
    data = DataReader(
        encoding_type=encoding_type).process_from_file(paths)

    if load_embed:
#        if args.language != 'french':
#            
#        embed_french = BertEmbedding(data.get_vocab('words'), model_dir_or_name='dbmdz/bert-base-french-europeana-cased',
#                    pool_method='last', requires_grad=True, layers='0,-1,-2,-3,-4,-5', 
#                    include_cls_sep=False, dropout=0.2, auto_truncate=True,
#                  word_dropout=0.01)
        embed_french = BertEmbedding(data.get_vocab('words'), model_dir_or_name=args.pre_trained_model,
                    pool_method='last', requires_grad=True, layers='0,-1,-2,-3,-4,-5', 
                    include_cls_sep=False, dropout=0.2, auto_truncate=True,
                  word_dropout=0.01)

#        embed_german = BertEmbedding(data.get_vocab('words'), model_dir_or_name='dbmdz/bert-base-german-europeana-cased',
#                    pool_method='last', requires_grad=True, layers='0,-1,-2,-3,-4,-5', 
#                    include_cls_sep=False, dropout=0.2, auto_truncate=True,
#                  word_dropout=0.01)
        
        # False
        # True

        embed_ner_french = CamembertEmbedding(data.get_vocab('words'), model_dir_or_name="Jean-Baptiste/camembert-ner",
                    pool_method='last', requires_grad=True, layers='0,-1,-2,-3,-4,-5', 
                    include_cls_sep=False, dropout=0.2, auto_truncate=True,
                  word_dropout=0.01)
#        
#        embed_ner_german = BertEmbedding(data.get_vocab('words'), model_dir_or_name='fhswf/bert_de_ner',
#                    pool_method='last', requires_grad=True, layers='0,-1,-2,-3,-4,-5', 
#                    include_cls_sep=False, dropout=0.2, auto_truncate=True,
#                  word_dropout=0.01)
        
#        embed_multi = BertEmbedding(data.get_vocab('words'), model_dir_or_name='bert-base-multilingual-cased',
#                    pool_method='last', requires_grad=False, layers='0,-1,-2,-3,-4,-5', 
#                    include_cls_sep=False, dropout=0.2, auto_truncate=True,
#                  word_dropout=0.01)
#        
#        
        #dslim/bert-large-NER
        #dbmdz/bert-base-german-europeana-cased

#        embed = StackEmbedding([embed_french, embed_german, embed_multi, 
#                                embed_ner_french, embed_ner_german], dropout=0.0, word_dropout=0.02)
        embed = StackEmbedding([embed_french, embed_ner_french], dropout=0.0, word_dropout=0.02)
#        else:
        
            # embed_large = CamembertEmbedding(vocab=data.get_vocab(
            #     'doc'), model_dir_or_name=args.pre_trained_model, requires_grad=True, layers='0',
            #     auto_truncate=True)

        return data, embed, embed
    return data


data_bundle, embed, embed_doc = load_data(paths, load_embed=True)
print(data_bundle.get_dataset('test')[:10])

#import pdb;pdb.set_trace()
def predict(path, data_bundle, predictor, predict_on='test', do_eval=False):

    if do_eval:
        #        print(path)
        paths = {'train': path}
        data_bundle_test = DataReader(
            encoding_type=encoding_type, vocabulary=data_bundle.get_vocab('words')).process_from_file(paths)
        dataset_test = data_bundle_test.get_dataset('train')
        predictions = predictor.predict(dataset_test)
        predictions_path = path.replace(dataset_dir, output_dir)
    else:
        print('Predicting on {}:'.format(predict_on))
        dataset_test = data_bundle.get_dataset(predict_on)
        predictions = predictor.predict(dataset_test)
        predictions_path = path
        
    with open(predictions_path, 'w') as f:
        f.write('TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC\n')
#        for i, j, j1 in zip(dataset_test, predictions['pred'], predictions['pred1']):
        for i, j, j1, j2, j3, j4, j5 in zip(dataset_test, predictions['pred'], predictions['pred1'], predictions['pred2'], predictions['pred3'],
                        predictions['pred4'], predictions['pred5']):
            if type(j[0]) == int:
#                import pdb;pdb.set_trace()
                f.write(str(i['raw_words'][0]) +
                        '\tO\tO\tO\tO\tO\t_\t_\t_\t_\n')
            else:
                labels = list([data_bundle.get_vocab('target').idx2word[x] for x in j[0]])
                labels += ['O']*len(i['raw_words'])
                labels1 = list([data_bundle.get_vocab('target1').idx2word[x] for x in j1[0]])
                labels1 += ['O']*len(i['raw_words'])
                labels2 = list([data_bundle.get_vocab('target2').idx2word[x] for x in j2[0]])
                labels2 += ['O']*len(i['raw_words'])
                labels3 = list([data_bundle.get_vocab('target3').idx2word[x] for x in j3[0]])
                labels3 += ['O']*len(i['raw_words'])
                labels4 = list([data_bundle.get_vocab('target4').idx2word[x] for x in j4[0]])
                labels4 += ['O']*len(i['raw_words'])
                labels5 = list([data_bundle.get_vocab('target5').idx2word[x] for x in j5[0]])
                labels5 += ['O']*len(i['raw_words'])
#                
#                for word, label, label1 in zip(i['raw_words'], labels, labels1):
#                    f.write(str(word) + '\t' + str(label) + '\t' + str(label1) + '\tO\tO\tO\t_\t_\t_\t_\n')
                for word, label, label1, label2, label3, label4, label5 in zip(i['raw_words'], labels, labels1, labels2, labels3, labels4, labels5):
                    f.write(str(word) + '\t' + str(label) + '\t' + str(label1) + '\t' + str(label2) + '\t' + str(label3) + '\t' + str(label4) + '\t' + str(label5) + '\t_\t_\t_\n')
            f.write('\n')
    return predictions


def main():
    if args.do_eval:
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    if args.model == 'bert':

        model = BertCRF(embed,#, data_bundle.get_vocab('target1')
                        [data_bundle.get_vocab('target')],
                        encoding_type='bioes')

    else:
        model = StackedTransformersCRF(tag_vocabs=[data_bundle.get_vocab('target'),
                                                   data_bundle.get_vocab('target1'),
                                                   data_bundle.get_vocab('target2'),
                                                   data_bundle.get_vocab('target3'),
                                                   data_bundle.get_vocab('target4'),
                                                   data_bundle.get_vocab('target5')],
                                       embed=embed, embed_doc=embed_doc, num_layers=num_layers,
                                       d_model=d_model, n_head=n_heads,
                                       feedforward_dim=dim_feedforward, dropout=trans_dropout,
                                       after_norm=after_norm, attn_type=attn_type,
                                       bi_embed=None,
                                       fc_dropout=fc_dropout,
                                       pos_embed=pos_embed,
                                       scale=attn_type == 'transformer')
        model = torch.nn.DataParallel(model)

    if args.do_eval:
        if os.path.exists(os.path.expanduser(args.saved_model)):
            print("Load checkpoint from {}".format(
                os.path.expanduser(args.saved_model)))
            model = torch.load(args.saved_model)
            model.to('cuda')
            print('model to CUDA')

    optimizer = AdamW(model.parameters(),
                      lr=lr,
                      eps=1e-8
                      )

    callbacks = []
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
    evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))#, batch_size=8, use_cuda=False)
    checkpoint_callback = CheckPointCallback(
        os.path.join(directory, 'model.pth'), delete_when_train_finish=False,
        recovery_fitlog=True)

    if warmup_steps > 0:
        warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
        callbacks.append(warmup_callback)
    callbacks.extend([clip_callback, checkpoint_callback])#, evaluate_callback])

    if not args.do_eval:
        trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer,
                          batch_size=batch_size, sampler=BucketSampler(num_buckets=10, 
                                                                       batch_size=batch_size),
                          num_workers=no_cpu, n_epochs=args.n_epochs,
                          dev_data=data_bundle.get_dataset('dev'),
                          metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'),
                                                    encoding_type=encoding_type),
                          dev_batch_size=batch_size,
                          callbacks=callbacks,
                          device=args.device,
                          test_use_tqdm=True,
                          use_tqdm=True,
                          print_every=1,
                          save_path=os.path.join(directory, 'best'))

        trainer.train(load_best_model=True)

        predictor = Predictor(model)
        predict(os.path.join(directory, 'predictions_dev.tsv'),
                data_bundle, predictor, 'dev')
        predict(os.path.join(directory, 'predictions_test.tsv'),
                data_bundle, predictor, 'test')

    else:
        print('Predicting')
        # predictions of multiple files
        torch.multiprocessing.freeze_support()
        model.share_memory()
        predictor = Predictor(model)

        if len(files) > multiprocessing.cpu_count():
            with torch.multiprocessing.Pool(processes=no_cpu) as p:
                with tqdm(total=len(files)) as pbar:
                    for i, _ in enumerate(p.imap_unordered(partial(predict,
                                                                   data_bundle=data_bundle,
                                                                   predictor=predictor,
                                                                   predict_on='train',
                                                                   do_eval=args.do_eval), files)):
                        pbar.update()
        else:
            for file in tqdm(files):
                predict(file, data_bundle, predictor, 'train', args.do_eval)


if __name__ == '__main__':
    main()
