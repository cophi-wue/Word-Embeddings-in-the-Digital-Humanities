import argparse
import sys

import numpy as np
import pandas
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModel, BertTokenizer


class AllSet:

    def __contains__(self, item):
        return True


def generate_type_embeddings(embeddings, vectorizations, poolings, aggregations):
    def meannorm(vectors):
        normalized_input = torch.stack([v / torch.norm(v) for v in vectors])
        mean = torch.mean(normalized_input, axis=0)
        return mean / torch.norm(mean)

    def fractional_medoid(vectors, p):
        distances = torch.cdist(vectors, vectors, p=p).sum(axis=1)
        assert all(distances != np.inf)
        return vectors[torch.argmin(distances)]

    def embeddings_to_subword_vectors(embeddings):
        num_contexts, num_subwords, _, _ = embeddings.shape
        if 'alllayers' in vectorizations: yield 'alllayers', embeddings.reshape(num_contexts, num_subwords, 13 * 768)
        if 'layer1to4' in vectorizations: yield 'layer1to4', embeddings[:,:,[1,2,3,4],:].reshape(num_contexts, num_subwords, 4*768)
        if 'layer9to12' in vectorizations: yield 'layer9to12', embeddings[:,:,[9,10,11,12],:].reshape(num_contexts, num_subwords, 4*768)

        if 'sum' in vectorizations: yield 'sum', embeddings.sum(axis=2)
        if 'inputemb' in vectorizations: yield 'inputemb', embeddings[:, :, 0, :]
        for i in range(1, 13):
            if 'layer' + str(i) in vectorizations: yield 'layer' + str(i), embeddings[:, :, i, :]

    def pool_subword_vectors_to_token_vectors(embedding_subword_vectors):
        num_contexts, num_subwords, vector_len = embedding_subword_vectors.shape
        if 'nopooling' in poolings: yield 'nopooling', embedding_subword_vectors.reshape(num_contexts * num_subwords, vector_len)

        if 'mean' in poolings: yield 'mean', torch.mean(embedding_subword_vectors, dim=1)
        if 'median' in poolings: yield 'median', torch.quantile(embedding_subword_vectors, dim=1, q=0.5)
        if 'meannorm' in poolings: yield 'meannorm', torch.stack([meannorm(v) for v in embedding_subword_vectors])
        if 'first' in poolings: yield 'first', embedding_subword_vectors[:, 0, :]
        if 'last' in poolings: yield 'last', embedding_subword_vectors[:, -1, :]
        if 'l0.5medoid' in poolings: yield 'l0.5medoid', torch.stack([fractional_medoid(v, 0.5) for v in embedding_subword_vectors])
        if 'l1medoid' in poolings: yield 'l1medoid', torch.stack([fractional_medoid(v, 1) for v in embedding_subword_vectors])
        if 'l2medoid' in poolings: yield 'l2medoid', torch.stack([fractional_medoid(v, 2) for v in embedding_subword_vectors])

    def aggregate_token_vectors(token_vectors):
        if 'mean' in aggregations: yield 'mean', torch.mean(token_vectors, dim=0)
        if 'median' in aggregations: yield 'median', torch.quantile(token_vectors, dim=0, q=0.5)
        if 'meannorm' in aggregations: yield 'meannorm', meannorm(token_vectors)
        if 'l0.5medoid' in aggregations: yield 'l0.5medoid', fractional_medoid(token_vectors, 0.5)
        if 'l1medoid' in aggregations: yield 'l1medoid', fractional_medoid(token_vectors, 1)
        if 'l2medoid' in aggregations: yield 'l2medoid', fractional_medoid(token_vectors, 2)

    for vectorization, subword_vectors in embeddings_to_subword_vectors(embeddings):
        for pooling, token_vectors in pool_subword_vectors_to_token_vectors(subword_vectors):
            for aggregation, type_vector in aggregate_token_vectors(token_vectors):
                yield '-'.join([vectorization, pooling, aggregation]), type_vector


def forwardpass(inputs, model, tokenizer):
    for i in inputs:
        assert len(i) <= 510

    input_seq = [torch.tensor(tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(i))) for i in
                 inputs]
    input_cuda = pad_sequence(input_seq, batch_first=True).to('cuda')
    attention_tensor = (input_cuda != 0).int().to('cuda')
    outputs = model(input_cuda, attention_mask=attention_tensor, output_hidden_states=True)
    inner_layers = outputs[2]
    permuted = torch.stack(list(inner_layers)).permute(1, 2, 0, 3).detach()
    del input_cuda
    del attention_tensor
    return permuted, input_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-prefix', required=True, type=str,
                        help='File prefix for embedding output files; binary word2vec format')
    parser.add_argument('--contexts', required=True, type=argparse.FileType('rb'),
                        help='Contexts, as generated from contexts.py')
    parser.add_argument('--vectorizations', type=str,
                        help='Comma-separated list of vectorizations to perform, or \'all\'', default='all')
    parser.add_argument('--poolings', type=str, help='Comma-separated list of poolings to perform, or \'all\'',
                        default='all')
    parser.add_argument('--aggregations', type=str, help='Comma-separated list of aggregations to perform, or \'all\'',
                        default='all')
    args = parser.parse_args()
    print(args, file=sys.stderr)

    if args.vectorizations == 'all':
        vectorizations = AllSet()
    else:
        vectorizations = args.vectorizations.split(',')
    if args.poolings == 'all':
        poolings = AllSet()
    else:
        poolings = args.poolings.split(',')
    if args.aggregations == 'all':
        aggregations = AllSet()
    else:
        aggregations = args.aggregations.split(',')

    print(vectorizations, poolings, aggregations, file=sys.stderr, flush=True)

    tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
    tokenizer.do_basic_tokenize = False

    model = AutoModel.from_pretrained("deepset/gbert-base")
    model.eval()
    model.to('cuda')
    torch.no_grad()
    torch.set_grad_enabled(False)

    print("loading input", file=sys.stderr, flush=True)
    input_contexts = pandas.read_csv(args.contexts, sep='\t', index_col=False, quoting=3)

    vocab = list(input_contexts['token'].unique())
    embeddings_files = dict()

    with tqdm(total=len(input_contexts)) as pbar:
        for focus_word, contexts in input_contexts.groupby('token'):
            focus_len = contexts.iloc[0]['focus_len']

            context_embeddings = torch.empty((len(contexts), focus_len, 13, 768)).cuda()
            counter = 0

            for chunk in [contexts.iloc[i:i + 20] for i in range(0, len(contexts), 20)]:
                pieces = [x.split(' ') for x in chunk['context']]
                focus_index = chunk['focus_index'].values

                output, input_seq = forwardpass(pieces, model, tokenizer)
                for j in range(len(chunk)):
                    # adjust for added start token in sequence
                    sl = slice(focus_index[j] + 1, focus_index[j] + 1 + focus_len)
                    # assert tokenizer.convert_ids_to_tokens(list(input_seq[j][sl]))\
                    #   == pieces[j][focus_index[j]:focus_index[j] + focus_len]
                    # assert tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(list(input_seq[j][sl])))\
                    #   == focus_word
                    context_embeddings[counter] = output[j][sl].detach()
                    counter += 1

            for embedding_name, type_vector in generate_type_embeddings(context_embeddings,
                                                                        vectorizations=vectorizations,
                                                                        poolings=poolings, aggregations=aggregations):
                if embedding_name not in embeddings_files.keys():
                    embeddings_files[embedding_name] = open(args.output_prefix + embedding_name + '.bin', 'wb')
                    embeddings_files[embedding_name].write(f"{len(vocab)} {type_vector.shape[0]}\n".encode('utf8'))
                embeddings_files[embedding_name].write(f"{focus_word} ".encode('utf8') + np.array(type_vector.cpu()).astype(np.float32).tobytes())

            del context_embeddings
            pbar.update(len(contexts))

    for ef in embeddings_files.values():
        ef.close()
