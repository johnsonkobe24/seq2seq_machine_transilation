import argparse
import time
import torch
from typing import *
from util import load_data, initialize_seq2seq_params, build_seq2seq_model
from time import time
from torch import optim
from core import Seq2SeqModel, encode_all, SOS_token, EOS_token
from math import exp


def decode(prev_hidden: torch.tensor, input: int, model: Seq2SeqModel) -> (torch.tensor, torch.tensor):
    """ Run the decoder AND the output layer for a single step.

    (This function will be used in both log_likelihood and translate_greedy_search.)

    :param prev_hidden: tensor of shape [L, hidden_dim] - the decoder's previous hidden state, denoted H^{dec}_{t-1}
                        in the assignment
    :param input: int - the word being inputted to the decoder.
                            during log-likelihood computation, this is y_{t-1}
                            during greedy decoding, this is yhat_{t-1}
    :param model: a Seq2Seq model
    :return: (1) a tensor `probs` of shape [target_vocab_size], denoted p(y_t | x_1 ... x_S, y_1 .. y_{t-1})
             (2) a tensor `hidden` of shape [L, hidden_dim], denoted H^{dec}_t in the assignment
    """
    h=model.decoder_gru.forward(model.target_embedding_matrix[input],prev_hidden)
    return model.output_layer.forward(h[-1]), h

def log_likelihood(source_sentence: List[int], target_sentence: List[int], model: Seq2SeqModel) -> torch.Tensor:
    """ Compute the log-likelihood for a (source_sentence, target_sentence) pair.

    :param source_sentence: the source sentence, as a list of words
    :param target_sentence: the target sentence, as a list of words
    :return: conditional log-likelihood of the (source_sentence, target_sentence) pair
    """
    res=0
    prev_h=encode_all(source_sentence, model)[-1]
    for i in range(len(target_sentence)):
        if i==0:
            pd, prev_h=decode(prev_h, SOS_token,model)
            p=pd[target_sentence[i]]
        else:
            pd, prev_h=decode(prev_h, target_sentence[i-1],model)
            p=pd[target_sentence[i]]
        res+=torch.log(p)
    return res


def translate_greedy_search(source_sentence: List[int], model: Seq2SeqModel, max_length=10) -> List[int]:
    """ Translate a source sentence using greedy decoding.

    :param source_sentence: the source sentence, as a list of words
    :param max_length: the maximum length that the target sentence could be
    :return: the translated sentence as a list of words
    """

    res=[]
    prev_h=encode_all(source_sentence,model)[-1]
    for i in range(max_length):
        if i==0:
            pd, prev_h=decode(prev_h, SOS_token,model)
            y_hat=torch.argmax(pd)
            res.append(y_hat.item())
        else:
            pd, prev_h=decode(prev_h, y_hat,model)
            y_hat=torch.argmax(pd)
            res.append(y_hat.item())
        if y_hat==EOS_token:
            break
    print(res)
    return res


def perplexity(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel):
    """ Compute the perplexity of an entire dataset under a seq2seq model.  Refer to the write-up for the
    definition of perplexity.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: seq2seq model
    :return: perplexity of the dataset
    """
    LL=0
    T=0
    for i in range(len(sentences)):
        LL+=log_likelihood(sentences[i][0],sentences[i][1],model)
        T+=len(sentences[i][1])
    return torch.exp(-LL/T).detach()


def train_epoch(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel,
                epoch: int, print_every: int = 100, learning_rate: float = 0.0001, gradient_clip=5):
    """ Train the model for an epoch.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: a Seq2Seq model
    :param epoch: which epoch we're at
    """
    print("epoch\titer\tavg loss\telapsed secs")
    total_loss = 0
    start_time = time()
    optimizer = optim.Adam(model_params.values(), lr=learning_rate)
    for i, (source_sentence, target_sentence) in enumerate(sentences):
        optimizer.zero_grad()
        theloss = -log_likelihood(source_sentence, target_sentence, model)
        total_loss += theloss
        theloss.backward()

        torch.nn.utils.clip_grad_norm_(model_params.values(), gradient_clip)

        optimizer.step()

        if i % print_every == 0:
            avg_loss = total_loss / print_every
            total_loss = 0
            elapsed_secs = time() - start_time
            print("{}\t{}\t{:.3f}\t{:.3f}".format(epoch, i, avg_loss, elapsed_secs))

    return model_params


def print_translations(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel,
                       source_vocab: Dict[int, str], target_vocab: Dict[int, str]):
    """ Iterate through a dataset, printing (1) the source sentence, (2) the actual target sentence, and (3)
    the translation according to our model.

    :param sentences: a list of (source sentence, target sentence) pairs
    :param model: a Seq2Seq model
    :param source_vocab: the mapping from word index to word string, in the source language
    :param target_vocab: the mapping from word index to word string, in the target language
    """
    for (source_sentence, target_sentence) in sentences:
        translation = translate_greedy_search(source_sentence, model)

        print("source sentence:" + " ".join([source_vocab[word] for word in source_sentence]))
        print("target sentence:" + " ".join([target_vocab[word] for word in target_sentence]))
        print("translation:\t" + " ".join([target_vocab[word] for word in translation]))
        print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Seq2Seq Homework Assignment')
    parser.add_argument("action", type=str,
                        choices=["train", "finetune", "train_perplexity", "test_perplexity",
                                 "print_train_translations", "print_test_translations"])
    parser.add_argument("--load_model", type=str,
                        help="path to saved model on disk.  if this arg is unset, the weights are initialized randomly")
    parser.add_argument("--save_model_prefix", type=str, help="prefix to save model with, if you're training")
    args = parser.parse_args()

    # load train/test data, and source/target vocabularies
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    # load model weights (if path is specified) or else initialize weights randomly
    model_params = initialize_seq2seq_params() if args.load_model is None \
        else torch.load(args.load_model)  # type: Dict[str, torch.Tensor]

    # build a Seq2SeqModel object
    model = build_seq2seq_model(model_params)  # type: Seq2SeqModel

    if args.action == 'train':
        for epoch in range(10):
            train_epoch(train_sentences, model, epoch)
            torch.save(model_params, '{}_{}.pth'.format(args.save_model_prefix, epoch))
    elif args.action == 'finetune':
        train_epoch(train_sentences[:1000], model, 0, learning_rate=1e-5)
        torch.save(model_params, "{}.pth".format(args.save_model_prefix))
    elif args.action == "print_train_translations":
        print_translations(train_sentences, model, source_vocab, target_vocab)
    elif args.action == "print_test_translations":
        print_translations(test_sentences, model, source_vocab, target_vocab)
    elif args.action == "train_perplexity":
        print("perplexity: {}".format(perplexity(train_sentences[:1000], model)))
    elif args.action == "test_perplexity":
        print("perplexity: {}".format(perplexity(test_sentences, model)))
