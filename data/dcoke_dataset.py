""" data reader for CoKE
"""
from torch.utils.data import Dataset

import six
import collections
import logging
import torch
import random
from nltk.tokenize import word_tokenize

# from .mask import prepare_batch_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

Example = collections.namedtuple("RawExample", ["token_ids", "pos_ids", "mask_label", "mask_pos"])


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.strip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


def load_desc_vocab(desc_file):
    desc_dict = dict()
    for line in open(desc_file, 'r', encoding='utf-8'):
        try:
            ent, desc = line.strip().split('\t')
        except:
            u = line.strip().split('\t')
            ent, desc = u[0], u[1:]
            desc = ' '.join(desc)
        desc_dict[ent] = desc
    return desc_dict


def load_stop_word(stop_word_file):
    stop_word = set()
    for line in open(stop_word_file, 'r', encoding='utf-8'):
        word = line.strip()
        stop_word.add(word)
    return stop_word


from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel, RobertaTokenizer, RobertaConfig


def load_relation_map(relation_alias_file):
    rel_dict = dict()
    for line in open(relation_alias_file, 'r', encoding='utf-8'):
        u = line.strip().split('\t')
        rel_num, rel_name = u[0], u[1]
        rel_dict[rel_num] = rel_name
    return rel_dict


def load_entity_map(entity_alias_file):
    entity_dict = dict()
    for line in open(entity_alias_file, 'r', encoding='utf-8'):
        u = line.strip().split('\t')
        ent_num, ent_name = u[0], u[1]
        entity_dict[ent_num] = ent_name
    return entity_dict


class KBCDataset(Dataset):
    """ DataReader
    """

    def __init__(self, data_path, desc_path, relation_file, entity_file, max_seq_len=3, vocab_size=-1):
        self.tokenizer = AutoTokenizer.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/roberta-base/")
        self.vocab = self.tokenizer.get_vocab()
        self.desc_vocab = load_desc_vocab(desc_path)
        self.relation_map = load_relation_map(relation_file)
        self.entity_map = load_entity_map(entity_file)
        self.stop_words = load_stop_word("/data/private/wanghuadong/mnt/liangshihao/wikidata5m/stop_words.txt")
        # for key in self.vocab2:
        #     print(key)
        self.sep_id = self.tokenizer.convert_tokens_to_ids("<s>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<mask>")
        # if vocab_size > 0:
        #     assert len(self.vocab) == vocab_size, \
        #         "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
        #         (vocab_size, len(self.vocab))
        self.max_seq_len = max_seq_len
        self.examples = self.read_example(data_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        token_ids, pos_ids, mask_label, mask_pos = example[0], example[1], example[2], example[3]
        return torch.tensor(token_ids, dtype=torch.long), \
               torch.tensor(pos_ids, dtype=torch.long), \
               torch.tensor(mask_label, dtype=torch.long), \
               torch.tensor(mask_pos, dtype=torch.long)

    def convert_entity_to_dedsc(self, entity):
        if entity in self.desc_vocab:
            return self.desc_vocab[entity]
        elif entity in self.relation_map:
            return self.relation_map[entity]
        else:
            print("Item {} do not have description!".format(entity))
            return entity

    def convert_tokens_to_ids(self, vocab, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        output = []
        # print(tokens)
        for item in tokens:
            tokenized = self.tokenizer(item, return_tensors="np")
            output.append(tokenized['input_ids'][0])
            output.append(self.sep_id)
            # output.append(vocab[item])
        # print(output)
        return output

    def line2tokens(self, line):
        tokens = line.split("\t")
        return tokens

    def read_example(self, input_file):
        """Reads the input file into a list of examples."""
        examples = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                line = convert_to_unicode(line.strip())
                head, rel, tail = self.line2tokens(line)
                if head in self.entity_map:
                    head_name = self.entity_map[head]
                else:
                    head_name = head
                    print("Cannot find {} in entity alias!".format(head))
                mask_type = 'MASK_HEAD'
                head_desc = self.convert_entity_to_dedsc(head)
                tail_desc = self.convert_entity_to_dedsc(tail)
                try:
                    rel_desc = self.relation_map[rel]
                except:
                    print("Cannot find {} in relation alias!".format(rel))
                    continue
                # TODO 如何在实体描述中进行mask
                # 随机选某个单词
                if mask_type == 'MASK_HEAD':
                    seg_list = word_tokenize(head_desc, language='english')
                elif mask_type == 'MASK_TAIL':
                    seg_list = word_tokenize(tail_desc, language='english')
                else:
                    seg_list = word_tokenize(rel_desc, language='english')
                # 随机mask
                # num = random.randint(0, len(seg_list)-1)
                # mask_str = seg_list[num]
                # seg_list[num] = '<mask>'
                # is are was were
                mask_str = ''
                for idx, word in enumerate(seg_list):
                    if word not in self.stop_words:
                        mask_str += (' ' + word)
                    else:
                        break
                mask_str = mask_str.strip()
                mask_ids = self.tokenizer.encode(mask_str)[1:-1]
                # print(mask_ids)
                head_ids = self.tokenizer.encode(head_desc)
                tail_ids = self.tokenizer.encode(tail_desc)
                rel_ids = self.tokenizer.encode(rel_desc)
                pos_ids = [0] * len(head_ids) + [1] * len(rel_ids) + [2] * len(tail_ids) + \
                          [3] * (self.max_seq_len - len(head_ids) - len(rel_ids) - len(tail_ids))
                mask_pos = [i for i, idx in enumerate(head_ids) if id == self.mask_id]
                head_ids.extend(rel_ids)
                head_ids.extend(tail_ids)

                if len(head_ids) > self.max_seq_len:
                    continue
                else:
                    while len(head_ids) < self.max_seq_len:
                        head_ids.append(self.pad_id)
                    while len(mask_ids) < 100:
                        mask_ids.append(-1)
                    while len(mask_pos) < 100:
                        mask_pos.append(-1)
                if len(head_ids) <= 0:
                    continue
                examples.append(
                    Example(
                        token_ids=head_ids, pos_ids=pos_ids, mask_label=mask_ids, mask_pos=mask_pos))
        return examples


class PathqueryDataset(KBCDataset):
    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 vocab_size=-1):
        KBCDataset.__init__(self, vocab_path, data_path, max_seq_len, vocab_size)

    def line2tokens(self, line):
        tokens = []
        s, path, o, mask_type = line.split("\t")
        path_tokens = path.split(",")
        tokens.append(s)
        tokens.extend(path_tokens)
        tokens.append(o)
        tokens.append(mask_type)
        return tokens


if __name__ == '__main__':
    train_dataset = KBCDataset(data_path='/data/private/wanghuadong/mnt/liangshihao/wikidata5m/test.txt',
                               desc_path='/data/private/wanghuadong/mnt/liangshihao/wikidata5m/wikidata5m_text.txt',
                               relation_file='/data/private/wanghuadong/mnt/liangshihao/wikidata5m/wikidata5m_relation.txt',
                               entity_file='/data/private/wanghuadong/mnt/liangshihao/wikidata5m/wikidata5m_entity.txt',
                               max_seq_len=1024,
                               vocab_size=16396)
