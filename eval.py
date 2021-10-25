import os
import re
import json
import shutil

from tqdm import tqdm
from konlpy.tag import Okt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from PIL import ImageFile

pass_config = {
    "text": {
        "vocab_pass": "text_vocab_1.json",
        "cp_pass": "resnext_lb_text_1.pth"},
    "hashtag": {
        "vocab_pass": "hashtag_vocab_1.json",
        "cp_pass": "resnext_lb_hashtag_1.pth"}
}

picy_dir = '/content/drive/MyDrive/gh/picy_lab/'

# 이미지 파일 복사후 이름변경, 경로에 이동
file_list = os.listdir(picy_dir + 'pics')
for idx, img in enumerate(file_list):
    os.rename(
        picy_dir + f"pics/{img}",
        picy_dir + f"pics/{idx}.{img.split('.')[-1]}")

# 이미지가 들어오면 input 에 맞는 json 파일 생성
file_list = os.listdir(picy_dir + 'pics')
forms = []
for f in file_list:
    form = {
        'hashtag': ['#서울'],
        'image_path': f,
        'text': '테스트'
    }
    forms.append(form)

with open(picy_dir + 'test.json', 'w') as outfile:
    json.dump(forms, outfile)

for img in file_list:
    shutil.copy(
        picy_dir + f"pics/{img}",
        picy_dir)

text_list = []
hashtag_list = []

# dataloader

okt = Okt()


def tokenize_fn(text):
    tokens = okt.pos(text, norm=True, join=True)
    return tokens


ImageFile.LOAD_TRUNCATED_IMAGES = True

target_size = 224
default_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PICYDataset(Dataset):
    def __init__(self, json_file, vocab, transform=default_transform, type='hashtag', tokenize_fn=None,
                 load_on_ram=False):
        assert any(json_file.endswith(file_type) for file_type in [
                   'total.json', 'train.json', 'val.json', 'test.json'])
        assert type in ['hashtag', 'text']
        assert '<UNK>' in vocab
        assert '<start>' in vocab
        assert '<end>' in vocab

        if type == 'text':
            assert tokenize_fn is not None
        self.vocab = vocab

        self.root_dir = '/'.join(json_file.split('/')[:-1])
        self.type = type
        self.vocab = vocab

        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self.load_on_ram = load_on_ram

        with open(json_file, encoding='utf8') as fr:
            # remove empty data
            d = json.load(fr)
            self.json = []
            for item in d:
                if type not in item:
                    continue
                if type == 'text':
                    # check vocab
                    target = [token for token in tokenize_fn(
                        item['text']) if token in self.vocab]
                else:
                    target = [t for t in item['hashtag'] if t in self.vocab]
                if target != []:
                    # check max_len
                    if type == 'text':
                        if len(target) > Config.max_text_length:
                            continue
                    elif type == 'hashtag':
                        if len(target) > Config.max_hashtag_length:
                            continue
                    self.json.append(item)

        # create target
        self.targets = []
        self.tokens = []
        for item in self.json:
            if type == 'text':
                text = item['text']
                tokens = self.tokenize_fn(text)
                self.tokens.append(tokens)
                UNK_idx = self.vocab['<UNK>']
                target = [self.vocab.get(token, UNK_idx) for token in tokens]
            elif type == 'hashtag':
                # check hashtag in vocab
                hashtags = item['hashtag']
                self.tokens.append(hashtags)
                target = [self.vocab.get(hashtag)
                          for hashtag in hashtags if hashtag in self.vocab]
            self.targets.append(target)

        # load images
        if self.load_on_ram:
            self.images = []
            for item in self.json:
                image = Image.open(os.path.join(
                    self.root_dir, item['image_path']))
                if self.transform is not None:
                    image = self.transform(image)
                self.images.append(image)

    def __getitem__(self, index):
        item = self.json[index]

        # load target = hashtag or text
        target = self.targets[index]
        target = [self.vocab.get('<start>')] + target + \
            [self.vocab.get('<end>')]
        target = torch.LongTensor(target)

        # load image
        if self.load_on_ram:
            image = self.images[index]
        else:
            image = Image.open(os.path.join(self.root_dir, item['image_path']))
            if self.transform is not None:
                image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.json)


def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, targets = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge targets (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in targets]
    padded_targets = torch.zeros(len(targets), max(lengths)).long()
    for i, cap in enumerate(targets):
        end = lengths[i]
        padded_targets[i, :end] = cap[:end]
    return images, padded_targets, lengths


def get_dataloader(json_file, vocab, transform=default_transform, type='hashtag', tokenize_fn=None,
                   batch_size=32, shuffle=True, num_workers=-1, load_on_ram=False):

    dataset = PICYDataset(json_file, vocab, transform=transform,
                          type=type, tokenize_fn=tokenize_fn, load_on_ram=load_on_ram)

    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return loader


# model

class ResNextEncoder(nn.Module):
    def __init__(self, encoded_size=14, encoder_finetune=False):
        super().__init__()

        self.encoded_size = encoded_size

        #resnext_wsl = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        #resnext_wsl = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        resnext_wsl = torch.hub.load(
            'pytorch/vision:v0.8.0', 'resnext101_32x8d', pretrained=True)

        layers_to_use = list(resnext_wsl.children())[:-3]

        self.conv_net = nn.Sequential(*layers_to_use)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        self.encoder_finetune = encoder_finetune
        if not encoder_finetune:
            self.conv_net.eval()

    def forward(self, images):
        if not self.encoder_finetune:
            with torch.no_grad():
                x = self.conv_net(images)
        else:
            # (batch_size, encoder_dim, image_size/32, image_size/32
            x = self.conv_net(images)
        # (batch_size, encoder_dim, self.encoded_size, self.encoded_size
        x = self.adaptive_pool(x)
        # (batch_size, self.encoded_size, self.encoded_size, encoder_dim
        x = x.permute(0, 2, 3, 1)
        # since shape (batch_size, self.encoded_size ** 2, encoder_dim) will be used in decoder, do permutation

        batch_size = x.shape[0]
        encoder_dim = x.shape[-1]
        x = x.view(batch_size, -1, encoder_dim)  # (batch_size, L, D)
        # each point l in encoded image has vector with D-dim that represents that point
        # self.encoded_size ** 2 will be L and encoder_dim will be D in original paper's notation
        return x


class LookBackAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_context_vector_attention = nn.Linear(
            decoder_dim + encoder_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_output, decoder_hidden, context_vector):
        encoder_att = self.encoder_attention(
            encoder_output)  # (batch_size, L, attention_dim)
        decoder_context_vector_att = self.decoder_context_vector_attention(torch.cat(
            [decoder_hidden, context_vector], dim=1))  # (batch_size, attention_dim)
        encoder_plus_decoder_att = encoder_att + \
            decoder_context_vector_att.unsqueeze(
                1)  # (batch_size, L, attention_dim)
        attention = self.attention(
            F.relu(encoder_plus_decoder_att)).squeeze(2)  # (batch_size, L)
        alpha = self.softmax(attention)  # (batch_size, L)
        context_vector = (encoder_output * alpha.unsqueeze(2)
                          ).sum(dim=1)  # (batch_size, encoder_dim)
        # sum(dim=1) means summing over L
        # context vector is z_hat in original paper, calculated from h_t-1, and encoder output a

        return context_vector, alpha


class LookBackDecoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, embed_dim, vocab_size,
                 dropout=0.5, embedding_finetune=True):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout

        self.look_back_attention = LookBackAttention(
            encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTMCell(
            self.encoder_dim + self.embed_dim, self.decoder_dim, bias=True)

        self.init_h = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.init_c = nn.Linear(encoder_dim, decoder_dim, bias=False)

        # deep output layers
        self.L_h = nn.Linear(decoder_dim, embed_dim, bias=False)
        self.L_z = nn.Linear(encoder_dim, embed_dim, bias=False)
        self.L_o = nn.Linear(embed_dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        if not embedding_finetune:
            # always set embedding_finetune == True when not using pretrained embeddings
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.embedding_finetune = embedding_finetune

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def load_embedding(self, embedding):
        self.embedding.from_pretrained(
            embedding, freeze=not self.embedding_finetune)

    def init_hidden_states(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1)  # mean over L
        init_h = self.init_h(mean_encoder_output)
        init_c = self.init_c(mean_encoder_output)

        return init_h, init_c

    def deep_output_layer(self, embedded_caption, h, context_vector):
        scores = self.L_o(self.dropout(embedded_caption +
                          self.L_h(h) + self.L_z(context_vector)))
        return scores

    def forward(self, encoder_output, captions):
        batch_size = encoder_output.shape[0]
        num_pixels = encoder_output.shape[1]
        max_caption_length = captions.shape[-1]

        predictions = torch.zeros(
            batch_size, max_caption_length - 1, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max_caption_length -
                             1, num_pixels)  # save attention

        # (batch_size, max_caption_length, embed_dim)
        embedded_captions = self.embedding(captions)

        h, c = self.init_hidden_states(encoder_output)

        # initial context vector
        context_vector = encoder_output.mean(dim=1)
        # don't need prediction when y_t-1 is <end>
        for t in range(max_caption_length - 1):
            # (batch_size, embed_dim)
            embedded_caption_t = embedded_captions[:, t, :]
            context_vector, alpha = self.look_back_attention(
                encoder_output, h, context_vector)
            # context vector has size (batch_size, encoder_dim)
            h, c = self.lstm(torch.cat([embedded_caption_t, context_vector], dim=1),  # lstm input has shape (batch_size, embed_dim + encoder_dim)
                             (h, c))
            preds = self.deep_output_layer(
                embedded_caption_t, h, context_vector)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas

    def generate_caption_greedily(self, encoder_output, start_token, end_token):
        #
        self.eval()
        h, c = self.init_hidden_states(encoder_output)
        captions = [start_token]
        alphas = []

        # initial context vector
        context_vector = encoder_output.mean(dim=1)
        with torch.no_grad():
            while captions[-1] != end_token and len(captions) < 30:  # 1 is '.'
                caption = captions[-1]
                embedded_caption = self.embedding(torch.LongTensor(
                    [caption]).to(self.device))  # (1, embed_dim)
                context_vector, alpha = self.look_back_attention(
                    encoder_output, h, context_vector)  # (1, encoder_dim)
                h, c = self.lstm(torch.cat([embedded_caption, context_vector], dim=1),
                                 (h, c))
                preds = self.deep_output_layer(
                    embedded_caption, h, context_vector)  # (1, vocab_size)
                next_word = int(torch.argmax(
                    preds, dim=1, keepdim=True).squeeze())
                captions.append(next_word)
                alphas.append(alpha)

        self.train()

        return captions, alphas


def test_text(encoder, decoder, test_dataloader, vocab):
    actuals = test_dataloader.dataset.tokens
    preds = []
    idx2word = dict([(v, k) for k, v in vocab.items()])

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for ix, (image, target, length) in tqdm(enumerate(test_dataloader)):
            image = image.to(Config.device)
            target = target.to(Config.device)

            encoded_image = encoder(image)
            prediction, alphas = decoder.generate_caption_greedily(encoded_image,
                                                                   test_dataloader.dataset.vocab['<start>'],
                                                                   test_dataloader.dataset.vocab['<end>'])

            target = target[0, 1:-1]
            target = target.tolist()

            prediction = prediction[1:-1]
            prediction = [idx2word[idx] for idx in prediction]

            actuals.append(target)
            preds.append(prediction)

            # 조건에 따라 이어붙이기.
            # 함수화 후 util로 옮기기
            print()
            my_list = []
            for i in preds[-1]:
                my_list.extend(re.compile('[가-힣]+').findall(i))
            s = "".join(my_list)
            print(s)
            ###
            text_list.append(s)


def test_hashtag(encoder, decoder, test_dataloader, vocab):
    #actuals = test_dataloader.dataset.tokens
    preds = []
    idx2word = dict([(v, k) for k, v in vocab.items()])

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for ix, (image, target, length) in tqdm(enumerate(test_dataloader)):
            image = image.to(Config.device)
            target = target.to(Config.device)

            encoded_image = encoder(image)
            prediction, alphas = decoder.generate_caption_greedily(encoded_image,
                                                                   test_dataloader.dataset.vocab['<start>'],
                                                                   test_dataloader.dataset.vocab['<end>'])

            target = target[0, 1:-1]
            target = target.tolist()

            prediction = prediction[1:-1]
            prediction = [idx2word[idx] for idx in prediction]

            preds.append(prediction)
            print()
            print(preds[-1])
            hashtag_list.append(preds[-1])

###
# config


for k, v in pass_config.items():
    vocab_pass = picy_dir + f'vocabs/{v["vocab_pass"]}'
    cp_pass = picy_dir + f'checkpoint/{v["cp_pass"]}'

    JSON_FILES = {
        'test': picy_dir + 'test.json',
    }

    class args:
        model = 'resnext_lb'
        mode = 'test'
        target_type = f'{k}'
        vocab_file = vocab_pass
        checkpoint_load_path = cp_pass
        load_image_on_ram = True

    class Config:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # encoder
        encoded_size = 14
        encoder_finetune = False
        # decoder
        encoder_dim = 1024
        decoder_dim = 512
        attention_dim = 512
        dropout = 0.5
        embed_dim = 400
        embedding_finetune = True

        lr = 5e-4

        # dataloader
        batch_size = 30
        num_workers = 4
        max_text_length = 30
        max_hashtag_length = 25

    ###
    # Do test
    assert args.vocab_file is not None
    assert args.checkpoint_load_path is not None

    print("Loading vocab...")
    with open(args.vocab_file) as fr:
        vocab = json.load(fr)

    print("Loading model...")
    print(args.model)

    if args.model == 'resnext_lb':
        encoder = ResNextEncoder(Config.encoded_size)
        decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim,
                                  Config.attention_dim, Config.embed_dim, len(vocab))

    encoder = encoder.to(Config.device)
    decoder = decoder.to(Config.device)

    def load_model(encoder, decoder, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    load_model(encoder, decoder, args.checkpoint_load_path)

    encoder.eval()
    decoder.eval()

    test_dataloader = get_dataloader(JSON_FILES['test'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                     batch_size=1, num_workers=Config.num_workers, load_on_ram=args.load_image_on_ram,
                                     shuffle=False)

    print("Running test...")
    if args.target_type == 'hashtag':
        test_hashtag(encoder, decoder, test_dataloader, vocab)
    elif args.target_type == 'text':
        test_text(encoder, decoder, test_dataloader, vocab)


# 아래의 변수 활용
print(hashtag_list)
print(text_list)
