# Copyright \u00a9 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8

import sys
import argparse
import time
import re
import string
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForPreTraining, AutoConfig

class TextDataset(Dataset):

    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding='max_length')
        return inputs

class TextCollator:

    def __init__(self, tokenizer, padding, device):
        self.tokenizer = tokenizer
        self.padding = True
        self.max_length = None
        self.device = device

    def __call__(self, features):
        batch_size = len(features)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, -1).to(self.device) for k, v in batch.items()}
        return batch

class TextCleaner(object):

    def __init__(self, model, tokenizer, args):

        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model is not None and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = None
        self.batch_size = args.batchsize
        self.device = model.device if model is not None else 'cpu'

    def create_data_loader(self, texts, batch_size=None):
        if batch_size is not None: self.batch_size = batch_size
        collate_fn = TextCollator(self.tokenizer, padding=True, device=self.device)
        dataset = TextDataset(texts, self.tokenizer)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def compute_batch_log_perplexity(self, texts=None):

        if texts is not None:
            self.create_data_loader(texts, min(len(texts), self.batch_size))

        log_perplexity = []
        for batch in self.data_loader:

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            with torch.no_grad():

                outputs = self.model(**batch, labels=input_ids, output_hidden_states=True)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                flatten_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                flatten_shift_labels = shift_labels.view(-1)

                assert input_ids.max() < self.tokenizer.vocab_size, "Found an input_id greater than vocab_size"
                assert input_ids.min() >= 0, "Found an input_id less than 0"

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id, reduction='none')
                losses = loss_fct(flatten_shift_logits, flatten_shift_labels)
                losses = losses.reshape(logits.shape[0], -1)
                sentence_losses = losses.sum(dim=-1) / attention_mask.sum(dim=-1)

                log_perplexity.extend(sentence_losses.tolist())

        return log_perplexity

    @staticmethod
    def punctuation_percentage(sentence, discarded=None, additional=None):
        english_punctuation = set(string.punctuation)
        chinese_punctuation = set(chr(i) for i in range(0x3000, 0x303F+1))
        special_symbols = {
            # Currency symbols: Dollar, Euro, Pound, Yen, Cent, Indian Rupee
            "$", "\u20ac", "\u00a3", "\u00a5", "\u00a2", "\u20b9",
            # Mathematical symbols
            "=", "+", "-", "\u00d7", "\u00f7", "%", "<", ">", "\u2264", "\u2265", "\u2260", "\u221e",
            # Other common symbols
            "@", "#", "&", "*", "~", "|", ":", ";", "?", "!", "^", "_",
        }
        additional = [] if additional is None else additional
        combined_set = set().union(*[english_punctuation, chinese_punctuation, special_symbols, additional])
        if discarded is not None:
            for ch in discarded:
                combined_set.discard(ch)
        punctuation_count = sum(1 for char in sentence if char in combined_set)
        total_characters = len(sentence)
        percentage = (punctuation_count / total_characters) if total_characters > 0 else 0
        return percentage

    @staticmethod
    def de_whitespace(sentence):
        sentence = re.sub(r'\n', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence.lstrip()

    @staticmethod
    def de_html(sentence):
        sentence = re.sub(r'<a\s+href=[^>]+>(.*?)</a>', ' ', sentence)
        sentence = re.sub(r'([\u4e00-\u9fffA-Za-z0-9:]+\|)+[\u4e00-\u9fffA-Za-z0-9:]+', ' ', sentence)
        sentence = re.sub(r'<\/?(?:div|br|p|span|center|hr|MARQUEE|a|A)\s*\/?>', ' ', sentence)
        # quoted_string = re.findall(r'\"(.*?)\"', sentence)
        # if len(quoted_string) != 0:
        #     sentence = re.sub(r'<!DOCTYPE.*?-->', quoted_string[0], sentence)
        sentence = re.sub(r'(?:!DOCTYPE|\uff01DOCTYPE\uff0c|DOCTYPE)', ' ', sentence)
        return sentence

    @staticmethod
    def de_url(sentence):
        sentence = re.sub(r'https?://\S{30,}', ' ', sentence)
        sentence = re.sub(r'ftp://\S{15,}', ' ', sentence)
        sentence = re.sub(r'file://\S{15,}', ' ', sentence)
        return sentence

    @staticmethod
    def de_url_all(sentence):
        sentence = re.sub(r'(ftp|https?|file)?://(?:[\u4e00-\u9fffa-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', sentence)
        sentence = re.sub(r'www\.(?:[\u4e00-\u9fffa-zA-Z\uff21-\uff3a]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', sentence)
        sentence = re.sub(r'\b[\w%]+=([\w%]+&)*[\w%]+\b', ' ', sentence)
        return sentence

    @staticmethod
    def de_long_strings(sentence):
        # sentence = re.sub(r'\b([^\W-]{0,}[^\w\s]{3,}[^\W-]{0,}|[^a-zA-Z0-9_-]{50,})\b', ' ', sentence)
        # sentence = re.sub(r'\b\w[\w\-.:/#]{39,}\b', ' ', sentence)
        sentence = re.sub(r'\b(\w*[-.:/#]\w*){4,}\b', ' ', sentence)
        return sentence

    @staticmethod
    def de_punctuations(sentence):
        sentence = re.sub(r'[^\w\s.,"]{3}', ' ', sentence)
        # sentence = re.sub(r'(\s[,\u201c:0-9r]+\s)+', ' ', sentence)
        sentence = re.sub(r'(\s[\u254b\u2501\u2505\u3010\u3011\u300e \u300f\uff0c\u3002\uff01,;.''\s]+\s)+', ' ', sentence)
        return sentence

    @staticmethod
    def de_baike(sentence):
        sentence = re.sub(r'<!(ENTITY|SHORTREF|USEMAP).*?>', ' ', sentence)
        sentence = re.sub(r'<\s*(NITFPUBLIC|[\w-]+\s*PUBLIC).*?>', ' ', sentence)
        sentence = re.sub(r'\u5df2\u6295\u7968\s\d+', ' ', sentence)
        sentence = re.sub(r'\u6536\u85cf \u67e5\u770b\u6211\u7684\u6536\u85cf\s\d+', ' ', sentence)
        sentence = re.sub(r'\u6709\u7528\+\d+', ' ', sentence)
        sentence = re.sub(r'\u5c55\u5f00\u5168\u90e8|\u6536\u8d77', ' ', sentence)
        return sentence

    @staticmethod
    def de_zh_books(sentence):
        sentence = re.sub(r'<MARQUEE*?>', ' ', sentence)
        sentence = re.sub(r'<A.*?/A>', ' ', sentence)
        sentence = re.sub(r'<.{,100}?>', ' ', sentence)
        sentence = re.sub(r'\u3010.{,100}?\u3011', ' ', sentence)
        sentence = re.sub(r'@.{,100}?@', ' ', sentence)

        sina = [
            '\u9996\u9875 \u6587\u53f2 \u56fe\u4e66\u8fde\u8f7d \u539f\u521b\u6587\u5b66 VIP\u5168\u672c \u5bfc\u822a \u8bfb\u4e66\u98ce\u4e91\u699c\u5386\u53f2\u9986\u8d22\u7ecf\u9986\u5c0f\u8bf4\u9986\u9752\u6625\u9986\u60c5\u611f\u9986\u5973\u4eba\u9986\u751f\u6d3b\u9986|\u6eda\u52a8\u65b0\u95fb\u8d44\u8baf\u4e66\u6458\u4e66\u8bc4\u56fe\u5e93\u540d\u4eba\u5802\u535a\u5ba2\u8bba\u575b\u7535\u5b50\u6742\u5fd7\u4e13\u9898 \u65b0\u77e5\u65e7\u95fb \u5386\u53f2\u89e3\u5bc6 \u8001\u7167\u7247 \u4eba\u7269 \u6545\u4e8b\u65b0\u7f16 \u767e\u59d3\u8bb2\u8ff0 \u8fa3\u8bc4 \u5e7d\u9ed8\u7b11\u8bdd | \u6587\u5316\u535a\u5ba2 \u56fe\u5e93 \u4eba\u6587\u4e66\u5e93 \u6587\u5316\u8bba\u575b \u5927\u8bdd\u6625\u79cb \u539f\u521b\u57fa\u5730 \u56fe\u4e66\u8fde\u8f7d\u70b9\u51fb\u699c |\u90fd\u5e02\u5c0f\u8bf4\u5b98\u573a\u5546\u6218\u5c0f\u8bf4\u519b\u4e8b\u5386\u53f2\u5c0f\u8bf4\u60ac\u7591\u5947\u5e7b\u5c0f\u8bf4\u8a00\u60c5\u5c0f\u8bf4\u6821\u56ed\u9752\u6625\u5c0f\u8bf4\u4f20\u8bb0\u5386\u53f2\u5730\u7406\u91d1\u878d\u6295\u8d44\u5a31\u4e50\u65f6\u5c1a\u7ecf\u6d4e\u4e0e\u7ba1\u7406 \u52b1\u5fd7\u6210\u529f \u539f\u521b\u70b9\u51fb\u699c | \u5947\u5e7b\u6b66\u4fa0\u90fd\u5e02\u60c5\u611f\u519b\u4e8b\u5386\u53f2\u9752\u6625\u8a00\u60c5\u6e38\u620f\u7ade\u6280\u6050\u6016\u7075\u5f02\u79d1\u5e7b\u5c0f\u8bf4\u7f8e\u6587\u5176\u4ed6 | \u4e66\u5e93\u5927\u5168\u6295\u7a3f\u5e2e\u52a9\u624b\u518c\u95ee\u9898\u7b54\u7591\u4ea4\u6d41\u8bba\u575b\u5145\u503c\u6210vip \u6536\u8d39\u4e66\u5e93 \u540d\u5bb6\u7cbe\u54c1 V\u5468\u520a|\u5145\u503c\u6210\u4e3aVIP \u4f1a\u5458\u7279\u6743 \u65b0\u624b\u5165\u95e8 \u4f5c\u8005\u624b\u518c \u5e38\u89c1\u95ee\u9898 \u5728\u7ebf\u7b54\u7591 \u4f7f\u7528\u672c\u529f\u80fd\u9700\u8981\u767b\u5f55 \u7533\u8bf7\u6210\u4e3a\u65b0\u6d6a\u8bfb\u4e66\u8bfb\u8005\uff0c\u4eab\u53d7\u5c0a\u8d35\u65e0\u4e0a\u7684\u670d\u52a1 \u8d2d\u4e70\u6536\u8d39\u7ae0\u8282\uff0c\u9605\u8bfb\u66f4\u591a\u7cbe\u5f69\u5185\u5bb9 \u7ed9\u4f5c\u54c1\u6295\u7968\uff0c\u4e3a\u559c\u7231\u7684\u4f5c\u5bb6\u52a0\u6cb9 \u6536\u85cf\u4f5c\u54c1\uff0c\u62e5\u6709\u81ea\u5df1\u7684\u7f51\u4e0a\u85cf\u4e66\u9601 \u2014\u2014 \u4f53\u9a8c\u9605\u8bfb\u5e26\u6765\u7684\u5feb\u4e50',
            '\u8bfb\u4e66\u653b\u7565 \u4e2a\u4eba\u4e66\u67b6 \u63a8\u8350\u7968 \u7ad9\u5185\u77ed\u4fe1\u606f \u79ef\u5206\u83b7\u53d6 \u804c\u4e1a\u664b\u7ea7 \u7533\u8bf7\u4f5c\u5bb6 \u4f5c\u8005\u653b\u7565 \u53d1\u8868\u4f5c\u54c1 \u4f5c\u54c1\u7ba1\u7406 \u7ad9\u5185\u77ed\u4fe1\u606f \u7533\u8bf7\u7b7e\u7ea6 \u7a3f\u916c\u7ed3\u7b97 \u65b0\u6d6a\u7528\u6237\u8bf7\u76f4\u63a5\u767b\u5f55 \u7528\u6237\u540d\uff1a \u5bc6 \u7801\uff1a \u5fd8\u8bb0\u5bc6\u7801 \u6ca1\u6709\u65b0\u6d6a\u5e10\u53f7\uff1f \u9a6c\u4e0a\u6ce8\u518c\u65b0\u6d6a\u901a\u884c\u8bc1 \u4ec0\u4e48\u662f\u65b0\u6d6a\u901a\u884c\u8bc1? \u5e10\u53f7\u6ce8\u518c\u6210\u529f\u540e\uff0c\u4f5c\u8005\u7b14\u540d\u662f\u5426\u53ef\u4ee5\u4fee\u6539\uff1f \u5982\u4f55\u627e\u56de\u6211\u7684\u5e10\u53f7\u5bc6\u7801\uff1f \u8fdb\u5165\u7528\u6237\u624b\u518c>> \u65b0\u6d6a\u7f51\u8bfb\u4e66\u9891\u9053\u7f51\u53cb\u610f\u89c1 \u7f16\u8f91\u90e8\u7535\u8bdd\uff1a \u5ba2\u670d\u7535\u8bdd\uff1a \u6b22\u8fce\u6279\u8bc4\u6307\u6b63 \u65b0\u6d6a\u7b80\u4ecb | About Sina | \u5e7f\u544a\u670d\u52a1 | \u8054\u7cfb\u6211\u4eec | \u62db\u8058\u4fe1\u606f | \u7f51\u7ad9\u5f8b\u5e08 | SINA English | \u7528\u6237\u6ce8\u518c | \u4ea7\u54c1\u7b54\u7591 Copyright \u00a9 1996 - 2011 SINA Corporation, All Rights Reserved \u65b0\u6d6a\u516c\u53f8 \u7248\u6743\u6240\u6709 ',
            '\u66f4\u591a\u514d\u8d39txt\u7535\u5b50\u4e66\uff0c\u6b22\u8fce\u60a8\u5230\u4e0b\u8f7d \u58f0\u660e\uff1a\u672c\u7535\u5b50\u4e66\u4ec5\u4f9b\u8bfb\u8005\u9884\u89c8,\u8bf7\u5728\u4e0b\u8f7d24\u5c0f\u65f6\u5185\u5220\u9664\uff0c\u4e0d\u5f97\u7528\u4f5c\u5546\u4e1a\u7528\u9014\uff1b\u5982\u679c\u559c\u6b22\u8bf7\u8d2d\u4e70\u6b63\u7248\u56fe\u4e66\uff01'
        ]
        for text in sina:
            try: sentence = re.sub(re.escape(text), ' ', sentence)
            except: continue

        templates = [
            r'\*\* \u4f5c\u8005\uff1a(\S+)\u6240\u5199\u7684(.*?)\u4e3a\u8f6c\u8f7d\u4f5c\u54c1\uff0c\u6536\u96c6\u4e8e(.*?)',
            r'\*\* \u5982\u679c\u60a8\u662f(.*?)\u4f5c\u54c1\u7684\u7248\u6743\u6240\u6709\u8005\u4f46\u4e0d\u613f\u610f\u6211\u4eec\u8f6c\u8f7d\u60a8\u7684\u4f5c\u54c1\uff0c\u8bf7\u901a\u77e5\u6211\u4eec\u5220\u9664',
            r'\*\* (.*?)\u4ec5\u4ee3\u8868\u4f5c\u8005\u4e2a\u4eba\u7684\u89c2\u70b9\uff0c\u4e0e(.*?)',
            r'\u5543\u6587\u4e66\u5e93\ >\ (.*?)\ >\ (.*?)',
            r'\u5c06(.*?)\u52a0\u5165\u4e66\u67b6\ \|\ \u6211\u7684\u4e66\u67b6\ \|\ \u8fd4\u56de\u4e66\u9875\ (.*?)\ \u4f5c\u8005\uff1a\u614e\u72ec\u884c\ \|\ TXT\u5168\u6587\u4e0b\u8f7d\ \|\ TXT\u5355\u7ae0\u4e0b\u8f7d\ \|\ UMD\u4e0b\u8f7d\ \|\ JAR\u4e0b\u8f7d\ \|\ JAD\u4e0b\u8f7d',
        ]
        for temp in templates:
            try: sentence = re.sub(temp, ' ', sentence)
            except: continue

        zh_book = [
            '\u6b22\u8fce\u8bbf\u95ee:', '\u672c\u4e66\u6765\u81ea', '\u60a8\u4e0b\u8f7d\u7684\u8be5\u7535\u5b50\u4e66\u6765\u81ea:', '\u5c0f\u8bf4\u5728\u7ebf\u9605\u8bfb\u4e0b\u8f7d', '\u89c2\u770b\u5730\u5740', '\u8d34\u5427\u5730\u5740', '\u7fa4\u53f7\u7801', '\u624b\u673a\u9605\u8bfb\u672c\u7ae0\u8282\u8bf7\u767b\u9646',
            '\u58f0\u660e\uff1a', '\u672c\u7535\u5b50\u4e66\u4ec5\u4f9b\u8bfb\u8005\u9884\u89c8', '\u5185\u5bb9\u7248\u6743\u5f52\u4f5c\u8005\u6240\u6709', '\u5982\u679c\u559c\u6b22\u8bf7\u8d2d\u4e70\u6b63\u7248\u56fe\u4e66', '\u8bf7\u5728\u4e0b\u8f7d\u540e24\u5c0f\u65f6\u5185\u5220\u9664', '\u8bf7\u5728\u4e0b\u8f7d24\u5c0f\u65f6\u5185\u5220\u9664\uff0c',
            '\u4e0d\u5f97\u7528\u4f5c\u5546\u4e1a\u7528\u9014', 'TXT\u4e66\u5e93', 'TXT\u8d5b\u770b', 'TXT\u8bba\u575b', 'TXT BBS', '\u6b22\u8fce\u60a8\u6765TXTBBS\u63a8\u8350\u597d\u4e66', '( \u5e73\u5357\u6587\u5b66\u7f51)', '\u672c\u4e66\u4e0b\u8f7d\u5730\u5740\uff1a',
            '\u725bbb\u5c0f\u8bf4\u9605\u8bfb\u7f51', '\u725b\uff42\uff42\u5c0f\u8bf4\u7f51', '\uff37\uff37\uff37.\uff2e\uff29\uff35\uff22\uff22.\uff2e\uff25\uff34', '\u4e0a3Q\u5c0f\u8bf4\u7f51', 'www.\uff4e\uff49\uff55\uff42\uff42.\uff2e\uff25\uff34', 'WWW\uff0eniubb.net',
            '\u767e\u5ea6\u641c\u7d22\u2192\u7231\u53bb\u5c0f\u8bf4\u7f51wwW.AiQuXsM', '\u5c0f\u8bf4\u9605\u8bfb\u4e0b\u8f7d\u5c3d\u5728\u4e2d\u6587\u7f51\u66f4\u65b0\u8d85\u5feb\u5c0f\u8bf4\u66f4\u591a\uff1a', '\u9996\u53d1.com', 'xx\u6f47\u6e58\u9996\u53d1\uff0c\u8f6c\u8f7d\u5fc5\u7a76xx',
            '\u7b14\u4e0b\u6587\u5b66', '\u672c\u4e66\u9996\u53d1', '\u8150\u4e16\u4e4b\u5dc5', '\u4e91\u4e0a\u7b19\u6b4c', '{\u95ea\u821e\u5c0f\u8bf4\u7f51 }', '{\u541e\u566c\u5c0f\u8bf4\u7f51 }', '/xshuotxt/', 'xshuotxt',
            '\u641c\u522e\u5404\u7c7bTXT\u5c0f\u8bf4', '\u6b22\u8fce\u60a8\u6765\u63a8\u8350\u597d\u4e66', '\u514d\u8d39txt\u5c0f\u8bf4\u4e0b\u8f7d\u7ad9', '(\u624b\u6253\u5427 \u9996\u53d1)', '( \u9996\u53d1)', '\u4e0a3Q\u5c0f\u8bf4\u7f51', '\u771f \u610f \u4e66 \u76df',
            '\u66f4\u591a\u514d\u8d39txt\u7535\u5b50\u4e66\uff0c\u6b22\u8fce\u60a8\u5230\u4e0b\u8f7d', '\u66f4\u591a\u514d\u8d39txt\u7535\u5b50\u4e66', '\u6b22\u8fce\u60a8\u5230\u4e0b\u8f7d', '{ \u9996\u53d1 \u624b.\u6253/\u5427}', 'Www\u3002', 'Shouda8m',
            '\u8bbf\u95ee,\u8bf7\u7262\u8bb0bxwx\u5c0f\u8bf4\u7f51\uff0cbxwx.net', '\u8bf7\u7262\u8bb0bxwx\u5c0f\u8bf4\u7f51', 'bxwx.net', '\u9996\u53d1BXzw.com', '\uff37\uff37\uff37.bxwx.org', 'bXwX.Org \u5c0f\u8bf4\u7f51',
            '(\u8bf7\u8bb0\u4f4f\u6211\u4eec\u7684\u7f51\u5740.56\u4e66.\u5e93)', '\u624b\u673atxt\u5c0f\u8bf4-\u963f\u5df4\u8fbe', '\u63d0\u4f9b\u4e0b\u8f7d', '\u5c0f\u8bf4\u6392\u884c\u699c\uff1a', '\u6700\u65b0\u66f4\u65b0\u5c0f\u8bf4\uff1a', '\u7eff\u300e\u8272\u300f\u5c0f\u8bf4\u7f51',
            '\u624b\u673a\u88c5\u6709\u4e3b\u6d41\u9605\u89c8\u5668\u53ef\u4ee5\u76f4\u63a5\u8bbf\u95ee\u4e0b\u8f7d\u7535\u5b50\u4e66', '\u5144\u9ac1\u9adf\u7766\u6994\u9aba\u68a2\u707e\u82ef\u81c3\u68a6\u6c0f\u7565\u6c10\u7f31\u90ae', '\u65e0\u5f39\u7a97\u5c0f\u8bf4\u7f51', '23us\uff0ecom', '\u66f4\u65b0\u6700\u5feb', 'egegengxin',
            '[\u8bb0\u4f4f\u7f51\u5740 \u4e09\u4e94\u4e2d\u6587\u7f51]', '(\u6309\u4f4f\u4e09\u79d2\u590d\u5236)', '\u4e0b\u8f7d\u514d\u8d39\u9605\u8bfb\u5668!!', '[\u98ce\u4e91\u5c0f\u8bf4\u7f51]', 'bxwx \u5c0f\u8bf4\u7f51', 'shuokehuduan', 'appxsyd',
            '\u90fd\u5e02\u5168\u80fd\u5de8\u661f\u7531\u8d77-\u70b9\u6b63\u7248\u9996\u53d1\uff0c\u7b2c\u4e00\u65f6\u95f4\u66f4\u65b0\u6700\u65b0\u7ae0\u8282\uff0c\u6c42\u8ba2\u9605\u3001\u6c42\u6708\u7968\u652f', 'TXT\u5c0f\u8bf4\u5929\u5802 \uff0c\u6700\u6709\u6587\u827a\u6c14\u606f\u7684\u6587\u5b66\u7f51\u7ad9', 'w w w.2 7 t x t.c o m',
            '\u7eaf\u7eff\u8272\u6e05\u723d\u9605\u8bfb\u3002\u656c\u8bf7\u8bb0\u4f4f\u6211\u4eec\u6700\u65b0\u7f51\u57409?9?9?w??o?m', '\uff57\u00e0\uff50.1\u2479\u03baxOM', '\uff08\u672c\u7ad9\u5b98\u65b9\u624b\u673a\u6700\u65b0\u9605\u8bfb\u5668APP\u4e0a\u67b6\u4e86\uff01', '\u4e0b\u8f7d\u624b\u673a\u5ba2\u6237\u7aef',
            '(\u5168\u6587\u5b57\u5c0f\u8bf4\u9605\u8bfb\u5c3d\u5728\u6587\u5b66\u7f51)', '(\u672c\u4e66\u8f6c\u8f7d1\u2479\u6587\u5b66\u7f51\u24746\uff4b\uff38\uff33.\uff43\uff2f\u041c)', '(\u672c\u4e66\u8f6c\u8f7d\u039a\uff58\uff53\u6587\u5b66\u7f51)', 'siluke', 'info\u66f4\u65b0\u6700\u5feb\u7684\u5c0f\u8bf4\u7f51',
            '\u00b7\u5c0f\u8bf4\u4e0b\u8f7d\u7f51 - \u963f\u5df4\u8fbe \u00b7\u5c0f\u8bf4\u6392\u884c\u699c - /to', '\u9605\u8bfb\u8be5\u6587\u8bf7\u5230\u767e\u5ea6\u641c\u7d22\u201c\u5927\u4f17\u5c0f\u8bf4\u7f51', '{Www\u3002Shouda8.Com \u9996\u53d1 \u624b.\u6253/\u5427}',
            '\u6700\u65b0\u6700\u5feb\u7684\u66f4\u65b0\u5c3d\u5728..\u65b0\u4e16\u7eaa\u5c0f\u8bf4\u7f51', '2100xs.com', '\u541e\u566c\u661f\u7a7a,\u5927\u5468\u7687\u65cf\u7b49\u70ed\u95e8\u6d41\u884c\u5c0f\u8bf4\u514d\u8d39\u9605\u8bfb\u548ctxt\u7535\u5b50\u4e66\u514d\u8d39\u4e0b\u8f7d\u3002',
            '\u66f4\u591a\u597d\u4e66\u5c3d\u5728\u6050\u6016\u5c0f\u8bf4\u8bba\u575b kbtxt.', '\uff08\u4e0b\u8f7d\u6949', '\u672c\u624b\u673a\u79fb\u52a8\u7aef\u9996\u53d1\u5730\u5740\uff1aM.', '(\u672c\u4e66\u91c7\u96c6\u6765\u6e90\u7f51\u7ad9\u6e05\u6670\u3001\u65e0\u5f39\u7a97\u3001\u66f4\u65b0\u5ea6\u5feb)',
            '\uff0dshuhaigem \u9996\u53d1 \u4f5c\u54c1\u76f8\u5173', '\u672c\u4f5c\u54c1\u7531WWW.94TXT.CN\u5f7c\u5cb8TXT\u7535\u5b50\u4e66\u8bba\u575b \u6574\u7406\u6536\u85cf', '\u79c0\u4e66\u7f51\u4e13\u4e1a\u63d0\u4f9b\u624b\u673a\u7535\u5b50\u4e66/\u7535\u5b50\u4e66\u4e0b\u8f7d', 'xiushu.com',
            'Jun\uff5aitang.co\uff4d\u9996\u53d1', '\u514d\u8d39\u63d0\u4f9b\uff0c\u8bf7\u591a\u53bb\u5149\u987e\u6b64\u7f51\u7ad9\u54e6\uff01', '+d80ok0bo >', '\uff0d\uff0a\ s\ h\ u\ h\ a\ i\ g\ e\ \uff0acm\ \u4e66\u53cb\u4e0a\u4f20/\uff0dshuhaigem',
            '\u624b\u673a\u514d\u8d39\u5ba2\u6237\u7aef\u6b63\u5f0f\u4e0a\u7ebf\uff01', ' \u5ba2\u6237\u7aef\u662f\u4e00\u6b3e\u4e13\u4e3a\u5e7f\u5927 \u8ff7\u6253\u9020\u7684\u4e13\u5c5e\u9605\u8bfb\u5668\uff0c\u6c47\u805a\u6d77\u91cf \u8d44\u6e90\uff0c\u5206\u7c7b\u7cbe\u7ec6\uff0c\u6392\u7248\u6e05\u6670\uff0c\u9605\u8bfb\u6548\u679c\u6781\u597d\uff01',
            '\u7b14\u4e0b\u4e66\u53cb\u6b63\u5728\u9605\u8bfb:', '\u5927\u4e3b\u5bb0 \u83bd\u8352\u7eaa \u7edd\u4e16\u5510\u95e8 \u661f\u6cb3\u5927\u5e1d \u7075\u57df \u6211\u7684\u8d34\u8eab\u6821\u82b1 \u5b9d\u9274 \u6b66\u9006', '\u533b\u9053\u5b98\u9014 \u7ea2\u8272\u4ed5\u9014 \u5b98\u672f \u65e0\u5c3d\u5251\u88c5 \u6700\u5f3a\u5f03\u5c11',
            '\u4e3e\u62a5\uff1a\u5185\u5bb9\u51fa\u9519 / \u5176\u5b83\u95ee\u9898 \u4e0a\u4e00\u9875 \u8fd4\u56de', '\u76ee\u5f55 \u4e0b\u4e00\u9875 \u52a0\u5165\u4e66\u7b7e \u63a8\u8350\u672c\u4e66', '\u5b57\u5927 \u8c03\u8282', '\u9605\u8bfb\u4eae\u5ea6\u8c03\u6574', 'zaixianxiaoshuo', 'w w w.z h e n 1.c o m',
            '\u672c\u7ad9\u91cd\u8981\u901a\u77e5\uff1a\u672c\u7ad9\u7684\u514d\u8d39 APP\uff0c\u65e0\u5e7f\u544a\u3001\u65e0\u9519\u8bef\u3001\u66f4\u65b0\u5feb\uff0c\u4f1a\u5458\u540c\u6b65\u4e66\u67b6\uff0c', '\u66f4\u597d\u7684\u9605\u8bfb\u4f53\u9a8c\uff0c\u8bf7\u5173\u6ce8\u5fae\u4fe1', '\u4e0b\u8f7d\u514d\u8d39\u9605\u8bfb\u5668', '\u5173\u6ce8\u5fae\u4fe1\u516c\u4f17 (.*?)',
            '\u7231\u597d\u8005\u63d0\u4f9b\u5f02\u4e16\u90aa\u541b,\u906e\u5929,\u6c38\u751f,\u4ed9\u9006,\u5929\u73e0\u53d8,\u541e\u566c\u661f\u7a7a,\u5927\u5468\u7687\u65cf\u7b49\u70ed\u95e8\u6d41\u884c \u514d\u8d39\u9605\u8bfb\u548ctxt\u7535\u5b50\u4e66\u514d\u8d39\u4e0b\u8f7d\u3002',
        ]
        for text in zh_book:
            try: sentence = re.sub(re.escape(text), ' ', sentence)
            except: continue

        return sentence

    @staticmethod
    def de_commons(sentence, commons):
        cc_adv = commons
        for text in cc_adv:
            sentence = re.sub(re.escape(text), ' ', sentence)
        return sentence

def main():

    parser = argparse.ArgumentParser(description='A Pre-trained Language Model Based Data Filter.')
    parser.add_argument('-i', '--input', default=None, type=str, help='input file path')
    parser.add_argument('-n', '--dataset', default=None, type=str, help='dataset name in the meta file. e.g., "pile-webtext"')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file path')
    parser.add_argument('-m', '--meta', default='meta_config.json', type=str, help='meta data file path')
    parser.add_argument('-s', '--batchsize', default=4, type=int, help='the batch size for llm inference')
    parser.add_argument('-d', '--device', default='cuda', type=str, help='the device for computation')
    parser.add_argument('-l', '--min_len', default=10, type=int, help='the minimal length of a training sentence')
    parser.add_argument('-f', '--filter', default=None, type=str, help='the filtering llm for quick filtering of high quality data')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available:
        print('CUDA not available. Switch to CPU runtime.')
        args.device = 'cpu'

    if args.input:

        print('<=== Start: Loading Meta Files ===>')

        with open(args.meta, 'r', encoding='utf-8') as fmeta:
            meta_data = json.load(fmeta)

        dataset = args.dataset
        name = args.input

        suffix = '_removed'
        if suffix in name:
            splits = name.split('.')
            name = splits[0][:-len(suffix)] + '.' + splits[1]

        for data_name, data_info in meta_data.items():
            if name in data_info['path']:
                dataset = data_name

        if dataset is None:
            raise Exception('Please specify the name of the dataset or make sure the path corresponds to the meta file.')

        print('<=== Start: Loading Meta Files ===>')

        print('<=== Start: Loading Language Models ===>')

        tokenizer, model = None, None
        model_name = args.filter
        if meta_data[dataset]['lm_filter'] != 'null':
            model_name = meta_data[dataset]['lm_filter']
        if model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
            model.to(args.device)
        cleaner = TextCleaner(model, tokenizer, args)

        print('<=== Done: Loading Language Models ===>')

        print('<=== Start: Loading Raw Text Data ===>')

        texts = []
        corpus = []
        cnt = 0
        discarded = meta_data[dataset]['special_punc']['discarded']
        additional = meta_data[dataset]['special_punc']['additional']
        punc_threshold = meta_data[dataset]['punc_filter']

        with open(args.input, 'r', encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line)
                sentence = sample['content']
                for pipeline in meta_data[dataset]['pipeline']:
                    sentence = getattr(cleaner, pipeline)(sentence)
                if len(sentence) < args.min_len: continue

                if punc_threshold < 1.00:
                    punc_filter = cleaner.punctuation_percentage(sentence, discarded, additional)
                    if punc_filter >= punc_threshold:
                        cnt += 1
                        continue
                sample['content'] = sentence
                corpus.append(sentence)
                texts.append(sample)

        print('{} entries are filtered due to a punctuation percentage larger than {}.'.format(cnt, punc_threshold))
        print('{} entries loaded and pre-processed for next steps.'.format(len(texts)))

        print('<=== Done: Loading Raw Text Data ===>')

        print('<=== Start: Text Processing ===>')

        filtered = []
        ppl_threshold = meta_data[dataset]['ppl_filter']
        if model_name is not None:
            log_perplexity = cleaner.compute_batch_log_perplexity(corpus)
            for i in range(len(log_perplexity)):
                log_perplexity[i] /= len(corpus[i])
            log_perplexity = np.array(log_perplexity)
            filtered = np.where(log_perplexity > ppl_threshold)[0]
            filtered = filtered.tolist()
        print('{} entries are filtered due to a normed log perplexity percentage larger than {}.'.format(len(filtered), ppl_threshold))

        print('<=== Done: Text Processing ===>')

        print('<=== Start: Writing Output Text ===>')

        outfile = args.output if args.output is not None else args.input.split('.')[0] + '_filtered.jsonl'
        with open(outfile, 'w', encoding='utf-8') as json_file:
            for i, text in enumerate(texts):
                if i in filtered: continue
                json_file.write(json.dumps(text) + '\n')

        print('<=== Done: Writing Output Text ===>')

    else:
        raise Exception('Please specify input files or directories.')

if __name__ == '__main__':
    sys.exit(main())

