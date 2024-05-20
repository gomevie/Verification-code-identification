# coding=utf-8

import torch
import data_generator

captcha_list = data_generator.captcha_list
captcha_size = data_generator.captcha_size

def text2vec(text):
    vec = torch.zeros(captcha_size, len(captcha_list))
    for i in range(len(text)):
        vec[i][captcha_list.index(text[i])] = 1
    return vec

def vec2text(vec):
    vec = torch.argmax(vec, dim=1)
    text = ''
    for i in vec:
        text += captcha_list[i]
    return text

if __name__ == '__main__':
    text = 'abcd'
    vec = text2vec(text)
    print(vec, vec.shape)
    print(vec2text(vec))



