# coding=utf-8
import random
import string
from captcha.image import ImageCaptcha
import time


# 字符集合：包括数字、字母大小写
captcha_list = string.digits + string.ascii_letters
captcha_size = 4


def data_generator(path, nums):
    for i in range(nums):
        image = ImageCaptcha()
        image_text = "".join(random.sample(captcha_list, captcha_size))
        image_path = path + '/{}_{}.png'.format(image_text, int(time.time()))
        image.write(image_text, image_path)


if __name__ == '__main__':
    data_generator("../data/train", 20000)
    data_generator("../data/test", 2000)

    # print(captcha_list, len(captcha_list))
    # for i in captcha_list:
    #     print(i)
