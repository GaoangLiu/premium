import re

import codefast as cf
from codefast.argparser import ArgParser


def sshmac():
    ip = re.search(r'\((192.168.*)\)', cf.shell('arp -a |grep 18:65:90:ce'))
    if not ip:
        cf.error('Device was not found.')
    else:
        cf.info(f'logging in mac: ssh alpha@{ip.group(1)}')


def main():
    ap = ArgParser()
    ap.input('-sshmac',
             '-sshmac',
             sub_args=[],
             description='Login MacPro when connected to same WIFI.')
    ap.input('-jcut',
             'jieba_cut',
             description=
             'jieba segment result for one sentence. \nUsage: zz -jcut 天气不错.')

    ap.input('-hanlp', description='Hanlp segment result.')
    ap.parse()

    if ap.sshmac:
        sshmac()
    elif ap.jieba_cut:
        import jieba
        print(jieba.lcut(ap.jieba_cut.value))

    elif ap.hanlp:
        from pyhanlp import HanLP
        print([term.word for term in HanLP.segment(ap.hanlp.value)])

    else:
        ap.help()
