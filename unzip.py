import zipfile
import os
import sys

def unzip(file, destination):
    with zipfile.ZipFile(file, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
            print(member.filename)
            zf.extract(member, destination)

if __name__ == "__main__":
    unzip('./자유대화(노인남녀)/Validation/[원천]1.AI챗봇.zip', './자유대화(노인남녀)/Validation/AI챗봇/data/')
  