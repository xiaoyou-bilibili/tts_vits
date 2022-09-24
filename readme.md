# 语音合成项目
> 原项目地址：https://github.com/AlexandaJerry/vits-mandarin-biaobei
> 
> 最原始项目：https://github.com/jaywalnut310/vits
> 
> 本项目只是添加注释和套一层web壳

## 效果展示
![](./source/01.png)

合成效果：
[派蒙](./source/paimon.wav)
[默认](./source/default.wav)


## 项目运行

请把模型放到model里面，然后改成 `default.pth` 即可，因为模型将近500M，实在太大，暂不提供，请自行训练

```shell
# 安装必要依赖
pip install -r requirements.txt
sudo apt-get install espeak -y
# 构建依赖
cd monotonic_align
python setup.py build_ext --inplace
cd ..
# 运行项目
python main.py
```

## 训练自己的模型

这里会分别介绍如何使用标贝的数据集和使用原神游戏的数据集来进行训练

> 注意，pytorch必须是1.6，其他版本运行会有问题，不过跑模型没这个限制

### 标贝数据集训练

因为标贝的数据集非常的标准，而且还有音频对应的文本，所以训练起来也比较简单

数据集地址：https://www.data-baker.com/data/index/TNtts

默认项目就已经配置好文本信息了，我们只需要音频文件即可，也就是`Wave`文件夹

然后我们可以软链接 `ln -s xxx/Wave vits-mandarin-biaobei/biaobei` 把数据给映射到项目中，到这里我们就可以开始训练了

训练代码在`vits-mandarin-biaobei`里，同时这个文件夹就是原项目代码，你也可以自己手动克隆项目
```shell
# 安装必要依赖
pip install -r requirements.txt
sudo apt-get install espeak -y
# 安装一下音频处理工具
sudo apt-get install -y sox
# 进入音频文件夹
cd voice
# 对原始音频进行处理，把采样率转换为22050（这里需要把下面的代码写到shell脚本里去）
for x in ./*.wav
do 
  b=${x##*/}
  sox $b -r 22050 tmp_$b
  rm -rf $b
  mv tmp_$b $b
done
# 还需要再构建一下依赖
cd ../monotonic_align
python setup.py build_ext --inplace
cd ..
# 对原始数据进行处理，后续训练是拿处理后的数据来训练的（原项目已经处理过了，可以不处理）
python preprocess.py --text_index 1 --text_cleaners chinese_cleaners1 --filelists filelists/train_filelist.txt filelists/val_filelist.txt
# 开始训练
python train.py -c configs/biaobei_base.json -m biaobei_base
```
训练完后的模型会在`logs`里

### 原神数据训练

原神数据集训练比较麻烦，首先我们得准备好数据集，这里为了避免版权争议，我只提供思路
- 首先自己把游戏里面所有的角色音频文件全部提取出来，大概有10w条
- 然后我们可以使用语音识别来获取都对应音频文件的文本信息（网上已经有现成的了）
- 有了文本信息后，我们还需要判断一下每句话到底是谁说的，这里可以使用我之前的一个[声纹识别](https://github.com/xiaoyou-bilibili/voice_recognize)项目，不过这个方法的准确率没那么高，但是基本上是能用的
- 使用声纹识别后提取出派蒙的语音条数大概在9000左右

然后就是音频处理，这里我们需要调整一下音量，要不然转换的时候会有警告
```shell
for x in ./*.wav
do 
  b=${x##*/}
  sox -v 0.90 $b -r 22050 tmp_$b
  rm -rf $b
  mv tmp_$b $b
done
```
转换完后还是不能直接训练，有部分音频是有问题的，所以我们还需要把这些有问题的音频找出来并删除，代码如下
```python
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence
import os
import commons
import torch
from mel_processing import spectrogram_torch
sampling_rate = 22050
max_wav_value = 32768.0
filter_length = 1024
hop_length = 256
win_length = 1024
cleaned_text=True
text_cleaners=["chinese_cleaners1"]
add_blank=True

def get_audio(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, sampling_rate))
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")
    print(spec_filename)
    if os.path.exists(spec_filename):
        spec = torch.load(spec_filename)
    else:
        spec = spectrogram_torch(audio_norm, filter_length,
                                 sampling_rate, hop_length, win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filename)
    return spec, audio_norm


def get_text(text):
    if cleaned_text:
        text_norm = cleaned_text_to_sequence(text)
    else:
        text_norm = text_to_sequence(text, text_cleaners)
    if add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

with open("filelists/paimon_train.txt.cleaned") as f:
    with open("filelists/paimon_train.txt.cleaned1","w") as f2:
        for file in f.read().split("\n"):
            name = file.split("|")
            if len(name) == 2:
                try:
                    get_audio(name[0])
                    get_text(name[1])
                    f2.write("{}\n".format(file))
                except Exception as e:
                    print(e)
                    print(file)
```

最后处理完毕后就可以训练了，配置文件和标贝的可以保持一致，只需要修改里面的文件位置即可，使用`V100-16G`训练一晚上就基本上可以使用了

## 算法原理


## 相关问题
### sndfile library not found

```shell
apt-get install -y libsndfile1
```

### AttributeError: module ‘distutils‘ has no attribute ‘version
```shell
pip uninstall setuptools
pip install setuptools==59.5.0
```

