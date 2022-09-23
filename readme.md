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
```shell
# 安装必要依赖
pip install -r requirements.txt
sudo apt-get install espeak -y

```




## 相关问题
### sndfile library not found

```shell
apt-get install -y libsndfile1
```