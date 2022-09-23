import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

# 全局模型信息
_model_info = {}
# 加载配置文件
hps = utils.get_hparams_from_file("configs/base.json")


# 获取文本信息
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# 加载模型
def load_model(name):
    global _model_info
    # 读取第一个模型
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    model.eval()
    utils.load_checkpoint("model/{}.pth".format(name), model, None)
    _model_info[name] = model


def generate(name, text):
    # 如果模型不存在就手动加载
    if name not in _model_info:
        load_model(name)
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = _model_info[name].infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()
        write("web/static/output.wav", 22050, audio)
