from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json
from core.tts import generate

# 初始化flaskAPP
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 返回JSON字符串
def return_json(data):
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


# 语音合成
@app.route('/tts/generate', methods=['POST'])
def recognize_image():
    content = request.form.get('content')
    name = request.form.get('model')
    print(name)
    # 直接把音频信息保存为文件
    generate(name, content)
    # # 返回json类型字符串
    return return_json({
        'audio': '/static/output.wav'
    })



# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')