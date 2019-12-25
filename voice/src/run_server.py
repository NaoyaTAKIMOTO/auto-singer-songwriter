import flask
import os
import sys
pwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pwd)
print(sys.path)
from tts import text2wav
# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
app.config["JSON_AS_ASCII"]=False
WAV_MIMETYPE="audio/wav"

def load_model():
    print(" * Loading pre-trained model ...")
    print("root:{}".format(ROOT))
    print(' * Loading end')

@app.route("/tts", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read feature from json
        input_text = flask.request.get_json().get("text","社交辞令")
        filename = flask.request.get_json().get("filename", "sample")
        text2wav(input_text, filename)
        download_file = os.path.join(ROOT, "wav", filename+".wav")
        # indicate that the request was a success
        response["success"] = True
    # return the data dictionary as a JSON response
    return flask.send_file(download_file, as_attachment = True, mimetype = WAV_MIMETYPE)


if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0',port=5001)