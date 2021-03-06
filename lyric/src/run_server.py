import flask
import os
import sys
ROOT = os.path.dirname(__file__)
sys.path.append(ROOT)
import json
from text_generator import text_generate
# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
app.config["JSON_AS_ASCII"]=False



def load_model():
    print(" * Loading pre-trained model ...")

    print(' * Loading end')

@app.route("/generate", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read feature from json
        input_text = flask.request.get_json().get("text")
        

        response["generated_text"]=text_generate(input_text)

        # indicate that the request was a success
        response["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(response)


if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0', port=5000)