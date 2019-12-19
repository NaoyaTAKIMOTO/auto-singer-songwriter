import flask
from lyric_generator import *
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer,MecabTokenizer
from transformers.modeling_bert import BertForMaskedLM

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
app.config["JSON_AS_ASCII"]=False

tokenizer=None
model=None
beam_decoder=None


def load_model():
    global tokenizer, model,beam_decoder
    print(" * Loading pre-trained model ...")
    tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
    model = BertForMaskedLM.from_pretrained('bert-base-japanese-whole-word-masking')
    beam_decoder = BeamDecoder(model, tokenizer)
    print(' * Loading end')

@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read feature from json
        seed_text = flask.request.get_json().get("seed_text")
        beam_depth=int(flask.request.get_json().get("beam_depth"))
        input_ids = beam_decoder.encode(seed_text, beam_depth)

        # predict additional tokens
        response["n_best"] =beam_decoder.decode(input_ids)

        # indicate that the request was a success
        response["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(response)


if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0',port=5000)