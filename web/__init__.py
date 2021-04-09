import json
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from tensorflow import keras

from ml_recommend_web import get_ml_recommend
from ml_embeddings_web import get_ml_embeddings


app = Flask(__name__)

if __name__ != "__main__":
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.DEBUG)


@app.route("/")
def api():
    cube_name = request.args.get("cube_name")
    num_recs = request.args.get("num_recs", 30000)
    root = request.args.get("root", "https://www.cubecobra.com")
    if not (cube_name and num_recs):
        error = "Need cube_name and num_recs as parameters!"
        app.logger.error(error)
        return error
    try:
        num_recs = int(num_recs)
    except ValueError:
        error = "num_recs needs to be an integer!"
        app.logger.error(error)
        return error

    try:
        results = get_ml_recommend(model, int_to_card, card_to_int, cube_name, num_recs, root)
    except Exception as e:
        app.logger.error(e)
        raise e
    return jsonify(results)

@app.route("/embeddings/",methods=['POST'])
def embeddings():
    cards = request.json.get("cards")
    n_decimals = request.args.get("n_decimals", 5)
    results = get_ml_embeddings(model, int_to_card, card_to_int, cards, n_decimals=n_decimals)
    return jsonify(results)

if __name__ == "__main__":
    model_path = Path('ml_files/20210409')
    with open(model_path / 'int_to_card.json', 'rb') as map_file:
        int_to_card = json.load(map_file)
    int_to_card = {int(k): v for k, v in enumerate(int_to_card)}
    card_to_int = {v: k for k, v in int_to_card.items()}

    model = keras.models.load_model(model_path)
    app.run(host="0.0.0.0", port=8000, threaded=True)
