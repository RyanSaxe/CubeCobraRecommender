from flask import Flask, request, jsonify

from web.ml_recommend_web import get_ml_recommend


app = Flask(__name__)


@app.route('/')
def api():
    cube_name = request.args.get("cube_name")
    try:
        num_recs = int(request.args.get("num_recs"))
    except ValueError:
        return "num_recs needs to be an integer!"
    if cube_name and num_recs:
        results = get_ml_recommend(cube_name, num_recs)
        return jsonify(results)
    else:
        return "Need cube_name and num_recs as parameters!"
