from flask import Flask, request, jsonify

from web.ml_recommend_web import get_ml_recommend


app = Flask(__name__)


@app.route('/')
def api():
    cube_name = request.args.get("cube_name")
    num_recs = request.args.get("num_recs")
    root = request.args.get("root", "https://www.cubecobra.com")
    if not (cube_name and num_recs):
        return "Need cube_name and num_recs as parameters!"
    try:
        num_recs = int(num_recs)
    except ValueError:
        return "num_recs needs to be an integer!"

    results = get_ml_recommend(cube_name, num_recs, root)
    return jsonify(results)
