import logging

from flask import Flask, request, jsonify

from .ml_recommend_web import get_ml_recommend


app = Flask(__name__)

if __name__ != "__main__":
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.DEBUG)


@app.route("/")
def api():
    cube_name = request.args.get("cube_name")
    num_recs = request.args.get("num_recs",30000)
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
        results = get_ml_recommend(cube_name, num_recs, root)
    except Exception as e:
        app.logger.error(e)
        raise e
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
