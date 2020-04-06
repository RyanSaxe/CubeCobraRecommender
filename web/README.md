# CubeCobra Recommender API

This is a very quick and dirty Flask API that will return the
CubeCobra Recommeder recommendation in an json response.

For example http://127.0.0.1:8000/?cube_name=thepaupercube&num_recs=5 will return
the following json:

```json
{
    "sylvan might": "0.22489496",
    "goblin smuggler": "0.21615778",
    "squad captain": "0.20287196",
    "herald of the dreadhorde": "0.19540468",
    "compulsive research": "0.1574262",
}
```

To get this up and running, install requirements from the root directory:

```bash
$ pip install -r web/requirements.txt
```

And run the `gunicorn` http server:

```bash
$ gunicorn --reload web:app
```

*Note* The `reload` flag is only needed for local developement and will
reload the HTTP server when a file changes.
