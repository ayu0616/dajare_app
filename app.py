from flask import Flask, flash, redirect, render_template, request, url_for

from predict import predict

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/judge", methods=["GET"])
def judge():
    raw_sentence = request.args.get("sentence")
    if raw_sentence is None:
        flash("文章を入力してください")
        return redirect(url_for("index"))
    res = predict(raw_sentence)
    return render_template("judge.html", res=res, sentence=raw_sentence)


if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")
