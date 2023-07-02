import pickle
import subprocess

from word import Sentence


def predict(raw_sentence: str):
    """駄洒落かどうかを判定する

    Parameters
    ----------
    - raw_sentence: ユーザーが入力した文章
    """
    dic_dir = subprocess.run(["mecab-config", "--dicdir"], capture_output=True, text=True).stdout.replace("\n", "")
    tagged_sentence = subprocess.run(
        ["mecab", "-Ochasen", "-d", f"{dic_dir}/mecab-ipadic-neologd"],
        input=raw_sentence,
        text=True,
        capture_output=True,
    ).stdout
    sentence = Sentence.from_sentence(tagged_sentence)
    with open("./models/pre_process.pkl", "rb") as f:
        pre_process = pickle.load(f)
    x = pre_process.transform([sentence])
    with open("./models/model.pkl", "rb") as f:
        model = pickle.load(f)
    res = model.predict(x)
    return bool(res)
