import hashlib
import random
import requests


def baidu_translate(text, from_lang, to_lang, appid, key):
    salt = random.randint(10000, 99999)
    sign = hashlib.md5(f"{appid}{text}{salt}{key}".encode("utf-8")).hexdigest()

    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": appid,
        "salt": salt,
        "sign": sign
    }

    r = requests.get(url, params=params, timeout=10).json()
    if "trans_result" not in r:
        raise RuntimeError(str(r))

    return "\n".join(x["dst"] for x in r["trans_result"])
