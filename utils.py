import json
import pickle
import os
import traceback


def parse(content, separator):
    if len(separator) == 0:
        return content.strip()
    else:
        return [parse(c, separator[1:]) for c in content.strip().split(separator[0])]


def write(path, string):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(string)


def read(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def json_dump(data, path):
    json.dump(data, open(path, "w", encoding="utf8"), ensure_ascii=False)


def json_load(path):
    return json.load(open(path, "r", encoding="utf8"))


def split_basename(path):
    return os.path.splitext(os.path.basename(path))


def get_extension(path):
    return split_basename(path)[-1]


def get_name(path):
    return split_basename(path)[0]


def pkl_load(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def make_dict(*expr):
    (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
    begin = text.find('make_dict(') + len('make_dict(')
    end = text.find(')', begin)
    text = [name.strip() for name in text[begin:end].split(',')]
    return dict(zip(text, expr))


class ObjectDict:
    def __init__(self, attrs=None, **kwargs):
        if attrs is None:
            attrs = kwargs
        else:
            attrs.update(kwargs)
        for name, value in attrs.items():
            setattr(self, name, value)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __str__(self):
        return self.__dict__.__str__()

    def save(self, path):
        json_dump(self.__dict__, path)

    @classmethod
    def load(cls, path):
        attrs = json_load(path)
        return cls(attrs)

    def items(self):
        return self.__dict__.items()