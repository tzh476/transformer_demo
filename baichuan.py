import openai
from bytedance import servicediscovery
import random


class APIWrapper:
    def __init__(self, PSM="P.S.M", DC=None) -> None:
        self.PSM = PSM
        if DC is not None:
            self.PSM += f".service.{DC}"

    def get_api_base(self):
        instances = servicediscovery.lookup(self.PSM)
        weighted_instances = []
        for instance in instances:
            weight = int(instance['Tags'].get('weight', 1))
            weighted_instances.extend([instance] * weight)
        instance = random.choice(weighted_instances)
        return f"http://{instance['Host']}:{instance['Port']}/v1"

    def __str__(self):
        return self.get_api_base()


def hook_openai(PSM="yangxinyu.715.infer", DC="lq"):
    import openai
    openai.api_base = APIWrapper(PSM=PSM, DC=DC)
    openai.api_key = "---"


hook_openai()
model = "baichuan-7b"


def test_completion(prompt="Once upon a time,"):
    completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
    print(prompt + completion.choices[0].text)


def test_embedding():
    embedding = openai.Embedding.create(model=model, input="Hello world!")
    print(len(embedding["data"][0]["embedding"]))


if __name__ == "__main__":
    test_completion()
    test_embedding()