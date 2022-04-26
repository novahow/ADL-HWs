import csv
import pandas as pd
import datasets


_CITATION = """\
@inproceedings{zellers2018swagaf,
    title={SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
    author={Zellers, Rowan and Bisk, Yonatan and Schwartz, Roy and Choi, Yejin},
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year={2018}
}
"""

_DESCRIPTION = """\
Given a partial description like "she opened the hood of the car,"
humans can reason about the situation and anticipate what might come
next ("then, she examined the engine"). SWAG (Situations With Adversarial Generations)
is a large-scale dataset for this task of grounded commonsense
inference, unifying natural language inference and physically grounded reasoning.
The dataset consists of 113k multiple choice questions about grounded situations
(73k training, 20k validation, 20k test).
Each question is a video caption from LSMDC or ActivityNet Captions,
with four answer choices about what might happen next in the scene.
The correct answer is the (real) video caption for the next event in the video;
the three incorrect answers are adversarially generated and human verified,
so as to fool machines but not humans. SWAG aims to be a benchmark for
evaluating grounded commonsense NLI and for learning representations.
The full data contain more information,
but the regular configuration will be more interesting for modeling
(note that the regular data are shuffled). The test set for leaderboard submission
is under the regular configuration.
"""

_LICENSE = "Unknown"
'''
_URLs = {
    "full": {
        "train": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/train_full.csv",
        "val": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/val_full.csv",
    },
    "regular": {
        "train": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/train.csv",
        "val": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/val.csv",
        "test": "https://raw.githubusercontent.com/rowanz/swagaf/master/data/test.csv",
    },
}
'''


class Qads(datasets.GeneratorBasedBuilder):
    """SWAG dataset"""


    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="regular", description="The configuration to use for modeling."),
        datasets.BuilderConfig(name="full", description="The full data."),
    ]

    DEFAULT_CONFIG_NAME = "regular"

    def _info(self):
        if self.config.name == "regular":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "qusetion": datasets.Value("string"),
                    "p0": datasets.Value("int"),
                    "p1": datasets.Value("int"),
                    "p2": datasets.Value("int"),
                    "p3": datasets.Value("int"),
                    "label": datasets.ClassLabel(names=["0", "1", "2", "3"]),
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "qusetion": datasets.Value("string"),
                    "p0": datasets.Value("int"),
                    "p1": datasets.Value("int"),
                    "p2": datasets.Value("int"),
                    "p3": datasets.Value("int"),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=None,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # my_urls = _URLs[self.config.name]
        data_dir = {'train': '../train.json', 'val': '../valid.json', 'test': '../test.json'}

        splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["val"],
                    "split": "val",
                },
            ),
        ]
        if self.config.name == "regular":
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": data_dir["test"], "split": "test"},
                )
            )

        return splits

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        df = pd.read_json(filepath)

        for i in range(len(df)):
            if self.config.name == "regular":
                yield i, {
                    "id": df.iloc[i]['id'],
                    "p0": df.iloc[i]['paragraphs'][0],
                    "p1": df.iloc[i]['paragraphs'][1],
                    "p2": df.iloc[i]['paragraphs'][2],
                    "p3": df.iloc[i]['paragraphs'][3],
                    "qusetion": df.iloc[i]['qusetion'],
                    "label": -1 if split == "test" else df.iloc[i]['relevant'],
                }
                