

import jsonlines

import datasets
from typing import List


logger = datasets.logging.get_logger(__name__)


class Hw3Config(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Hw3Config, self).__init__(**kwargs)


class SUMDS(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "context": datasets.features.Sequence(datasets.Value("string")),
                    'id': datasets.Value('int32'),
                    'persona': datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(self, file_paths):
        return [datasets.SplitGenerator(name=e, gen_kwargs={'filepath': self.config.data_files[e]}) for e in self.config.data_files.keys()]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        print('49', filepath)
        with jsonlines.open(filepath[0]) as f:
            for article in f:
                dialog = article['dialog']
                aid = article['id']
                persona = article['personas'] if 'personas' in article.keys() else []
                yield aid, {
                        "context": dialog,
                        'id': aid,
                        'persona': persona,
                    }
