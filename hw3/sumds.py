

import jsonlines

import datasets


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
                    "title": datasets.Value("string"),
                    "maintext": datasets.Value("string"),
                    'id': datasets.Value('int32')
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
                title = article['title'] if 'title' in article.keys() else ''
                aid = article['id']
                context = article['maintext'].strip().replace('\n', '').replace('\r', '')\
                                                .replace(' ', '').replace('`', '')
                yield aid, {
                        "title": title,
                        "maintext": context,
                        'id': aid,
                    }
