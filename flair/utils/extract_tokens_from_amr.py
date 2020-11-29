from stog.data.dataset_readers import AbstractMeaningRepresentationDatasetReader
import sys
from stog.utils import logging

logger = logging.init_logger()
def extract_amr_token(file_path):
    dataset_reader = AbstractMeaningRepresentationDatasetReader()
    for instance in dataset_reader.read(file_path):
        amr_tokens = instance.fields["amr_tokens"]["decoder_tokens"]
        yield " ".join(amr_tokens)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""Usage:
    python {} [amr_file]

The output will in stdout.
              """)
    for filename in sys.argv[1:]:
        for line in extract_amr_token(filename):
            sys.stdout.write(line + "\n")



