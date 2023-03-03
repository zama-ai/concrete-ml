import os
import subprocess
from pathlib import Path


def main():
    path_of_script = Path(__file__).parent.resolve()
    os.environ["TRANSFORMERS_CACHE"] = str((path_of_script / "hf_cache").resolve())
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load the tokenizer (converts text to tokens)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Load the pre-trained model
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    del tokenizer
    del transformer_model

    # Build image
    os.chdir(path_of_script)
    command = f'docker build --tag cml_client_sentiment_analysis --file "{path_of_script}/Dockerfile.client" .'
    print(command)
    subprocess.check_output(command, shell=True)


if __name__ == "__main__":
    main()
