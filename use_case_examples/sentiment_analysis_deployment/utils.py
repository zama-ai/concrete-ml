import numpy
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Function that transforms a list of texts to their representation
# learned by the transformer.
def text_to_tensor(
    list_text_X_train: list,
    transformer_model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
) -> numpy.ndarray:
    # Tokenize each text in the list one by one
    tokenized_text_X_train_split = []
    for text_x_train in list_text_X_train:
        tokenized_text_X_train_split.append(tokenizer.encode(text_x_train, return_tensors="pt"))

    # Send the model to the device
    transformer_model = transformer_model.to(device)
    output_hidden_states_list = []

    for tokenized_x in tqdm.tqdm(tokenized_text_X_train_split):
        # Pass the tokens through the transformer model and get the hidden states
        # Only keep the last hidden layer state for now
        output_hidden_states = transformer_model(tokenized_x.to(device), output_hidden_states=True)[
            1
        ][-1]
        # Average over the tokens axis to get a representation at the text level.
        output_hidden_states = output_hidden_states.mean(dim=1)
        output_hidden_states = output_hidden_states.detach().cpu().numpy()
        output_hidden_states_list.append(output_hidden_states)

    return numpy.concatenate(output_hidden_states_list, axis=0)
