import os 

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# load tokenizer
mistral_tokenizer = MistralTokenizer.from_file(os.path.expanduser("~")+"/mistral_7b_instruct_v3/tokenizer.model.v3")
# chat completion request
completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
# encode message
tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens
# load model
model = Transformer.from_folder(os.path.expanduser("~")+"/mistral_7b_instruct_v3")
# generate results
out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)
# decode generated tokens
result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
print(result)