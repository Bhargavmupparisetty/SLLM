from sllm import HuggingFaceBPETokenizer

tokenizer = HuggingFaceBPETokenizer(vocab_size=5000)

with open("C:/Users/bharg/Downloads/merged_final.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

tokenizer.train(corpus)
tokenizer.save("C:/Users/bharg/OneDrive/Desktop/paradox/bpe_tokenizer.json")  # Must be saved like this
print("Tokenizer saved successfully...!")
