from huggingface_hub import Repository

repo = Repository(local_dir="huggingface-hub", clone_from="https://huggingface.co/spaces/openpecha/word_vectors_literary_bo")
