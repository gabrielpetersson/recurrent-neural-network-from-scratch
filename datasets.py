import os
path_to_module = os.path.dirname(__file__)

datasets_path = os.path.join(path_to_module, "resources", "datasets")
processed_data_path = os.path.join(path_to_module, "resources", "processeddata")
embeddings_data_path = os.path.join(path_to_module, "resources", "embeddings")

BOOKS = os.path.join(datasets_path, "books.txt")
SIMPSONS = os.path.join(datasets_path, "simpson.txt")

BOOKS_PROCESSED = os.path.join(processed_data_path, "books_processed.txt")
SIMPSONS_PROCESSED = os.path.join(processed_data_path, "simpson_processed.txt")
SIMPSONS_SHORT_PROCESSED = os.path.join(processed_data_path, "simpson_short_processed.txt")

SIMPSONS_EMBEDDINGS = os.path.join(embeddings_data_path, "simpson_embeddings.npy")
SIMPSONS_EMBEDDINGS1 = os.path.join(embeddings_data_path, "simpson_embeddings1.npy")
