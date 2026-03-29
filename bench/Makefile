DATASETS_DIR = datasets
RESULTS_DIR = results

# Dataset URLs (HuggingFace mirrors for reliability)
GLOVE100_URL = https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/glove-100-angular.hdf5
SIFT128_URL = https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift-128-euclidean.hdf5
SIFTSMALL_URL = https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/siftsmall-128-euclidean.hdf5
CODESEARCH768_URL = https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/codesearchnet-jina-768-cosine.hdf5
SIMPLEWIKI3072_URL = https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/simplewiki-openai-3072-normalized.hdf5

.PHONY: all datasets bench clean

all: datasets bench

# === Dataset Downloads ===

datasets: dataset-siftsmall dataset-glove100 dataset-sift128

datasets-all: datasets dataset-codesearchnet768 dataset-simplewiki3072

dataset-siftsmall:
	@mkdir -p $(DATASETS_DIR)
	@[ -f $(DATASETS_DIR)/siftsmall-128-euclidean.hdf5 ] || \
		(echo "Downloading siftsmall-128-euclidean (5 MB)..." && \
		curl -L -o $(DATASETS_DIR)/siftsmall-128-euclidean.hdf5 $(SIFTSMALL_URL))

dataset-glove100:
	@mkdir -p $(DATASETS_DIR)
	@[ -f $(DATASETS_DIR)/glove-100-angular.hdf5 ] || \
		(echo "Downloading glove-100-angular (485 MB)..." && \
		curl -L -o $(DATASETS_DIR)/glove-100-angular.hdf5 $(GLOVE100_URL))

dataset-sift128:
	@mkdir -p $(DATASETS_DIR)
	@[ -f $(DATASETS_DIR)/sift-128-euclidean.hdf5 ] || \
		(echo "Downloading sift-128-euclidean (525 MB)..." && \
		curl -L -o $(DATASETS_DIR)/sift-128-euclidean.hdf5 $(SIFT128_URL))

dataset-codesearchnet768:
	@mkdir -p $(DATASETS_DIR)
	@[ -f $(DATASETS_DIR)/codesearchnet-jina-768-cosine.hdf5 ] || \
		(echo "Downloading codesearchnet-jina-768-cosine (~4 GB)..." && \
		curl -L -o $(DATASETS_DIR)/codesearchnet-jina-768-cosine.hdf5 $(CODESEARCH768_URL))

dataset-simplewiki3072:
	@mkdir -p $(DATASETS_DIR)
	@[ -f $(DATASETS_DIR)/simplewiki-openai-3072-normalized.hdf5 ] || \
		(echo "Downloading simplewiki-openai-3072-normalized (~3 GB)..." && \
		curl -L -o $(DATASETS_DIR)/simplewiki-openai-3072-normalized.hdf5 $(SIMPLEWIKI3072_URL))

# === Benchmarks ===

bench: bench-siftsmall

bench-siftsmall: dataset-siftsmall
	@echo "Running siftsmall benchmark..."
	cd .. && go run ./bench/cmd/tqdb-annbench --dataset bench/datasets/siftsmall-128-euclidean.hdf5 --output bench/results/siftsmall.md

bench-glove100: dataset-glove100
	@echo "Running glove-100 benchmark..."
	cd .. && go run ./bench/cmd/tqdb-annbench --dataset bench/datasets/glove-100-angular.hdf5 --output bench/results/glove100.md

bench-sift128: dataset-sift128
	@echo "Running sift-128 benchmark..."
	cd .. && go run ./bench/cmd/tqdb-annbench --dataset bench/datasets/sift-128-euclidean.hdf5 --output bench/results/sift128.md

bench-all: bench-siftsmall bench-glove100 bench-sift128

# === Cleanup ===

clean:
	rm -rf $(RESULTS_DIR)/*.md

clean-all: clean
	rm -rf $(DATASETS_DIR)/*.hdf5
