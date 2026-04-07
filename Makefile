PYBIND_INCLUDES=$(shell python3 -m pybind11 --includes)
PYTHON_EXT=$(shell python3-config --extension-suffix)

NVCC_FLAGS=-std=c++20 -g -arch=native -Xcompiler -fPIC
SOURCE_FILE=cuda_transformer/cuda_transformer.cu

# Compile cuda_transformer module natively binding into a unified library to be loaded explicitly in Python.
all: cuda_transformer

clean:
	rm -f cuda_transformer*.so cuda_transformer.pyi

cuda_transformer: $(SOURCE_FILE) FORCE
	nvcc ${NVCC_FLAGS} --shared -I. -Icuda_transformer ${PYBIND_INCLUDES} ${SOURCE_FILE} -o cuda_transformer${PYTHON_EXT}
	stubgen -m cuda_transformer -o .

%: %.cu FORCE
	nvcc ${NVCC_FLAGS} -I. -I${PYBIND_INCLUDES} $< -o $@
	./$@

FORCE: