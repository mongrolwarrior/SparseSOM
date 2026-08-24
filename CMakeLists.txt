cmake_minimum_required(VERSION 3.22)
project(SparseSOM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ada (RTX 4090, SM 8.9) + Hopper (H100/H200, SM 9.0).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89;90")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for IntelliSense / clangd

find_package(CUDAToolkit REQUIRED)

# Shared host harness (IO, codebook, training loop) — fill in from MedSOM.
add_library(sparsesom_core STATIC
  src/core/codebook.cpp
  src/core/dataset.cpp
  src/core/codebook_io.cpp)
target_include_directories(sparsesom_core PUBLIC include)

# Novel feature-major sparse-binary backend (CUDA).
add_library(sparsesom_featuremajor STATIC
  src/backend_featuremajor/fill_kernel.cu
  src/backend_featuremajor/device_dataset.cu
  src/backend_featuremajor/feature_major_codebook.cu
  src/backend_featuremajor/update_factorized.cu
  src/backend_featuremajor/bmu.cu
  src/backend_featuremajor/training.cu
  src/backend_featuremajor/kaski_lagus.cu
  src/backend_featuremajor/pca_init.cu)
target_include_directories(sparsesom_featuremajor PUBLIC include)
target_link_libraries(sparsesom_featuremajor PUBLIC CUDA::cudart CUDA::curand)

add_executable(sparsesom cli/main.cpp)
target_link_libraries(sparsesom PRIVATE sparsesom_core sparsesom_featuremajor)

# Converter: generic CSR (.csr) -> SparseSOM .sbcsr (host-only).
add_executable(csr_convert cli/csr_convert.cpp)
target_link_libraries(csr_convert PRIVATE sparsesom_core)

# Inspector: summarise a saved .somw codebook (host-only).
add_executable(som_inspect cli/som_inspect.cpp)
target_link_libraries(som_inspect PRIVATE sparsesom_core)

enable_testing()
add_executable(sparsesom_smoke tests/test_smoke.cpp)
target_link_libraries(sparsesom_smoke PRIVATE sparsesom_featuremajor)
add_test(NAME smoke COMMAND sparsesom_smoke)

add_executable(sparsesom_factorized_update tests/test_factorized_update.cpp)
target_link_libraries(sparsesom_factorized_update PRIVATE sparsesom_featuremajor)
add_test(NAME factorized_update COMMAND sparsesom_factorized_update)

add_executable(sparsesom_factorized_update_multichunk tests/test_factorized_update_multichunk.cpp)
target_link_libraries(sparsesom_factorized_update_multichunk PRIVATE sparsesom_featuremajor)
add_test(NAME factorized_update_multichunk COMMAND sparsesom_factorized_update_multichunk)

add_executable(sparsesom_bmu tests/test_bmu.cpp)
target_link_libraries(sparsesom_bmu PRIVATE sparsesom_featuremajor)
add_test(NAME bmu COMMAND sparsesom_bmu)

add_executable(sparsesom_training tests/test_training.cpp)
target_link_libraries(sparsesom_training PRIVATE sparsesom_featuremajor)
add_test(NAME training COMMAND sparsesom_training)

add_executable(sparsesom_kaski_lagus tests/test_kaski_lagus.cpp)
target_link_libraries(sparsesom_kaski_lagus PRIVATE sparsesom_featuremajor)
add_test(NAME kaski_lagus COMMAND sparsesom_kaski_lagus)

add_executable(sparsesom_diagnostics tests/test_diagnostics.cpp)
target_link_libraries(sparsesom_diagnostics PRIVATE sparsesom_featuremajor)
add_test(NAME diagnostics COMMAND sparsesom_diagnostics)

add_executable(sparsesom_dataset tests/test_dataset.cpp)
target_link_libraries(sparsesom_dataset PRIVATE sparsesom_core)
add_test(NAME dataset COMMAND sparsesom_dataset)

add_executable(sparsesom_csr_convert tests/test_csr_convert.cpp)
target_link_libraries(sparsesom_csr_convert PRIVATE sparsesom_core)
add_test(NAME csr_convert COMMAND sparsesom_csr_convert)

add_executable(sparsesom_codebook_io tests/test_codebook_io.cpp)
target_link_libraries(sparsesom_codebook_io PRIVATE sparsesom_core)
add_test(NAME codebook_io COMMAND sparsesom_codebook_io)

add_executable(sparsesom_reupload tests/test_reupload.cpp)
target_link_libraries(sparsesom_reupload PRIVATE sparsesom_core sparsesom_featuremajor)
add_test(NAME reupload COMMAND sparsesom_reupload)

# End-to-end: load .sbcsr -> upload -> train. Also writes the corpus the cli test uses.
add_executable(sparsesom_pipeline tests/test_pipeline.cpp)
target_link_libraries(sparsesom_pipeline PRIVATE sparsesom_core sparsesom_featuremajor)
add_test(NAME pipeline COMMAND sparsesom_pipeline)
set_tests_properties(pipeline PROPERTIES FIXTURES_SETUP corpus)

# Drive the actual CLI binary on the corpus the pipeline test produced, saving the trained
# codebook; then inspect it. The cli test requires the corpus fixture and provides the
# weights fixture that the inspect test consumes.
add_test(NAME cli COMMAND sparsesom /tmp/sparsesom_cli_corpus.sbcsr
         --rows 6 --cols 6 --epochs 60 --sigma-init 3 --sigma-min 0.4 --wd-path-frac 1
         --save-weights /tmp/sparsesom_cli_weights.somw)
set_tests_properties(cli PROPERTIES FIXTURES_REQUIRED corpus FIXTURES_SETUP weights)

add_test(NAME inspect COMMAND som_inspect /tmp/sparsesom_cli_weights.somw)
set_tests_properties(inspect PROPERTIES FIXTURES_REQUIRED weights)

# Restore the saved codebook onto the GPU and assign the corpus (eval-only, no training).
add_test(NAME eval COMMAND sparsesom /tmp/sparsesom_cli_corpus.sbcsr
         --load-weights /tmp/sparsesom_cli_weights.somw --eval)
set_tests_properties(eval PROPERTIES FIXTURES_REQUIRED "corpus;weights")