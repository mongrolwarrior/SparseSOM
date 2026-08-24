#pragma once
#include <cstddef>
#include "sparsesom/config.hpp"
namespace sparsesom {
// Size of the codebook implied by a config: neuron count × feature dimension.
struct CodebookInfo { std::size_t neurons; std::size_t features; };
// Geometry of the codebook for a given config (defaults to the MEDLINE headline config).
CodebookInfo planned_codebook(const SomConfig& cfg = {});
}
