#include "sparsesom/core.hpp"
namespace sparsesom {
// Codebook geometry is fully determined by the config: one weight per (neuron, feature).
CodebookInfo planned_codebook(const SomConfig& cfg) {
    return CodebookInfo{ cfg.n_neurons(), cfg.n_features };
}
}
