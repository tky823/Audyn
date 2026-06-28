#ifndef AUDYN_H
#define AUDYN_H

#include <torch/extension.h>

namespace audyn
{
    at::Tensor search_monotonic_alignment_by_viterbi(const at::Tensor &probs, const at::Tensor &tgt_lengths, const at::Tensor &src_lengths, const bool take_log);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_bipartite_match_precision_recall_fscore(
        torch::Tensor input,
        torch::Tensor target,
        torch::Tensor input_lengths,
        torch::Tensor target_lengths,
        const float tolerance);
}

#endif // AUDYN_H
