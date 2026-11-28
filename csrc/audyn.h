#ifndef AUDYN_H
#define AUDYN_H

#include <torch/extension.h>

namespace audyn
{
    at::Tensor search_monotonic_alignment_by_viterbi(const at::Tensor &probs, const at::Tensor &tgt_lengths, const at::Tensor &src_lengths, const bool take_log);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_bipartite_match_precision_recall_fscore(
        at::Tensor input,
        at::Tensor target,
        at::Tensor input_lengths,
        at::Tensor target_lengths,
        const float tolerance);
    at::Tensor hungarian(const at::Tensor &cost);
}

#endif // AUDYN_H
