#ifndef AUDYN_H
#define AUDYN_H

#include <torch/extension.h>

namespace audyn
{
    at::Tensor search_monotonic_alignment_by_viterbi(const at::Tensor &probs, const at::Tensor &tgt_lengths, const at::Tensor &src_lengths, const bool take_log);
}

#endif // AUDYN_H
