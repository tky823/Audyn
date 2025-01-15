/*
    for C++ extension with torch >= 2.4
*/

#include "audyn.h"

namespace audyn
{
    at::Tensor search_monotonic_alignment_by_viterbi(
        const at::Tensor &probs, const at::Tensor &tgt_lengths, const at::Tensor &src_lengths, const bool take_log = false)
    {
        TORCH_CHECK(probs.dim() == 3, "probs should be 3 dim.");
        TORCH_CHECK(tgt_lengths.dim() == 1, "tgt_lengths should be 1 dim.");
        TORCH_CHECK(src_lengths.dim() == 1, "src_lengths should be 1 dim.");

        int64_t batch_size = probs.size(0);
        int64_t max_tgt_length = probs.size(1);
        int64_t max_src_length = probs.size(2);

        int64_t num_threads = torch::get_num_threads();
        int64_t grain_size = std::ceil((batch_size - 1) / num_threads) + 1;

        double_t inf = std::numeric_limits<double_t>::infinity();
        torch::TensorOptions int64options = torch::TensorOptions().dtype(torch::kInt64).device(probs.device());

        at::Tensor log_probs;
        at::Tensor hard_align = torch::zeros({batch_size, max_tgt_length, max_src_length}, int64options);
        int64_t *hard_align_ptr = hard_align.data_ptr<int64_t>();

        if (take_log)
        {
            log_probs = torch::log(probs).contiguous();
        }
        else
        {
            log_probs = probs.contiguous();
        }

        float *log_probs_ptr = log_probs.data_ptr<float>();
        int64_t *tgt_lengths_ptr = tgt_lengths.data_ptr<int64_t>();
        int64_t *src_lengths_ptr = src_lengths.data_ptr<int64_t>();

        torch::parallel_for(
            0, batch_size, grain_size,
            [&](int64_t start, int64_t end)
            {
                for (int64_t batch_idx = start; batch_idx < end; batch_idx++)
                {
                    float *log_prob_ptr = log_probs_ptr + batch_idx * max_tgt_length * max_src_length;
                    int64_t tgt_length = tgt_lengths_ptr[batch_idx];
                    int64_t src_length = src_lengths_ptr[batch_idx];
                    int64_t tgt_idx, src_idx, start_src_idx, end_src_idx, min_src_idx, max_src_idx;

                    std::vector<float> log_seq_prob(max_tgt_length * max_src_length, -inf);

                    TORCH_CHECK(tgt_length >= src_length, "tgt_length should be greater than or equal src_length.");

                    // forward
                    log_seq_prob[0 * max_src_length + 0] = 0;

                    for (tgt_idx = 1; tgt_idx < tgt_length; tgt_idx++)
                    {
                        start_src_idx = std::max((int64_t)0, src_length - tgt_length + tgt_idx);
                        end_src_idx = std::min(src_length, tgt_idx + 1);

                        for (src_idx = start_src_idx; src_idx < end_src_idx; src_idx++)
                        {
                            min_src_idx = std::max((int64_t)0, src_idx - 1);
                            max_src_idx = std::min(tgt_idx - 1, src_idx);

                            auto slice_start = log_seq_prob.begin() + (tgt_idx - 1) * max_src_length + min_src_idx;
                            auto slice_end = log_seq_prob.begin() + (tgt_idx - 1) * max_src_length + max_src_idx + 1;
                            auto max_prev_log_seq_prob = std::max_element(slice_start, slice_end);
                            log_seq_prob[tgt_idx * max_src_length + src_idx] = *max_prev_log_seq_prob + log_prob_ptr[tgt_idx * max_src_length + src_idx];
                        }
                    }

                    // back track
                    src_idx = src_length - 1;
                    hard_align_ptr[batch_idx * max_tgt_length * max_src_length + (tgt_length - 1) * max_src_length + src_idx] = 1;

                    for (tgt_idx = tgt_length - 1; tgt_idx > 0; tgt_idx--)
                    {
                        min_src_idx = std::max((int64_t)0, src_idx - 1);
                        max_src_idx = std::min(tgt_idx - 1, src_idx);
                        auto slice_start = log_seq_prob.begin() + (tgt_idx - 1) * max_src_length + min_src_idx;
                        auto slice_end = log_seq_prob.begin() + (tgt_idx - 1) * max_src_length + max_src_idx + 1;
                        auto max_prev_log_seq_prob = std::max_element(slice_start, slice_end);
                        src_idx = std::distance(log_seq_prob.begin() + (tgt_idx - 1) * max_src_length, max_prev_log_seq_prob);
                        hard_align_ptr[batch_idx * max_tgt_length * max_src_length + (tgt_idx - 1) * max_src_length + src_idx] = 1;
                    }
                }
            });

        return hard_align;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(audyn, m)
{
    m.def("search_monotonic_alignment_by_viterbi(Tensor probs, Tensor tgt_lengths, Tensor src_lengths, bool take_log) -> Tensor");
}

TORCH_LIBRARY_IMPL(audyn, CPU, m)
{
    m.impl("search_monotonic_alignment_by_viterbi", &audyn::search_monotonic_alignment_by_viterbi);
}
