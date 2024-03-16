// NOTE: Some utilities are defined in not 'torch', but 'at' (ATen).

#include <torch/extension.h>

torch::Tensor search_monotonic_alignment_by_viterbi(
    torch::Tensor probs, torch::Tensor tgt_lengths, torch::Tensor src_lengths, bool take_log = false)
{
    int64_t batch_size = probs.size(0);
    int64_t max_tgt_length = probs.size(1);
    int64_t max_src_length = probs.size(2);

    int64_t num_threads = torch::get_num_threads();
    int64_t grain_size = std::ceil((batch_size - 1) / num_threads) + 1;

    double_t inf = std::numeric_limits<double_t>::infinity();
    torch::TensorOptions float32options = probs.options();
    torch::TensorOptions int64options = torch::TensorOptions().dtype(torch::kInt64).device(probs.device());

    torch::Tensor log_probs;
    std::vector<int64_t> flattened_hard_align(batch_size * max_tgt_length * max_src_length, 0);

    if (take_log)
    {
        log_probs = torch::log(probs);
    }
    else
    {
        log_probs = probs;
    }

    torch::parallel_for(
        0, batch_size, grain_size,
        [&](int64_t start, int64_t end)
        {
            for (int64_t batch_idx = start; batch_idx < end; batch_idx++)
            {
                torch::Tensor log_prob = log_probs.index({batch_idx}).contiguous();
                int64_t tgt_length = tgt_lengths.index({batch_idx}).item<int64_t>();
                int64_t src_length = src_lengths.index({batch_idx}).item<int64_t>();
                int64_t tgt_idx, src_idx, start_src_idx, end_src_idx, min_src_idx, max_src_idx;

                float *data_ptr = log_prob.data_ptr<float>();
                std::vector<float> flattened_log_prob(max_tgt_length * max_src_length);
                std::vector<float> log_seq_prob(max_tgt_length * max_src_length, -inf);

                flattened_log_prob.assign(data_ptr, data_ptr + max_tgt_length * max_src_length);

                assert(tgt_length >= src_length);

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
                        log_seq_prob[tgt_idx * max_src_length + src_idx] = *max_prev_log_seq_prob + flattened_log_prob[tgt_idx * max_src_length + src_idx];
                    }
                }

                // back track
                src_idx = src_length - 1;
                flattened_hard_align[batch_idx * max_tgt_length * max_src_length + (tgt_length - 1) * max_src_length + src_idx] = 1;

                for (tgt_idx = tgt_length - 1; tgt_idx > 0; tgt_idx--)
                {
                    min_src_idx = std::max((int64_t)0, src_idx - 1);
                    max_src_idx = std::min(tgt_idx - 1, src_idx);
                    auto slice_start = log_seq_prob.begin() + (tgt_idx - 1) * max_src_length + min_src_idx;
                    auto slice_end = log_seq_prob.begin() + (tgt_idx - 1) * max_src_length + max_src_idx + 1;
                    auto max_prev_log_seq_prob = std::max_element(slice_start, slice_end);
                    src_idx = std::distance(log_seq_prob.begin() + (tgt_idx - 1) * max_src_length, max_prev_log_seq_prob);
                    flattened_hard_align[batch_idx * max_tgt_length * max_src_length + (tgt_idx - 1) * max_src_length + src_idx] = 1;
                }
            }
        });

    torch::Tensor hard_align = torch::from_blob(flattened_hard_align.data(), {batch_size, max_tgt_length, max_src_length}, int64options);

    return hard_align.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("search_monotonic_alignment_by_viterbi", &search_monotonic_alignment_by_viterbi,
          "Search monotonic alignment by Viterbi algorithm");
}
