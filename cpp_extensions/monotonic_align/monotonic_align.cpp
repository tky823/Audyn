// NOTE: Some utilities are defined in not 'torch', but 'at' (ATen).

#include <torch/extension.h>

torch::Tensor search_monotonic_alignment_by_viterbi(torch::Tensor probs,
                                                    bool take_log = false)
{
    int64_t batch_size = probs.size(0);
    int64_t max_tgt_length = probs.size(1);
    int64_t max_src_length = probs.size(2);

    double_t inf = std::numeric_limits<double_t>::infinity();
    torch::TensorOptions float32options = probs.options();
    torch::TensorOptions int64options =
        torch::TensorOptions().dtype(torch::kInt64).device(probs.device());

    torch::Tensor hard_align = torch::zeros(
        {batch_size, max_tgt_length, max_src_length}, int64options);

    torch::parallel_for(
        0, batch_size, torch::internal::GRAIN_SIZE,
        [&](int64_t start, int64_t end)
        {
            for (int64_t batch_idx = start; batch_idx < end; batch_idx++)
            {
                torch::Tensor prob = probs.index({batch_idx});
                torch::Tensor log_prob, sum_prob;
                torch::Tensor log_seq_prob, prev_log_seq_prob, log_p;

                int64_t tgt_length, src_length;
                int64_t tgt_idx, src_idx;
                int64_t start_src_idx, end_src_idx;
                int64_t min_src_idx, max_src_idx;
                torch::indexing::Slice slice;

                if (take_log)
                {
                    log_prob = torch::log(prob);
                }
                else
                {
                    log_prob = prob;
                    prob = torch::exp(log_prob);
                }

                sum_prob = prob.sum(/*dim=*/1);
                tgt_length = torch::count_nonzero(sum_prob).item<int64_t>();

                sum_prob = prob.sum(/*dim=*/0);
                src_length = torch::count_nonzero(sum_prob).item<int64_t>();

                assert(tgt_length >= src_length);

                log_seq_prob = torch::full({max_tgt_length, max_src_length},
                                           /*fill_value=*/-inf, float32options);

                // forward
                log_seq_prob.index_put_({0, 0}, 0);

                for (tgt_idx = 1; tgt_idx < tgt_length; tgt_idx++)
                {
                    start_src_idx =
                        std::max((int64_t)0, src_length - tgt_length + tgt_idx);
                    end_src_idx = std::min(src_length, tgt_idx + 1);

                    for (src_idx = start_src_idx; src_idx < end_src_idx;
                         src_idx++)
                    {
                        min_src_idx = std::max((int64_t)0, src_idx - 1);
                        max_src_idx = std::min(tgt_idx - 1, src_idx);

                        slice = torch::indexing::Slice(min_src_idx,
                                                       max_src_idx + 1);
                        prev_log_seq_prob =
                            log_seq_prob.index({tgt_idx - 1, slice});
                        log_p = torch::max(prev_log_seq_prob) +
                                log_prob.index({tgt_idx, src_idx});
                        log_seq_prob.index_put_({tgt_idx, src_idx}, log_p);
                    }
                }

                // back track
                src_idx = src_length - 1;
                hard_align.index_put_({batch_idx, tgt_length - 1, src_idx}, 1);

                for (tgt_idx = tgt_length - 1; tgt_idx > 0; tgt_idx--)
                {
                    min_src_idx = std::max((int64_t)0, src_idx - 1);
                    max_src_idx = std::min(tgt_idx - 1, src_idx);
                    slice =
                        torch::indexing::Slice(min_src_idx, max_src_idx + 1);
                    prev_log_seq_prob =
                        log_seq_prob.index({tgt_idx - 1, slice});
                    src_idx = min_src_idx +
                              torch::argmax(prev_log_seq_prob).item<int64_t>();
                    hard_align.index_put_({batch_idx, tgt_idx - 1, src_idx}, 1);
                }
            }
        });

    return hard_align;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("search_monotonic_alignment_by_viterbi", &search_monotonic_alignment_by_viterbi,
          "Search monotonic alignment by Viterbi algorithm");
}
