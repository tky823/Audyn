#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <pybind11/stl.h>
#include <cmath>
#include <tuple>
#include <algorithm>

#include "audyn.h"

namespace audyn
{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_bipartite_match_precision_recall_fscore(
        torch::Tensor input,
        torch::Tensor target,
        torch::Tensor input_lengths,
        torch::Tensor target_lengths,
        const float tolerance)
    {
        TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor of shape (batch_size, max_input_len)");
        TORCH_CHECK(target.dim() == 2, "target must be a 2D tensor of shape (batch_size, max_target_len)");
        TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be a 1D tensor");
        TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be a 1D tensor");

        int64_t batch_size = input.size(0);

        TORCH_CHECK(target.size(0) == batch_size, "batch size mismatch between input and target");
        TORCH_CHECK(input_lengths.size(0) == batch_size, "batch size mismatch for input_lengths");
        TORCH_CHECK(target_lengths.size(0) == batch_size, "batch size mismatch for target_lengths");

        input = input.contiguous().to(torch::kFloat32);
        target = target.contiguous().to(torch::kFloat32);
        input_lengths = input_lengths.contiguous().to(torch::kInt64);
        target_lengths = target_lengths.contiguous().to(torch::kInt64);

        int64_t max_input_nodes = input.size(1);
        int64_t max_target_nodes = target.size(1);

        const int64_t *input_lengths_ptr = input_lengths.data_ptr<int64_t>();
        const int64_t *target_lengths_ptr = target_lengths.data_ptr<int64_t>();
        auto float32options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
        torch::Tensor precision = torch::zeros({batch_size}, float32options);
        torch::Tensor recall = torch::zeros({batch_size}, float32options);
        torch::Tensor f_score = torch::zeros({batch_size}, float32options);

        const float *input_ptr = input.data_ptr<float>();
        const float *target_ptr = target.data_ptr<float>();
        float *precision_ptr = precision.data_ptr<float>();
        float *recall_ptr = recall.data_ptr<float>();
        float *f_score_ptr = f_score.data_ptr<float>();

        int64_t num_threads = torch::get_num_threads();
        int64_t grain_size = std::ceil((batch_size - 1) / num_threads) + 1;

        torch::parallel_for(0, batch_size, grain_size, [&](int64_t start, int64_t end)
                            {
            for (int64_t batch_idx = start; batch_idx < end; batch_idx++)
            {
                const float *_input = input_ptr + batch_idx * max_input_nodes;
                const float *_target = target_ptr + batch_idx * max_target_nodes;

                int64_t num_inputs = input_lengths_ptr[batch_idx];
                int64_t num_targets = target_lengths_ptr[batch_idx];

                num_inputs = std::min(num_inputs, max_input_nodes);
                num_targets = std::min(num_targets, max_target_nodes);

                int64_t i = 0;
                int64_t j = 0;
                int64_t tp = 0;

                while (i < num_inputs && j < num_targets)
                {
                    float x_i = _input[i];
                    float y_j = _target[j];

                    if (std::abs(x_i - y_j) <= tolerance)
                    {
                        tp++;
                        i++;
                        j++;
                    }
                    else if (x_i < y_j)
                    {
                        i++;
                    }
                    else
                    {
                        j++;
                    }
                }

                float _precision = 0.0f;
                float _recall = 0.0f;
                float _f_score = 0.0f;

                if (num_inputs > 0)
                {
                    _precision = static_cast<float>(tp) / num_inputs;
                }

                if (num_targets > 0)
                {
                    _recall = static_cast<float>(tp) / num_targets;
                }

                if (_precision + _recall > 0.0f)
                {
                    _f_score = (2.0f * _precision * _recall) / (_precision + _recall);
                }

                precision_ptr[batch_idx] = _precision;
                recall_ptr[batch_idx] = _recall;
                f_score_ptr[batch_idx] = _f_score;
            } });

        return {precision, recall, f_score};
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_bipartite_match_precision_recall_fscore",
          &audyn::compute_bipartite_match_precision_recall_fscore,
          "Compute precision, recall, and F-score using bipartite matching");
}
