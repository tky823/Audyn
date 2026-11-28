#include <ATen/Parallel.h>

#include "audyn.h"

#include <cstdint>
#include <vector>

namespace
{
    struct State
    {
        std::vector<float> row_potential, column_potential;
        std::vector<int64_t> assigned_column, assigned_row;
    };

    State initialize(const float *cost, int64_t num_rows)
    {
        State state{std::vector<float>(num_rows), std::vector<float>(num_rows, 0.0f), std::vector<int64_t>(num_rows, -1), std::vector<int64_t>(num_rows, -1)};
        for (int64_t row_idx = 0; row_idx < num_rows; row_idx++)
        {
            state.row_potential[row_idx] = cost[row_idx * num_rows];

            for (int64_t column_idx = 1; column_idx < num_rows; column_idx++)
            {
                if (cost[row_idx * num_rows + column_idx] <
                    state.row_potential[row_idx])
                {
                    state.row_potential[row_idx] = cost[row_idx * num_rows + column_idx];
                }
            }
        }

        return state;
    }

    int64_t search(const float *cost, int64_t num_rows, int64_t tgt_row_idx, State &state, std::vector<int64_t> &prev_rows)
    {
        std::vector<bool> is_row_visited(num_rows, false), is_column_visited(num_rows, false);
        std::vector<float> min_slack(num_rows);
        is_row_visited[tgt_row_idx] = true;

        for (int64_t column_idx = 0; column_idx < num_rows; column_idx++)
        {
            min_slack[column_idx] = cost[tgt_row_idx * num_rows + column_idx] - state.row_potential[tgt_row_idx] - state.column_potential[column_idx];
            prev_rows[column_idx] = tgt_row_idx;
        }

        for (int64_t step = 0; step < num_rows; step++)
        {
            int64_t column_idx = -1;

            for (int64_t _column_idx = 0; _column_idx < num_rows; _column_idx++)
            {
                if (!is_column_visited[_column_idx] && (column_idx == -1 || min_slack[_column_idx] < min_slack[column_idx]))
                {
                    column_idx = _column_idx;
                }
            }

            const float _min_slack = min_slack[column_idx];

            for (int64_t row_idx = 0; row_idx < num_rows; row_idx++)
            {
                if (is_row_visited[row_idx])
                {
                    state.row_potential[row_idx] += _min_slack;
                }
            }

            for (int64_t _column_idx = 0; _column_idx < num_rows; _column_idx++)
            {
                if (is_column_visited[_column_idx])
                {
                    state.column_potential[_column_idx] -= _min_slack;
                }
                else
                {
                    min_slack[_column_idx] -= _min_slack;
                }
            }

            if (state.assigned_row[column_idx] == -1)
            {
                return column_idx;
            }

            const int64_t row_idx = state.assigned_row[column_idx];
            is_row_visited[row_idx] = true;
            is_column_visited[column_idx] = true;

            for (int64_t _column_idx = 0; _column_idx < num_rows; _column_idx++)
            {
                if (is_column_visited[_column_idx])
                {
                    continue;
                }

                const float reduced_cost = cost[row_idx * num_rows + _column_idx] - state.row_potential[row_idx] - state.column_potential[_column_idx];

                if (reduced_cost < min_slack[_column_idx])
                {
                    min_slack[_column_idx] = reduced_cost;
                    prev_rows[_column_idx] = row_idx;
                }
            }
        }

        TORCH_CHECK(false, "Hungarian search failed to find an augmenting path");

        return -1;
    }

    void traverse(int64_t tgt_row_idx, int64_t column_idx,
                  const std::vector<int64_t> &prev_rows, State &state)
    {
        while (true)
        {
            const int64_t row_idx = prev_rows[column_idx];
            const int64_t _column_idx = state.assigned_column[row_idx];
            state.assigned_column[row_idx] = column_idx;
            state.assigned_row[column_idx] = row_idx;

            if (row_idx == tgt_row_idx)
            {
                return;
            }

            column_idx = _column_idx;
        }
    }
} // namespace

namespace audyn
{
    at::Tensor hungarian(const at::Tensor &cost)
    {
        // based on Kuhn-Munkres method
        TORCH_CHECK(cost.dim() == 3, "cost must be a 3-dimensional tensor");
        TORCH_CHECK(cost.size(1) == cost.size(2), "cost must contain square matrices");
        TORCH_CHECK(cost.device().is_cpu(), "cost must be on the CPU");
        TORCH_CHECK(cost.scalar_type() == torch::kFloat32, "cost must have float32 dtype");
        TORCH_CHECK(cost.is_contiguous(), "cost must be contiguous");

        const int64_t batch_size = cost.size(0);
        const int64_t num_rows = cost.size(1);
        auto output = torch::empty({batch_size, num_rows}, cost.options().dtype(torch::kInt64));

        if (batch_size == 0 || num_rows == 0)
        {
            return output;
        }

        const float *cost_ptr = cost.data_ptr<float>();
        int64_t *output_ptr = output.data_ptr<int64_t>();
        const int64_t num_threads = torch::get_num_threads();
        const int64_t grain_size = (batch_size - 1) / num_threads + 1;

        torch::parallel_for(
            0, batch_size, grain_size, [&](int64_t start, int64_t end)
            {
        for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
          const float *_cost_ptr = cost_ptr + batch_idx * num_rows * num_rows;
          auto state = initialize(_cost_ptr, num_rows);
          std::vector<int64_t> prev_rows(num_rows);

          for (int64_t tgt_row_idx = 0; tgt_row_idx < num_rows; tgt_row_idx++) {
            const int64_t column_idx = search(_cost_ptr, num_rows, tgt_row_idx, state, prev_rows);
            traverse(tgt_row_idx, column_idx, prev_rows, state);
          }

          for (int64_t row_idx = 0; row_idx < num_rows; row_idx++) {
            output_ptr[batch_idx * num_rows + row_idx] = state.assigned_column[row_idx];
          }
        } });

        return output;
    }
} // namespace audyn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hungarian", &audyn::hungarian,
          "Minimum-cost assignment for batched square matrices");
}
