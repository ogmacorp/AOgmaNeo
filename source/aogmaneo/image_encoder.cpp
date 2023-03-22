// ----------------------------------------------------------------------------
//  aogma_neo
//  copyright(c) 2020-2023 ogma intelligent systems corp. all rights reserved.
//
//  this copy of aogma_neo is licensed to you under the terms described
//  in the aogmaneo_license.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "image_encoder.h"

using namespace aon;

void image_encoder::forward(
    const int2 &column_pos,
    const array<const byte_buffer*> &inputs,
    bool learn_enabled
) {
    int hidden_column_index = address2(column_pos, int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int max_index = -1;
    float max_activation = limit_min;

    const float byte_inv = 1.0f / 255.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum = 0.0f;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            visible_layer &vl = visible_layers[vli];
            const visible_layer_desc &vld = visible_layer_descs[vli];

            int diam = vld.radius * 2 + 1;

            // projection
            float2 h_to_v = float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

            int2 visible_center = project(column_pos, h_to_v);

            // lower corner
            int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

            // bounds of receptive field, clamped to input size
            int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
            int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(int2(ix, iy), int2(vld.size.x, vld.size.y));

                    int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                    int i_start = vld.size.z * (iy + ix * vld.size.y);

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wi_start;

                        float input = (*inputs[vli])[vc + i_start] * byte_inv;

                        float w = vl.protos[wi] * byte_inv;

                        float delta = input - w;

                        sum -= delta * delta;
                    }
                }
        }

        if (sum > max_activation || max_index == -1) {
            max_activation = sum;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;

    if (learn_enabled) {
        for (int dhc = -1; dhc <= 1; dhc++) {
            int hc = hidden_cis[hidden_column_index] + dhc;

            if (hc < 0 || hc >= hidden_size.z)
                continue;

            int hidden_cell_index = hc + hidden_cells_start;

            float rate = hidden_rates[hidden_cell_index];

            for (int vli = 0; vli < visible_layers.size(); vli++) {
                visible_layer &vl = visible_layers[vli];
                const visible_layer_desc &vld = visible_layer_descs[vli];

                int diam = vld.radius * 2 + 1;

                // projection
                float2 h_to_v = float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

                int2 visible_center = project(column_pos, h_to_v);

                // lower corner
                int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

                // bounds of receptive field, clamped to input size
                int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
                int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int visible_column_index = address2(int2(ix, iy), int2(vld.size.x, vld.size.y));

                        int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                        int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                        int i_start = vld.size.z * (iy + ix * vld.size.y);

                        for (int vc = 0; vc < vld.size.z; vc++) {
                            int wi = vc + wi_start;

                            float input = (*inputs[vli])[vc + i_start];

                            vl.protos[wi] = min(255, max(0, vl.protos[wi] + roundf(rate * (input - vl.protos[wi]))));
                        }
                    }
            }

            hidden_rates[hidden_cell_index] -= lr * rate;
        }
    }
}

void image_encoder::reconstruct(
    const int2 &column_pos,
    const int_buffer* recon_cis,
    int vli
) {
    visible_layer &vl = visible_layers[vli];
    visible_layer_desc &vld = visible_layer_descs[vli];

    int diam = vld.radius * 2 + 1;

    int visible_column_index = address2(column_pos, int2(vld.size.x, vld.size.y));

    int visible_cells_start = visible_column_index * vld.size.z;

    // projection
    float2 v_to_h = float2(static_cast<float>(hidden_size.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hidden_size.y) / static_cast<float>(vld.size.y));

    float2 h_to_v = float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));
                
    int2 reverse_radii(ceilf(v_to_h.x * (vld.radius * 2 + 1) * 0.5f), ceilf(v_to_h.y * (vld.radius * 2 + 1) * 0.5f));

    int2 hidden_center = project(column_pos, v_to_h);

    // lower corner
    int2 field_lower_bound(hidden_center.x - reverse_radii.x, hidden_center.y - reverse_radii.y);

    // bounds of receptive field, clamped to input size
    int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    int2 iter_upper_bound(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));
    
    // find current max
    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        float sum = 0.0f;
        float total = 0.0f;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 hidden_pos = int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, int2(hidden_size.x, hidden_size.y));
                int hidden_cell_index = address3(int3(hidden_pos.x, hidden_pos.y, (*recon_cis)[hidden_column_index]), hidden_size);

                int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, int2(visible_center.x - vld.radius, visible_center.y - vld.radius), int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                    float dist_x = static_cast<float>(abs(column_pos.x - visible_center.x)) / static_cast<float>(vld.radius + 1);
                    float dist_y = static_cast<float>(abs(column_pos.y - visible_center.y)) / static_cast<float>(vld.radius + 1);

                    float strength = min(1.0f - dist_x, 1.0f - dist_y);

                    sum += strength * vl.protos[wi];
                    total += strength;
                }
            }

        sum /= max(0.0001f, total);

        vl.reconstruction[visible_cell_index] = roundf(sum);
    }
}

void image_encoder::init_random(
    const int3 &hidden_size,
    const array<visible_layer_desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        visible_layer &vl = visible_layers[vli];
        const visible_layer_desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.protos.resize(num_hidden_cells * area * vld.size.z);

        // initialize to random values
        for (int i = 0; i < vl.protos.size(); i++)
            vl.protos[i] = rand() % 256;

        vl.reconstruction = byte_buffer(num_visible_cells, 0);
    }

    hidden_rates = float_buffer(num_hidden_cells, 0.5f);

    hidden_cis = int_buffer(num_hidden_columns, 0);
}

void image_encoder::step(
    const array<const byte_buffer*> &inputs,
    bool learn_enabled
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    #pragma omp parallel for
    for (int i = 0; i < num_hidden_columns; i++)
        forward(int2(i / hidden_size.y, i % hidden_size.y), inputs, learn_enabled);
}

void image_encoder::reconstruct(
    const int_buffer* recon_cis
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const visible_layer_desc &vld = visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        #pragma omp parallel for
        for (int i = 0; i < num_visible_columns; i++)
            reconstruct(int2(i / vld.size.y, i % vld.size.y), recon_cis, vli);
    }
}

int image_encoder::size() const {
    int size = sizeof(int3) + sizeof(float) + hidden_rates.size() * sizeof(float) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const visible_layer &vl = visible_layers[vli];
        const visible_layer_desc &vld = visible_layer_descs[vli];

        size += sizeof(visible_layer_desc) + vl.protos.size() * sizeof(byte);
    }

    return size;
}

void image_encoder::write(
    stream_writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&hidden_size), sizeof(int3));

    writer.write(reinterpret_cast<const void*>(&lr), sizeof(float));
    
    writer.write(reinterpret_cast<const void*>(&hidden_rates[0]), hidden_rates.size() * sizeof(float));

    writer.write(reinterpret_cast<const void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));
    
    int num_visible_layers = visible_layers.size();

    writer.write(reinterpret_cast<const void*>(&num_visible_layers), sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const visible_layer &vl = visible_layers[vli];
        const visible_layer_desc &vld = visible_layer_descs[vli];

        writer.write(reinterpret_cast<const void*>(&vld), sizeof(visible_layer_desc));

        writer.write(reinterpret_cast<const void*>(&vl.protos[0]), vl.protos.size() * sizeof(byte));
    }
}

void image_encoder::read(
    stream_reader &reader
) {
    reader.read(reinterpret_cast<void*>(&hidden_size), sizeof(int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    reader.read(reinterpret_cast<void*>(&lr), sizeof(float));

    hidden_rates.resize(num_hidden_cells);

    reader.read(reinterpret_cast<void*>(&hidden_rates[0]), hidden_rates.size() * sizeof(float));

    hidden_cis.resize(num_hidden_columns);

    reader.read(reinterpret_cast<void*>(&hidden_cis[0]), hidden_cis.size() * sizeof(int));

    int num_visible_layers;

    reader.read(reinterpret_cast<void*>(&num_visible_layers), sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        visible_layer &vl = visible_layers[vli];
        visible_layer_desc &vld = visible_layer_descs[vli];

        reader.read(reinterpret_cast<void*>(&vld), sizeof(visible_layer_desc));

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.protos.resize(num_hidden_cells * area * vld.size.z);

        reader.read(reinterpret_cast<void*>(&vl.protos[0]), vl.protos.size() * sizeof(byte));

        vl.reconstruction = byte_buffer(num_visible_cells, 0);
    }
}
