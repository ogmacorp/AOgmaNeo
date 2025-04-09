// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "encoder.h"

using namespace aon;

void Encoder::forward(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    float count = 0.0f;
    float total_importance = 0.0f;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int diam = vld.radius * 2 + 1;

        // projection
        Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

        Int2 visible_center = project(column_pos, h_to_v);

        // lower corner
        Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

        // bounds of receptive field, clamped to input size
        Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        Int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

        count += vl.importance * (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        total_importance += vl.importance;

        Int_Buffer_View vl_input_cis = input_cis[vli];

        for (int hc = 0; hc < hidden_size.z; hc++) {
            int hidden_cell_index = hc + hidden_cells_start;

            vl.hidden_sums[hidden_cell_index] = 0.0f;
        }

        const float vld_size_z_inv = 1.0f / vld.size.z;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                int in_ci = vl_input_cis[visible_column_index];

                float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                int wi_start = hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                for (int hc = 0; hc < hidden_size.z; hc++) {
                    int hidden_cell_index = hc + hidden_cells_start;

                    int wi = hc + wi_start;

                    float delta = in_value - vl.centroids[wi];

                    vl.hidden_sums[hidden_cell_index] += delta * delta;
                }
            }
    }

    count /= max(limit_small, total_importance);

    int max_index = -1;
    float max_activation = 0.0f;

    int max_complete_index = 0;
    float max_complete_activation = 0.0f;
    
    float max_match = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum = 0.0f;

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            sum += vl.hidden_sums[hidden_cell_index] * vl.importance;
        }

        sum /= max(limit_small, total_importance);

        hidden_norms[hidden_cell_index] = sqrtf(sum);

        float match = 1.0f - max(hidden_radii[hidden_cell_index], hidden_norms[hidden_cell_index]);

        float activation = (1.0f - max(hidden_radii[hidden_cell_index], hidden_norms[hidden_cell_index])) / (1.0f - hidden_radii[hidden_cell_index] + params.choice);

        hidden_learn_flags[hidden_cell_index] = (match >= params.category_vigilance);

        if (hidden_learn_flags[hidden_cell_index] && activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }

        if (activation > max_complete_activation) {
            max_complete_activation = activation;
            max_complete_index = hc;
        }

        max_match = max(max_match, match);
    }

    hidden_comparisons[hidden_column_index] = (max_match >= params.compare_vigilance ? max_complete_activation : 0.0f);

    hidden_cis[hidden_column_index] = (max_index == -1 ? max_complete_index : max_index);
}

void Encoder::learn(
    const Int2 &column_pos,
    const Array<Int_Buffer_View> &input_cis,
    const Params &params
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    float hidden_max = hidden_comparisons[hidden_column_index];

    int num_higher = 0;
    int count = 1; // start at 1 since self is skipped

    for (int dcx = -params.l_radius; dcx <= params.l_radius; dcx++)
        for (int dcy = -params.l_radius; dcy <= params.l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            Int2 other_column_pos(column_pos.x + dcx, column_pos.y + dcy);

            if (in_bounds0(other_column_pos, Int2(hidden_size.x, hidden_size.y))) {
                int other_hidden_column_index = address2(other_column_pos, Int2(hidden_size.x, hidden_size.y));

                if (hidden_comparisons[other_hidden_column_index] >= hidden_max)
                    num_higher++;

                count++;
            }
        }

    float ratio = static_cast<float>(num_higher) / static_cast<float>(count);

    if (ratio > params.active_ratio)
        return;

    int hidden_ci = hidden_cis[hidden_column_index];

    for (int dhc = -params.n_radius; dhc <= params.n_radius; dhc++) {
        int hc = hidden_ci + dhc;

        if (hc < 0 || hc >= hidden_size.z)
            continue;

        // spatial
        int hidden_cell_index = hc + hidden_cells_start;

        if (!hidden_learn_flags[hidden_cell_index])
            continue;

        float base_rate = (hidden_radii[hidden_cell_index] > 0.0f ? params.lr : 1.0f) * powf(params.falloff, abs(dhc));

        float centroid_rate = base_rate * (1.0f - min(hidden_radii[hidden_cell_index], hidden_norms[hidden_cell_index]) / max(limit_small, hidden_norms[hidden_cell_index]));

        hidden_radii[hidden_cell_index] += base_rate * (max(hidden_radii[hidden_cell_index], hidden_norms[hidden_cell_index]) - hidden_radii[hidden_cell_index]);

        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int diam = vld.radius * 2 + 1;

            // projection
            Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

            Int2 visible_center = project(column_pos, h_to_v);

            // lower corner
            Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

            // bounds of receptive field, clamped to input size
            Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
            Int2 iter_upper_bound(min(vld.size.x - 1, visible_center.x + vld.radius), min(vld.size.y - 1, visible_center.y + vld.radius));

            Int_Buffer_View vl_input_cis = input_cis[vli];

            const float vld_size_z_inv = 1.0f / vld.size.z;

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    int in_ci = vl_input_cis[visible_column_index];

                    float in_value = (in_ci + 0.5f) * vld_size_z_inv;

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi = hc + hidden_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

                    vl.centroids[wi] += centroid_rate * (in_value - vl.centroids[wi]);
                }
        }
    }
}

void Encoder::init_random(
    const Int3 &hidden_size,
    const Array<Visible_Layer_Desc> &visible_layer_descs
) {
    this->visible_layer_descs = visible_layer_descs;

    this->hidden_size = hidden_size;

    visible_layers.resize(visible_layer_descs.size());

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    // create layers
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = this->visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.centroids.resize(num_hidden_cells * area);

        for (int i = 0; i < vl.centroids.size(); i++)
            vl.centroids[i] = randf();

        vl.hidden_sums.resize(num_hidden_cells);
        vl.hidden_totals.resize(num_hidden_cells);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);

    hidden_norms.resize(num_hidden_cells);

    hidden_radii = Float_Buffer(num_hidden_cells, 0.0f);

    hidden_learn_flags.resize(num_hidden_cells);

    hidden_comparisons.resize(num_hidden_columns);
}

void Encoder::step(
    const Array<Int_Buffer_View> &input_cis,
    bool learn_enabled,
    const Params &params
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);

    if (learn_enabled) {
        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), input_cis, params);
    }
}

void Encoder::clear_state() {
    hidden_cis.fill(0);
}

long Encoder::size() const {
    long size = sizeof(Int3) + hidden_cis.size() * sizeof(int) + hidden_radii.size() * sizeof(float) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += sizeof(Visible_Layer_Desc) + vl.centroids.size() * sizeof(float) + sizeof(float);
    }

    return size;
}

long Encoder::state_size() const {
    return 2 * hidden_cis.size() * sizeof(int);
}

long Encoder::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += 2 * vl.centroids.size() * sizeof(float);
    }

    return size;
}

void Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));

    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    writer.write(&hidden_radii[0], hidden_radii.size() * sizeof(float));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.centroids[0], vl.centroids.size() * sizeof(float));

        writer.write(&vl.importance, sizeof(float));
    }
}

void Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));

    // pre-compute dimensions
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    hidden_cis.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    hidden_learn_flags.resize(num_hidden_cells);

    hidden_radii.resize(num_hidden_cells);

    reader.read(&hidden_radii[0], hidden_radii.size() * sizeof(float));

    hidden_comparisons.resize(num_hidden_columns);

    int num_visible_layers = visible_layers.size();

    reader.read(&num_visible_layers, sizeof(int));

    visible_layers.resize(num_visible_layers);
    visible_layer_descs.resize(num_visible_layers);
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];
        Visible_Layer_Desc &vld = visible_layer_descs[vli];

        reader.read(&vld, sizeof(Visible_Layer_Desc));

        int num_visible_columns = vld.size.x * vld.size.y;
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.centroids.resize(num_hidden_cells * area);

        reader.read(&vl.centroids[0], vl.centroids.size() * sizeof(float));

        vl.hidden_sums.resize(num_hidden_cells);
        vl.hidden_totals.resize(num_hidden_cells);

        reader.read(&vl.importance, sizeof(float));
    }
}

void Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
}

void Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
}

void Encoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.centroids[0], vl.centroids.size() * sizeof(float));
    }

    writer.write(&hidden_radii[0], hidden_radii.size() * sizeof(float));
}

void Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.centroids[0], vl.centroids.size() * sizeof(float));
    }
    
    reader.read(&hidden_radii[0], hidden_radii.size() * sizeof(float));
}

void Encoder::merge(
    const Array<Encoder*> &encoders,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.centroids.size(); i++) {
                int e = rand() % encoders.size();                

                vl.centroids[i] = encoders[e]->visible_layers[vli].centroids[i];
            }
        }

        for (int i = 0; i < hidden_radii.size(); i++) {
            int e = rand() % encoders.size();                

            hidden_radii[i] = encoders[e]->hidden_radii[i];
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.centroids.size(); i++) {
                float total = 0.0f;

                for (int e = 0; e < encoders.size(); e++)
                    total += encoders[e]->visible_layers[vli].centroids[i];

                vl.centroids[i] = total / encoders.size();
            }
        }

        for (int i = 0; i < hidden_radii.size(); i++) {
            float total = 0.0f;

            for (int e = 0; e < encoders.size(); e++)
                total += encoders[e]->hidden_radii[i];

            hidden_radii[i] = total / encoders.size();
        }

        break;
    }
}
