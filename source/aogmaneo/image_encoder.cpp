// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "image_encoder.h"

using namespace aon;

void Image_Encoder::forward(
    const Int2 &column_pos,
    const Array<Byte_Buffer_View> &inputs,
    bool learn_enabled
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int max_index = -1;
    float max_activation = 0.0f;

    int max_complete_index = 0;
    float max_complete_activation = 0.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        int sum = 0;
        int count = 0;
        int total = 0;

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

            count += (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * vld.size.z;

            Byte_Buffer_View vl_inputs = inputs[vli];

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                    int i_start = vld.size.z * (iy + ix * vld.size.y);

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wi_start;

                        int input = vl_inputs[vc + i_start];

                        sum += min(input, static_cast<int>(vl.weights0[wi])) + min(255 - input, static_cast<int>(vl.weights1[wi]));
                        total += vl.weights0[wi];
                        total += vl.weights1[wi];
                    }
                }
        }

        float match = (sum / 255.0f) / count;

        float activation = (sum / 255.0f) / (params.choice + total / 255.0f);

        if (match >= params.vigilance) {
            if (activation > max_activation) {
                max_activation = activation;
                max_index = hc;
            }
        }

        if (activation > max_complete_activation) {
            max_complete_activation = activation;
            max_complete_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = (max_index == -1 ? max_complete_index : max_index);

    if (learn_enabled && max_index != -1) {
        int hidden_cell_index_max = max_index + hidden_cells_start;

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

            Byte_Buffer_View vl_inputs = inputs[vli];

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int visible_column_index = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                    Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index_max));

                    int i_start = vld.size.z * (iy + ix * vld.size.y);

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wi_start;

                        int input = vl_inputs[vc + i_start];

                        vl.weights0[wi] = max(0, vl.weights0[wi] + ceilf(params.lr * (min(input, static_cast<int>(vl.weights0[wi])) - vl.weights0[wi])));
                        vl.weights1[wi] = max(0, vl.weights1[wi] + ceilf(params.lr * (min(255 - input, static_cast<int>(vl.weights1[wi])) - vl.weights1[wi])));
                    }
                }
        }
    }
}

void Image_Encoder::learn_reconstruction(
    const Int2 &column_pos,
    Byte_Buffer_View inputs,
    int vli,
    unsigned long* state
) {
    Visible_Layer &vl = visible_layers[vli];
    Visible_Layer_Desc &vld = visible_layer_descs[vli];

    int diam = vld.radius * 2 + 1;

    int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

    int visible_cells_start = visible_column_index * vld.size.z;

    // projection
    Float2 v_to_h = Float2(static_cast<float>(hidden_size.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hidden_size.y) / static_cast<float>(vld.size.y));

    Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));
                
    Int2 reverse_radii(ceilf(v_to_h.x * (vld.radius * 2 + 1) * 0.5f), ceilf(v_to_h.y * (vld.radius * 2 + 1) * 0.5f));

    Int2 hidden_center = project(column_pos, v_to_h);

    // lower corner
    Int2 field_lower_bound(hidden_center.x - reverse_radii.x, hidden_center.y - reverse_radii.y);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));
    
    const float byte_inv = 1.0f / 255.0f;

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        float sum = 0.0f;
        int count = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                Int2 hidden_pos = Int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

                Int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    int hidden_cell_index = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                    sum += vl.weights_recon[wi];
                    count++;
                }
            }

        sum /= max(1, count * 255);

        float target = inputs[visible_cell_index] * byte_inv;

        int delta = rand_roundf(params.rr * (target - min(1.0f, max(0.0f, (sum - 0.5f) * 2.0f * params.scale + 0.5f))) * 255.0f, state);

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                Int2 hidden_pos = Int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

                Int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    int hidden_cell_index = hidden_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                    vl.weights_recon[wi] = min(255, max(0, vl.weights_recon[wi] + delta));
                }
            }
    }
}

void Image_Encoder::reconstruct(
    const Int2 &column_pos,
    Int_Buffer_View recon_cis,
    int vli
) {
    Visible_Layer &vl = visible_layers[vli];
    Visible_Layer_Desc &vld = visible_layer_descs[vli];

    int diam = vld.radius * 2 + 1;

    int visible_column_index = address2(column_pos, Int2(vld.size.x, vld.size.y));

    int visible_cells_start = visible_column_index * vld.size.z;

    // projection
    Float2 v_to_h = Float2(static_cast<float>(hidden_size.x) / static_cast<float>(vld.size.x),
        static_cast<float>(hidden_size.y) / static_cast<float>(vld.size.y));

    Float2 h_to_v = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));
                
    Int2 reverse_radii(ceilf(v_to_h.x * (vld.radius * 2 + 1) * 0.5f), ceilf(v_to_h.y * (vld.radius * 2 + 1) * 0.5f));

    Int2 hidden_center = project(column_pos, v_to_h);

    // lower corner
    Int2 field_lower_bound(hidden_center.x - reverse_radii.x, hidden_center.y - reverse_radii.y);

    // bounds of receptive field, clamped to input size
    Int2 iter_lower_bound(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    Int2 iter_upper_bound(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));
    
    // find current max
    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        float sum = 0.0f;
        int count = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                Int2 hidden_pos = Int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

                Int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    int hidden_cell_index = recon_cis[hidden_column_index] + hidden_column_index * hidden_size.z;

                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                    sum += vl.weights_recon[wi];
                    count++;
                }
            }

        sum /= max(1, count) * 255;

        vl.reconstruction[visible_cell_index] = roundf(min(1.0f, max(0.0f, (sum - 0.5f) * 2.0f * params.scale + 0.5f)) * 255.0f);
    }
}

void Image_Encoder::init_random(
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
        int num_visible_cells = num_visible_columns * vld.size.z;

        int diam = vld.radius * 2 + 1;
        int area = diam * diam;

        vl.weights0.resize(num_hidden_cells * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());
        vl.weights_recon.resize(vl.weights0.size());

        // initialize to random values
        for (int i = 0; i < vl.weights0.size(); i++) {
            vl.weights0[i] = 255 - (rand() % init_weight_noisei);
            vl.weights1[i] = 255 - (rand() % init_weight_noisei);
            vl.weights_recon[i] = 128;
        }

        vl.reconstruction = Byte_Buffer(num_visible_cells, 0);
    }

    hidden_cis = Int_Buffer(num_hidden_columns, 0);
}

void Image_Encoder::step(
    const Array<Byte_Buffer_View> &inputs,
    bool learn_enabled,
    bool learn_recon
) {
    int num_hidden_columns = hidden_size.x * hidden_size.y;
    
    PARALLEL_FOR
    for (int i = 0; i < num_hidden_columns; i++)
        forward(Int2(i / hidden_size.y, i % hidden_size.y), inputs, learn_enabled);

    if (learn_enabled && learn_recon) {
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            unsigned int base_state = rand();

            PARALLEL_FOR
            for (int i = 0; i < num_visible_columns; i++) {
                unsigned long state = rand_get_state(base_state + i * rand_subseed_offset);

                learn_reconstruction(Int2(i / vld.size.y, i % vld.size.y), inputs[vli], vli, &state);
            }
        }
    }
}

void Image_Encoder::reconstruct(
    Int_Buffer_View recon_cis
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        int num_visible_columns = vld.size.x * vld.size.y;

        PARALLEL_FOR
        for (int i = 0; i < num_visible_columns; i++)
            reconstruct(Int2(i / vld.size.y, i % vld.size.y), recon_cis, vli);
    }
}

long Image_Encoder::size() const {
    long size = sizeof(Int3) + sizeof(Params) + hidden_cis.size() * sizeof(int) + sizeof(int);

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        size += sizeof(Visible_Layer_Desc) + 3 * vl.weights0.size() * sizeof(Byte);
    }

    return size;
}

long Image_Encoder::state_size() const {
    return hidden_cis.size() * sizeof(int);
}

long Image_Encoder::weights_size() const {
    long size = 0;

    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        size += 3 * vl.weights0.size() * sizeof(Byte);
    }

    return size;
}

void Image_Encoder::write(
    Stream_Writer &writer
) const {
    writer.write(&hidden_size, sizeof(Int3));

    writer.write(&params, sizeof(Params));
    
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    int num_visible_layers = visible_layers.size();

    writer.write(&num_visible_layers, sizeof(int));
    
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];
        const Visible_Layer_Desc &vld = visible_layer_descs[vli];

        writer.write(&vld, sizeof(Visible_Layer_Desc));

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
        writer.write(&vl.weights_recon[0], vl.weights_recon.size() * sizeof(Byte));
    }
}

void Image_Encoder::read(
    Stream_Reader &reader
) {
    reader.read(&hidden_size, sizeof(Int3));

    int num_hidden_columns = hidden_size.x * hidden_size.y;
    int num_hidden_cells = num_hidden_columns * hidden_size.z;

    reader.read(&params, sizeof(Params));

    hidden_cis.resize(num_hidden_columns);

    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));

    int num_visible_layers;

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

        vl.weights0.resize(num_hidden_cells * area * vld.size.z);
        vl.weights1.resize(vl.weights0.size());
        vl.weights_recon.resize(vl.weights0.size());

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
        reader.read(&vl.weights_recon[0], vl.weights_recon.size() * sizeof(Byte));

        vl.reconstruction = Byte_Buffer(num_visible_cells, 0);
    }
}

void Image_Encoder::write_state(
    Stream_Writer &writer
) const {
    writer.write(&hidden_cis[0], hidden_cis.size() * sizeof(int));
}

void Image_Encoder::read_state(
    Stream_Reader &reader
) {
    reader.read(&hidden_cis[0], hidden_cis.size() * sizeof(int));
}

void Image_Encoder::write_weights(
    Stream_Writer &writer
) const {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        const Visible_Layer &vl = visible_layers[vli];

        writer.write(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        writer.write(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
        writer.write(&vl.weights_recon[0], vl.weights_recon.size() * sizeof(Byte));
    }
}

void Image_Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights0[0], vl.weights0.size() * sizeof(Byte));
        reader.read(&vl.weights1[0], vl.weights1.size() * sizeof(Byte));
        reader.read(&vl.weights_recon[0], vl.weights_recon.size() * sizeof(Byte));
    }
}

void Image_Encoder::merge(
    const Array<Image_Encoder*> &image_encoders,
    Merge_Mode mode
) {
    switch (mode) {
    case merge_random:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights0.size(); i++) {
                int e = rand() % image_encoders.size();                

                vl.weights0[i] = image_encoders[e]->visible_layers[vli].weights0[i];
                vl.weights1[i] = image_encoders[e]->visible_layers[vli].weights1[i];
                vl.weights_recon[i] = image_encoders[e]->visible_layers[vli].weights_recon[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights0.size(); i++) {
                float total0 = 0.0f;
                float total1 = 0.0f;
                float total_recon = 0.0f;

                for (int e = 0; e < image_encoders.size(); e++) {
                    total0 += image_encoders[e]->visible_layers[vli].weights0[i];
                    total1 += image_encoders[e]->visible_layers[vli].weights1[i];
                    total_recon += image_encoders[e]->visible_layers[vli].weights_recon[i];
                }

                vl.weights0[i] = roundf(total0 / image_encoders.size());
                vl.weights1[i] = roundf(total1 / image_encoders.size());
                vl.weights_recon[i] = roundf(total_recon / image_encoders.size());
            }
        }

        break;
    }
}
