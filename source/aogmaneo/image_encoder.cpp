// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "image_encoder.h"

using namespace aon;

void Image_Encoder::forward(
    const Int2 &column_pos,
    const Array<Byte_Buffer_View> &inputs
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int max_index = 0;
    float max_activation = 0.0f;

    const float byte_inv = 1.0f / 255.0f;

    for (int hc = 0; hc < hidden_size.z; hc++) {
        int hidden_cell_index = hc + hidden_cells_start;

        float sum = 0.0f;
        int count = 0;

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

                    int visible_cells_start = vld.size.z * (iy + ix * vld.size.y);

                    for (int vc = 0; vc < vld.size.z; vc++) {
                        int wi = vc + wi_start;

                        sum += min(vl.weights_act[wi], vl_inputs[vc + visible_cells_start]);
                    }
                }
        }

        float activation = sum / count;

        if (activation > max_activation) {
            max_activation = activation;
            max_index = hc;
        }
    }

    hidden_cis[hidden_column_index] = max_index;
}

void Image_Encoder::backward(
    const Int2 &column_pos,
    int vli
) {
    Visible_Layer &vl = visible_layers[vli];
    const Visible_Layer_Desc &vld = visible_layer_descs[vli];

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

    for (int vc = 0; vc < vld.size.z; vc++) {
        int visible_cell_index = vc + visible_cells_start;

        int recon_match = 0;
        int recon_act = 0;
        int count = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                Int2 hidden_pos = Int2(ix, iy);

                int hidden_column_index = address2(hidden_pos, Int2(hidden_size.x, hidden_size.y));

                Int2 visible_center = project(hidden_pos, h_to_v);

                if (in_bounds(column_pos, Int2(visible_center.x - vld.radius, visible_center.y - vld.radius), Int2(visible_center.x + vld.radius + 1, visible_center.y + vld.radius + 1))) {
                    int hidden_ci = hidden_cis[hidden_column_index];

                    int hidden_cell_index_max = hidden_ci + hidden_column_index * hidden_size.z;

                    Int2 offset(column_pos.x - visible_center.x + vld.radius, column_pos.y - visible_center.y + vld.radius);

                    int wi = vc + vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index_max));

                    recon_match += vl.weights_match[wi];
                    recon_act += vl.weights_act[wi];
                    count++;
                }
            }

        vl.recons_match[visible_cell_index] = recon_match / max(1, count);
        vl.recons_act[visible_cell_index] = recon_act / max(1, count);
    }
}

void Image_Encoder::learn(
    const Int2 &column_pos,
    const Array<Byte_Buffer_View> &inputs
) {
    int hidden_column_index = address2(column_pos, Int2(hidden_size.x, hidden_size.y));

    int hidden_cells_start = hidden_column_index * hidden_size.z;

    int hidden_ci = hidden_cis[hidden_column_index];

    int hidden_cell_index_max = hidden_ci + hidden_cells_start;

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

                int visible_cells_start = vld.size.z * (iy + ix * vld.size.y);

                for (int vc = 0; vc < vld.size.z; vc++) {
                    int visible_cell_index = vc + visible_cells_start;

                    Byte recon_match = vl.recons_match[visible_cell_index];
                    Byte recon_act = vl.recons_act[visible_cell_index];

                    Byte input = vl_inputs[vc + visible_cells_start];

                    int wi = vc + wi_start;

                    vl.weights_act[wi] = max(0, vl.weights_act[wi] + roundf2i(params.lr * (min(recon_act, max(recon_match, vl.weights_match[wi])) - recon_act)));
                    vl.weights_match[wi] = max(vl.weights_match[wi], input);
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

        int delta = rand_roundf(params.rr * 255.0f * (target - min(1.0f, max(0.0f, (sum - 0.5f) * 2.0f * params.scale + 0.5f))), state);

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

        sum /= max(1, count * 255);

        vl.reconstruction[visible_cell_index] = roundf2b(255.0f * min(1.0f, max(0.0f, (sum - 0.5f) * 2.0f * params.scale + 0.5f)));
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

        vl.weights_match.resize(num_hidden_cells * area * vld.size.z);
        vl.weights_act.resize(vl.weights_match.size());
        vl.weights_recon.resize(vl.weights_match.size());

        // initialize to random values
        for (int i = 0; i < vl.weights_match.size(); i++) {
            vl.weights_match[i] = 0;
            vl.weights_act[i] = 255 - (rand() % init_weight_noisei);
            vl.weights_recon[i] = 128;
        }

        vl.recons_match.resize(num_visible_cells);
        vl.recons_act.resize(num_visible_cells);

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
        forward(Int2(i / hidden_size.y, i % hidden_size.y), inputs);

    if (learn_enabled) {
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];

            int num_visible_columns = vld.size.x * vld.size.y;

            PARALLEL_FOR
            for (int i = 0; i < num_visible_columns; i++)
                backward(Int2(i / vld.size.y, i % vld.size.y), vli);
        }

        PARALLEL_FOR
        for (int i = 0; i < num_hidden_columns; i++)
            learn(Int2(i / hidden_size.y, i % hidden_size.y), inputs);

        if (learn_recon) {
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

        size += sizeof(Visible_Layer_Desc) + 3 * vl.weights_match.size() * sizeof(Byte);
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

        size += 3 * vl.weights_match.size() * sizeof(Byte);
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

        writer.write(&vl.weights_match[0], vl.weights_match.size() * sizeof(Byte));
        writer.write(&vl.weights_act[0], vl.weights_act.size() * sizeof(Byte));
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

        vl.weights_match.resize(num_hidden_cells * area * vld.size.z);
        vl.weights_act.resize(vl.weights_match.size());
        vl.weights_recon.resize(vl.weights_match.size());

        reader.read(&vl.weights_match[0], vl.weights_match.size() * sizeof(Byte));
        reader.read(&vl.weights_act[0], vl.weights_act.size() * sizeof(Byte));
        reader.read(&vl.weights_recon[0], vl.weights_recon.size() * sizeof(Byte));

        vl.recons_match.resize(num_visible_cells);
        vl.recons_act.resize(num_visible_cells);

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

        writer.write(&vl.weights_match[0], vl.weights_match.size() * sizeof(Byte));
        writer.write(&vl.weights_act[0], vl.weights_act.size() * sizeof(Byte));
        writer.write(&vl.weights_recon[0], vl.weights_recon.size() * sizeof(Byte));
    }
}

void Image_Encoder::read_weights(
    Stream_Reader &reader
) {
    for (int vli = 0; vli < visible_layers.size(); vli++) {
        Visible_Layer &vl = visible_layers[vli];

        reader.read(&vl.weights_match[0], vl.weights_match.size() * sizeof(Byte));
        reader.read(&vl.weights_act[0], vl.weights_act.size() * sizeof(Byte));
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
        
            for (int i = 0; i < vl.weights_match.size(); i++) {
                int e = rand() % image_encoders.size();                

                vl.weights_match[i] = image_encoders[e]->visible_layers[vli].weights_match[i];
                vl.weights_act[i] = image_encoders[e]->visible_layers[vli].weights_act[i];
                vl.weights_recon[i] = image_encoders[e]->visible_layers[vli].weights_recon[i];
            }
        }

        break;
    case merge_average:
        for (int vli = 0; vli < visible_layers.size(); vli++) {
            Visible_Layer &vl = visible_layers[vli];
            const Visible_Layer_Desc &vld = visible_layer_descs[vli];
        
            for (int i = 0; i < vl.weights_match.size(); i++) {
                float total_match = 0.0f;
                float total_act = 0.0f;
                float total_recon = 0.0f;

                for (int e = 0; e < image_encoders.size(); e++) {
                    total_match += image_encoders[e]->visible_layers[vli].weights_match[i];
                    total_act += image_encoders[e]->visible_layers[vli].weights_act[i];
                    total_recon += image_encoders[e]->visible_layers[vli].weights_recon[i];
                }

                vl.weights_match[i] = roundf2b(total_match / image_encoders.size());
                vl.weights_act[i] = roundf2b(total_act / image_encoders.size());
                vl.weights_recon[i] = roundf2b(total_recon / image_encoders.size());
            }
        }

        break;
    }
}
