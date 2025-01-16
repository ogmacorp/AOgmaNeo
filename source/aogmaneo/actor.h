// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "helpers.h"

namespace aon {
// a reinforcement learning layer
class Actor {
public:
    // visible layer descriptor
    struct Visible_Layer_Desc {
        Int3 size; // visible/input size

        int radius; // radius onto input

        // defaults
        Visible_Layer_Desc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Float_Buffer value_weights;
        Float_Buffer adv_weights;
    };

    // history sample for delayed updates
    struct History_Sample {
        Array<Int_Buffer> input_cis;
        Int_Buffer hidden_target_cis_prev;

        float reward;
    };

    struct Params {
        float scale; // softmax scale
        float vlr; // value learning rate
        float alr; // advantage learning rate
        float leak; // ReLU leak
        float discount; // discount fActor
        float td_scale_decay; // decay on max abs td error scaler
        int n_steps; // q steps
        int history_iters; // number of iterations over samples

        Params()
        :
        scale(4.0f),
        vlr(0.001f),
        alr(0.004f),
        leak(0.01f),
        discount(0.99f),
        td_scale_decay(0.999f),
        n_steps(8),
        history_iters(16)
        {}
    };

private:
    Int3 hidden_size; // hidden/output/action size
    int num_dendrites_per_cell;

    // current history size - fixed after initialization. determines length of wait before updating
    int history_size;

    Int_Buffer hidden_cis; // hidden states

    Float_Buffer hidden_advs;

    Float_Buffer dendrite_advs;

    Float_Buffer hidden_td_scales;

    Circle_Buffer<History_Sample> history_samples; // history buffer, fixed length

    // visible layers and descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        unsigned long* state,
        const Params &params
    );

    void learn(
        const Int2 &column_pos,
        int t,
        const Params &params
    );

public:
    // initialized randomly
    void init_random(
        const Int3 &hidden_size,
        int num_dendrites_per_cell,
        int history_capacity,
        const Array<Visible_Layer_Desc> &visible_layer_descs
    );

    // step (get actions and update)
    void step(
        const Array<Int_Buffer_View> &input_cis,
        Int_Buffer_View hidden_target_cis_prev,
        bool learn_enabled,
        float reward,
        const Params &params
    );

    void clear_state();

    // serialization
    long size() const; // returns size in bytes
    long state_size() const; // returns size of state in bytes
    long weights_size() const; // returns size of weights in bytes

    void write(
        Stream_Writer &writer
    ) const;

    void read(
        Stream_Reader &reader
    );

    void write_state(
        Stream_Writer &writer
    ) const;

    void read_state(
        Stream_Reader &reader
    );

    void write_weights(
        Stream_Writer &writer
    ) const;

    void read_weights(
        Stream_Reader &reader
    );

    // get number of visible layers
    int get_num_visible_layers() const {
        return visible_layers.size();
    }

    // get a visible layer
    const Visible_Layer &get_visible_layer(
        int i // index of layer
    ) const {
        return visible_layers[i];
    }

    // get a visible layer descriptor
    const Visible_Layer_Desc &get_visible_layer_desc(
        int i // index of layer
    ) const {
        return visible_layer_descs[i];
    }

    // get hidden state/output/actions
    const Int_Buffer &get_hidden_cis() const {
        return hidden_cis;
    }

    // get hidden advs
    const Float_Buffer &get_hidden_advs() const {
        return hidden_advs;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }

    int get_history_capacity() const {
        return history_samples.size();
    }

    int get_history_size() const {
        return history_size;
    }

    // merge list of decoders and write to this one
    void merge(
        const Array<Actor*> &actors,
        Merge_Mode mode
    );
};
}
