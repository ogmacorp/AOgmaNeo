// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
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
        Float_Buffer policy_weights;
        Float_Buffer policy_traces;
        Float_Buffer value_weights;
        Float_Buffer value_weights_delayed;
        Float_Buffer value_traces;

        Int_Buffer input_cis_prev;
    };

    struct Params {
        float vlr; // value learning rate
        float plr; // policy learning rate
        float leak; // dendrite ReLU leak
        float value_rate; // rate of delayed value weights
        float discount; // discount factor
        float trace_curve; // curvature of trace to prevent getting too large
        float trace_decay; // eligibility trace decay

        Params()
        :
        vlr(0.01f),
        plr(0.01f),
        leak(0.01f),
        value_rate(0.01f),
        discount(0.99f),
        trace_curve(16.0f),
        trace_decay(0.97f)
        {}
    };

private:
    Int3 hidden_size; // hidden/output/action size
    int num_dendrites_per_cell;

    Int_Buffer hidden_cis; // hidden states

    Float_Buffer hidden_acts;
    Float_Buffer hidden_acts_prev;

    Float_Buffer policy_dendrite_acts;
    Float_Buffer policy_dendrite_acts_prev;
    Float_Buffer value_dendrite_acts;
    Float_Buffer value_dendrite_acts_delayed;
    Float_Buffer value_dendrite_acts_prev;

    Float_Buffer hidden_values; // hidden value function output buffer

    // visible layers and descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        Int_Buffer_View hidden_target_cis_prev,
        float reward,
        float mimic,
        bool learn_enabled,
        unsigned long* state,
        const Params &params
    );

public:
    // initialized randomly
    void init_random(
        const Int3 &hidden_size,
        int num_dendrites_per_cell,
        const Array<Visible_Layer_Desc> &visible_layer_descs
    );

    // step (get actions and update)
    void step(
        const Array<Int_Buffer_View> &input_cis,
        Int_Buffer_View hidden_target_cis_prev,
        bool learn_enabled,
        float reward,
        float mimic,
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

    // get hidden activations (probabilities) for actions
    const Float_Buffer &get_hidden_acts() const {
        return hidden_acts;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }

    // merge list of decoders and write to this one
    void merge(
        const Array<Actor*> &actors,
        Merge_Mode mode
    );
};
}
