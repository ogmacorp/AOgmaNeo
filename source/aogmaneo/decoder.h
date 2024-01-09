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
// a prediction layer (predicts x_(t+1))
class Decoder {
public:
    // visible layer descriptor
    struct Visible_Layer_Desc {
        Int3 size; // size of input

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
        Byte_Buffer weights;

        Int_Buffer input_cis_prev; // previous timestep (prev) input states

        Float_Buffer gates;
    };

    struct Params {
        float scale; // scale of softmax
        float lr; // learning rate
        float gcurve;

        Params()
        :
        scale(64.0f),
        lr(0.1f),
        gcurve(16.0f)
        {}
    };

private:
    Int3 hidden_size; // size of the output/hidden/prediction
    int num_dendrites_per_cell;

    Int_Buffer hidden_cis; // hidden state

    Int_Buffer hidden_cell_dis;

    Float_Buffer hidden_acts;

    Float_Buffer dendrite_acts;

    Float_Buffer dendrite_deltas;

    // visible layers and descs
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Array<Int3> visible_pos_vlis; // for parallelization, cartesian product of column coordinates and visible layers

    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        const Params &params
    );

    void update_gates(
        const Int2 &column_pos,
        int vli,
        const Params &params
    );

    void learn(
        const Int2 &column_pos,
        Int_Buffer_View hidden_target_cis,
        unsigned long* state,
        const Params &params
    );

public:
    // create with random initialization
    void init_random(
        const Int3 &hidden_size, // hidden/output/prediction size
        int num_dendrites_per_cell,
        const Array<Visible_Layer_Desc> &visible_layer_descs
    );

    // activate the predictor (predict values)
    void step(
        const Array<Int_Buffer_View> &input_cis,
        Int_Buffer_View hidden_target_cis,
        bool learn_enabled,
        const Params &params
    );

    void clear_state();

    // serialization
    long size() const; // returns size in Bytes
    long state_size() const; // returns size of state in Bytes
    long weights_size() const; // returns size of weights in Bytes

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
    Visible_Layer &get_visible_layer(
        int i // index of visible layer
    ) {
        return visible_layers[i];
    }

    // get a visible layer
    const Visible_Layer &get_visible_layer(
        int i // index of visible layer
    ) const {
        return visible_layers[i];
    }

    // get a visible layer descriptor
    const Visible_Layer_Desc &get_visible_layer_desc(
        int i // index of visible layer
    ) const {
        return visible_layer_descs[i];
    }

    // get the hidden states (predictions)
    const Int_Buffer &get_hidden_cis() const {
        return hidden_cis;
    }

    // get the hidden activations
    const Float_Buffer &get_hidden_acts() const {
        return hidden_acts;
    }

    // get the dendrite states
    const Float_Buffer &get_dendrite_acts() const {
        return dendrite_acts;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }

    // merge list of decoders and write to this one
    void merge(
        const Array<Decoder*> &decoders,
        Merge_Mode mode
    );
};
}
