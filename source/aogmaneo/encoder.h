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
// sparse coder
class Encoder {
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

        Int_Buffer recon_sums;

        float importance;

        Visible_Layer()
        :
        importance(1.0f)
        {}
    };

    struct Params {
        float scale; // recon curve
        float lr; // learning rate
        int spatial_recon_tolerance;
        int recurrent_recon_tolerance;

        Params()
        :
        scale(4.0f),
        lr(0.05f),
        spatial_recon_tolerance(2),
        recurrent_recon_tolerance(2)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer
    int spatial_activity;
    int recurrent_radius;

    Int_Buffer spatial_cis;
    Int_Buffer hidden_cis;
    Int_Buffer hidden_cis_prev;

    Float_Buffer hidden_acts;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    Array<Int3> visible_pos_vlis; // for parallelization, cartesian product of column coordinates and visible layers
    
    Byte_Buffer recurrent_weights;

    Int_Buffer recurrent_recon_sums;

    // --- kernels ---

    void forward_spatial(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        const Params &params
    );

    void forward_recurrent(
        const Int2 &column_pos,
        const Params &params
    );

    void learn_spatial(
        const Int2 &column_pos,
        Int_Buffer_View input_cis,
        int vli,
        unsigned long* state,
        const Params &params
    );

    void learn_recurrent(
        const Int2 &column_pos,
        unsigned long* state,
        const Params &params
    );

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int3 &hidden_size, // hidden/output size
        int spatial_activity,
        int recurrent_radius,
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    );

    void step(
        const Array<Int_Buffer_View> &input_cis, // input states
        bool learn_enabled, // whether to learn
        const Params &params // parameters
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

    // get the number of visible layers
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

    // get the hidden states
    const Int_Buffer &get_hidden_cis() const {
        return hidden_cis;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }

    // merge list of encoders and write to this one
    void merge(
        const Array<Encoder*> &encoders,
        Merge_Mode mode
    );
};
}
