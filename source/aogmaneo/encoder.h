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
        Int4 size; // size of input

        int radius; // radius onto input

        // defaults
        Visible_Layer_Desc()
        :
        size(4, 4, 4, 16),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Int_Buffer input_cis;

        S_Byte_Buffer visible_code_vecs;

        S_Byte_Buffer visible_pos_vecs; // positional encodings

        S_Byte_Buffer input_vecs;

        Int_Buffer visible_bundle_buffer;
        Int_Buffer hidden_bundle_buffer;

        Int_Buffer recon_cis;

        S_Byte_Buffer recon_bundle_vecs;

        bool use_input;
        bool up_to_date;
        
        float importance;

        Visible_Layer()
        :
        use_input(false),
        up_to_date(false),
        importance(1.0f)
        {}
    };

    struct Params {
        float lr; // code learning rate
        int resonate_iters;

        Params()
        :
        lr(0.01f),
        resonate_iters(16)
        {}
    };

private:
    Int4 hidden_size; // size of hidden/output layer
    int vec_size;

    Int_Buffer hidden_cis;

    Float_Buffer hidden_learn_vecs;
    S_Byte_Buffer hidden_code_vecs;
    S_Byte_Buffer hidden_corr_mats; // correlation matrices

    S_Byte_Buffer hidden_bundle_vecs;
    S_Byte_Buffer hidden_temp_vecs;

    Int_Buffer hidden_comparisons;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    // --- kernels ---
    
    void bind_inputs(
        const Int2 &column_pos,
        int vli
    );

    void forward(
        const Int2 &column_pos,
        bool learn_enabled,
        const Params &params
    );

    void reconstruct(
        const Int2 &column_pos,
        int vli
    );

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int4 &hidden_size, // hidden/output size
        int vec_size,
        float positional_scale, // positional encoding scale
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    );

    void set_ignore(
        int vli
    ) {
        visible_layers[vli].use_input = false;
    }

    void set_input_cis(
        int vli,
        Int_Buffer_View input_cis
    ) {
        assert(input_cis.size() == visible_layers[vli].input_cis.size());

        visible_layers[vli].use_input = true;
        visible_layers[vli].up_to_date = false;
        visible_layers[vli].input_cis = input_cis;
    }

    void step(
        bool learn_enabled, // whether to learn
        const Params &params // parameters
    );

    void reconstruct(
        int vli
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
    const Int4 &get_hidden_size() const {
        return hidden_size;
    }
};
}
