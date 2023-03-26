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
        Int_Buffer weight_indices;
        Byte_Buffer weights;

        Float_Buffer hidden_partial_acts;
        
        Int_Buffer input_cis;
        Int_Buffer recon_cis;
        
        float importance;

        Byte use_input;
        Byte needs_update;

        Visible_Layer()
        :
        importance(1.0f),
        use_input(false),
        needs_update(true)
        {}
    };

    struct Params {
        float choice; // Choice parameter
        float vigilance; // ART vigilance
        float lr; // learning rate
        int l_radius; // Second stage inhibition radius

        Params()
        :
        choice(0.1f),
        vigilance(0.9f),
        lr(0.1f),
        l_radius(2)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis;

    Int_Buffer learn_cis;

    Float_Buffer hidden_totals;

    Float_Buffer hidden_max_acts;

    Int_Buffer hidden_commits;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    // --- kernels ---
    
    void forward(
        const Int2 &column_pos,
        unsigned int* state,
        const Params &params
    );

    void learn(
        const Int2 &column_pos,
        const Params &params
    );

    void reconstruct(
        const Int2 &column_pos,
        int vli
    );

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int3 &hidden_size, // hidden/output size
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    );

    void set_input_cis(
        const Int_Buffer* input_cis,
        int vli
    );

    void activate(
        const Params &params // parameters
    );

    void learn(
        const Params &params // parameters
    );

    void reconstruct(
        int vli
    );

    void clear_state() {
        hidden_cis.fill(0);
    }

    // serialization
    int size() const; // returns size in bytes
    int state_size() const; // returns size of state in bytes

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

    // get the number of committed cells per column
    const Int_Buffer &get_hidden_commits() const {
        return hidden_commits;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }
};
}
