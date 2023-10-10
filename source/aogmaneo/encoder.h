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
        Float_Buffer weights0; // regular weights
        Float_Buffer weights1; // complement weights
        
        float importance;

        Visible_Layer()
        :
        importance(1.0f)
        {}
    };

    struct Params {
        float choice; // choice parameter
        float vigilance; // ART vigilance
        float falloff; // falloff for mini-SOM part
        float lr; // learning rate
        int l_radius; // second stage inhibition radius

        Params()
        :
        choice(0.01f),
        vigilance(0.95f),
        falloff(0.99f),
        lr(0.5f),
        l_radius(2)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis;

    Int_Buffer learn_cis;

    Float_Buffer hidden_matches;
    Float_Buffer hidden_totals;
    Byte_Buffer hidden_commits;

    Float_Buffer hidden_maxs;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<const Int_Buffer*> &input_cis,
        const Params &params
    );

    void learn(
        const Int2 &column_pos,
        const Array<const Int_Buffer*> &input_cis,
        const Params &params
    );

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int3 &hidden_size, // hidden/output size
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    );

    void step(
        const Array<const Int_Buffer*> &input_cis, // input states
        bool learn_enabled, // whether to learn
        const Params &params // parameters
    );

    void clear_state();

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

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }
};
}
