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

        Int_Buffer hidden_sums;

        Int_Buffer input_cis;
        Int_Buffer recon_cis;

        Int_Buffer hidden_totals;

        Int_Buffer recon_sums;

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
        float choice; // choice parameter, higher makes it select matchier columns over ones with less overall weights (total)
        float vigilance; // ART vigilance
        float lr; // learning rate
        float active_ratio; // 2nd stage inhibition activity ratio
        int l_radius; // second stage inhibition radius
        int min_matches;

        Params()
        :
        choice(0.1f),
        vigilance(0.95f),
        lr(0.1f),
        active_ratio(0.05f),
        l_radius(3),
        min_matches(3)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis;

    Int_Buffer learn_cis;

    Float_Buffer hidden_comparisons;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    // --- kernels ---
    
    void forward(
        const Int2 &column_pos,
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
