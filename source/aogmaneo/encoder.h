// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
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
        size(5, 5, 16),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Byte_Buffer weights;
        
        Int_Buffer hidden_sums;
        Int_Buffer hidden_totals;
        Int_Buffer hidden_counts;

        float importance;

        Visible_Layer()
        :
        importance(1.0f)
        {}
    };

    struct Params {
        float choice; // choice parameter, higher makes it select matchier columns over ones with less overall weights (total)
        float category_vigilance; // standard ART vigilance
        float compare_vigilance; // vigilance value used cross-column comparison (2nd stage inhibition)
        float lr; // learning rate
        float active_ratio; // 2nd stage inhibition activity ratio
        int l_radius; // second stage inhibition radius

        Params()
        :
        choice(0.0001f),
        category_vigilance(0.95f),
        compare_vigilance(0.9f),
        lr(0.5f),
        active_ratio(0.1f),
        l_radius(2)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer
    int temporal_size; // spatial region size (this must evenly divide hidden_size.z)
    int recurrent_radius; // radius of recurrent connections

    Int_Buffer hidden_cis;
    Int_Buffer temporal_cis;
    Int_Buffer temporal_cis_prev;

    Byte_Buffer hidden_learn_flags;
    Byte_Buffer temporal_learn_flags;

    Float_Buffer hidden_comparisons;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Int_Buffer recurrent_sums;
    Byte_Buffer recurrent_weights;
    Int_Buffer recurrent_totals;
    
    // --- kernels ---
    
    void forward(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        const Params &params
    );

    void learn(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        const Params &params
    );

public:
    // create a sparse coding layer with random initialization
    void init_random(
        const Int3 &hidden_size, // hidden/output size
        int temporal_size,
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

    // get the hidden states
    const Int_Buffer &get_temporal_cis() const {
        return temporal_cis;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }

    int get_temporal_size() const {
        return temporal_size;
    }

    int get_recurrent_radius() const {
        return recurrent_radius;
    }

    // merge list of encoders and write to this one
    void merge(
        const Array<Encoder*> &encoders,
        Merge_Mode mode
    );
};
}
