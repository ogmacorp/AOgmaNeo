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
class Layer {
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

        Int_Buffer recon_cis;

        float importance;

        Visible_Layer()
        :
        importance(1.0f)
        {}
    };

    struct Params {
        float scale; // recon curve
        float rlr; // recon learning rate
        float tlr; // transition learning rate
        int early_stop_cells; // if target of reconstruction is in top <this number> cells, stop early

        Params()
        :
        scale(4.0f),
        rlr(0.01f),
        tlr(0.1f),
        early_stop_cells(1)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis;

    Float_Buffer hidden_acts;

    // planning stuff
    Float_Buffer hidden_plan_dists;
    Byte_Buffer hidden_plan_opens;
    Int_Buffer hidden_plan_prevs;
    Int_Buffer hidden_plan_cis;
    Byte_Buffer hidden_transitions;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    Array<Int3> visible_pos_vlis; // for parallelization, cartesian product of column coordinates and visible layers
    
    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        bool learn_enabled,
        const Params &params
    );

    void plan(
        const Int2 &column_pos,
        Int_Buffer_View goal_cis,
        int t,
        const Params &params
    );

    void learn_reconstruction(
        const Int2 &column_pos,
        Int_Buffer_View input_cis,
        int vli,
        unsigned long* state,
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

    void step(
        const Array<Int_Buffer_View> &input_cis, // input states
        bool learn_enabled, // whether to learn
        const Params &params // parameters
    );

    void plan(
        Int_Buffer_View goal_cis,
        const Params &params
    );

    void reconstruct(
        int vli
    );

    const Int_Buffer &get_reconstruction(
        int vli
    ) const {
        return visible_layers[vli].recon_cis;
    }

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
};
}
