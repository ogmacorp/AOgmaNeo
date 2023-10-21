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
class Predictor {
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
        Float_Buffer weights;

        Float_Buffer errors;
    };

    struct Params {
        float lr; // learning rate

        Params()
        :
        lr(0.02f)
        {}
    };

private:
    Int3 hidden_size; // size of the output/hidden/prediction

    Int_Buffer hidden_cis; // hidden state

    Float_Buffer hidden_acts;

    Float_Buffer hidden_errors;

    // visible layers and descs
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    Array<Int3> visible_pos_vlis; // for parallelization, cartesian product of column coordinates and visible layers

    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        const Array<Float_Buffer_View> &input_acts,
        const Params &params
    );

    void backward(
        const Int2 &column_pos,
        Int_Buffer_View input_cis,
        Float_Buffer_View input_acts,
        Int_Buffer_View target_cis,
        int vli,
        bool learn_enabled,
        const Params &params
    );

public:
    // create with random initialization
    void init_random(
        const Int3 &hidden_size, // hidden/output/prediction size
        const Array<Visible_Layer_Desc> &visible_layer_descs
    );

    // activate the predictor (predict values)
    void forward(
        const Array<Int_Buffer_View> &input_cis,
        const Array<Float_Buffer_View> &input_acts,
        const Params &params
    );

    void backward(
        const Array<Int_Buffer_View> &input_cis,
        const Array<Float_Buffer_View> &input_acts,
        Int_Buffer_View target_cis,
        bool learn_enabled,
        const Params &params
    );

    void clear_state();

    // serialization
    int size() const; // returns size in Bytes
    int state_size() const; // returns size of state in Bytes

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

    // get the hidden states (predictions)
    const Float_Buffer &get_hidden_acts() const {
        return hidden_acts;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }
};
}
