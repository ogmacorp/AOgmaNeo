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
// A reinforcement learning layer
class Actor {
public:
    // Visible layer descriptor
    struct Visible_Layer_Desc {
        Int3 size; // Visible/input size

        int radius; // Radius onto input

        // Defaults
        Visible_Layer_Desc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // Visible layer
    struct Visible_Layer {
        Float_Buffer weights;
        Float_Buffer traces;

        Int_Buffer input_cis_prev;
    };

    struct Params {
        float lr;
        float discount;
        float trace_decay;

        // Defaults
        Params()
        :
        lr(0.1f),
        discount(0.99f),
        trace_decay(0.97f)
        {}
    };

private:
    Int3 hidden_size; // Hidden/output/action size

    Int_Buffer hidden_cis; // Hidden states

    Float_Buffer hidden_values;

    // Visible layers and descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    // --- Kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<Int_Buffer_View> &input_cis,
        Int_Buffer_View hidden_target_cis,
        float reward,
        bool learn_enabled,
        const Params &params
    );

public:
    // Initialized randomly
    void init_random(
        const Int3 &hidden_size,
        const Array<Visible_Layer_Desc> &visible_layer_descs
    );

    // Step (get actions and update)
    void step(
        const Array<Int_Buffer_View> &input_cis,
        Int_Buffer_View hidden_target_cis,
        float reward,
        bool learn_enabled,
        const Params &params
    );

    void clear_state();

    // Serialization
    int size() const; // Returns size in bytes
    int state_size() const; // Returns size of state in bytes

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

    // Get number of visible layers
    int get_num_visible_layers() const {
        return visible_layers.size();
    }

    // Get a visible layer
    const Visible_Layer &get_visible_layer(
        int i // Index of layer
    ) const {
        return visible_layers[i];
    }

    // Get a visible layer descriptor
    const Visible_Layer_Desc &get_visible_layer_desc(
        int i // Index of layer
    ) const {
        return visible_layer_descs[i];
    }

    // Get hidden state/output/actions
    const Int_Buffer &get_hidden_cis() const {
        return hidden_cis;
    }

    // Get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }
};
}
