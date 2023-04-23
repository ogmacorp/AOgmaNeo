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
        Float_Buffer weights_value;
        Float_Buffer weights_advantage;
    };

    // History sample for delayed updates
    struct History_Sample {
        Array<Int_Buffer> input_cis;
        Int_Buffer hidden_target_cis_prev;

        float reward;
    };

    struct Params {
        float vlr; // Value learning rate
        float alr; // Advantage learning rate
        float discount;
        int n_steps;
        int history_iters;

        // Defaults
        Params()
        :
        vlr(0.01f),
        alr(0.01f),
        discount(0.99f),
        n_steps(8),
        history_iters(16)
        {}
    };

private:
    Int3 hidden_size; // Hidden/output/action size

    // Current history size - fixed after initialization. Determines length of wait before updating
    int history_size;

    Int_Buffer hidden_cis; // Hidden states

    Float_Buffer hidden_advantages;

    Circle_Buffer<History_Sample> history_samples; // History buffer, fixed length

    // Visible layers and descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;

    // --- Kernels ---

    void forward(
        const Int2 &column_pos,
        const Array<const Int_Buffer*> &input_cis
    );

    void learn(
        const Int2 &column_pos,
        int t,
        float r,
        float d,
        const Params &params
    );

public:
    // Initialized randomly
    void init_random(
        const Int3 &hidden_size,
        int history_capacity,
        const Array<Visible_Layer_Desc> &visible_layer_descs
    );

    // Step (get actions and update)
    void step(
        const Array<const Int_Buffer*> &input_cis,
        const Int_Buffer* hidden_target_cis_prev,
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

    int get_history_capacity() const {
        return history_samples.size();
    }

    int get_history_size() const {
        return history_size;
    }
};
}
