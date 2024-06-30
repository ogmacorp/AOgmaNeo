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
// a prediction layer (predicts x_(t+1))
class Searcher {
public:
    struct Params {
        float lr; // weight learning rate
        float leak; // relu leak
        float exploration; // exploration amount

        Params()
        :
        lr(0.01f),
        leak(0.01f),
        exploration(0.1f)
        {}
    };

private:
    Int3 config_size; // size of the configuration
    int num_dendrites_per_cell;
    int radius;

    Int_Buffer pred_config_cis; // hidden state

    Float_Buffer pred_config_acts;

    Float_Buffer dendrite_acts;

    Float_Buffer dendrite_deltas;

    // visible layers and descs
    Float_Buffer weights;

    // --- kernels ---

    void forward(
        const Int2 &column_pos,
        Int_Buffer_View actual_config_cis,
        float reward,
        bool learn_enabled,
        unsigned long* state,
        const Params &params
    );

public:
    // create with random initialization
    void init_random(
        const Int3 &config_size, // configuration size
        int num_dendrites_per_cell,
        int radius
    );

    // step the search
    void step(
        Int_Buffer_View actual_config_cis,
        float reward,
        bool learn_enabled,
        const Params &params
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

    // get the hidden states (predictions)
    const Int_Buffer &get_pred_config_cis() const {
        return pred_config_cis;
    }

    // get the hidden size
    const Int3 &get_config_size() const {
        return config_size;
    }

    // merge list of decoders and write to this one
    void merge(
        const Array<Searcher*> &decoders,
        Merge_Mode mode
    );
};
}
