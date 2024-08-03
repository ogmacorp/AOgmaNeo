// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "encoder.h"
#include "decoder.h"

namespace aon {
// type of hierarchy input layer
enum IO_Type {
    none = 0,
    prediction = 1
};

// a sph
class Hierarchy {
public:
    struct IO_Desc {
        Int3 size;
        IO_Type type;

        int num_dendrites_per_cell; // also for policy

        int up_radius; // encoder radius
        int down_radius; // decoder radius, also shared with actor if there is one

        IO_Desc(
            const Int3 &size = Int3(4, 4, 16),
            IO_Type type = prediction,
            int num_dendrites_per_cell = 4,
            int up_radius = 2,
            int down_radius = 2
        )
        :
        size(size),
        type(type),
        num_dendrites_per_cell(num_dendrites_per_cell),
        up_radius(up_radius),
        down_radius(down_radius)
        {}
    };

    // describes a layer for construction. for the first layer, the IO_Desc overrides the parameters that are the same name
    struct Layer_Desc {
        Int3 hidden_size; // size of hidden layer

        int num_dendrites_per_cell;

        int up_radius; // encoder radius
        int down_radius; // decoder radius, also shared with actor if there is one

        int ticks_per_update; // number of ticks a layer takes to update (relative to previous layer)
        int temporal_horizon; // temporal distance into the past addressed by the layer. should be greater than or equal to ticks_per_update

        Layer_Desc(
            const Int3 &hidden_size = Int3(4, 4, 16),
            int num_dendrites_per_cell = 4,
            int up_radius = 2,
            int down_radius = 2,
            int ticks_per_update = 2,
            int temporal_horizon = 2
        )
        :
        hidden_size(hidden_size),
        num_dendrites_per_cell(num_dendrites_per_cell),
        up_radius(up_radius),
        down_radius(down_radius),
        ticks_per_update(ticks_per_update),
        temporal_horizon(temporal_horizon)
        {}
    };

    struct Layer_Params {
        Decoder::Params decoder;
        Encoder::Params encoder;
    };

    struct IO_Params {
        Decoder::Params decoder;

        // additional
        float importance;

        IO_Params()
        :
        importance(1.0f)
        {}
    };

    struct Params {
        Array<Layer_Params> layers;
        Array<IO_Params> ios;
    };

private:
    // layers
    Array<Encoder> encoders;
    Array<Array<Decoder>> decoders;
    Array<Int_Buffer> hidden_cis_prev;

    // for mapping first layer Decoders
    Int_Buffer i_indices;
    Int_Buffer d_indices;

    // histories
    Array<Array<Circle_Buffer<Int_Buffer>>> histories;

    // per-layer values
    Byte_Buffer updates;

    Int_Buffer ticks;
    Int_Buffer ticks_per_update;

    // input dimensions
    Array<Int3> io_sizes;
    Array<Byte> io_types;

    // importance control
    void set_input_importance(
        int i,
        float importance
    ) {
        for (int t = 0; t < histories[0][i].size(); t++)
            encoders[0].get_visible_layer(i * histories[0][i].size() + t).importance = importance;
    }

public:
    // parameters
    Params params;

    // default
    Hierarchy() {}

    Hierarchy(
        const Array<IO_Desc> &io_descs, // input-output descriptors
        const Array<Layer_Desc> &layer_descs // descriptors for layers
    ) {
        init_random(io_descs, layer_descs);
    }
    
    // create a randomly initialized hierarchy
    void init_random(
        const Array<IO_Desc> &io_descs, // input-output descriptors
        const Array<Layer_Desc> &layer_descs // descriptors for layers
    );

    // simulation step/tick
    void step(
        const Array<Int_Buffer_View> &input_cis, // inputs to remember
        Int_Buffer_View top_feedback_cis, // top feed back ("program")
        bool learn_enabled = true // whether learning is enabled
    );

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

    // get the number of layers (encoders)
    int get_num_layers() const {
        return encoders.size();
    }

    bool io_layer_exists(
        int i
    ) const {
        return d_indices[i] != -1;
    }

    // retrieve predictions
    const Int_Buffer &get_prediction_cis(
        int i
    ) const {
        return decoders[0][d_indices[i]].get_hidden_cis();
    }

    const Int_Buffer &get_top_hidden_cis() const {
        return encoders[encoders.size() - 1].get_hidden_cis();
    }

    // whether this layer received on update this timestep
    bool get_update(
        int l
    ) const {
        return updates[l];
    }

    // get current layer ticks, relative to previous layer
    int get_ticks(
        int l
    ) const {
        return ticks[l];
    }

    // get layer ticks per update, relative to previous layer
    int get_ticks_per_update(
        int l
    ) const {
        return ticks_per_update[l];
    }

    // number of io layers
    int get_num_io() const {
        return io_sizes.size();
    }

    // get input/output sizes
    const Int3 &get_io_size(
        int i
    ) const {
        return io_sizes[i];
    }

    // get input/output types
    IO_Type get_io_type(
        int i
    ) const {
        return static_cast<IO_Type>(io_types[i]);
    }

    int get_num_encoder_visible_layers(
        int l
    ) const {
        return encoders[l].get_num_visible_layers();
    }

    // retrieve a sparse coding layer
    Encoder &get_encoder(
        int l
    ) {
        return encoders[l];
    }

    // retrieve a sparse coding layer, const version
    const Encoder &get_encoder(
        int l
    ) const {
        return encoders[l];
    }

    int get_num_decoders(
        int l
    ) const {
        return decoders[l].size();
    }

    // retrieve by index
    Decoder &get_decoder(
        int l,
        int i
    ) {
        if (l == 0)
            return decoders[l][d_indices[i]];

        return decoders[l][i];
    }

    const Decoder &get_decoder(
        int l,
        int i
    ) const {
        if (l == 0)
            return decoders[l][d_indices[i]];

        return decoders[l][i];
    }

    const Int_Buffer &get_i_indices() const {
        return i_indices;
    }

    const Int_Buffer &get_d_indices() const {
        return d_indices;
    }

    const Array<Circle_Buffer<Int_Buffer>> &get_histories(
        int l
    ) const {
        return histories[l];
    }

    // merge list of hierarchies and write to this one
    void merge(
        const Array<Hierarchy*> &hierarchies,
        Merge_Mode mode
    );
};
}
