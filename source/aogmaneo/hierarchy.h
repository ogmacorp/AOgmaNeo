// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "encoder.h"
#include "routed_layer.h"
#include "predictor.h"
#include "actor.h"

namespace aon {
// type of hierarchy input layer
enum IO_Type {
    none = 0,
    prediction = 1,
    action = 2
};

// a sph
class Hierarchy {
public:
    struct IO_Desc {
        Int3 size;
        IO_Type type;

        int up_radius; // encoder radius
        int down_radius; // decoder radius, also shared with actor if there is one

        int history_capacity; // actor history max window size

        IO_Desc(
            const Int3 &size = Int3(4, 4, 16),
            IO_Type type = prediction,
            int up_radius = 2,
            int down_radius = 2,
            int history_capacity = 128
        )
        :
        size(size),
        type(type),
        up_radius(up_radius),
        down_radius(down_radius),
        history_capacity(history_capacity)
        {}
    };

    // describes a layer for construction. for the first layer, the IO_Desc overrides the parameters that are the same name
    struct Layer_Desc {
        Int3 hidden_size; // size of hidden layer

        int up_radius; // encoder radius
        int down_radius; // decoder radius, also shared with actor if there is one

        int ticks_per_update;
        int temporal_horizon;

        Layer_Desc(
            const Int3 &hidden_size = Int3(4, 4, 16),
            int up_radius = 2,
            int down_radius = 2,
            int ticks_per_update = 2,
            int temporal_horizon = 2
        )
        :
        hidden_size(hidden_size),
        up_radius(up_radius),
        down_radius(down_radius),
        ticks_per_update(ticks_per_update),
        temporal_horizon(temporal_horizon)
        {}
    };

    struct Layer_Params {
        Routed_Layer::Params routed_layer;
        Encoder::Params encoder;
    };

    struct IO_Params {
        Predictor::Params predictor;
        Actor::Params actor;

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
    Array<Array<Routed_Layer>> routed_layers;
    Array<Predictor> predictors;
    Array<Actor> actors;

    // histories
    Array<Array<Circle_Buffer<Int_Buffer>>> histories;

    // per-layer values
    Byte_Buffer updates;

    Int_Buffer ticks;
    Int_Buffer ticks_per_update;

    // for mapping first layer Decoders
    Int_Buffer i_indices;
    Int_Buffer o_indices;

    // input dimensions
    Array<Int3> io_sizes;
    Array<Byte> io_types;

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
        bool learn_enabled = true, // whether learning is enabled
        float reward = 0.0f // reward
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

    // get the number of layers (encoders)
    int get_num_layers() const {
        return encoders.size();
    }

    bool io_layer_exists(
        int i
    ) const {
        return o_indices[i] != -1;
    }

    bool is_layer_recurrent(
        int l
    ) const {
        return (l == 0 ? encoders[l].get_num_visible_layers() > io_sizes.size() : encoders[l].get_num_visible_layers() > 1);
    }

    // retrieve predictions
    const Int_Buffer &get_prediction_cis(
        int i
    ) const {
        if (io_types[i] == action)
            return actors[o_indices[i]].get_hidden_cis();

        return predictors[o_indices[i]].get_hidden_cis();
    }

    // retrieve prediction activations
    const Float_Buffer &get_prediction_acts(
        int i
    ) const {
        if (io_types[i] == action)
            return actors[o_indices[i]].get_hidden_acts();

        return predictors[o_indices[i]].get_hidden_acts();
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

    // retrieve by index
    Routed_Layer &get_routed_layer(
        int l,
        int i
    ) {
        return routed_layers[l][i];
    }

    const Routed_Layer &get_routed_layer(
        int l,
        int i
    ) const {
        return routed_layers[l][i];
    }

    Predictor &get_predictor(
        int i
    ) {
        return predictors[o_indices[i]];
    }

    const Predictor &get_predictor(
        int i
    ) const {
        return predictors[o_indices[i]];
    }

    Actor &get_actor(
        int i
    ) {
        return actors[o_indices[i]];
    }

    const Actor &get_actor(
        int i
    ) const {
        return actors[o_indices[i]];
    }

    const Int_Buffer &get_i_indices() const {
        return i_indices;
    }

    const Int_Buffer &get_o_indices() const {
        return o_indices;
    }
};
}
