// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "layer.h"

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

        int radius; // layer up radius

        IO_Desc(
            const Int3 &size = Int3(4, 4, 16),
            IO_Type type = prediction,
            int radius = 2
        )
        :
        size(size),
        type(type),
        radius(radius)
        {}
    };

    // describes a layer for construction. for the first layer, the IO_Desc overrides the parameters that are the same name
    struct Layer_Desc {
        Int3 hidden_size; // size of hidden layer

        int radius; // layer up radius

        int ticks_per_update; // number of ticks a layer takes to update (relative to previous layer)
        int temporal_horizon; // temporal distance into the past addressed by the layer. should be greater than or equal to ticks_per_update

        Layer_Desc(
            const Int3 &hidden_size = Int3(4, 4, 16),
            int radius = 2,
            int ticks_per_update = 2,
            int temporal_horizon = 2
        )
        :
        hidden_size(hidden_size),
        radius(radius),
        ticks_per_update(ticks_per_update),
        temporal_horizon(temporal_horizon)
        {}
    };

    struct IO_Params {
        // additional
        float importance;

        IO_Params()
        :
        importance(1.0f)
        {}
    };

    struct Params {
        Array<Layer::Params> layers;
        Array<IO_Params> ios;
    };

private:
    // layers
    Array<Layer> layers;

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
            layers[0].get_visible_layer(i * histories[0][i].size() + t).importance = importance;
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
        Int_Buffer_View top_goal_cis,
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

    // get the number of layers (layers)
    int get_num_layers() const {
        return layers.size();
    }

    // retrieve predictions
    const Int_Buffer &get_prediction_cis(
        int i
    ) const {
        return layers[0].get_reconstruction(i * histories[0][i].size());
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

    int get_num_layer_visible_layers(
        int l
    ) const {
        return layers[l].get_num_visible_layers();
    }

    // retrieve a sparse coding layer
    Layer &get_layer(
        int l
    ) {
        return layers[l];
    }

    // retrieve a sparse coding layer, const version
    const Layer &get_layer(
        int l
    ) const {
        return layers[l];
    }

    const Array<Circle_Buffer<Int_Buffer>> &get_histories(
        int l
    ) const {
        return histories[l];
    }

    const Int_Buffer &get_top_hidden_cis() const {
        return layers[layers.size() - 1].get_hidden_cis();
    }
};
}
