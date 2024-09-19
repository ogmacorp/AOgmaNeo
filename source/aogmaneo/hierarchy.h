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
template<int S, int L>
class Hierarchy {
public:
    struct IO_Desc {
        Int2 size;
        IO_Type type;

        int radius; // layer radius

        IO_Desc(
            const Int2 &size = Int2(4, 4),
            IO_Type type = prediction,
            int radius = 2,
            int down_radius = 2
        )
        :
        size(size),
        type(type),
        radius(radius)
        {}
    };

    // describes a layer for construction. for the first layer, the IO_Desc overrides the parameters that are the same name
    struct Layer_Desc {
        Int2 hidden_size; // size of hidden layer
        int num_hidden; // number of dendrites per cell in predictor

        int radius; // layer radius

        float positional_scale; // positional embedding scale

        Layer_Desc(
            const Int2 &hidden_size = Int2(4, 4),
            int num_hidden = 128,
            int radius = 2,
            float positional_scale = 1.0f
        )
        :
        hidden_size(hidden_size),
        num_hidden(num_hidden),
        radius(radius),
        positional_scale(positional_scale)
        {}
    };

    struct IO_Params {};

    struct Params {
        Array<typename Layer<S, L>::Params> layers;
        Array<IO_Params> ios;
    };

private:
    // layers
    Array<Layer<S, L>> layers;

    // input dimensions
    Array<Int2> io_sizes;
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
    ) {
        // create layers
        layers.resize(layer_descs.size());

        // cache input sizes
        io_sizes.resize(io_descs.size());
        io_types.resize(io_descs.size());

        for (int i = 0; i < io_descs.size(); i++) {
            io_sizes[i] = io_descs[i].size;
            io_types[i] = io_descs[i].type;
        }

        // iterate through layers
        for (int l = 0; l < layer_descs.size(); l++) {
            // create sparse coder visible layer descriptors
            Array<typename Layer<S, L>::Visible_Layer_Desc> visible_layer_descs;

            // if first layer
            if (l == 0) {
                visible_layer_descs.resize(io_sizes.size());

                for (int i = 0; i < io_sizes.size(); i++) {
                    visible_layer_descs[i].size = io_sizes[i];
                    visible_layer_descs[i].radius = io_descs[i].radius;

                    if (io_types[i] == prediction)
                        visible_layer_descs[i].predictable = true;
                }
            }
            else {
                visible_layer_descs.resize(1);

                visible_layer_descs[0].size = layer_descs[l - 1].hidden_size;
                visible_layer_descs[0].radius = layer_descs[l].radius;

                visible_layer_descs[0].predictable = true;
            }
            
            // create the sparse coding layer
            layers[l].init_random(layer_descs[l].hidden_size, layer_descs[l].num_hidden, layer_descs[l].positional_scale, visible_layer_descs);
        }

        // initialize params
        params.layers = Array<typename Layer<S, L>::Params>(layer_descs.size());
        params.ios = Array<IO_Params>(io_descs.size());
    }

    // simulation step/tick
    void step(
        const Array<Array_View<Vec<S, L>>> &input_vecs, // inputs to remember
        bool learn_enabled = true // whether learning is enabled
    ) {
        assert(params.layers.size() == layers.size());
        assert(params.ios.size() == io_sizes.size());

        // forward
        for (int l = 0; l < layers.size(); l++) {
            Array<Array_View<Vec<S, L>>> layer_inputs(layers[l].get_num_visible_layers());

            // set inputs
            if (l == 0) {
                for (int i = 0; i < io_sizes.size(); i++)
                    layer_inputs[i] = input_vecs[i];
            }
            else
                layer_inputs[0] = layers[l - 1].get_hidden_vecs();

            layers[l].forward(layer_inputs);
        }

        // backward
        for (int l = layers.size() - 1; l >= 0; l--) {
            Array_View<Vec<S, L>> feedback;

            if (l < layers.size() - 1) {
                layers[l + 1].backward(0);

                feedback = layers[l + 1].get_visible_layer(0).pred_vecs;
            }

            layers[l].predict(feedback, learn_enabled, params.layers[l]);
        }
    }

    void clear_state() {
        for (int l = 0; l < layers.size(); l++)
            layers[l].clear_state();
    }

    // serialization
    long size() const { // returns size in bytes
        long size = 2 * sizeof(int) + io_sizes.size() * sizeof(Int2) + io_types.size() * sizeof(Byte);

        for (int l = 0; l < layers.size(); l++)
            size += layers[l].size();

        // params
        size += layers.size() * sizeof(typename Layer<S, L>::Params);
        size += io_sizes.size() * sizeof(IO_Params);

        return size;
    }

    long state_size() const { // returns size of state in bytes
        long size = 0;

        for (int l = 0; l < layers.size(); l++)
            size += layers[l].state_size();

        return size;
    }

    long weights_size() const { // returns size of weights in bytes
        long size = 0;

        for (int l = 0; l < layers.size(); l++)
            size += layers[l].weights_size();

        return size;
    }

    void write(
        Stream_Writer &writer
    ) const {
        int num_layers = layers.size();

        writer.write(&num_layers, sizeof(int));

        int num_io = io_sizes.size();

        writer.write(&num_io, sizeof(int));

        writer.write(&io_sizes[0], num_io * sizeof(Int2));
        writer.write(&io_types[0], num_io * sizeof(Byte));

        for (int l = 0; l < num_layers; l++)
            layers[l].write(writer);
        
        // params
        for (int l = 0; l < layers.size(); l++)
            writer.write(&params.layers[l], sizeof(typename Layer<S, L>::Params));

        for (int i = 0; i < io_sizes.size(); i++)
            writer.write(&params.ios[i], sizeof(IO_Params));
    }

    void read(
        Stream_Reader &reader
    ) {
        int num_layers;

        reader.read(&num_layers, sizeof(int));

        int num_io;

        reader.read(&num_io, sizeof(int));

        io_sizes.resize(num_io);
        io_types.resize(num_io);

        reader.read(&io_sizes[0], num_io * sizeof(Int2));
        reader.read(&io_types[0], num_io * sizeof(Byte));

        layers.resize(num_layers);

        for (int l = 0; l < num_layers; l++)
            layers[l].read(reader);

        params.layers.resize(num_layers);
        params.ios.resize(num_io);

        for (int l = 0; l < num_layers; l++)
            reader.read(&params.layers[l], sizeof(typename Layer<S, L>::Params));

        for (int i = 0; i < num_io; i++)
            reader.read(&params.ios[i], sizeof(IO_Params));
    }

    void write_state(
        Stream_Writer &writer
    ) const {
        for (int l = 0; l < layers.size(); l++)
            layers[l].write_state(writer);
    }

    void read_state(
        Stream_Reader &reader
    ) {
        for (int l = 0; l < layers.size(); l++)
            layers[l].read_state(reader);
    }

    void write_weights(
        Stream_Writer &writer
    ) const {
        for (int l = 0; l < layers.size(); l++)
            layers[l].write_weights(writer);
    }

    void read_weights(
        Stream_Reader &reader
    ) {
        for (int l = 0; l < layers.size(); l++)
            layers[l].read_weights(reader);
    }

    // get the number of layers (layers)
    int get_num_layers() const {
        return layers.size();
    }

    // retrieve predictions
    const Array<Vec<S, L>> &get_prediction_vecs(
        int i
    ) {
        layers[0].backward(i);

        return layers[0].get_visible_layer(i).pred_vecs;
    }

    // number of io layers
    int get_num_io() const {
        return io_sizes.size();
    }

    // get input/output sizes
    const Int2 &get_io_size(
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
    Layer<S, L> &get_layer(
        int l
    ) {
        return layers[l];
    }

    // retrieve a sparse coding layer, const version
    const Layer<S, L> &get_layer(
        int l
    ) const {
        return layers[l];
    }
};
}
