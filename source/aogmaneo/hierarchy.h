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

struct IO_Desc {
    Int2 size;
    IO_Type type;

    int radius; // layer radius

    float positional_scale;

    IO_Desc(
        const Int2 &size = Int2(4, 4),
        IO_Type type = prediction,
        int radius = 2,
        float positional_scale = 1.0f
    )
    :
    size(size),
    type(type),
    radius(radius),
    positional_scale(positional_scale)
    {}
};

// describes a layer for construction. for the first layer, the IO_Desc overrides the parameters that are the same name
struct Layer_Desc {
    Int2 hidden_size; // size of hidden layer

    int radius; // layer radius

    int ticks_per_update;
    int temporal_horizon;

    float positional_scale; // positional embedding scale

    Layer_Desc(
        const Int2 &hidden_size = Int2(4, 4),
        int radius = 2,
        int ticks_per_update = 2,
        int temporal_horizon = 2,
        float positional_scale = 1.0f
    )
    :
    hidden_size(hidden_size),
    radius(radius),
    ticks_per_update(ticks_per_update),
    temporal_horizon(temporal_horizon),
    positional_scale(positional_scale)
    {}
};

struct IO_Params {};

struct Params {
    Array<Layer_Params> layers;
    Array<IO_Params> ios;
};

// a sph
template<int S, int L>
class Hierarchy {
private:
    // layers
    Array<Layer<S, L>> layers;

    // histories
    Array<Array<Circle_Buffer<Array<Vec<S, L>>>>> histories;

    // per-layer values
    Byte_Buffer updates;

    Int_Buffer ticks;
    Int_Buffer ticks_per_update;

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

        ticks = Int_Buffer(layer_descs.size(), 0);

        histories.resize(layer_descs.size());
        
        // default update state is no update
        updates = Byte_Buffer(layer_descs.size(), false);

        // cache input sizes
        io_sizes.resize(io_descs.size());
        io_types.resize(io_descs.size());

        for (int i = 0; i < io_descs.size(); i++) {
            io_sizes[i] = io_descs[i].size;
            io_types[i] = io_descs[i].type;
        }

        ticks_per_update.resize(layer_descs.size());

        // determine ticks per update, first layer is always 1
        for (int l = 0; l < layer_descs.size(); l++)
            ticks_per_update[l] = (l == 0 ? 1 : layer_descs[l].ticks_per_update);

        // iterate through layers
        for (int l = 0; l < layer_descs.size(); l++) {
            // create sparse coder visible layer descriptors
            Array<typename Layer<S, L>::Visible_Layer_Desc> visible_layer_descs;

            // if first layer
            if (l == 0) {
                // initialize history buffers
                histories[l].resize(io_sizes.size());

                for (int i = 0; i < histories[l].size(); i++) {
                    int in_size = io_sizes[i].x * io_sizes[i].y;

                    histories[l][i].resize(layer_descs[l].temporal_horizon);
                    
                    for (int t = 0; t < histories[l][i].size(); t++)
                        histories[l][i][t] = Array<Vec<S, L>>(in_size, 0);
                }

                visible_layer_descs.resize(io_sizes.size() * layer_descs[l].temporal_horizon);

                for (int i = 0; i < io_sizes.size(); i++) {
                    for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                        int index = t + layer_descs[l].temporal_horizon * i;

                        visible_layer_descs[index].size = io_sizes[i];
                        visible_layer_descs[index].radius = io_descs[i].radius;
                        visible_layer_descs[index].positional_scale = io_descs[i].positional_scale;

                        if (t == 0 && io_types[i] == prediction)
                            visible_layer_descs[index].predictable = true;
                    }
                }
            }
            else {
                histories[l].resize(1);

                int in_size = layer_descs[l - 1].hidden_size.x * layer_descs[l - 1].hidden_size.y;

                histories[l][0].resize(layer_descs[l].temporal_horizon);

                for (int t = 0; t < histories[l][0].size(); t++)
                    histories[l][0][t] = Array<Vec<S, L>>(in_size, 0);

                visible_layer_descs.resize(layer_descs[l].temporal_horizon);

                for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                    visible_layer_descs[t].size = layer_descs[l - 1].hidden_size;
                    visible_layer_descs[t].radius = layer_descs[l].radius;
                    visible_layer_descs[t].positional_scale = layer_descs[l].positional_scale;

                    if (t < layer_descs[l].ticks_per_update)
                        visible_layer_descs[t].predictable = true;
                }
            }
            
            // create the sparse coding layer
            layers[l].init_random(layer_descs[l].hidden_size, layer_descs[l].hidden_segments, layer_descs[l].hidden_length, visible_layer_descs);
        }

        // initialize params
        params.layers = Array<Layer_Params>(layer_descs.size());
        params.ios = Array<IO_Params>(io_descs.size());
    }

    // simulation step/tick
    void step(
        const Array<Array_View<Vec<S, L>>> &input_vecs, // inputs to remember
        bool learn_enabled = true // whether learning is enabled
    ) {
        assert(params.layers.size() == layers.size());
        assert(params.ios.size() == io_sizes.size());

        // first tick is always 0
        ticks[0] = 0;

        // add input to first layer history   
        for (int i = 0; i < io_sizes.size(); i++) {
            histories[0][i].push_front();

            histories[0][i][0] = input_vecs[i];
        }

        // set all updates to no update, will be set to true if an update occurred later
        updates.fill(false);

        // forward
        for (int l = 0; l < layers.size(); l++) {
            // if is time for layer to tick
            if (l == 0 || ticks[l] >= ticks_per_update[l]) {
                // reset tick
                ticks[l] = 0;

                // updated
                updates[l] = true;

                Array<Array_View<Vec<S, L>>> layer_inputs(layers[l].get_num_visible_layers());

                // set inputs
                int index = 0;

                for (int i = 0; i < histories[l].size(); i++) {
                    for (int t = 0; t < histories[l][i].size(); t++) {
                        layer_inputs[index] = histories[l][i][t];

                        index++;
                    }
                }

                layers[l].forward(layer_inputs);

                // add to next layer's history
                if (l < layers.size() - 1) {
                    int l_next = l + 1;

                    histories[l_next][0].push_front();

                    histories[l_next][0][0] = layers[l].get_hidden_vecs();

                    ticks[l_next]++;
                }
            }
        }

        // backward
        for (int l = layers.size() - 1; l >= 0; l--) {
            if (updates[l]) {
                Array_View<Vec<S, L>> feedback;

                if (l < layers.size() - 1) {
                    int t = ticks_per_update[l + 1] - 1 - ticks[l + 1];

                    layers[l + 1].backward(t);

                    feedback = layers[l + 1].get_visible_layer(t).pred_vecs;
                }

                layers[l].predict(feedback, learn_enabled, params.layers[l]);
            }
        }
    }

    void clear_state() {
        updates.fill(false);
        ticks.fill(0);

        for (int l = 0; l < layers.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    histories[l][i][t].fill(0);
            }

            layers[l].clear_state();
        }
    }

    // serialization
    long size() const { // returns size in bytes
        long size = 2 * sizeof(int) + io_sizes.size() * sizeof(Int2) + io_types.size() * sizeof(Byte) + updates.size() * sizeof(Byte) + 2 * ticks.size() * sizeof(int);

        for (int l = 0; l < layers.size(); l++) {
            size += sizeof(int);

            for (int i = 0; i < histories[l].size(); i++) {
                size += 2 * sizeof(int);

                for (int t = 0; t < histories[l][i].size(); t++)
                    size += sizeof(int) + histories[l][i][t].size() * sizeof(int);
            }

            size += layers[l].size();
        }

        // params
        size += layers.size() * sizeof(Layer_Params);
        size += io_sizes.size() * sizeof(IO_Params);

        return size;
    }

    long state_size() const { // returns size of state in bytes
        long size = updates.size() * sizeof(Byte) + ticks.size() * sizeof(int);

        for (int l = 0; l < layers.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                size += sizeof(int);

                for (int t = 0; t < histories[l][i].size(); t++)
                    size += histories[l][i][t].size() * sizeof(int);
            }

            size += layers[l].state_size();
        }

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

        writer.write(&updates[0], updates.size() * sizeof(Byte));
        writer.write(&ticks[0], ticks.size() * sizeof(int));
        writer.write(&ticks_per_update[0], ticks_per_update.size() * sizeof(int));

        for (int l = 0; l < num_layers; l++) {
            int num_layer_inputs = histories[l].size();

            writer.write(&num_layer_inputs, sizeof(int));

            for (int i = 0; i < histories[l].size(); i++) {
                int history_size = histories[l][i].size();

                writer.write(&history_size, sizeof(int));

                int history_start = histories[l][i].start;

                writer.write(&history_start, sizeof(int));

                for (int t = 0; t < histories[l][i].size(); t++) {
                    int buffer_size = histories[l][i][t].size();

                    writer.write(&buffer_size, sizeof(int));

                    writer.write(&histories[l][i][t][0], histories[l][i][t].size() * sizeof(int));
                }
            }

            layers[l].write(writer);
        }
        
        // params
        for (int l = 0; l < layers.size(); l++)
            writer.write(&params.layers[l], sizeof(Layer_Params));

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

        histories.resize(num_layers);
        
        updates.resize(num_layers);
        ticks.resize(num_layers);
        ticks_per_update.resize(num_layers);

        reader.read(&updates[0], updates.size() * sizeof(Byte));
        reader.read(&ticks[0], ticks.size() * sizeof(int));
        reader.read(&ticks_per_update[0], ticks_per_update.size() * sizeof(int));

        for (int l = 0; l < num_layers; l++) {
            int num_layer_inputs;
            
            reader.read(&num_layer_inputs, sizeof(int));

            histories[l].resize(num_layer_inputs);

            for (int i = 0; i < histories[l].size(); i++) {
                int history_size;

                reader.read(&history_size, sizeof(int));

                int history_start;
                
                reader.read(&history_start, sizeof(int));

                histories[l][i].resize(history_size);
                histories[l][i].start = history_start;

                for (int t = 0; t < histories[l][i].size(); t++) {
                    int buffer_size;

                    reader.read(&buffer_size, sizeof(int));

                    histories[l][i][t].resize(buffer_size);

                    reader.read(&histories[l][i][t][0], histories[l][i][t].size() * sizeof(int));
                }
            }

            layers[l].read(reader);
        }

        params.layers.resize(num_layers);
        params.ios.resize(num_io);

        for (int l = 0; l < num_layers; l++)
            reader.read(&params.layers[l], sizeof(Layer_Params));

        for (int i = 0; i < num_io; i++)
            reader.read(&params.ios[i], sizeof(IO_Params));
    }

    void write_state(
        Stream_Writer &writer
    ) const {
        writer.write(&updates[0], updates.size() * sizeof(Byte));
        writer.write(&ticks[0], ticks.size() * sizeof(int));

        for (int l = 0; l < layers.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                int history_start = histories[l][i].start;

                writer.write(&history_start, sizeof(int));

                for (int t = 0; t < histories[l][i].size(); t++)
                    writer.write(&histories[l][i][t][0], histories[l][i][t].size() * sizeof(int));
            }

            layers[l].write_state(writer);
        }
    }

    void read_state(
        Stream_Reader &reader
    ) {
        reader.read(&updates[0], updates.size() * sizeof(Byte));
        reader.read(&ticks[0], ticks.size() * sizeof(int));
        
        for (int l = 0; l < layers.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                int history_start;
                
                reader.read(&history_start, sizeof(int));

                histories[l][i].start = history_start;

                for (int t = 0; t < histories[l][i].size(); t++)
                    reader.read(&histories[l][i][t][0], histories[l][i][t].size() * sizeof(int));
            }

            layers[l].read_state(reader);
        }
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
        int index = i * histories[0][0].size();

        layers[0].backward(index);

        return layers[0].get_visible_layer(index).pred_vecs;
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

    int get_temporal_horizon(
        int l
    ) const {
        return histories[l][0].size();
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

    const Array<Circle_Buffer<Array<Vec<S, L>>>> &get_histories(
        int l
    ) const {
        return histories[l];
    }
};
}
