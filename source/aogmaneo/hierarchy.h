// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "encoder.h"

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

        int num_dendrites_per_cell; // also for policy
        int value_num_dendrites_per_cell; // just value

        int up_radius; // encoder radius
        int down_radius; // decoder radius, also shared with actor if there is one

        IO_Desc(
            const Int2 &size = Int2(4, 4),
            IO_Type type = prediction,
            int num_dendrites_per_cell = 4,
            int value_num_dendrites_per_cell = 8,
            int up_radius = 2,
            int down_radius = 2
        )
        :
        size(size),
        type(type),
        num_dendrites_per_cell(num_dendrites_per_cell),
        value_num_dendrites_per_cell(value_num_dendrites_per_cell),
        up_radius(up_radius),
        down_radius(down_radius)
        {}
    };

    // describes a layer for construction. for the first layer, the IO_Desc overrides the parameters that are the same name
    struct Layer_Desc {
        Int2 hidden_size; // size of hidden layer
        int HS;
        int HL;

        int up_radius; // encoder radius
        int down_radius; // decoder radius, also shared with actor if there is one

        int ticks_per_update; // number of ticks a layer takes to update (relative to previous layer)
        int temporal_horizon; // temporal distance into the past addressed by the layer. should be greater than or equal to ticks_per_update

        float positional_scale; // positional embedding scale

        Layer_Desc(
            const Int2 &hidden_size = Int2(4, 4),
            int HS = 16,
            int HL = 64,
            int up_radius = 2,
            int down_radius = 2,
            int ticks_per_update = 2,
            int temporal_horizon = 2,
            float positional_scale = 1.0f
        )
        :
        hidden_size(hidden_size),
        HS(HS),
        HL(HL),
        up_radius(up_radius),
        down_radius(down_radius),
        ticks_per_update(ticks_per_update),
        temporal_horizon(temporal_horizon),
        positional_scale(positional_scale)
        {}
    };

    struct Layer_Params {
        typename Encoder<S, L>::Params encoder;
    };

    struct IO_Params {};

    struct Params {
        Array<Layer_Params> layers;
        Array<IO_Params> ios;
    };

private:
    // layers
    Array<Encoder<S, L>> encoders;

    // for mapping first layer Decoders
    Int_Buffer i_indices;
    Int_Buffer d_indices;

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
        encoders.resize(layer_descs.size());

        ticks.resize(layer_descs.size(), 0);

        histories.resize(layer_descs.size());
        
        ticks_per_update.resize(layer_descs.size());

        // default update state is no update
        updates.resize(layer_descs.size(), false);

        // cache input sizes
        io_sizes.resize(io_descs.size());
        io_types.resize(io_descs.size());

        // determine ticks per update, first layer is always 1
        for (int l = 0; l < layer_descs.size(); l++)
            ticks_per_update[l] = (l == 0 ? 1 : layer_descs[l].ticks_per_update);

        int num_predictions = 0;

        for (int i = 0; i < io_sizes.size(); i++) {
            io_sizes[i] = io_descs[i].size;
            io_types[i] = static_cast<Byte>(io_descs[i].type);

            if (io_descs[i].type == prediction)
                num_predictions++;
        }

        // iterate through layers
        for (int l = 0; l < layer_descs.size(); l++) {
            // create sparse coder visible layer descriptors
            Array<typename Encoder<S, L>::Visible_Layer_Desc> e_visible_layer_descs;

            // if first layer
            if (l == 0) {
                i_indices.resize(io_sizes.size() * 2);
                d_indices = Int_Buffer(io_sizes.size(), -1);

                // initialize history buffers
                histories[l].resize(io_sizes.size());

                for (int i = 0; i < histories[l].size(); i++) {
                    int in_size = io_sizes[i].x * io_sizes[i].y;

                    histories[l][i].resize(layer_descs[l].temporal_horizon);
                    
                    for (int t = 0; t < histories[l][i].size(); t++)
                        histories[l][i][t] = Array<Vec<S, L>>(in_size, 0);
                }

                e_visible_layer_descs.resize(io_sizes.size() * layer_descs[l].temporal_horizon + num_predictions + (l < layer_descs.size() - 1));

                for (int i = 0; i < io_sizes.size(); i++) {
                    for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                        int index = t + layer_descs[l].temporal_horizon * i;

                        e_visible_layer_descs[index].size = io_sizes[i];
                        e_visible_layer_descs[index].radius = io_descs[i].up_radius;
                    }
                }

                int predictions_start = io_sizes.size() * layer_descs[l].temporal_horizon;
                int prediction_index = 0;

                for (int i = 0; i < io_sizes.size(); i++) {
                    if (io_descs[i].type == prediction) {
                        int index = predictions_start + prediction_index;

                        e_visible_layer_descs[index].size = io_sizes[i];
                        e_visible_layer_descs[index].radius = io_descs[i].up_radius;

                        i_indices[prediction_index] = i;
                        d_indices[i] = prediction_index;
                        prediction_index++;
                    }
                }

                if (l < layer_descs.size() - 1) {
                    int feedback_index = e_visible_layer_descs.size() - 1;
                    
                    e_visible_layer_descs[feedback_index].size = layer_descs[l].hidden_size;
                    e_visible_layer_descs[feedback_index].radius = layer_descs[l].down_radius;
                }
            }
            else {
                histories[l].resize(1);

                int in_size = layer_descs[l - 1].hidden_size.x * layer_descs[l - 1].hidden_size.y;

                histories[l][0].resize(layer_descs[l].temporal_horizon);

                for (int t = 0; t < histories[l][0].size(); t++)
                    histories[l][0][t] = Array<Vec<S, L>>(in_size, 0);

                e_visible_layer_descs.resize(layer_descs[l].temporal_horizon + layer_descs[l].ticks_per_update + (l < layer_descs.size() - 1));

                for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                    e_visible_layer_descs[t].size = layer_descs[l - 1].hidden_size;
                    e_visible_layer_descs[t].radius = layer_descs[l].up_radius;
                }

                for (int t = 0; t < layer_descs[l].ticks_per_update; t++) {
                    int index = layer_descs[l].temporal_horizon + t;

                    e_visible_layer_descs[index].size = layer_descs[l - 1].hidden_size;
                    e_visible_layer_descs[index].radius = layer_descs[l].up_radius;
                }

                if (l < layer_descs.size() - 1) {
                    int feedback_index = e_visible_layer_descs.size() - 1;
                    
                    e_visible_layer_descs[feedback_index].size = layer_descs[l].hidden_size;
                    e_visible_layer_descs[feedback_index].radius = layer_descs[l].down_radius;
                }
            }
            
            // create the sparse coding layer
            encoders[l].init_random(layer_descs[l].hidden_size, layer_descs[l].HS, layer_descs[l].HL, layer_descs[l].positional_scale, e_visible_layer_descs);
        }

        // initialize params
        params.layers = Array<Layer_Params>(layer_descs.size());
        params.ios = Array<IO_Params>(io_descs.size());
    }

    // simulation step/tick
    void step(
        const Array<Array_View<Vec<S, L>>> &input_vecs, // inputs to remember
        bool learn_enabled = true, // whether learning is enabled
        float reward = 0.0f, // reward
        float mimic = 0.0f // mimicry mode
    ) {
        assert(params.layers.size() == encoders.size());
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
        for (int l = 0; l < encoders.size(); l++) {
            // if is time for layer to tick
            if (l == 0 || ticks[l] >= ticks_per_update[l]) {
                // reset tick
                ticks[l] = 0;

                // updated
                updates[l] = true;

                if (learn_enabled && encoders[l].get_visible_layer(0).use_input) { // check if used at least once
                    if (l == 0) {
                        int predictions_start = io_sizes.size() * histories[l][0].size();
                        int prediction_index = 0;

                        for (int i = 0; i < io_sizes.size(); i++) {
                            if (io_types[i] == prediction) {
                                int index = predictions_start + prediction_index;

                                encoders[l].set_input_vecs(index, input_vecs[i]);

                                prediction_index++;
                            }
                        }
                    }
                    else {
                        for (int t = 0; t < ticks_per_update[l]; t++) {
                            int index = histories[l][0].size() + t;

                            encoders[l].set_input_vecs(index, histories[l][0][t]);
                        }
                    }

                    encoders[l].step(true, params.layers[l].encoder);
                }

                // ignore all (clear)
                for (int i = 0; i < encoders[l].get_num_visible_layers(); i++)
                    encoders[l].set_ignore(i);

                // set inputs
                int index = 0;

                for (int i = 0; i < histories[l].size(); i++) {
                    for (int t = 0; t < histories[l][i].size(); t++) {
                        encoders[l].set_input_vecs(index, histories[l][i][t]);

                        index++;
                    }
                }

                // activate sparse coder
                encoders[l].step(false, params.layers[l].encoder);

                // add to next layer's history
                if (l < encoders.size() - 1) {
                    int l_next = l + 1;

                    histories[l_next][0].push_front();

                    histories[l_next][0][0] = encoders[l].get_hidden_vecs();

                    ticks[l_next]++;
                }
            }
        }

        // backward
        for (int l = encoders.size() - 1; l >= 0; l--) {
            if (updates[l]) {
                if (l < encoders.size() - 1) {
                    int next_predictions_start = histories[l + 1][0].size(); // temporal horizon

                    // feedback
                    int feedback_index = encoders[l].get_num_visible_layers() - 1;

                    encoders[l].set_input_vecs(feedback_index, encoders[l + 1].get_visible_layer(next_predictions_start + ticks_per_update[l + 1] - 1 - ticks[l + 1]).recon_vecs);

                    encoders[l].step(false, params.layers[l].encoder);
                }

                // reconstruct
                if (l == 0) {
                    int predictions_start = io_sizes.size() * histories[l][0].size();
                    int prediction_index = 0;

                    for (int i = 0; i < io_sizes.size(); i++) {
                        if (io_types[i] == prediction) {
                            int index = predictions_start + prediction_index;

                            encoders[l].reconstruct(index, params.layers[l].encoder);

                            prediction_index++;
                        }
                    }
                }
                else {
                    for (int t = 0; t < ticks_per_update[l]; t++) {
                        int index = histories[l][0].size() + t;

                        encoders[l].reconstruct(index, params.layers[l].encoder);
                    }
                }
            }
        }
    }

    void clear_state() {
        updates.fill(false);
        ticks.fill(0);

        for (int l = 0; l < encoders.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++)
                    histories[l][i][t].fill(0);
            }

            encoders[l].clear_state();
        }
    }

    // serialization
    long size() const { // returns size in bytes
        long size = 2 * sizeof(int) + io_sizes.size() * sizeof(Int2) + io_types.size() * sizeof(Byte) + updates.size() * sizeof(Byte) + 2 * ticks.size() * sizeof(int) + i_indices.size() * sizeof(int) + d_indices.size() * sizeof(int);

        for (int l = 0; l < encoders.size(); l++) {
            size += sizeof(int);

            for (int i = 0; i < histories[l].size(); i++) {
                size += 2 * sizeof(int);

                for (int t = 0; t < histories[l][i].size(); t++)
                    size += sizeof(int) + histories[l][i][t].size() * sizeof(int);
            }

            size += encoders[l].size();
        }

        // params
        size += encoders.size() * sizeof(Layer_Params);
        size += io_sizes.size() * sizeof(IO_Params);

        return size;
    }

    long state_size() const { // returns size of state in bytes
        long size = updates.size() * sizeof(Byte) + ticks.size() * sizeof(int);

        for (int l = 0; l < encoders.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                size += sizeof(int);

                for (int t = 0; t < histories[l][i].size(); t++)
                    size += histories[l][i][t].size() * sizeof(int);
            }

            size += encoders[l].state_size();
        }

        return size;
    }

    long weights_size() const { // returns size of weights in bytes
        long size = 0;

        for (int l = 0; l < encoders.size(); l++) {
            size += encoders[l].weights_size();
        }

        return size;
    }

    void write(
        Stream_Writer &writer
    ) const {
        int num_layers = encoders.size();

        writer.write(&num_layers, sizeof(int));

        int num_io = io_sizes.size();

        writer.write(&num_io, sizeof(int));

        writer.write(&io_sizes[0], num_io * sizeof(Int2));
        writer.write(&io_types[0], num_io * sizeof(Byte));

        writer.write(&updates[0], updates.size() * sizeof(Byte));
        writer.write(&ticks[0], ticks.size() * sizeof(int));
        writer.write(&ticks_per_update[0], ticks_per_update.size() * sizeof(int));

        writer.write(&i_indices[0], i_indices.size() * sizeof(int));
        writer.write(&d_indices[0], d_indices.size() * sizeof(int));

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

            encoders[l].write(writer);
        }
        
        // params
        for (int l = 0; l < encoders.size(); l++)
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

        encoders.resize(num_layers);

        histories.resize(num_layers);
        
        updates.resize(num_layers);
        ticks.resize(num_layers);
        ticks_per_update.resize(num_layers);

        reader.read(&updates[0], updates.size() * sizeof(Byte));
        reader.read(&ticks[0], ticks.size() * sizeof(int));
        reader.read(&ticks_per_update[0], ticks_per_update.size() * sizeof(int));

        i_indices.resize(2 * num_io);
        d_indices.resize(num_io);

        reader.read(&i_indices[0], i_indices.size() * sizeof(int));
        reader.read(&d_indices[0], d_indices.size() * sizeof(int));
        
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

            encoders[l].read(reader);
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

        for (int l = 0; l < encoders.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                int history_start = histories[l][i].start;

                writer.write(&history_start, sizeof(int));

                for (int t = 0; t < histories[l][i].size(); t++)
                    writer.write(&histories[l][i][t][0], histories[l][i][t].size() * sizeof(int));
            }

            encoders[l].write_state(writer);
        }
    }

    void read_state(
        Stream_Reader &reader
    ) {
        reader.read(&updates[0], updates.size() * sizeof(Byte));
        reader.read(&ticks[0], ticks.size() * sizeof(int));
        
        for (int l = 0; l < encoders.size(); l++) {
            for (int i = 0; i < histories[l].size(); i++) {
                int history_start;
                
                reader.read(&history_start, sizeof(int));

                histories[l][i].start = history_start;

                for (int t = 0; t < histories[l][i].size(); t++)
                    reader.read(&histories[l][i][t][0], histories[l][i][t].size() * sizeof(int));
            }

            encoders[l].read_state(reader);
        }
    }

    void write_weights(
        Stream_Writer &writer
    ) const {
        for (int l = 0; l < encoders.size(); l++)
            encoders[l].write_weights(writer);
    }

    void read_weights(
        Stream_Reader &reader
    ) {
        for (int l = 0; l < encoders.size(); l++)
            encoders[l].read_weights(reader);
    }

    // get the number of layers (encoders)
    int get_num_layers() const {
        return encoders.size();
    }

    bool a_layer_exists(
        int i
    ) const {
        return d_indices[i] != -1;
    }

    // retrieve predictions
    const Array<Vec<S, L>> &get_prediction_vecs(
        int i
    ) const {
        int predictions_start = io_sizes.size() * histories[0][0].size();

        return encoders[0].get_visible_layer(predictions_start + d_indices[i]).recon_vecs;
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

    int get_num_encoder_visible_layers(
        int l
    ) const {
        return encoders[l].get_num_visible_layers();
    }

    // retrieve a sparse coding layer
    Encoder<S, L> &get_encoder(
        int l
    ) {
        return encoders[l];
    }

    // retrieve a sparse coding layer, const version
    const Encoder<S, L> &get_encoder(
        int l
    ) const {
        return encoders[l];
    }

    const Int_Buffer &get_i_indices() const {
        return i_indices;
    }

    const Int_Buffer &get_d_indices() const {
        return d_indices;
    }

    const Array<Circle_Buffer<Array<Vec<S, L>>>> &get_histories(
        int l
    ) const {
        return histories[l];
    }
};
}
