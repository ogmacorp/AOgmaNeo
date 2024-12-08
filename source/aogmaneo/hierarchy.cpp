// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "hierarchy.h"

using namespace aon;

void Hierarchy::init_random(
    const Array<IO_Desc> &io_descs,
    const Array<Layer_Desc> &layer_descs
) {
    // create layers
    layers.resize(layer_descs.size());

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

    for (int i = 0; i < io_sizes.size(); i++) {
        io_sizes[i] = io_descs[i].size;
        io_types[i] = static_cast<Byte>(io_descs[i].type);
    }

    // iterate through layers
    for (int l = 0; l < layer_descs.size(); l++) {
        // create sparse coder visible layer descriptors
        Array<Layer::Visible_Layer_Desc> visible_layer_descs;

        // if first layer
        if (l == 0) {
            visible_layer_descs.resize(io_sizes.size() * layer_descs[l].temporal_horizon);

            for (int i = 0; i < io_sizes.size(); i++) {
                for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                    int index = t + layer_descs[l].temporal_horizon * i;

                    visible_layer_descs[index].size = io_sizes[i];
                    visible_layer_descs[index].radius = io_descs[i].up_radius;
                }
            }
            
            // initialize history buffers
            histories[l].resize(io_sizes.size());

            for (int i = 0; i < histories[l].size(); i++) {
                int in_size = io_sizes[i].x * io_sizes[i].y;

                histories[l][i].resize(layer_descs[l].temporal_horizon);
                
                for (int t = 0; t < histories[l][i].size(); t++)
                    histories[l][i][t] = Int_Buffer(in_size, 0);
            }
        }
        else {
            visible_layer_descs.resize(layer_descs[l].temporal_horizon);

            for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                visible_layer_descs[t].size = layer_descs[l - 1].hidden_size;
                visible_layer_descs[t].radius = layer_descs[l].up_radius;
            }

            histories[l].resize(1);

            int in_size = layer_descs[l - 1].hidden_size.x * layer_descs[l - 1].hidden_size.y;

            histories[l][0].resize(layer_descs[l].temporal_horizon);

            for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = Int_Buffer(in_size, 0);
        }
        
        // create the sparse coding layer
        layers[l].init_random(layer_descs[l].hidden_size, visible_layer_descs);
    }

    // initialize params
    params.layers = Array<Layer::Params>(layer_descs.size());
    params.ios = Array<IO_Params>(io_descs.size());
}

void Hierarchy::step(
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View top_goal_cis,
    bool learn_enabled
) {
    assert(params.layers.size() == layers.size());
    assert(params.ios.size() == io_sizes.size());

    // set importances from params
    for (int i = 0; i < io_sizes.size(); i++)
        set_input_importance(i, params.ios[i].importance);

    // first tick is always 0
    ticks[0] = 0;

    // add input to first layer history   
    for (int i = 0; i < io_sizes.size(); i++) {
        histories[0][i].push_front();

        histories[0][i][0] = input_cis[i];
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

            Array<Int_Buffer_View> layer_input_cis(layers[l].get_num_visible_layers());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++) {
                    layer_input_cis[index] = histories[l][i][t];

                    index++;
                }
            }

            // activate sparse coder
            layers[l].step(layer_input_cis, learn_enabled, params.layers[l]);

            // add to next layer's history
            if (l < layers.size() - 1) {
                int l_next = l + 1;

                histories[l_next][0].push_front();

                histories[l_next][0][0] = layers[l].get_hidden_cis();

                ticks[l_next]++;
            }
        }
    }

    // backward
    for (int l = layers.size() - 1; l >= 0; l--) {
        Int_Buffer_View goal_cis;

        if (l < layers.size() - 1) {
            int vli = ticks_per_update[l + 1] - 1 - ticks[l + 1];

            layers[l + 1].reconstruct(vli);

            goal_cis = layers[l + 1].get_reconstruction(vli);
        }
        else
            goal_cis = top_goal_cis;

        layers[l].plan(goal_cis, params.layers[l]);
    }

    // reconstruct all output predictions
    for (int i = 0; i < io_sizes.size(); i++) {
        if (io_types[i] == prediction)
            layers[0].reconstruct(i * histories[0][i].size());
    }
}

void Hierarchy::clear_state() {
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

long Hierarchy::size() const {
    long size = 2 * sizeof(int) + io_sizes.size() * sizeof(Int3) + io_types.size() * sizeof(Byte) + updates.size() * sizeof(Byte) + 2 * ticks.size() * sizeof(int);

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
    size += layers.size() * sizeof(Layer::Params);
    size += io_sizes.size() * sizeof(IO_Params);

    return size;
}

long Hierarchy::state_size() const {
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

long Hierarchy::weights_size() const {
    long size = 0;

    for (int l = 0; l < layers.size(); l++)
        size += layers[l].weights_size();

    return size;
}

void Hierarchy::write(
    Stream_Writer &writer
) const {
    int num_layers = layers.size();

    writer.write(&num_layers, sizeof(int));

    int num_io = io_sizes.size();

    writer.write(&num_io, sizeof(int));

    writer.write(&io_sizes[0], num_io * sizeof(Int3));
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
        writer.write(&params.layers[l], sizeof(Layer::Params));

    for (int i = 0; i < io_sizes.size(); i++)
        writer.write(&params.ios[i], sizeof(IO_Params));
}

void Hierarchy::read(
    Stream_Reader &reader
) {
    int num_layers;

    reader.read(&num_layers, sizeof(int));

    int num_io;

    reader.read(&num_io, sizeof(int));

    io_sizes.resize(num_io);
    io_types.resize(num_io);

    reader.read(&io_sizes[0], num_io * sizeof(Int3));
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
        reader.read(&params.layers[l], sizeof(Layer::Params));

    for (int i = 0; i < num_io; i++)
        reader.read(&params.ios[i], sizeof(IO_Params));
}

void Hierarchy::write_state(
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

void Hierarchy::read_state(
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

void Hierarchy::write_weights(
    Stream_Writer &writer
) const {
    for (int l = 0; l < layers.size(); l++)
        layers[l].write_weights(writer);
}

void Hierarchy::read_weights(
    Stream_Reader &reader
) {
    for (int l = 0; l < layers.size(); l++)
        layers[l].read_weights(reader);
}
