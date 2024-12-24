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
    encoders.resize(layer_descs.size());
    decoders.resize(layer_descs.size());

    conditions.resize(layer_descs.size());
    
    // cache input sizes
    io_sizes.resize(io_descs.size());
    io_types.resize(io_descs.size());

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
        Array<Encoder::Visible_Layer_Desc> e_visible_layer_descs;

        // if first layer
        if (l == 0) {
            e_visible_layer_descs.resize(io_sizes.size() + (layer_descs[l].recurrent_radius > -1));

            for (int i = 0; i < io_sizes.size(); i++) {
                e_visible_layer_descs[i].size = io_sizes[i];
                e_visible_layer_descs[i].radius = io_descs[i].up_radius;
            }
            
            decoders[l].resize(num_predictions);

            i_indices.resize(io_sizes.size());
            d_indices = Int_Buffer(io_sizes.size(), -1);

            // create decoders and actors
            int d_index = 0;

            for (int i = 0; i < io_sizes.size(); i++) {
                if (io_descs[i].type == prediction) {
                    // decoder visible layer descriptors
                    Array<Decoder::Visible_Layer_Desc> d_visible_layer_descs(2);

                    d_visible_layer_descs[0].size = layer_descs[l].hidden_size;
                    d_visible_layer_descs[0].radius = io_descs[i].down_radius;
                    d_visible_layer_descs[1] = d_visible_layer_descs[0];

                    decoders[l][d_index].init_random(io_sizes[i], io_descs[i].num_dendrites_per_cell, d_visible_layer_descs);

                    i_indices[d_index] = i;
                    d_indices[i] = d_index;
                    d_index++;
                }
            }
        }
        else {
            e_visible_layer_descs.resize(1 + (layer_descs[l].recurrent_radius > -1));

            e_visible_layer_descs[0].size = layer_descs[l - 1].hidden_size;
            e_visible_layer_descs[0].radius = layer_descs[l].up_radius;

            decoders[l].resize(1);

            // decoder visible layer descriptors
            Array<Decoder::Visible_Layer_Desc> d_visible_layer_descs(2);

            d_visible_layer_descs[0].size = layer_descs[l].hidden_size;
            d_visible_layer_descs[0].radius = layer_descs[l].down_radius;
            d_visible_layer_descs[1] = d_visible_layer_descs[0];

            // create decoders
            for (int t = 0; t < decoders[l].size(); t++)
                decoders[l][t].init_random(layer_descs[l - 1].hidden_size, layer_descs[l].num_dendrites_per_cell, d_visible_layer_descs);
        }
        
        // create the sparse coding layer
        encoders[l].init_random(layer_descs[l].hidden_size, e_visible_layer_descs);

        conditions[l].resize(layer_descs[l].conditioning_horizon);

        for (int t = 0; t < conditions[l].size(); t++)
            conditions[l][t] = encoders[l].get_hidden_cis();
    }

    // initialize params
    params.layers = Array<Layer_Params>(layer_descs.size());
    params.ios = Array<IO_Params>(io_descs.size());
}

void Hierarchy::step(
    const Array<Int_Buffer_View> &input_cis,
    Int_Buffer_View top_feedback_cis,
    bool learn_enabled
) {
    assert(params.layers.size() == encoders.size());
    assert(params.ios.size() == io_sizes.size());

    // set importances from params
    for (int i = 0; i < io_sizes.size(); i++)
        set_input_importance(i, params.ios[i].importance);

    // forward
    for (int l = 0; l < encoders.size(); l++) {
        // if is time for layer to tick
        if (l == 0 || ticks[l] >= ticks_per_update[l]) {
            // reset tick
            ticks[l] = 0;

            // updated
            updates[l] = true;

            Array<Int_Buffer_View> layer_input_cis(encoders[l].get_num_visible_layers());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < temporal_horizons[l]; t++) {
                    layer_input_cis[index] = histories[l][i][t];

                    index++;
                }
            }

            // copy to prevs/conditions
            conditions[l].push_front();

            conditions[l][0] = encoders[l].get_hidden_cis();

            // activate sparse coder
            encoders[l].step(layer_input_cis, learn_enabled, params.layers[l].encoder);

            // add to next layer's history
            if (l < encoders.size() - 1) {
                int l_next = l + 1;

                histories[l_next][0].push_front();

                histories[l_next][0][0] = encoders[l].get_hidden_cis();

                ticks[l_next]++;
            }
        }
    }

    // backward
    for (int l = encoders.size() - 1; l >= 0; l--) {
        Array<Int_Buffer_View> layer_input_cis(2);

        if (updates[l] && learn_enabled) {
            layer_input_cis[1] = encoders[l].get_hidden_cis();

            for (int t = conditions[l].size() - 1; t >= 0; t--) {
                layer_input_cis[0] = conditions[l][t];
                
                for (int d = 0; d < decoders[l].size(); d++) {
                    decoders[l][d].activate(layer_input_cis, (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));

                    decoders[l][d].learn(layer_input_cis, histories[l][l == 0 ? i_indices[d] : 0][(l == 0 ? 0 : d) + t * ticks_per_update[l]], (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));
                }
            }
        }

        layer_input_cis[0] = encoders[l].get_hidden_cis();
        
        if (l < encoders.size() - 1)
            layer_input_cis[1] = decoders[l + 1][ticks_per_update[l + 1] - 1 - ticks[l + 1]].get_hidden_cis();
        else
            layer_input_cis[1] = top_feedback_cis;

        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].activate(layer_input_cis, (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));
    }
}

void Hierarchy::clear_state() {
    updates.fill(false);
    ticks.fill(0);

    for (int l = 0; l < encoders.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            for (int t = 0; t < histories[l][i].size(); t++)
                histories[l][i][t].fill(0);
        }

        encoders[l].clear_state();
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].clear_state();
    }
}

long Hierarchy::size() const {
    long size = 4 * sizeof(int) + io_sizes.size() * sizeof(Int3) + io_types.size() * sizeof(Byte) + updates.size() * sizeof(Byte) + 2 * ticks.size() * sizeof(int) + i_indices.size() * sizeof(int) + d_indices.size() * sizeof(int);

    for (int l = 0; l < encoders.size(); l++) {
        size += sizeof(int);

        for (int i = 0; i < histories[l].size(); i++) {
            size += 2 * sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += sizeof(int) + histories[l][i][t].size() * sizeof(int);
        }

        size += encoders[l].size();

        for (int d = 0; d < decoders[l].size(); d++)
            size += decoders[l][d].size();
    }

    // params
    size += encoders.size() * sizeof(Layer_Params);
    size += io_sizes.size() * sizeof(IO_Params);
    size += sizeof(Byte);

    return size;
}

long Hierarchy::state_size() const {
    long size = updates.size() * sizeof(Byte) + ticks.size() * sizeof(int);

    for (int l = 0; l < encoders.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            size += sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += histories[l][i][t].size() * sizeof(int);
        }

        size += encoders[l].state_size();
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            size += decoders[l][d].state_size();
    }

    return size;
}

long Hierarchy::weights_size() const {
    long size = 0;

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].weights_size();

        for (int d = 0; d < decoders[l].size(); d++)
            size += decoders[l][d].weights_size();
    }

    return size;
}

void Hierarchy::write(
    Stream_Writer &writer
) const {
    int num_layers = encoders.size();

    writer.write(&num_layers, sizeof(int));

    int num_io = io_sizes.size();

    writer.write(&num_io, sizeof(int));

    int num_predictions = decoders[0].size();

    writer.write(&num_predictions, sizeof(int));

    writer.write(&io_sizes[0], num_io * sizeof(Int3));
    writer.write(&io_types[0], num_io * sizeof(Byte));

    writer.write(&updates[0], updates.size() * sizeof(Byte));
    writer.write(&ticks[0], ticks.size() * sizeof(int));
    writer.write(&ticks_per_update[0], ticks_per_update.size() * sizeof(int));
    writer.write(&temporal_horizons[0], temporal_horizons.size() * sizeof(int));

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

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].write(writer);

        int conditioning_horizon = conditions[l].size();

        writer.write(&conditioning_horizon, sizeof(int));

        for (int t = 0; t < conditions[l].size(); t++)
            writer.write(&conditions[l][t][0], conditions[l][t].size() * sizeof(int));
    }
    
    // params
    for (int l = 0; l < encoders.size(); l++)
        writer.write(&params.layers[l], sizeof(Layer_Params));

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

    int num_predictions;

    reader.read(&num_predictions, sizeof(int));

    io_sizes.resize(num_io);
    io_types.resize(num_io);

    reader.read(&io_sizes[0], num_io * sizeof(Int3));
    reader.read(&io_types[0], num_io * sizeof(Byte));

    encoders.resize(num_layers);
    decoders.resize(num_layers);

    histories.resize(num_layers);
    conditions.resize(num_layers);
    
    updates.resize(num_layers);
    ticks.resize(num_layers);
    ticks_per_update.resize(num_layers);
    temporal_horizons.resize(num_layers);

    reader.read(&updates[0], updates.size() * sizeof(Byte));
    reader.read(&ticks[0], ticks.size() * sizeof(int));
    reader.read(&ticks_per_update[0], ticks_per_update.size() * sizeof(int));
    reader.read(&temporal_horizons[0], temporal_horizons.size() * sizeof(int));

    i_indices.resize(num_io);
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
        
        decoders[l].resize(l == 0 ? num_predictions : ticks_per_update[l]);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].read(reader);

        int conditioning_horizon;

        reader.read(&conditioning_horizon, sizeof(int));

        conditions[l].resize(conditioning_horizon);

        for (int t = 0; t < conditioning_horizon; t++) {
            conditions[l][t].resize(encoders[l].get_hidden_cis().size());;

            reader.read(&conditions[l][t][0], conditions[l][t].size() * sizeof(int));
        }
    }

    params.layers.resize(num_layers);
    params.ios.resize(num_io);

    for (int l = 0; l < num_layers; l++)
        reader.read(&params.layers[l], sizeof(Layer_Params));

    for (int i = 0; i < num_io; i++)
        reader.read(&params.ios[i], sizeof(IO_Params));
}

void Hierarchy::write_state(
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

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].write_state(writer);
    }
}

void Hierarchy::read_state(
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
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].read_state(reader);
    }
}

void Hierarchy::write_weights(
    Stream_Writer &writer
) const {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].write_weights(writer);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].write_weights(writer);
    }
}

void Hierarchy::read_weights(
    Stream_Reader &reader
) {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].read_weights(reader);
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].read_weights(reader);
    }
}

void Hierarchy::merge(
    const Array<Hierarchy*> &hierarchies,
    Merge_Mode mode
) {
    Array<Encoder*> merge_encoders(hierarchies.size());
    Array<Decoder*> merge_decoders(hierarchies.size());

    for (int l = 0; l < encoders.size(); l++) {
        for (int h = 0; h < hierarchies.size(); h++)
            merge_encoders[h] = &hierarchies[h]->encoders[l];

        encoders[l].merge(merge_encoders, mode);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++) {
            for (int h = 0; h < hierarchies.size(); h++)
                merge_decoders[h] = &hierarchies[h]->decoders[l][d];

            decoders[l][d].merge(merge_decoders, mode);
        }
    }
}
