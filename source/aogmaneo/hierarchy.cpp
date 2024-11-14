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
    hidden_cis_prev.resize(layer_descs.size());
    feedback_cis_prev.resize(layer_descs.size() - 1);
    errors.resize(layer_descs.size());

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
    int num_actions = 0;

    for (int i = 0; i < io_sizes.size(); i++) {
        io_sizes[i] = io_descs[i].size;
        io_types[i] = static_cast<Byte>(io_descs[i].type);

        if (io_descs[i].type == prediction)
            num_predictions++;
        else if (io_descs[i].type == action) {
            num_actions++;
            num_predictions++;
        }
    }

    // iterate through layers
    for (int l = 0; l < layer_descs.size(); l++) {
        // create sparse coder visible layer descriptors
        Array<Encoder::Visible_Layer_Desc> e_visible_layer_descs;

        // if first layer
        if (l == 0) {
            e_visible_layer_descs.resize(io_sizes.size() * layer_descs[l].temporal_horizon);

            for (int i = 0; i < io_sizes.size(); i++) {
                for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                    int index = t + layer_descs[l].temporal_horizon * i;

                    e_visible_layer_descs[index].size = io_sizes[i];
                    e_visible_layer_descs[index].radius = io_descs[i].up_radius;
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

            decoders[l].resize(num_predictions);
            actors.resize(num_actions);

            i_indices.resize(io_sizes.size() * 2);
            d_indices = Int_Buffer(io_sizes.size(), -1);

            // create decoders and actors
            int d_index = 0;

            for (int i = 0; i < io_sizes.size(); i++) {
                if (io_descs[i].type == prediction || io_descs[i].type == action) {
                    // decoder visible layer descriptors
                    Array<Decoder::Visible_Layer_Desc> d_visible_layer_descs(1 + (l < encoders.size() - 1));

                    d_visible_layer_descs[0].size = layer_descs[l].hidden_size;
                    d_visible_layer_descs[0].radius = io_descs[i].down_radius;

                    if (l < encoders.size() - 1)
                        d_visible_layer_descs[1] = d_visible_layer_descs[0];

                    decoders[l][d_index].init_random(io_sizes[i], d_visible_layer_descs);

                    i_indices[d_index] = i;
                    d_indices[i] = d_index;
                    d_index++;
                }
            }

            d_index = 0;

            for (int i = 0; i < io_sizes.size(); i++) {
                if (io_descs[i].type == action) {
                    // decoder visible layer descriptors
                    Array<Actor::Visible_Layer_Desc> a_visible_layer_descs(1 + (l < encoders.size() - 1));

                    a_visible_layer_descs[0].size = layer_descs[l].hidden_size;
                    a_visible_layer_descs[0].radius = io_descs[i].down_radius;

                    if (l < encoders.size() - 1)
                        a_visible_layer_descs[1] = a_visible_layer_descs[0];

                    actors[d_index].init_random(io_sizes[i], io_descs[i].history_capacity, a_visible_layer_descs);

                    i_indices[io_sizes.size() + d_index] = i;
                    d_indices[i] = d_index;
                    d_index++;
                }
            }
        }
        else {
            e_visible_layer_descs.resize(layer_descs[l].temporal_horizon);

            for (int t = 0; t < layer_descs[l].temporal_horizon; t++) {
                e_visible_layer_descs[t].size = layer_descs[l - 1].hidden_size;
                e_visible_layer_descs[t].radius = layer_descs[l].up_radius;
            }

            histories[l].resize(1);

            int in_size = layer_descs[l - 1].hidden_size.x * layer_descs[l - 1].hidden_size.y;

            histories[l][0].resize(layer_descs[l].temporal_horizon);

            for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = Int_Buffer(in_size, 0);

            decoders[l].resize(layer_descs[l].ticks_per_update);

            // decoder visible layer descriptors
            Array<Decoder::Visible_Layer_Desc> d_visible_layer_descs(1 + (l < encoders.size() - 1));

            d_visible_layer_descs[0].size = layer_descs[l].hidden_size;
            d_visible_layer_descs[0].radius = layer_descs[l].down_radius;

            if (l < encoders.size() - 1)
                d_visible_layer_descs[1] = d_visible_layer_descs[0];

            // create decoders
            for (int t = 0; t < decoders[l].size(); t++)
                decoders[l][t].init_random(layer_descs[l - 1].hidden_size, d_visible_layer_descs);
        }
        
        // create the sparse coding layer
        encoders[l].init_random(layer_descs[l].hidden_size, e_visible_layer_descs);

        hidden_cis_prev[l] = encoders[l].get_hidden_cis();

        if (l < encoders.size() - 1)
            feedback_cis_prev[l] = encoders[l].get_hidden_cis();

        errors[l].resize(hidden_cis_prev[l].size());
    }

    // initialize params
    params.layers = Array<Layer_Params>(layer_descs.size());
    params.ios = Array<IO_Params>(io_descs.size());
}

void Hierarchy::step(
    const Array<Int_Buffer_View> &input_cis,
    bool learn_enabled,
    float reward,
    float mimic
) {
    assert(params.layers.size() == encoders.size());
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
    for (int l = 0; l < encoders.size(); l++) {
        // if is time for layer to tick
        if (l == 0 || ticks[l] >= ticks_per_update[l]) {
            // reset tick
            ticks[l] = 0;

            // updated
            updates[l] = true;

            // copy to prev
            hidden_cis_prev[l] = encoders[l].get_hidden_cis();

            if (l < encoders.size() - 1)
                feedback_cis_prev[l] = decoders[l + 1][ticks_per_update[l + 1] - 1 - ticks[l + 1]].get_hidden_cis();

            Array<Int_Buffer_View> layer_input_cis(encoders[l].get_num_visible_layers());

            int index = 0;

            for (int i = 0; i < histories[l].size(); i++) {
                for (int t = 0; t < histories[l][i].size(); t++) {
                    layer_input_cis[index] = histories[l][i][t];

                    index++;
                }
            }

            if (learn_enabled) {
                errors[l].fill(0.0f);

                for (int d = 0; d < decoders[l].size(); d++)
                    decoders[l][d].generate_errors(hidden_cis_prev[l], histories[l][l == 0 ? i_indices[d] : 0][l == 0 ? 0 : d], errors[l], 0, params.layers[l].decoder);

                // Rescale
                float decoders_inv = 1.0f / decoders[l].size();

                for (int i = 0; i < errors[l].size(); i++)
                    errors[l][i] *= decoders_inv;
            }

            // activate sparse coder
            encoders[l].step(layer_input_cis, errors[l], learn_enabled, params.layers[l].encoder);

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
    for (int l = decoders.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            Array<Int_Buffer_View> layer_input_cis(1 + (l < encoders.size() - 1));

            if (learn_enabled) {
                layer_input_cis[0] = hidden_cis_prev[l];
                
                if (l < encoders.size() - 1) {
                    // learn on feedback
                    layer_input_cis[1] = feedback_cis_prev[l];

                    for (int d = 0; d < decoders[l].size(); d++)
                        decoders[l][d].learn(layer_input_cis, histories[l][l == 0 ? i_indices[d] : 0][l == 0 ? 0 : d], (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));

                    if (params.anticipation) {
                        // learn on actual
                        layer_input_cis[1] = encoders[l].get_hidden_cis();

                        for (int d = 0; d < decoders[l].size(); d++) {
                            // need to re-activate for this
                            decoders[l][d].activate(layer_input_cis, (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));

                            decoders[l][d].learn(layer_input_cis, histories[l][l == 0 ? i_indices[d] : 0][l == 0 ? 0 : d], (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));
                        }
                    }
                }
                else {
                    for (int d = 0; d < decoders[l].size(); d++)
                        decoders[l][d].learn(layer_input_cis, histories[l][l == 0 ? i_indices[d] : 0][l == 0 ? 0 : d], (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));
                }
            }

            layer_input_cis[0] = encoders[l].get_hidden_cis();
            
            if (l < encoders.size() - 1)
                layer_input_cis[1] = decoders[l + 1][ticks_per_update[l + 1] - 1 - ticks[l + 1]].get_hidden_cis();

            for (int d = 0; d < decoders[l].size(); d++)
                decoders[l][d].activate(layer_input_cis, (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));

            if (l == 0) {
                for (int d = 0; d < actors.size(); d++)
                    actors[d].step(layer_input_cis, input_cis[i_indices[d + io_sizes.size()]], learn_enabled, reward, mimic, params.ios[i_indices[d + io_sizes.size()]].actor);
            }
        }
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

    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].clear_state();
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

    // actors
    for (int d = 0; d < actors.size(); d++)
        size += actors[d].size();

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

    // actors
    for (int d = 0; d < actors.size(); d++)
        size += actors[d].state_size();

    return size;
}

long Hierarchy::weights_size() const {
    long size = 0;

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].weights_size();

        for (int d = 0; d < decoders[l].size(); d++)
            size += decoders[l][d].weights_size();
    }

    // actors
    for (int d = 0; d < actors.size(); d++)
        size += actors[d].weights_size();

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
    int num_actions = actors.size();

    writer.write(&num_predictions, sizeof(int));
    writer.write(&num_actions, sizeof(int));

    writer.write(&io_sizes[0], num_io * sizeof(Int3));
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

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].write(writer);
    }
    
    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].write(writer);

    // params
    for (int l = 0; l < encoders.size(); l++)
        writer.write(&params.layers[l], sizeof(Layer_Params));

    for (int i = 0; i < io_sizes.size(); i++)
        writer.write(&params.ios[i], sizeof(IO_Params));

    writer.write(&params.anticipation, sizeof(Byte));
}

void Hierarchy::read(
    Stream_Reader &reader
) {
    int num_layers;

    reader.read(&num_layers, sizeof(int));

    int num_io;

    reader.read(&num_io, sizeof(int));

    int num_predictions;
    int num_actions;

    reader.read(&num_predictions, sizeof(int));
    reader.read(&num_actions, sizeof(int));

    io_sizes.resize(num_io);
    io_types.resize(num_io);

    reader.read(&io_sizes[0], num_io * sizeof(Int3));
    reader.read(&io_types[0], num_io * sizeof(Byte));

    encoders.resize(num_layers);
    decoders.resize(num_layers);
    hidden_cis_prev.resize(num_layers);
    feedback_cis_prev.resize(num_layers - 1);
    errors.resize(num_layers);

    histories.resize(num_layers);
    
    updates.resize(num_layers);
    ticks.resize(num_layers);
    ticks_per_update.resize(num_layers);

    reader.read(&updates[0], updates.size() * sizeof(Byte));
    reader.read(&ticks[0], ticks.size() * sizeof(int));
    reader.read(&ticks_per_update[0], ticks_per_update.size() * sizeof(int));

    i_indices.resize(num_io * 2);
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

        hidden_cis_prev[l] = encoders[l].get_hidden_cis();

        if (l < encoders.size() - 1)
            feedback_cis_prev[l] = encoders[l].get_hidden_cis();

        errors[l].resize(hidden_cis_prev[l].size());
    }

    actors.resize(num_actions);

    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].read(reader);

    params.layers.resize(num_layers);
    params.ios.resize(num_io);

    for (int l = 0; l < num_layers; l++)
        reader.read(&params.layers[l], sizeof(Layer_Params));

    for (int i = 0; i < num_io; i++)
        reader.read(&params.ios[i], sizeof(IO_Params));

    reader.read(&params.anticipation, sizeof(Byte));
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

    for (int d = 0; d < actors.size(); d++)
        actors[d].write_state(writer);
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

    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].read_state(reader);
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

    for (int d = 0; d < actors.size(); d++)
        actors[d].write_weights(writer);
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

    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].read_weights(reader);
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
    
    // actors
    Array<Actor*> merge_actors(hierarchies.size());

    for (int d = 0; d < actors.size(); d++) {
        for (int h = 0; h < hierarchies.size(); h++)
            merge_actors[h] = &hierarchies[h]->actors[d];

        actors[d].merge(merge_actors, mode);
    }
}
