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
            // initialize history buffers
            histories[l].resize(io_sizes.size());

            for (int i = 0; i < histories[l].size(); i++) {
                int in_size = io_sizes[i].x * io_sizes[i].y;

                histories[l][i].resize(layer_descs[l].temporal_horizon);
                
                for (int t = 0; t < histories[l][i].size(); t++)
                    histories[l][i][t] = Int_Buffer(in_size, 0);
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
                if (io_descs[i].type == prediction || io_descs[i].type == action) {
                    int index = predictions_start + prediction_index;

                    e_visible_layer_descs[index].size = io_sizes[i];
                    e_visible_layer_descs[index].radius = io_descs[i].up_radius;

                    prediction_index++;
                }
            }

            if (l < layer_descs.size() - 1) {
                int feedback_index = e_visible_layer_descs.size() - 1;
                
                e_visible_layer_descs[feedback_index].size = layer_descs[l].hidden_size;
                e_visible_layer_descs[feedback_index].radius = layer_descs[l].down_radius;
            }

            actors.resize(num_actions);

            i_indices.resize(io_sizes.size());
            a_indices = Int_Buffer(io_sizes.size(), -1);

            // create actors
            int a_index = 0;

            for (int i = 0; i < io_sizes.size(); i++) {
                if (io_descs[i].type == action) {
                    // decoder visible layer descriptors
                    Array<Actor::Visible_Layer_Desc> a_visible_layer_descs(1 + (l < encoders.size() - 1));

                    a_visible_layer_descs[0].size = layer_descs[l].hidden_size;
                    a_visible_layer_descs[0].radius = io_descs[i].down_radius;

                    if (l < encoders.size() - 1)
                        a_visible_layer_descs[1] = a_visible_layer_descs[0];

                    actors[a_index].init_random(io_sizes[i], io_descs[i].num_dendrites_per_cell, io_descs[i].value_num_dendrites_per_cell, a_visible_layer_descs);

                    i_indices[a_index] = i;
                    a_indices[i] = a_index;
                    a_index++;
                }
            }
        }
        else {
            histories[l].resize(1);

            int in_size = layer_descs[l - 1].hidden_size.x * layer_descs[l - 1].hidden_size.y;

            histories[l][0].resize(layer_descs[l].temporal_horizon);

            for (int t = 0; t < histories[l][0].size(); t++)
                histories[l][0][t] = Int_Buffer(in_size, 0);

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
        encoders[l].init_random(layer_descs[l].hidden_size, layer_descs[l].base_vigilance, e_visible_layer_descs);
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

            if (learn_enabled && encoders[l].get_visible_layer(0).use_input) { // check if used at least once
                if (l == 0) {
                    int predictions_start = io_sizes.size() * histories[l][0].size();
                    int prediction_index = 0;

                    for (int i = 0; i < io_sizes.size(); i++) {
                        if (io_types[i] == prediction || io_types[i] == action) {
                            int index = predictions_start + prediction_index;

                            encoders[l].set_input_cis(index, input_cis[i]);

                            prediction_index++;
                        }
                    }
                }
                else {
                    for (int t = 0; t < ticks_per_update[l]; t++) {
                        int index = histories[l][0].size() + t;

                        encoders[l].set_input_cis(index, histories[l][0][t]);
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
                    encoders[l].set_input_cis(index, histories[l][i][t]);

                    index++;
                }
            }

            // activate sparse coder
            encoders[l].step(false, params.layers[l].encoder);

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
        if (updates[l]) {
            if (l < encoders.size() - 1) {
                int next_predictions_start = histories[l + 1][0].size(); // temporal horizon

                // learn on feedback
                int feedback_index = encoders[l].get_num_visible_layers() - 1;

                encoders[l].set_input_cis(feedback_index, encoders[l + 1].get_visible_layer(next_predictions_start + ticks_per_update[l + 1] - 1 - ticks[l + 1]).recon_cis);

                encoders[l].step(false, params.layers[l].encoder);
            }

            // reconstruct
            if (l == 0) {
                int predictions_start = io_sizes.size() * histories[l][0].size();
                int prediction_index = 0;

                for (int i = 0; i < io_sizes.size(); i++) {
                    if (io_types[i] == prediction || io_types[i] == action) {
                        int index = predictions_start + prediction_index;

                        encoders[l].reconstruct(index);

                        prediction_index++;
                    }
                }
            }
            else {
                for (int t = 0; t < ticks_per_update[l]; t++) {
                    int index = histories[l][0].size() + t;

                    encoders[l].reconstruct(index);
                }
            }

            if (l == 0 && actors.size() > 0) {
                int next_predictions_start = histories[l + 1][0].size(); // temporal horizon

                Array<Int_Buffer_View> layer_input_cis(1 + (encoders.size() > 1));

                layer_input_cis[0] = encoders[l].get_hidden_cis();

                if (encoders.size() > 1)
                    layer_input_cis[1] = encoders[l + 1].get_visible_layer(next_predictions_start + ticks_per_update[l + 1] - 1 - ticks[l + 1]).recon_cis;

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
    }

    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].clear_state();
}

long Hierarchy::size() const {
    long size = 2 * sizeof(int) + io_sizes.size() * sizeof(Int3) + io_types.size() * sizeof(Byte) + updates.size() * sizeof(Byte) + 2 * ticks.size() * sizeof(int) + i_indices.size() * sizeof(int) + a_indices.size() * sizeof(int);

    for (int l = 0; l < encoders.size(); l++) {
        size += sizeof(int);

        for (int i = 0; i < histories[l].size(); i++) {
            size += 2 * sizeof(int);

            for (int t = 0; t < histories[l][i].size(); t++)
                size += sizeof(int) + histories[l][i][t].size() * sizeof(int);
        }

        size += encoders[l].size();
    }

    // actors
    for (int d = 0; d < actors.size(); d++)
        size += actors[d].size();

    // params
    size += encoders.size() * sizeof(Layer_Params);
    size += io_sizes.size() * sizeof(IO_Params);

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

    writer.write(reinterpret_cast<const void*>(&num_layers), sizeof(int));

    int num_io = io_sizes.size();

    writer.write(reinterpret_cast<const void*>(&num_io), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&io_sizes[0]), num_io * sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&io_types[0]), num_io * sizeof(Byte));

    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(Byte));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&ticks_per_update[0]), ticks_per_update.size() * sizeof(int));

    writer.write(reinterpret_cast<const void*>(&i_indices[0]), i_indices.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&a_indices[0]), a_indices.size() * sizeof(int));

    for (int l = 0; l < num_layers; l++) {
        int num_layer_inputs = histories[l].size();

        writer.write(reinterpret_cast<const void*>(&num_layer_inputs), sizeof(int));

        for (int i = 0; i < histories[l].size(); i++) {
            int history_size = histories[l][i].size();

            writer.write(reinterpret_cast<const void*>(&history_size), sizeof(int));

            int history_start = histories[l][i].start;

            writer.write(reinterpret_cast<const void*>(&history_start), sizeof(int));

            for (int t = 0; t < histories[l][i].size(); t++) {
                int buffer_size = histories[l][i][t].size();

                writer.write(reinterpret_cast<const void*>(&buffer_size), sizeof(int));

                writer.write(reinterpret_cast<const void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
            }
        }

        encoders[l].write(writer);
    }
    
    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].write(writer);

    // params
    for (int l = 0; l < encoders.size(); l++)
        writer.write(reinterpret_cast<const void*>(&params.layers[l]), sizeof(Layer_Params));

    for (int i = 0; i < io_sizes.size(); i++)
        writer.write(reinterpret_cast<const void*>(&params.ios[i]), sizeof(IO_Params));
}

void Hierarchy::read(
    Stream_Reader &reader
) {
    int num_layers;

    reader.read(reinterpret_cast<void*>(&num_layers), sizeof(int));

    int num_io;

    reader.read(reinterpret_cast<void*>(&num_io), sizeof(int));

    io_sizes.resize(num_io);
    io_types.resize(num_io);

    reader.read(reinterpret_cast<void*>(&io_sizes[0]), num_io * sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&io_types[0]), num_io * sizeof(Byte));

    int num_predictions = 0;
    int num_actions = 0;

    for (int i = 0; i < io_sizes.size(); i++) {
        if (io_types[i]== prediction)
            num_predictions++;
        else if (io_types[i]== action) {
            num_actions++;
            num_predictions++;
        }
    }

    encoders.resize(num_layers);

    histories.resize(num_layers);
    
    updates.resize(num_layers);
    ticks.resize(num_layers);
    ticks_per_update.resize(num_layers);

    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(Byte));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&ticks_per_update[0]), ticks_per_update.size() * sizeof(int));

    i_indices.resize(num_io);
    a_indices.resize(num_io);

    reader.read(reinterpret_cast<void*>(&i_indices[0]), i_indices.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&a_indices[0]), a_indices.size() * sizeof(int));
    
    for (int l = 0; l < num_layers; l++) {
        int num_layer_inputs;
        
        reader.read(reinterpret_cast<void*>(&num_layer_inputs), sizeof(int));

        histories[l].resize(num_layer_inputs);

        for (int i = 0; i < histories[l].size(); i++) {
            int history_size;

            reader.read(reinterpret_cast<void*>(&history_size), sizeof(int));

            int history_start;
            
            reader.read(reinterpret_cast<void*>(&history_start), sizeof(int));

            histories[l][i].resize(history_size);
            histories[l][i].start = history_start;

            for (int t = 0; t < histories[l][i].size(); t++) {
                int buffer_size;

                reader.read(reinterpret_cast<void*>(&buffer_size), sizeof(int));

                histories[l][i][t].resize(buffer_size);

                reader.read(reinterpret_cast<void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
            }
        }

        encoders[l].read(reader);
    }

    actors.resize(num_actions);

    // actors
    for (int d = 0; d < actors.size(); d++)
        actors[d].read(reader);

    params.layers.resize(num_layers);
    params.ios.resize(num_io);

    for (int l = 0; l < num_layers; l++)
        reader.read(reinterpret_cast<void*>(&params.layers[l]), sizeof(Layer_Params));

    for (int i = 0; i < num_io; i++)
        reader.read(reinterpret_cast<void*>(&params.ios[i]), sizeof(IO_Params));
}

void Hierarchy::write_state(
    Stream_Writer &writer
) const {
    writer.write(reinterpret_cast<const void*>(&updates[0]), updates.size() * sizeof(Byte));
    writer.write(reinterpret_cast<const void*>(&ticks[0]), ticks.size() * sizeof(int));

    for (int l = 0; l < encoders.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int history_start = histories[l][i].start;

            writer.write(reinterpret_cast<const void*>(&history_start), sizeof(int));

            for (int t = 0; t < histories[l][i].size(); t++)
                writer.write(reinterpret_cast<const void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        encoders[l].write_state(writer);
    }

    for (int d = 0; d < actors.size(); d++)
        actors[d].write_state(writer);
}

void Hierarchy::read_state(
    Stream_Reader &reader
) {
    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(Byte));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    
    for (int l = 0; l < encoders.size(); l++) {
        for (int i = 0; i < histories[l].size(); i++) {
            int history_start;
            
            reader.read(reinterpret_cast<void*>(&history_start), sizeof(int));

            histories[l][i].start = history_start;

            for (int t = 0; t < histories[l][i].size(); t++)
                reader.read(reinterpret_cast<void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
        }

        encoders[l].read_state(reader);
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
    }

    for (int d = 0; d < actors.size(); d++)
        actors[d].write_weights(writer);
}

void Hierarchy::read_weights(
    Stream_Reader &reader
) {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].read_weights(reader);
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

    for (int l = 0; l < encoders.size(); l++) {
        for (int h = 0; h < hierarchies.size(); h++)
            merge_encoders[h] = &hierarchies[h]->encoders[l];

        encoders[l].merge(merge_encoders, mode);
    }
    
    // actors
    Array<Actor*> merge_actors(hierarchies.size());

    for (int d = 0; d < actors.size(); d++) {
        for (int h = 0; h < hierarchies.size(); h++)
            merge_actors[h] = &hierarchies[h]->actors[d];

        actors[d].merge(merge_actors, mode);
    }
}
