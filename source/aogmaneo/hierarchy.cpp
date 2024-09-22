// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
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
    routed_layers.resize(layer_descs.size() - 1);

    hidden_cis_prev.resize(layer_descs.size());

    // cache input sizes
    io_sizes.resize(io_descs.size());
    io_types.resize(io_descs.size());

    int num_predictions = 0;
    int num_actions = 0;

    for (int i = 0; i < io_sizes.size(); i++) {
        io_sizes[i] = io_descs[i].size;
        io_types[i] = static_cast<Byte>(io_descs[i].type);

        if (io_descs[i].type == prediction)
            num_predictions++;
        else if (io_descs[i].type == action) {
            num_predictions++;
            num_actions++;
        }
    }

    // iterate through layers
    for (int l = 0; l < layer_descs.size(); l++) {
        // create sparse coder visible layer descriptors
        Array<Encoder::Visible_Layer_Desc> e_visible_layer_descs;

        // if first layer
        if (l == 0) {
            e_visible_layer_descs.resize(io_sizes.size() + (layer_descs[l].recurrent_radius >= 0));

            for (int i = 0; i < io_sizes.size(); i++) {
                e_visible_layer_descs[i].size = io_sizes[i];
                e_visible_layer_descs[i].radius = io_descs[i].up_radius;
            }

            predictors.resize(num_predictions);
            actors.resize(num_actions);

            i_indices.resize(io_sizes.size() * 2);
            o_indices = Int_Buffer(io_sizes.size(), -1);

            // create decoders and actors
            int o_index = 0;

            for (int i = 0; i < io_sizes.size(); i++) {
                if (io_descs[i].type == prediction || io_descs[i].type == action) {
                    // decoder visible layer descriptors
                    Array<Predictor::Visible_Layer_Desc> p_visible_layer_descs(1);

                    p_visible_layer_descs[0].size = layer_descs[l].hidden_size;
                    p_visible_layer_descs[0].radius = io_descs[i].down_radius;

                    predictors[o_index].init_random(io_sizes[i], p_visible_layer_descs);

                    i_indices[o_index] = i;
                    o_indices[i] = o_index;
                    o_index++;
                }
            }

            o_index = 0;

            for (int i = 0; i < io_sizes.size(); i++) {
                if (io_descs[i].type == action) {
                    // decoder visible layer descriptors
                    Array<Actor::Visible_Layer_Desc> a_visible_layer_descs(1);

                    a_visible_layer_descs[0].size = layer_descs[l].hidden_size;
                    a_visible_layer_descs[0].radius = io_descs[i].down_radius;

                    actors[o_index].init_random(io_sizes[i], io_descs[i].history_capacity, a_visible_layer_descs);

                    i_indices[io_sizes.size() + o_index] = i;
                    o_indices[i] = o_index;
                    o_index++;
                }
            }
        }
        else {
            e_visible_layer_descs.resize(1 + (layer_descs[l].recurrent_radius >= 0));

            e_visible_layer_descs[0].size = layer_descs[l - 1].hidden_size;
            e_visible_layer_descs[0].radius = layer_descs[l].up_radius;
        }
        
        if (layer_descs[l].recurrent_radius >= 0) {
            e_visible_layer_descs[e_visible_layer_descs.size() - 1].size = layer_descs[l].hidden_size;
            e_visible_layer_descs[e_visible_layer_descs.size() - 1].radius = layer_descs[l].recurrent_radius;
        }

        // create the sparse coding layer
        encoders[l].init_random(layer_descs[l].hidden_size, e_visible_layer_descs);

        if (l < layer_descs.size() - 1) {
            // routed layer visible layer descriptors
            Array<Routed_Layer::Visible_Layer_Desc> r_visible_layer_descs(1);

            r_visible_layer_descs[0].size = layer_descs[l + 1].hidden_size;
            r_visible_layer_descs[0].radius = layer_descs[l].down_radius;

            // create decoders
            routed_layers[l].init_random(layer_descs[l].hidden_size, r_visible_layer_descs);
        }

        hidden_cis_prev[l] = encoders[l].get_hidden_cis();
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

    // backward and learn predictors, also set importances
    int p_index = 0;

    for (int i = 0; i < io_sizes.size(); i++) {
        if (learn_enabled) {
            Array<Int_Buffer_View> layer_input_cis(1);
            Array<Float_Buffer_View> layer_input_acts(1);

            layer_input_cis[0] = encoders[0].get_hidden_cis();

            if (routed_layers.size() > 0)
                layer_input_acts[0] = routed_layers[0].get_hidden_acts();

            if (io_types[i] == prediction || io_types[i] == action) {
                predictors[p_index].backward(layer_input_cis, layer_input_acts, input_cis[i], true, params.ios[i].predictor);
                
                p_index++;
            }
        }

        encoders[0].get_visible_layer(i).importance = params.ios[i].importance;
    }

    Array<Int_Buffer_View> r_input_cis(1);
    Array<Float_Buffer_View> r_input_acts(1);

    // merge errors from predictors (into first one, which always exists)
    Float_Buffer_View errors0 = predictors[0].get_visible_layer(0).errors;

    for (int p = 1; p < predictors.size(); p++) {
        Float_Buffer_View errors = predictors[p].get_visible_layer(0).errors;

        for (int i = 0; i < errors.size(); i++)
            errors0[i] += errors[i];
    }

    float predictors_inv = 1.0f / predictors.size();

    for (int i = 0; i < errors0.size(); i++)
        errors0[i] *= predictors_inv;

    // up
    for (int l = 0; l < encoders.size(); l++) {
        hidden_cis_prev[l] = encoders[l].get_hidden_cis();

        if (learn_enabled && l < encoders.size() - 1) {
            r_input_cis[0] = encoders[l + 1].get_hidden_cis();

            if (l < routed_layers.size() - 1)
                r_input_acts[0] = routed_layers[l + 1].get_hidden_acts();
            else
                r_input_acts[0] = Float_Buffer_View(); // empty

            routed_layers[l].backward(r_input_cis, r_input_acts, encoders[l].get_hidden_cis(), (l == 0 ? errors0 : routed_layers[l - 1].get_visible_layer(0).errors), true, params.layers[l].routed_layer);
        }

        Array<Int_Buffer_View> e_input_cis(encoders[l].get_num_visible_layers());

        if (l == 0) {
            for (int i = 0; i < io_sizes.size(); i++)
                e_input_cis[i] = input_cis[i];

            if (e_input_cis.size() > io_sizes.size()) {
                e_input_cis[io_sizes.size()] = hidden_cis_prev[l];

                // set importance
                encoders[l].get_visible_layer(io_sizes.size()).importance = params.layers[l].recurrent_importance;
            }
        }
        else {
            e_input_cis[0] = encoders[l - 1].get_hidden_cis();

            if (e_input_cis.size() > 1) {
                e_input_cis[1] = hidden_cis_prev[l];

                // set importance
                encoders[l].get_visible_layer(1).importance = params.layers[l].recurrent_importance;
            }
        }

        // activate sparse coder
        encoders[l].step(e_input_cis, learn_enabled, params.layers[l].encoder);
    }

    // down
    for (int l = encoders.size() - 2; l >= 0; l--) {
        r_input_cis[0] = encoders[l + 1].get_hidden_cis();

        if (l < routed_layers.size() - 1) // only set if there is a next layer. Will treat as all 1's if there is no next layer
            r_input_acts[0] = routed_layers[l + 1].get_hidden_acts();
        else
            r_input_acts[0] = Float_Buffer_View(); // empty

        routed_layers[l].forward(r_input_cis, r_input_acts, encoders[l].get_hidden_cis(), params.layers[l].routed_layer);
    }

    // predictors and actors
    p_index = 0;
    int a_index = 0;

    for (int i = 0; i < io_sizes.size(); i++) {
        r_input_cis[0] = encoders[0].get_hidden_cis();
        
        if (routed_layers.size() > 0)
            r_input_acts[0] = routed_layers[0].get_hidden_acts();
        else
            r_input_acts[0] = Float_Buffer_View(); // empty

        if (io_types[i] == prediction || io_types[i] == action) {
            predictors[p_index].forward(r_input_cis, r_input_acts, params.ios[i].predictor);
            
            p_index++;
        }

        if (io_types[i] == action) {
            actors[a_index].step(r_input_cis, r_input_acts, input_cis[i], learn_enabled, reward, mimic, params.ios[i].actor);
            
            a_index++;
        }
    }
}

void Hierarchy::clear_state() {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].clear_state();

        if (l < encoders.size() - 1)
            routed_layers[l].clear_state();
    }

    // predictors
    for (int p = 0; p < predictors.size(); p++)
        predictors[p].clear_state();

    // actors
    for (int a = 0; a < actors.size(); a++)
        actors[a].clear_state();
}

long Hierarchy::size() const {
    long size = 4 * sizeof(int) + io_sizes.size() * sizeof(Int3) + io_types.size() * sizeof(Byte) + i_indices.size() * sizeof(int) + o_indices.size() * sizeof(int);

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].size();

        if (l < encoders.size() - 1)
            size += routed_layers[l].size();
    }

    // predictors
    for (int p = 0; p < predictors.size(); p++)
        size += predictors[p].size();

    // actors
    for (int a = 0; a < actors.size(); a++)
        size += actors[a].size();

    // params
    size += encoders.size() * sizeof(Layer_Params);
    size += io_sizes.size() * sizeof(IO_Params);

    return size;
}

long Hierarchy::state_size() const {
    long size = 0;

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].state_size();
        
        if (l < encoders.size() - 1)
            size += routed_layers[l].size();
    }

    // predictors
    for (int p = 0; p < actors.size(); p++)
        size += predictors[p].state_size();

    // actors
    for (int a = 0; a < actors.size(); a++)
        size += actors[a].state_size();

    return size;
}

long Hierarchy::weights_size() const {
    long size = 0;

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].weights_size();
        
        if (l < encoders.size() - 1)
            size += routed_layers[l].size();
    }

    // predictors
    for (int p = 0; p < actors.size(); p++)
        size += predictors[p].weights_size();

    // actors
    for (int a = 0; a < actors.size(); a++)
        size += actors[a].weights_size();

    return size;
}


void Hierarchy::write(
    Stream_Writer &writer
) const {
    int num_layers = encoders.size();

    writer.write(reinterpret_cast<const void*>(&num_layers), sizeof(int));

    int num_io = io_sizes.size();

    writer.write(reinterpret_cast<const void*>(&num_io), sizeof(int));

    int num_predictions = predictors.size();
    int num_actions = actors.size();

    writer.write(reinterpret_cast<const void*>(&num_predictions), sizeof(int));
    writer.write(reinterpret_cast<const void*>(&num_actions), sizeof(int));

    writer.write(reinterpret_cast<const void*>(&io_sizes[0]), num_io * sizeof(Int3));
    writer.write(reinterpret_cast<const void*>(&io_types[0]), num_io * sizeof(Byte));

    writer.write(reinterpret_cast<const void*>(&i_indices[0]), i_indices.size() * sizeof(int));
    writer.write(reinterpret_cast<const void*>(&o_indices[0]), o_indices.size() * sizeof(int));

    for (int l = 0; l < num_layers; l++) {
        encoders[l].write(writer);

        if (l < encoders.size() - 1)
            routed_layers[l].write(writer);
    }

    // predictors
    for (int p = 0; p < predictors.size(); p++)
        predictors[p].write(writer);
    
    // actors
    for (int a = 0; a < actors.size(); a++)
        actors[a].write(writer);

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

    int num_predictions;
    int num_actions;

    reader.read(reinterpret_cast<void*>(&num_predictions), sizeof(int));
    reader.read(reinterpret_cast<void*>(&num_actions), sizeof(int));

    io_sizes.resize(num_io);
    io_types.resize(num_io);

    reader.read(reinterpret_cast<void*>(&io_sizes[0]), num_io * sizeof(Int3));
    reader.read(reinterpret_cast<void*>(&io_types[0]), num_io * sizeof(Byte));

    encoders.resize(num_layers);
    routed_layers.resize(num_layers - 1);

    hidden_cis_prev.resize(num_layers);

    i_indices.resize(num_io * 2);
    o_indices.resize(num_io);

    reader.read(reinterpret_cast<void*>(&i_indices[0]), i_indices.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&o_indices[0]), o_indices.size() * sizeof(int));
    
    for (int l = 0; l < num_layers; l++) {
        encoders[l].read(reader);
        
        if (l < encoders.size() - 1)
            routed_layers[l].read(reader);

        hidden_cis_prev[l] = encoders[l].get_hidden_cis();
    }

    predictors.resize(num_predictions);
    actors.resize(num_actions);

    // predictors
    for (int p = 0; p < predictors.size(); p++)
        predictors[p].read(reader);

    // actors
    for (int a = 0; a < actors.size(); a++)
        actors[a].read(reader);

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
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].write_state(writer);

        if (l < encoders.size() - 1)
            routed_layers[l].write_state(writer);
    }
    
    for (int p = 0; p < predictors.size(); p++)
        predictors[p].write_state(writer);

    for (int a = 0; a < actors.size(); a++)
        actors[a].write_state(writer);
}

void Hierarchy::read_state(
    Stream_Reader &reader
) {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].read_state(reader);
        
        if (l < encoders.size() - 1)
            routed_layers[l].read_state(reader);
    }

    for (int p = 0; p < predictors.size(); p++)
        predictors[p].read_state(reader);

    for (int a = 0; a < actors.size(); a++)
        actors[a].read_state(reader);
}

void Hierarchy::write_weights(
    Stream_Writer &writer
) const {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].write_weights(writer);

        if (l < encoders.size() - 1)
            routed_layers[l].write_weights(writer);
    }

    for (int p = 0; p < predictors.size(); p++)
        predictors[p].write_state(writer);

    for (int a = 0; a < actors.size(); a++)
        actors[a].write_state(writer);
}

void Hierarchy::read_weights(
    Stream_Reader &reader
) {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].read_weights(reader);
        
        if (l < encoders.size() - 1)
            routed_layers[l].read_weights(reader);
    }

    for (int p = 0; p < predictors.size(); p++)
        predictors[p].read_weights(reader);

    for (int a = 0; a < actors.size(); a++)
        actors[a].read_weights(reader);
}

void Hierarchy::merge(
    const Array<Hierarchy*> &hierarchies,
    Merge_Mode mode
) {
    Array<Encoder*> merge_encoders(hierarchies.size());
    Array<Routed_Layer*> merge_routed_layers(hierarchies.size());

    for (int l = 0; l < encoders.size(); l++) {
        for (int h = 0; h < hierarchies.size(); h++)
            merge_encoders[h] = &hierarchies[h]->encoders[l];

        encoders[l].merge(merge_encoders, mode);

        if (l < encoders.size() - 1) {
            for (int h = 0; h < hierarchies.size(); h++)
                merge_routed_layers[h] = &hierarchies[h]->routed_layers[l];

            routed_layers[l].merge(merge_routed_layers, mode);
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
