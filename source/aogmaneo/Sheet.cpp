// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Sheet.h"

using namespace aon;

void Sheet::initRandom(
    const Array<InputDesc> &inputDescs,
    int recurrentRadius,
    const Array<OutputDesc> &outputDescs,
    Int3 actorSize
) {
    Array<Actor::VisibleLayerDesc> aVlds(inputDescs.size() + 1);

    for (int i = 0; i < inputDescs.size(); i++) {
        aVlds[i].size = inputDescs[i].size;
        aVlds[i].radius = inputDescs[i].radius;
    }

    aVlds[inputDescs.size()].size = actorSize;
    aVlds[inputDescs.size()].radius = recurrentRadius;

    actor.initRandom(actorSize, aVlds);

    predictors.resize(outputDescs.size());

    for (int i = 0; i < outputDescs.size(); i++) {
        Array<Predictor::VisibleLayerDesc> pVlds(1);
        pVlds[0].size = actorSize;
        pVlds[0].radius = outputDescs[i].radius;

        predictors[i].initRandom(outputDescs[i].size, pVlds);
    }

    actorHiddenCsPrev = IntBuffer(actorSize.x * actorSize.y, 0);
    actorHiddenRewards = FloatBuffer(actorHiddenCsPrev.size(), 0.0f);
}

void Sheet::step(
    const Array<const IntBuffer*> &inputCs,
    const Array<const IntBuffer*> &targetCs,
    int subSteps,
    bool learnEnabled
) {
    actor.backup();

    Array<const IntBuffer*> actorInputCs(inputCs.size() + 1);

    for (int i = 0; i < inputCs.size(); i++)
        actorInputCs[i] = inputCs[i];

    actorInputCs[inputCs.size()] = &actorHiddenCsPrev;

    // Sub steps
    for (int ss = 0; ss < subSteps; ss++) {
        actor.activate(actorInputCs);

        actorHiddenCsPrev = actor.getHiddenCs();
    }

    Array<const IntBuffer*> predictorInputCs(1);
    predictorInputCs[0] = &actor.getHiddenCs();

    actorHiddenRewards.fill(0.0f);

    for (int i = 0; i < predictors.size(); i++) {
        if (learnEnabled) {
            predictors[i].generateRewards(targetCs[i], &actorHiddenRewards, 0);

            predictors[i].learn(targetCs[i]);
        }

        predictors[i].activate(predictorInputCs);
    }

    if (learnEnabled)
        actor.learn(&actorHiddenRewards);
}

void Sheet::write(
    StreamWriter &writer
) const {
    int numPredictors;

    writer.write(reinterpret_cast<const void*>(&numPredictors), sizeof(int));
    actor.write(writer);
    
    
}

void Sheet::read(
    StreamReader &reader
) {
    int numLayers;

    reader.read(reinterpret_cast<void*>(&numLayers), sizeof(int));

    int numInputs;

    reader.read(reinterpret_cast<void*>(&numInputs), sizeof(int));

    inputSizes.resize(numInputs);

    reader.read(reinterpret_cast<void*>(&inputSizes[0]), numInputs * sizeof(Int3));

    scLayers.resize(numLayers);
    pLayers.resize(numLayers);

    histories.resize(numLayers);
    
    updates.resize(numLayers);
    ticks.resize(numLayers);
    ticksPerUpdate.resize(numLayers);

    reader.read(reinterpret_cast<void*>(&updates[0]), updates.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&ticks[0]), ticks.size() * sizeof(int));
    reader.read(reinterpret_cast<void*>(&ticksPerUpdate[0]), ticksPerUpdate.size() * sizeof(int));
    
    for (int l = 0; l < numLayers; l++) {
        int numInputs;
        
        reader.read(reinterpret_cast<void*>(&numInputs), sizeof(int));

        histories[l].resize(numInputs);

        for (int i = 0; i < histories[l].size(); i++) {
            int historySize;

            reader.read(reinterpret_cast<void*>(&historySize), sizeof(int));

            int historyStart;
            
            reader.read(reinterpret_cast<void*>(&historyStart), sizeof(int));

            histories[l][i].resize(historySize);
            histories[l][i].start = historyStart;

            for (int t = 0; t < histories[l][i].size(); t++) {
                int bufferSize;

                reader.read(reinterpret_cast<void*>(&bufferSize), sizeof(int));

                histories[l][i][t].resize(bufferSize);

                reader.read(reinterpret_cast<void*>(&histories[l][i][t][0]), histories[l][i][t].size() * sizeof(int));
            }
        }

        scLayers[l].read(reader);
        
        pLayers[l].resize(l == 0 ? inputSizes.size() : ticksPerUpdate[l]);

        // Predictors
        for (int v = 0; v < pLayers[l].size(); v++) {
            char exists;

            reader.read(reinterpret_cast<void*>(&exists), sizeof(int));

            if (exists) {
                pLayers[l][v].make();
                pLayers[l][v]->read(reader);
            }
        }
    }

    // Actors
    aLayers.resize(inputSizes.size());

    for (int v = 0; v < aLayers.size(); v++) {
        char exists;

        reader.read(reinterpret_cast<void*>(&exists), sizeof(int));

        if (exists) {
            aLayers[v].make();
            aLayers[v]->read(reader);
        }
    }
}
