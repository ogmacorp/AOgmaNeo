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
        aVlds[i].recurrent = inputDescs[i].recurrent;
    }

    aVlds[inputDescs.size()].size = actorSize;
    aVlds[inputDescs.size()].radius = recurrentRadius;
    aVlds[inputDescs.size()].recurrent = true;

    actor.initRandom(actorSize, aVlds);

    predictors.resize(outputDescs.size());

    for (int i = 0; i < outputDescs.size(); i++) {
        Array<Predictor::VisibleLayerDesc> pVlds(1);
        pVlds[0].size = actorSize;
        pVlds[0].radius = outputDescs[i].radius;

        predictors[i].initRandom(outputDescs[i].size, pVlds);
    }

    actorHiddenCsPrev = IntBuffer(actorSize.x * actorSize.y, 0);
    actorHiddenErrors = FloatBuffer(actorHiddenCsPrev.size(), 0.0f);
}

void Sheet::step(
    const Array<const IntBuffer*> &inputCs,
    const Array<const IntBuffer*> &targetCs,
    int subSteps,
    bool learnEnabled
) {
    if (learnEnabled) {
        actorHiddenErrors.fill(0.0f);

        for (int i = 0; i < predictors.size(); i++) {
            predictors[i].generateErrors(targetCs[i], &actorHiddenErrors, 0);

            predictors[i].learn(targetCs[i]);
        }

        actor.learn(&actorHiddenErrors);
    }

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

    for (int i = 0; i < predictors.size(); i++)
        predictors[i].activate(predictorInputCs);
}

void Sheet::write(
    StreamWriter &writer
) const {
    int numPredictors = predictors.size();

    writer.write(reinterpret_cast<const void*>(&numPredictors), sizeof(int));

    actor.write(writer);
    
    for (int i = 0; i < predictors.size(); i++)
        predictors[i].write(writer);

    writer.write(reinterpret_cast<const void*>(&actorHiddenCsPrev[0]), actorHiddenCsPrev.size() * sizeof(int));
}

void Sheet::read(
    StreamReader &reader
) {
    int numPredictors;

    reader.read(reinterpret_cast<void*>(&numPredictors), sizeof(int));

    actor.read(reader);

    predictors.resize(numPredictors);

    for (int i = 0; i < predictors.size(); i++)
        predictors[i].read(reader);

    actorHiddenCsPrev.resize(actor.getHiddenCs().size());
    actorHiddenErrors = FloatBuffer(actor.getHiddenCs().size(), 0.0f);

    reader.read(reinterpret_cast<void*>(&actorHiddenCsPrev[0]), actorHiddenCsPrev.size() * sizeof(int));
}
