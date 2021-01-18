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
    const Int3 &actorSize
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

    actorHiddenCIsPrev = IntBuffer(actorSize.x * actorSize.y, 0);
    actorHiddenErrors = FloatBuffer(actorHiddenCIsPrev.size(), 0.0f);
}

void Sheet::step(
    const Array<const IntBuffer*> &inputCIs,
    const Array<const IntBuffer*> &targetCIs,
    int subSteps,
    bool learnEnabled,
    bool clearState
) {
    if (learnEnabled) {
        actorHiddenErrors.fill(0.0f);

        for (int i = 0; i < predictors.size(); i++) {
            predictors[i].generateErrors(targetCIs[i], &actorHiddenErrors, 0);

            predictors[i].learn(targetCIs[i]);
        }

        actor.learn(&actorHiddenErrors);
    }

    Array<const IntBuffer*> actorInputCIs(inputCIs.size() + 1);

    for (int i = 0; i < inputCIs.size(); i++)
        actorInputCIs[i] = inputCIs[i];

    actorInputCIs[inputCIs.size()] = &actorHiddenCIsPrev;

    if (clearState) {
        actorHiddenCIsPrev.fill(0);
        actor.clearTraces();
    }

    // Sub steps
    for (int ss = 0; ss < subSteps; ss++) {
        actor.activate(actorInputCIs);

        actorHiddenCIsPrev = actor.getHiddenCIs();
    }
    
    Array<const IntBuffer*> predictorInputCIs(1);
    predictorInputCIs[0] = &actor.getHiddenCIs();

    for (int i = 0; i < predictors.size(); i++)
        predictors[i].activate(predictorInputCIs);
}

void Sheet::step(
    const Array<const IntBuffer*> &inputCIs,
    const Array<const IntBuffer*> &targetCIs,
    Array<IntBuffer> &intermediates,
    bool learnEnabled,
    bool clearState
) {
    if (learnEnabled) {
        actorHiddenErrors.fill(0.0f);

        for (int i = 0; i < predictors.size(); i++) {
            predictors[i].generateErrors(targetCIs[i], &actorHiddenErrors, 0);

            predictors[i].learn(targetCIs[i]);
        }

        actor.learn(&actorHiddenErrors);
    }

    Array<const IntBuffer*> actorInputCIs(inputCIs.size() + 1);

    for (int i = 0; i < inputCIs.size(); i++)
        actorInputCIs[i] = inputCIs[i];

    actorInputCIs[inputCIs.size()] = &actorHiddenCIsPrev;

    if (clearState) {
        actorHiddenCIsPrev.fill(0);
        actor.clearTraces();
    }

    // Sub steps
    for (int ss = 0; ss < intermediates.size(); ss++) {
        actor.activate(actorInputCIs);

        actorHiddenCIsPrev = actor.getHiddenCIs();

        intermediates[ss] = actor.getHiddenCIs();
    }
    
    Array<const IntBuffer*> predictorInputCIs(1);
    predictorInputCIs[0] = &actor.getHiddenCIs();

    for (int i = 0; i < predictors.size(); i++)
        predictors[i].activate(predictorInputCIs);
}

void Sheet::write(
    StreamWriter &writer
) const {
    int numPredictors = predictors.size();

    writer.write(reinterpret_cast<const void*>(&numPredictors), sizeof(int));

    actor.write(writer);
    
    for (int i = 0; i < predictors.size(); i++)
        predictors[i].write(writer);

    writer.write(reinterpret_cast<const void*>(&actorHiddenCIsPrev[0]), actorHiddenCIsPrev.size() * sizeof(int));
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

    actorHiddenCIsPrev.resize(actor.getHiddenCIs().size());
    actorHiddenErrors = FloatBuffer(actor.getHiddenCIs().size(), 0.0f);

    reader.read(reinterpret_cast<void*>(&actorHiddenCIsPrev[0]), actorHiddenCIsPrev.size() * sizeof(int));
}
