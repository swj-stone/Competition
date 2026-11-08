"""Fixed random-threshold classifier ensemble, prior to any neural blend."""
import numpy as np

OFFSETS=(0,1009,2027)


def predict(factory, params, train, target, valid, test):
    validation=[];prediction=[]
    for offset in OFFSETS:
        member={**params,'random_state':params['random_state']+offset}
        model=factory(**member).fit(train,target)
        validation.append(model.predict_proba(valid))
        prediction.append(model.predict_proba(test))
    return np.mean(validation,axis=0),np.mean(prediction,axis=0)
