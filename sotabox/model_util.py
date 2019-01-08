print("[INFO] loading model_util")

def collect_trainable_weights(layer):
    """Collects all`trainable_weights` attributes,
    excluding any sublayers where `trainable` is set the `False`.
    """
    trainable = getattr(layer, 'trainable', True)
    if not trainable:
        return []
    weights = []
    if layer.__class__.__name__ == 'Model':
        for sublayer in layer.layers:
            weights += collect_trainable_weights(sublayer)
    elif layer.__class__.__name__ == 'Sequential':
        for sublayer in layer.flattened_layers:
            weights += collect_trainable_weights(sublayer)
    else:
        weights += layer.trainable_weights
    
    weights = list(set(weights))
    
    if weights:
        weights.sort(key=lambda x: x.name)
    return weights
    
    
def extract_weights(model):
    """Extract symbolic, trainable weights from a tf.keras Model."""
    trainable_weights = []
    for layer in model.layers:
        trainable_weights += collect_trainable_weights(layer)
    return trainable_weights