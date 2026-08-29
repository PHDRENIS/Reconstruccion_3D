import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_efficient_unet(input_shape=(480, 640, 5)):
    # Load pre-trained EfficientNetB0 as the encoder
    # We will extract features from specific layers for skip connections.
    base_model = applications.EfficientNetB0(
        include_top=False,  # Exclude the final classification layers
        weights='imagenet',  # Use ImageNet pre-trained weights
        input_shape=(input_shape[0], input_shape[1], 3)  # Note: We will adapt input channels later, for now use 3 channels
    )
    
    # Select layers for skip connections
    # We use the "_add" layers which represent the stable end of each stage.
    # This ensures that the dimensions match the Decoder's Upsampling.
    layer_names = [
        'block2b_add',       # Scale 1/4  (120 x 160) - End of Block 2
        'block3b_add',       # Scale 1/8  (60 x 80)   - End of Block 3
        'block5c_add',       # Scale 1/16 (30 x 40)   - End of Block 5
        'top_activation'     # Scale 1/32 (15 x 20)   - Bottleneck
    ]
    
    # Extract outputs
    base_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the feature extractor model
    feature_extractor = models.Model(inputs=base_model.input, outputs=base_outputs)
    feature_extractor.trainable = True

    # Adapt input to have 3 channels
    inputs = layers.Input(shape=input_shape) # (480, 640, 5)

    # Convert 5 channels -> 3 channels
    x_adapter = layers.Conv2D(3, (3, 3), padding='same', use_bias=False, name="adapter_conv")(inputs)

    # Connect to EfficientNet
    skips = feature_extractor(x_adapter)
    skip1, skip2, skip3, bottleneck = skips

    # Decoder

    #Images expected to ve 480x640, if not resize, so we will upsample back to that size


    # Block 1: From 1/32 (15x20) -> 1/16 (30x40)
    x = layers.UpSampling2D((2, 2))(bottleneck)
    x = layers.Concatenate()([x, skip3]) # skip3 is now (30x40), fits perfectly!
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Block 2: From 1/16 (30x40) -> 1/8 (60x80)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip2]) # skip2 is  (60x80)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Block 3: From 1/8 (60x80) -> 1/4 (120x160)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip1]) # skip1 is (120x160)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Block 4: From 1/4 (120x160) -> 1/2 (240x320)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Block 5: From 1/2 (240x320) -> Original (480x640)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # Final Output
    outputs = layers.Conv2D(1, (1, 1), activation='linear', name='depth_output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientDepth")
    return model