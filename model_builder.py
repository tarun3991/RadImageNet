from keras.applications import VGG16, VGG19, ResNet50, DenseNet121
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

class ModelBuilder:
    def __init__(self, base_model_name, input_shape, num_classes, lr, epochs):
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
       
    def build_model(self, num_classes):
        if self.base_model_name == "VGG16":
            base_model = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif self.base_model_name == "VGG19":
            base_model = VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif self.base_model_name == "ResNet50":
            base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif self.base_model_name == "DenseNet121":
            base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=self.input_shape)
        else:
        	raise ValueError(f"Unsupported base model: {self.base_model_name}")
        # Add more models as needed

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=output)

        for layer in base_model.layers:
            layer.trainable = False

        opt = Adam(learning_rate=self.lr, decay=self.lr/self.epochs)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

