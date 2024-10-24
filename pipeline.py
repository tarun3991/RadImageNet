from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import ModelTrainer
from evaluator import ModelEvaluator
import os
from keras.models import load_model

class Pipeline:
    def __init__(self, model_name, train_dir, test_dir, output_dir, lr, batch_size, epochs):
        self.model_name = model_name
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        # Load and preprocess training data
        print("Loading and preprocessing training data...")
        data_load_train = DataLoader(self.train_dir)
        x_train, y_train, class_names = data_load_train.load_train_data()
        print(f'Training data shape: {x_train.shape}, {y_train.shape}')
        
        # Build model
        print("Building Model ....")
        num_classes = y_train.shape[1]
        model_builder = ModelBuilder(self.model_name, (224, 224, 3), num_classes, self.lr, self.epochs)
        model = model_builder.build_model(num_classes)
        
        # Train and cross validate model
        print("Training and cross-validating model...")
        model_trainer = ModelTrainer(model, x_train, y_train, self.batch_size, self.epochs, self.output_dir)
        #model_trainer.train_and_evaluate()
        mean_accuracy, mean_loss, time_per_fold = model_trainer.cross_validate()
        model_trainer.train_on_entire_dataset()
        print(f"Cross-validation results - Mean Accuracy: {mean_accuracy:.3}%, Mean Loss: {mean_loss:.3f}")
        
        # Load and preprocess test data (unlabeled)
        print("Loading and preprocessing test data....")
        data_load_test = DataLoader(self.test_dir)
        x_test, image_paths = data_load_test.load_test_data()

        # Evaluate model on test data
        print("Evaluating model on unseen test data...")
        model_evaluator = ModelEvaluator(self.output_dir, x_test, image_paths)
        model_evaluator.evaluate(class_names)

