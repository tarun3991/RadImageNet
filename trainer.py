
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import numpy as np
import time

class ModelTrainer:
    def __init__(self, model, x_train, y_train, batch_size, epochs, output_dir):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir
        self.acc_per_fold = []
        self.loss_per_fold = []
        
        
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.tensorboard_dir = os.path.join(self.output_dir, 'tensorboard_logs')
        self.csv_logs_dir = os.path.join(self.output_dir, 'csv_logs')
        self.create_subdirectories()
        
    def create_subdirectories(self):
    	os.makedirs(self.models_dir, exist_ok=True)
    	os.makedirs(self.tensorboard_dir, exist_ok=True)
    	os.makedirs(self.csv_logs_dir, exist_ok=True)

    def get_callbacks(self, fold):
        mc_path = os.path.join(self.models_dir, f'model_{fold}.h5')
        model_checkpoint = ModelCheckpoint(filepath=mc_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
        csv_logger = CSVLogger(os.path.join(self.csv_logs_dir, f'training_log_{fold}.csv'))
        tensor_board = TensorBoard(log_dir=os.path.join(self.tensorboard_dir, f'tensorboard_logs_{fold}'))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

        return [model_checkpoint, early_stop, csv_logger, tensor_board, reduce_lr]

    def cross_validate(self, n_folds=5):
        acc_per_fold, loss_per_fold, val_acc_per_fold, val_loss_per_fold, time_per_fold = [], [], [], [], []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        total_time = 0

        for fold_no, (tr_idx, val_idx) in enumerate(kf.split(self.x_train, self.y_train), 1):
            print(f'Training fold {fold_no}...')
            
            start_time = time.time()
            
            x_tr, y_tr = self.x_train[tr_idx], self.y_train[tr_idx]
            x_val, y_val = self.x_train[val_idx], self.y_train[val_idx]

            aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
            history = self.model.fit(aug.flow(x_tr, y_tr, batch_size=self.batch_size), validation_data=(x_val, y_val),
                                     steps_per_epoch=len(x_tr) // self.batch_size, epochs=self.epochs,
                                     verbose=1, callbacks=self.get_callbacks(fold_no))

            #scores = self.model.evaluate(x_val, y_val, verbose=0)
            acc_per_fold.append(history.history['accuracy'][-1] * 100)
            loss_per_fold.append(history.history['loss'][-1])
            val_acc_per_fold.append(history.history['val_accuracy'][-1]*100)
            val_loss_per_fold.append(history.history['val_loss'][-1])
            
            
            end_time = time.time()
            fold_time = end_time - start_time
            total_time += fold_time
            time_per_fold.append(fold_time)
            
            print(f"Time taken for fold {fold_no}: {fold_time:.3f} seconds")
        print(f"Total training time for {n_folds} folds: {total_time:.3f} seconds")
            

        # Save cross-validation results
        self.save_cv_results(acc_per_fold, loss_per_fold, val_acc_per_fold, val_loss_per_fold, time_per_fold, total_time)
        return np.mean(acc_per_fold), np.mean(loss_per_fold), time_per_fold
		
		
    def save_cv_results(self, acc_per_fold, loss_per_fold, val_acc_per_fold, val_loss_per_fold, time_per_fold, total_time):
        results_df = pd.DataFrame({
            'Fold': [i + 1 for i in range(len(acc_per_fold))],
            'Training Accuracy': acc_per_fold,
            'Training Loss': loss_per_fold,
            'Validation Accuracy': val_acc_per_fold,
            'Validation Loss': val_loss_per_fold,
            'Mean_acc': np.mean(acc_per_fold),
            'Mean_loss': np.mean(loss_per_fold),
            'Time (seconds)': time_per_fold
        })
        results_df.to_csv(os.path.join(self.output_dir, 'cv_results.csv'), index=False)
        
        with open(os.path.join(self.output_dir, 'training_time.txt'), 'w') as f:
        	f.write(f'Total training time: {total_time:.3f} seconds\n')
        	
    def train_on_entire_dataset(self):
    	print("Training the model on the entire dataset...")
    	
    	callbacks = self.get_callbacks(fold='entire_dataset')
    	
    	
    	aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    	
    	start_time = time.time()

    	history = self.model.fit(aug.flow(self.x_train, self.y_train, batch_size=self.batch_size), steps_per_epoch=len(self.x_train) // self.batch_size, epochs=self.epochs, verbose=1, callbacks=callbacks)
    	
    	end_time = time.time()
    	total_time = end_time - start_time
    	
    	final_acc = history.history['accuracy'][-1] * 100
    	final_loss = history.history['loss'][-1]
    	
    	print(f"Time taken to train on the entire dataset: {total_time:.3f} seconds")
    	print(f"Final training accuracy on the entire dataset: {final_acc:.3f}%")
    	print(f"Final training loss on the entire dataset: {final_loss:.3f}")
    	
    	model_path = os.path.join(self.models_dir, 'final_model.h5')
    	self.model.save(model_path)
    	print(f"Final model saved at: {model_path}")
    	
    	self.save_entire_dataset_results(final_acc, final_loss, total_time)
    	
    def save_entire_dataset_results(self, final_acc, final_loss, total_time):
    	results_df = pd.DataFrame({"Final Accuracy (%)": [final_acc], "Final Loss": [final_loss], "Training Time (seconds)": [total_time]})
    	
    	results_df.to_csv(os.path.join(self.output_dir, 'entire_dataset_results.csv'), index=False)
    	with open(os.path.join(self.output_dir, 'entire_training_time.txt'), 'w') as f:
    		f.write(f"Final Accuracy: {final_acc:.3f}%\n")
    		f.write(f"Final Loss: {final_loss:.3f}\n")
    		f.write(f"Total training time: {total_time:.3f} seconds\n")
        
  
