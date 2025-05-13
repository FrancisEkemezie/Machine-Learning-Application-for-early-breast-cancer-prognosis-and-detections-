import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Multiply, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import cv2
import warnings
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import os
from tensorflow.keras.applications import MobileNetV3Small

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BreastCancerAISystem:
    """
    Integrated AI system for breast cancer diagnostics with two main components:
    1. FNA Nuclear Feature Analysis - Using Wisconsin Breast Cancer Dataset
    2. IDC Detection in Histopathological Images
    
    Both components include interpretability mechanisms and visualization tools.
    """
    
    def __init__(self):
        self.fna_model = None
        self.idc_model = None
        self.fna_scaler = StandardScaler()
        self.history_fna = None
        self.history_idc = None
        self.feature_names = None
    
    #################################
    # FNA Nuclear Feature Analysis  #
    #################################
    
    def load_wisconsin_data(self, path='../data/wdbc.csv'):
        """
        Load Wisconsin Breast Cancer Dataset. If actual file not available,
        use sklearn's built-in dataset for demonstration.
        """
        try:
            # Try to load actual dataset if available
            df = pd.read_csv(path)
            print(f"Loaded Wisconsin dataset from {path}")
        except:
            # Fall back to sklearn's dataset
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['diagnosis'] = data.target
            df['diagnosis'] = df['diagnosis'].map({0: 'M', 1: 'B'})  # Map 0->M, 1->B to match Wisconsin dataset format
            print("Using sklearn's built-in Wisconsin breast cancer dataset")
        
        # Store feature names for interpretability
        if 'diagnosis' in df.columns:
            self.feature_names = [col for col in df.columns if col != 'diagnosis']
        else:
            # Handle case where dataset might have different structure
            self.feature_names = df.columns[1:]  # Assuming first column is ID or diagnosis
        
        return df
    
    def preprocess_fna_data(self, df):
        """
        Preprocess Wisconsin dataset for training.
        """
        # Handle different dataset formats
        if 'diagnosis' in df.columns:
            X = df.drop('diagnosis', axis=1).values
            y = df['diagnosis'].map({'M': 1, 'B': 0}).values
        else:
            # Alternative format where first column might be ID and second is diagnosis
            X = df.iloc[:, 2:].values  # Skip ID and diagnosis
            y = (df.iloc[:, 1] == 'M').astype(int).values
        
        # Standardize features
        X_scaled = self.fna_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_fna_model(self, input_shape):
        """
        Build interpretable deep learning model for FNA analysis with attention mechanism.
        """
        inputs = Input(shape=input_shape)
        
        # First dense block with attention
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention mechanism for feature importance
        attention = Dense(128, activation='tanh')(x)
        attention = Dense(128, activation='sigmoid')(attention)
        x = Multiply()([x, attention])
        
        # Second dense block
        x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.fna_model = model
        return model
    
    def train_fna_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train the FNA model with early stopping and learning rate reduction.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('fna_model_best.h5', monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        self.history_fna = self.fna_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history_fna
    
    def visualize_fna_training(self):
        """
        Visualize training history of FNA model.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy plot
        ax1.plot(self.history_fna.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history_fna.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy - FNA Analysis')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Loss plot
        ax2.plot(self.history_fna.history['loss'], label='Train Loss')
        ax2.plot(self.history_fna.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss - FNA Analysis')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('fna_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self, X, feature_names=None):
        """
        Extract feature importance using attention weights.
        """
        # Get the attention layer (layer 3 in our architecture)
        attention_layer = self.fna_model.layers[3]
        
        # Create a model that outputs attention weights
        attention_model = Model(inputs=self.fna_model.input, 
                               outputs=attention_layer.output)
        
        # Get attention weights for all samples
        attention_weights = attention_model.predict(X)
        
        # Average the attention weights across all samples
        mean_attention = np.mean(attention_weights, axis=0)
        
        # Normalize to sum to 1 for easier interpretation
        importance_scores = mean_attention / np.sum(mean_attention)
        
        # If no feature names provided, use the stored ones or generate
        if feature_names is None:
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f'Feature {i}' for i in range(len(importance_scores))]
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importance_scores)],
            'Importance': importance_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def visualize_feature_importance(self, importance_df, top_n=10):
        """
        Create a visualization of feature importance.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar chart
        bar_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_features)))
        bars = plt.barh(top_features['Feature'], top_features['Importance'], color=bar_colors)
        
        # Add feature importance values
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{top_features["Importance"].iloc[i]:.4f}',
                    va='center', ha='left', fontsize=10)
        
        plt.title('Top Nuclear Features Contributing to Malignancy Prediction', fontsize=14)
        plt.xlabel('Relative Importance Score', fontsize=12)
        plt.ylabel('Nuclear Features', fontsize=12)
        plt.xlim(0, top_features['Importance'].max() * 1.15)  # Add some space for the text
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Add gradient colorbar to indicate importance level
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Importance Level', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('fna_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_fna_decision_boundary(self, X, y, feature1=0, feature2=1):
        """
        Visualize decision boundary using PCA for dimensionality reduction.
        """
        # Apply PCA to reduce dimensions to 2
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create a mesh grid
        h = 0.02  # Step size
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Create a temporary model for 2D input
        input_shape = (2,)
        temp_model = self.build_fna_model(input_shape)
        
        # Train on PCA-transformed data
        temp_model.fit(X_pca, y, epochs=20, batch_size=32, verbose=0)
        
        # Predict on mesh grid
        Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(12, 10))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu_r)
        
        # Plot training points
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                           edgecolors='k', s=80, cmap=plt.cm.RdBu_r)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        
        # Add informative labels
        plt.title('Decision Boundary on PCA-Reduced Features', fontsize=14)
        plt.xlabel(f'Principal Component 1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})', fontsize=12)
        plt.ylabel(f'Principal Component 2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2f})', fontsize=12)
        
        # Add a legend
        legend = plt.legend(*scatter.legend_elements(), title="Diagnosis", loc="upper right")
        legend.get_title().set_fontsize('12')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Diagnosis (0: Benign, 1: Malignant)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('fna_decision_boundary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_fna_model(self, X_test, y_test):
        """
        Evaluate FNA model and visualize results.
        """
        # Get predictions
        y_pred_prob = self.fna_model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xticklabels(['Benign', 'Malignant'])
        ax1.set_yticklabels(['Benign', 'Malignant'])
        
        # Plot ROC curve
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('Receiver Operating Characteristic', fontsize=14)
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('fna_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        df_report = pd.DataFrame(class_report).transpose()
        print(df_report.round(3))
        
        return conf_matrix, roc_auc, class_report

    def create_nuclear_feature_gradcam(self, X_sample, y_true):
        """
        Create a Grad-CAM-like visualization for nuclear features.
        This is an adaptation of Grad-CAM for tabular data.
        """
        # Reshape for batch input
        if len(X_sample.shape) == 1:
            X_sample = np.expand_dims(X_sample, axis=0)
        
        # Build a temporary model that outputs both prediction and last layer before final dense
        last_layer = self.fna_model.layers[-2].output
        output_layer = self.fna_model.layers[-1].output
        grad_model = Model(inputs=self.fna_model.input, outputs=[last_layer, output_layer])
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Cast input to float32
            inputs = tf.cast(X_sample, tf.float32)
            last_layer_output, preds = grad_model(inputs)
            class_idx = tf.argmax(preds[0])
            class_channel = preds[:, class_idx]
        
        # Gradient of output wrt the last layer output
        grads = tape.gradient(class_channel, last_layer_output)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=0)
        
        # Weight activation by importance
        last_layer_output = last_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        # Apply weighted activation to get feature importance
        for i in range(last_layer_output.shape[-1]):
            last_layer_output[i] *= pooled_grads[i]
        
        # Calculate average importance per feature
        feature_importance = np.mean(last_layer_output, axis=-1)
        
        # Get the original feature names
        if self.feature_names is None or len(self.feature_names) != X_sample.shape[1]:
            feature_names = [f'Feature {i}' for i in range(X_sample.shape[1])]
        else:
            feature_names = self.feature_names[:X_sample.shape[1]]
        
        # Normalize to 0-1 for better visualization
        feature_importance = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min() + 1e-10)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Sort by importance
        importance_idx = np.argsort(feature_importance)[::-1]
        sorted_features = [feature_names[i] for i in importance_idx]
        sorted_importance = feature_importance[importance_idx]
        
        # Plot
        bars = plt.barh(sorted_features, sorted_importance, color=plt.cm.viridis(sorted_importance))
        
        plt.title(f'GradCAM Visualization for Sample (True Class: {"Malignant" if y_true else "Benign"})', fontsize=14)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xlim(0, 1.1)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Add values
        for i, v in enumerate(sorted_importance):
            plt.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('nuclear_feature_gradcam.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return dict(zip(sorted_features, sorted_importance))
        
    def visualize_fna_feature_patterns(self, X, y):
        """
        Visualize feature patterns across malignant and benign samples
        """
        # Get feature names
        if self.feature_names is None or len(self.feature_names) != X.shape[1]:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names[:X.shape[1]]
        
        # Create a dataframe for easier manipulation
        df = pd.DataFrame(X, columns=feature_names)
        df['diagnosis'] = y
        
        # Select top features using a quick Random Forest
        if X.shape[1] > 10:  # Only if we have many features
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features = importance.head(8)['Feature'].tolist()
        else:
            top_features = feature_names
        
        # Create plots
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features[:8]):  # Plot top 8 features
            ax = axes[i]
            
            # Get values for benign and malignant
            benign_values = df[df['diagnosis'] == 0][feature]
            malign_values = df[df['diagnosis'] == 1][feature]
            
            # Create violin plots
            parts = ax.violinplot([benign_values, malign_values], showmeans=True, showmedians=True)
            
            # Style violins
            for pc in parts['bodies']:
                pc.set_facecolor('#D76A03')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            
            # Style mean and median lines
            parts['cmeans'].set_color('black')
            parts['cmedians'].set_color('blue')
            
            ax.set_title(feature, fontsize=10)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Benign', 'Malignant'])
            ax.set_ylabel('Value (Standardized)')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add a statistical test (t-test)
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(benign_values, malign_values)
            ax.text(0.5, 0.95, f'p-value: {p_val:.3f}', 
                    transform=ax.transAxes, ha='center', va='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('fna_feature_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    #################################
    # IDC Detection in Histopathology#
    #################################
    
    def build_idc_model(self, input_shape=(224, 224, 3)):
        """
        Build a lightweight CNN model for IDC detection with attention mechanism.
        Uses MobileNetV3Small as base model for resource efficiency.
        """
        # Base model - lightweight for resource constrained settings
        base_model = MobileNetV3Small(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        
        # Freeze early layers to prevent overfitting
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Create model
        inputs = Input(shape=input_shape)
        x = base_model(inputs)
        
        # Add attention mechanism for better interpretability
        attention = Conv2D(filters=128, kernel_size=1, activation='tanh')(x)
        attention = Conv2D(filters=128, kernel_size=1, activation='sigmoid')(attention)
        x = Multiply()([x, attention])
        
        # Global average pooling to reduce parameters
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.idc_model = model
        return model
    
    def train_idc_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """
        Train the IDC detection model with callbacks for better training.
        
        Note: For illustration purposes. In a real scenario, we would use:
        - Data generators for loading images
        - Image augmentation for robustness
        """
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('idc_model_best.h5', monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        self.history_idc = self.idc_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history_idc
    
    def simulate_idc_training(self, num_samples=100, image_size=(224, 224, 3)):
        """
        Simulate training when no actual histopathology images are available.
        Generates synthetic data for demonstration purposes.
        """
        print("Simulating IDC detection model training with synthetic data...")
        
        # Generate synthetic data
        X_train = np.random.rand(num_samples, *image_size)
        y_train = np.random.randint(0, 2, size=num_samples)
        
        X_val = np.random.rand(int(num_samples * 0.2), *image_size)
        y_val = np.random.randint(0, 2, size=int(num_samples * 0.2))
        
        # Build model
        self.build_idc_model(image_size)
        
        # Simulate training for a small number of epochs
        history = self.idc_model.fit(
            X_train, y_train,
            epochs=3,  # Just a few epochs for simulation
            batch_size=8,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        self.history_idc = history
        
        print("Simulation completed. In a real scenario with actual histopathology images:")
        print("1. Would train on a diverse dataset including various tissue preparation techniques")
        print("2. Would implement robustness testing under various staining conditions")
        print("3. Would validate across multiple magnification levels (40x to 400x)")
        
        return history
    
    def create_grad_cam(self, img, pred_index=None):
        """
        Generate Grad-CAM visualization to highlight regions the model focuses on.
        """
        # Ensure img is properly formatted
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        
        # Get the final convolutional layer
        final_conv_layer = None
        for layer in reversed(self.idc_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                final_conv_layer = layer
                break
        
        if final_conv_layer is None:
            print("Could not find convolutional layer")
            return None
        
        # Create grad model
        grad_model = Model(
            inputs=[self.idc_model.inputs],
            outputs=[final_conv_layer.output, self.idc_model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            
        # Extract gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Compute heatmap
        conv_outputs = conv_outputs[0]
        heatmap = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
        
        # Weight channels by corresponding gradients
        for i, w in enumerate(pooled_grads):
            heatmap += w.numpy() * conv_outputs[:, :, i].numpy()
        
        # Relu on heatmap
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
        
        # Normalize 
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        return heatmap
    
    def apply_heatmap(self, img, heatmap):
        """
        Apply heatmap to image for visualization.
        """
        # Ensure img is in proper range
        if img.max() > 1.0:
            img = img / 255.0
        
        # Resize heatmap if needed
        if heatmap.shape[:2] != img.shape[:2]:
            heatmap = cv2.cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    def apply_heatmap(self, img, heatmap):
        """
        Apply heatmap to image for visualization.
        """
        # Ensure img is in proper range
        if img.max() > 1.0:
            img = img / 255.0
        
        # Resize heatmap if needed
        if heatmap.shape[:2] != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = heatmap_colored / 255.0
        
        # Superimpose heatmap on image
        superimposed = 0.6 * img + 0.4 * heatmap_colored
        
        # Ensure superimposed is in proper range
        superimposed = np.clip(superimposed, 0, 1)
        
        return superimposed

    def visualize_gradcam(self, img, true_label, img_title=""):
        """
        Create GradCAM visualization for a histopathology image.
        """
        # Make prediction
        if len(img.shape) == 3:
            img_batch = np.expand_dims(img, axis=0)
        else:
            img_batch = img
            
        prediction = self.idc_model.predict(img_batch)[0][0]
        pred_label = 1 if prediction > 0.5 else 0
        
        # Generate heatmap
        heatmap = self.create_grad_cam(img_batch)
        
        # Apply heatmap to image
        superimposed = self.apply_heatmap(img, heatmap)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        ax1.imshow(img)
        ax1.set_title(f"Original Image\n{img_title}", fontsize=12)
        ax1.axis('off')
        
        # Heatmap only
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title("GradCAM Heatmap", fontsize=12)
        ax2.axis('off')
        
        # Superimposed
        ax3.imshow(superimposed)
        ax3.set_title(f"Prediction: {'Malignant' if pred_label else 'Benign'} ({prediction:.3f})\n"
                     f"True Label: {'Malignant' if true_label else 'Benign'}", fontsize=12)
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig('idc_gradcam_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return heatmap, superimposed
    
    def evaluate_idc_model(self, X_test, y_test):
        """
        Evaluate IDC model performance with visualizations.
        """
        # Get predictions
        y_pred_prob = self.idc_model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xticklabels(['Benign', 'Malignant'])
        ax1.set_yticklabels(['Benign', 'Malignant'])
        
        # Plot ROC curve
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('Receiver Operating Characteristic', fontsize=14)
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('idc_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        df_report = pd.DataFrame(class_report).transpose()
        print(df_report.round(3))
        
        return conf_matrix, roc_auc, class_report
    
    def visualize_idc_training(self):
        """
        Visualize training history of IDC model.
        """
        if self.history_idc is None:
            print("No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy plot
        ax1.plot(self.history_idc.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history_idc.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy - IDC Detection', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Loss plot
        ax2.plot(self.history_idc.history['loss'], label='Train Loss')
        ax2.plot(self.history_idc.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss - IDC Detection', fontsize=14)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('idc_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_misclassified_samples(self, X_test, y_test, n_samples=5):
        """
        Visualize misclassified histopathology images with GradCAM explanations.
        """
        # Make predictions
        y_pred_prob = self.idc_model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Find misclassified indices
        misclassified_idx = np.where(y_pred.flatten() != y_test)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassified samples found.")
            return
        
        # Select samples to visualize
        n_samples = min(n_samples, len(misclassified_idx))
        selected_idx = np.random.choice(misclassified_idx, n_samples, replace=False)
        
        for i, idx in enumerate(selected_idx):
            img = X_test[idx]
            true_label = y_test[idx]
            pred_label = y_pred[idx][0]
            confidence = y_pred_prob[idx][0]
            
            # Generate GradCAM
            heatmap = self.create_grad_cam(np.expand_dims(img, axis=0))
            superimposed = self.apply_heatmap(img, heatmap)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Original image
            ax1.imshow(img)
            ax1.set_title(f"True: {'Malignant' if true_label else 'Benign'}", fontsize=12)
            ax1.axis('off')
            
            # Superimposed
            ax2.imshow(superimposed)
            ax2.set_title(f"Predicted: {'Malignant' if pred_label else 'Benign'} ({confidence:.3f})", fontsize=12)
            ax2.axis('off')
            
            plt.suptitle(f"Misclassified Sample {i+1}/{n_samples}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f'idc_misclassified_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    #################################
    # Integrated System Functions   #
    #################################
    
    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Load and preprocess a histopathology image for prediction.
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, target_size)
            
            # Normalize
            img = img / 255.0
            
            return img
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def predict_histopathology(self, image, visualize=True):
        """
        Make prediction on histopathology image with visualization.
        """
        if self.idc_model is None:
            print("IDC model not initialized. Please train or load the model first.")
            return None
        
        # Preprocess image
        if isinstance(image, str):
            # If image is a file path
            img = self.load_and_preprocess_image(image)
        else:
            # If image is already loaded
            img = image
            
        if img is None:
            return None
            
        # Make prediction
        img_batch = np.expand_dims(img, axis=0)
        prediction = self.idc_model.predict(img_batch)[0][0]
        pred_class = "Malignant" if prediction > 0.5 else "Benign"
        
        # Visualize with GradCAM if requested
        if visualize:
            heatmap = self.create_grad_cam(img_batch)
            superimposed = self.apply_heatmap(img, heatmap)
            
            plt.figure(figsize=(14, 7))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image", fontsize=12)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(superimposed)
            plt.title(f"Prediction: {pred_class} ({prediction:.3f})", fontsize=12)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('histopathology_prediction.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            'prediction': pred_class,
            'confidence': float(prediction),
            'probability_malignant': float(prediction)
        }
    
    def predict_fna_features(self, features, visualize=True):
        """
        Make prediction on FNA nuclear features with interpretation.
        
        Parameters:
        -----------
        features : array-like
            Array of FNA nuclear features. Must match the expected input format.
        visualize : bool
            Whether to visualize the prediction with GradCAM.
            
        Returns:
        --------
        dict
            Prediction results and interpretation.
        """
        if self.fna_model is None:
            print("FNA model not initialized. Please train or load the model first.")
            return None
        
        # Convert to numpy array
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.fna_scaler.transform(features)
        
        # Make prediction
        prediction = self.fna_model.predict(features_scaled)[0][0]
        pred_class = "Malignant" if prediction > 0.5 else "Benign"
        
        # Get feature importance using GradCAM
        if visualize:
            feature_importance = self.create_nuclear_feature_gradcam(features_scaled, prediction > 0.5)
            
            # Identify top contributing features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Create interpretation
            interpretation = {
                'top_contributors': [{'feature': f, 'importance': i} for f, i in top_features],
                'interpretation': f"The diagnosis is {pred_class} with {prediction:.1%} confidence. " +
                                 f"The top contributing features are {', '.join([f[0] for f in top_features])}."
            }
        else:
            interpretation = {
                'interpretation': f"The diagnosis is {pred_class} with {prediction:.1%} confidence."
            }
        
        return {
            'prediction': pred_class,
            'confidence': float(prediction),
            'probability_malignant': float(prediction),
            **interpretation
        }
    
    def save_models(self, fna_path='fna_model.h5', idc_path='idc_model.h5'):
        """
        Save trained models to disk.
        """
        if self.fna_model is not None:
            self.fna_model.save(fna_path)
            print(f"FNA model saved to {fna_path}")
            
            # Save scaler for later use
            import joblib
            joblib.dump(self.fna_scaler, 'fna_scaler.pkl')
            print(f"FNA scaler saved to fna_scaler.pkl")
        
        if self.idc_model is not None:
            self.idc_model.save(idc_path)
            print(f"IDC model saved to {idc_path}")
    
    def load_models(self, fna_path='fna_model.h5', idc_path='idc_model.h5'):
        """
        Load trained models from disk.
        """
        try:
            self.fna_model = tf.keras.models.load_model(fna_path)
            print(f"FNA model loaded from {fna_path}")
            
            # Load scaler
            import joblib
            self.fna_scaler = joblib.load('fna_scaler.pkl')
            print(f"FNA scaler loaded from fna_scaler.pkl")
        except Exception as e:
            print(f"Error loading FNA model: {e}")
        
        try:
            self.idc_model = tf.keras.models.load_model(idc_path)
            print(f"IDC model loaded from {idc_path}")
        except Exception as e:
            print(f"Error loading IDC model: {e}")
    
    def combine_predictions(self, fna_result, idc_result, weights=(0.5, 0.5)):
        """
        Combine predictions from FNA and IDC models for a consensus diagnosis.
        Uses weighted average of malignancy probabilities.
        
        Parameters:
        -----------
        fna_result : dict
            Result from FNA prediction.
        idc_result : dict
            Result from IDC prediction.
        weights : tuple
            Weights for FNA and IDC results respectively.
            
        Returns:
        --------
        dict
            Combined prediction results.
        """
        if fna_result is None and idc_result is None:
            return None
        
        # Initialize probabilities
        fna_prob = 0
        idc_prob = 0
        
        # Get probabilities if available
        if fna_result is not None:
            fna_prob = fna_result['probability_malignant']
        
        if idc_result is not None:
            idc_prob = idc_result['probability_malignant']
        
        # Adjust weights if one result is missing
        if fna_result is None:
            weights = (0, 1)
        elif idc_result is None:
            weights = (1, 0)
            
        # Normalize weights
        w_sum = sum(weights)
        weights = (weights[0]/w_sum, weights[1]/w_sum)
        
        # Calculate weighted probability
        combined_prob = weights[0] * fna_prob + weights[1] * idc_prob
        combined_class = "Malignant" if combined_prob > 0.5 else "Benign"
        
        # Create visualization of combined prediction
        plt.figure(figsize=(8, 6))
        
        # Create bar chart
        bars = plt.bar(['FNA', 'IDC', 'Combined'], 
                    [fna_prob, idc_prob, combined_prob],
                    color=['#3498db', '#2ecc71', '#e74c3c'])
        
        # Add probability values
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        plt.ylabel('Probability of Malignancy')
        plt.title('Combined Breast Cancer Diagnosis')
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('combined_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'prediction': combined_class,
            'probability_malignant': float(combined_prob),
            'confidence': float(combined_prob) if combined_prob >= 0.5 else float(1 - combined_prob),
            'fna_contribution': {
                'weight': weights[0],
                'probability': float(fna_prob)
            },
            'idc_contribution': {
                'weight': weights[1],
                'probability': float(idc_prob)
            },
            'interpretation': f"The combined diagnosis is {combined_class} with " +
                            f"{combined_prob:.1%} confidence, based on " +
                            f"FNA (weight: {weights[0]:.2f}) and IDC (weight: {weights[1]:.2f}) analyses."
        }
    
    def visualize_data_manifold(self, X, y, method='tsne'):
        """
        Visualize high-dimensional data in 2D using t-SNE or PCA.
        Useful for exploring distribution of malignant and benign cases.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Labels (0=benign, 1=malignant).
        method : str
            Dimensionality reduction method ('tsne' or 'pca').
        """
        plt.figure(figsize=(10, 8))
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X)
            title = "t-SNE Visualization of Breast Cancer Data"
        else:  # PCA
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(X)
            explained_var = reducer.explained_variance_ratio_
            title = f"PCA Visualization of Breast Cancer Data\n" + \
                   f"(Explained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
        
        # Create scatter plot
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                           c=y, cmap='coolwarm', 
                           s=50, alpha=0.8, edgecolors='k')
        
        # Add legend and labels
        plt.colorbar(scatter, label='Diagnosis (0=Benign, 1=Malignant)')
        plt.title(title, fontsize=14)
        plt.xlabel("Dimension 1", fontsize=12)
        plt.ylabel("Dimension 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Add some statistics
        benign_count = np.sum(y == 0)
        malignant_count = np.sum(y == 1)
        plt.figtext(0.02, 0.02, f"Benign: {benign_count}, Malignant: {malignant_count}", 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{method}_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_demo(self):
        """
        Run a complete demonstration of the system using Wisconsin dataset.
        """
        print("=" * 80)
        print("BREAST CANCER AI DIAGNOSTIC SYSTEM DEMONSTRATION")
        print("=" * 80)
        
        # 1. Load Wisconsin dataset
        print("\n1. Loading Wisconsin Breast Cancer Dataset...")
        df = self.load_wisconsin_data()
        print(f"Dataset shape: {df.shape}")
        
        # 2. Preprocess data
        print("\n2. Preprocessing data...")
        X_train, X_test, y_train, y_test = self.preprocess_fna_data(df)
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # 3. Visualize data manifold
        print("\n3. Visualizing data distribution...")
        self.visualize_data_manifold(np.vstack([X_train, X_test]), 
                                    np.hstack([y_train, y_test]), 
                                    method='tsne')
        
        # 4. Build and train FNA model
        print("\n4. Building and training FNA model...")
        self.build_fna_model(input_shape=(X_train.shape[1],))
        self.train_fna_model(X_train, y_train, X_test, y_test, epochs=50)
        
        # 5. Visualize training history
        print("\n5. Visualizing training history...")
        self.visualize_fna_training()
        
        # 6. Evaluate FNA model
        print("\n6. Evaluating FNA model...")
        self.evaluate_fna_model(X_test, y_test)
        
        # 7. Feature importance analysis
        print("\n7. Analyzing feature importance...")
        importance_df = self.get_feature_importance(X_test)
        self.visualize_feature_importance(importance_df)
        
        # 8. Feature patterns analysis
        print("\n8. Analyzing feature patterns across diagnoses...")
        self.visualize_fna_feature_patterns(X_test, y_test)
        
        # 9. Decision boundary visualization
        print("\n9. Visualizing decision boundary...")
        self.visualize_fna_decision_boundary(X_test, y_test)
        
        # 10. Explain individual predictions
        print("\n10. Explaining individual predictions with GradCAM...")
        sample_idx = np.random.choice(len(X_test))
        self.create_nuclear_feature_gradcam(X_test[sample_idx], y_test[sample_idx])
        
        # 11. Simulate IDC model
        print("\n11. Simulating IDC histopathology model...")
        self.simulate_idc_training()
        
        # 12. Save models
        print("\n12. Saving trained models...")
        self.save_models()
        
        print("\nDemonstration completed successfully!")
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    # Initialize system
    bc_system = BreastCancerAISystem()
    
    # Run demonstration
    bc_system.run_demo()
    
    # Example of making predictions on new data
    print("\nExample: Making prediction on new FNA data...")
    
    # Generate synthetic sample for demonstration
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    random_idx = np.random.randint(0, len(data.data))
    sample = data.data[random_idx]
    
    result = bc_system.predict_fna_features(sample)
    print(f"Prediction: {result['prediction']} with {result['confidence']:.3f} confidence")
    print(f"Interpretation: {result['interpretation']}")