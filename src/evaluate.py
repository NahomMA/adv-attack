import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelPerformance:
    def __init__(self):
        self.metrics = {}

    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics for multi-class classification
        """
        # Calculate metrics using average='weighted' for multi-class
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        self.metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        self.metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')

        # Round metrics
        self.metrics = {
            'accuracy': round(self.metrics['accuracy'], 3),
            'precision': round(self.metrics['precision'], 3),
            'recall': round(self.metrics['recall'], 3),
            'f1_score': round(self.metrics['f1_score'], 3)
        }

    def plot_comparison(self, y_true, y_pred, model_name, class_names=None, fig_name=None):
        """
        Plot confusion matrix and metrics table for multi-class classification

        Parameters:
        - class_names: list of class names for the confusion matrix labels
        """
        # Calculate and store metrics
        self._calculate_metrics(y_true, y_pred)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Use class names if provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=class_names,
                    yticklabels=class_names)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title(f'Confusion Matrix - {model_name}')

        # Create the metrics table
        metrics_df = pd.DataFrame([self.metrics], index=[model_name])
        table = ax2.table(cellText=metrics_df.values,
                        colLabels=metrics_df.columns,
                        rowLabels=metrics_df.index,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax2.axis('off')
        ax2.set_title(f'Model Performance Metrics - {model_name}')

        plt.tight_layout()
          # Save the figure first if a filename is provided
        if fig_name:
            os.makedirs('figures', exist_ok=True)
            fig_path = os.path.join('figures', fig_name)
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        
        # Show the plot after saving
        plt.show()
        
        # Close the figure to free up memory
        plt.close(fig)