"""
Unit tests for the Enhanced Autoencoder module.
"""

import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rrr import Config, EnhancedAutoencoder, DataProcessor, AnomalyDetector


class TestConfig(unittest.TestCase):
    """Test the Configuration class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = Config()
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.latent_dim, 32)
        self.assertIsNotNone(config.encoding_dims)
    
    def test_config_post_init(self):
        """Test post-initialization behavior."""
        config = Config()
        self.assertIsInstance(config.encoding_dims, list)
        self.assertTrue(len(config.encoding_dims) > 0)


class TestEnhancedAutoencoder(unittest.TestCase):
    """Test the Enhanced Autoencoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.input_dim = 100
        self.autoencoder = EnhancedAutoencoder(self.input_dim, self.config)
    
    def test_initialization(self):
        """Test autoencoder initialization."""
        self.assertEqual(self.autoencoder.input_dim, self.input_dim)
        self.assertEqual(self.autoencoder.config, self.config)
        self.assertIsNone(self.autoencoder.model)
        self.assertIsNone(self.autoencoder.encoder)
    
    def test_build_model(self):
        """Test model building."""
        model = self.autoencoder.build_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape[1], self.input_dim)
        self.assertEqual(model.output_shape[1], self.input_dim)
        
        # Check if encoder and decoder are created
        self.assertIsNotNone(self.autoencoder.encoder)
        
        # Check compilation
        self.assertEqual(model.loss, 'mse')
    
    def test_model_architecture(self):
        """Test model architecture details."""
        model = self.autoencoder.build_model()
        
        # Count layers
        layer_count = len(model.layers)
        self.assertGreater(layer_count, 10)  # Should have many layers with BN and Dropout
        
        # Check for regularization layers
        layer_names = [layer.name for layer in model.layers]
        batch_norm_layers = [name for name in layer_names if 'batch_normalization' in name]
        dropout_layers = [name for name in layer_names if 'dropout' in name]
        
        self.assertGreater(len(batch_norm_layers), 0)
        self.assertGreater(len(dropout_layers), 0)
    
    @patch('enhanced_autoencoder.logger')
    def test_train_with_mock_data(self, mock_logger):
        """Test training with mock data."""
        # Create mock training data
        X_train = np.random.normal(0, 1, (1000, self.input_dim))
        X_val = np.random.normal(0, 1, (200, self.input_dim))
        
        # Build model
        self.autoencoder.build_model()
        
        # Mock the fit method to avoid actual training
        with patch.object(self.autoencoder.model, 'fit') as mock_fit:
            mock_history = MagicMock()
            mock_history.history = {
                'loss': [1.0, 0.8, 0.6],
                'val_loss': [1.1, 0.9, 0.7],
                'mae': [0.8, 0.6, 0.4],
                'val_mae': [0.9, 0.7, 0.5]
            }
            mock_fit.return_value = mock_history
            
            # Test training
            history = self.autoencoder.train(X_train, X_val)
            
            # Verify fit was called
            mock_fit.assert_called_once()
            self.assertIsNotNone(history)


class TestDataProcessor(unittest.TestCase):
    """Test the Data Processor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.processor = DataProcessor(self.config)
    
    def test_initialization(self):
        """Test data processor initialization."""
        self.assertEqual(self.processor.config, self.config)
        self.assertIsNone(self.processor.preprocessor)
        self.assertIsNone(self.processor.feature_names)
    
    @patch('enhanced_autoencoder.load_dataset')
    def test_load_data_success(self, mock_load_dataset):
        """Test successful data loading."""
        # Mock dataset
        mock_data = {
            'train': [{'feature1': 1, 'feature2': 2, 'class': 'normal'}] * 100,
            'test': [{'feature1': 1, 'feature2': 2, 'class': 'attack'}] * 50
        }
        mock_load_dataset.return_value = mock_data
        
        df_train, df_test = self.processor.load_nslkdd_data()
        
        self.assertEqual(len(df_train), 100)
        self.assertEqual(len(df_test), 50)
        mock_load_dataset.assert_called_once()
    
    @patch('enhanced_autoencoder.load_dataset')
    def test_load_data_failure(self, mock_load_dataset):
        """Test data loading failure."""
        mock_load_dataset.side_effect = Exception("Network error")
        
        with self.assertRaises(Exception):
            self.processor.load_nslkdd_data()
    
    def test_explore_data(self):
        """Test data exploration functionality."""
        # Create mock dataframes
        df_train = self.create_mock_dataframe(1000, include_attacks=False)
        df_test = self.create_mock_dataframe(500, include_attacks=True)
        
        analysis = self.processor.explore_data(df_train, df_test)
        
        # Check analysis results
        self.assertIn('train_shape', analysis)
        self.assertIn('test_shape', analysis)
        self.assertIn('train_class_dist', analysis)
        self.assertIn('numerical_cols', analysis)
        self.assertIn('categorical_cols', analysis)
        
        self.assertEqual(analysis['train_shape'], (1000, 6))
        self.assertEqual(analysis['test_shape'], (500, 6))
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create mock dataframes
        df_train = self.create_mock_dataframe(1000, include_attacks=False)
        df_test = self.create_mock_dataframe(500, include_attacks=True)
        
        X_train, X_val, X_test, y_test = self.processor.preprocess_data(df_train, df_test)
        
        # Check shapes
        self.assertGreater(X_train.shape[0], 0)
        self.assertGreater(X_val.shape[0], 0)
        self.assertEqual(X_test.shape[0], 500)
        self.assertEqual(len(y_test), 500)
        
        # Check feature dimensions match
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        
        # Check preprocessor is fitted
        self.assertIsNotNone(self.processor.preprocessor)
    
    def create_mock_dataframe(self, n_samples, include_attacks=True):
        """Create a mock dataframe for testing."""
        np.random.seed(42)
        
        data = {
            'duration': np.random.randint(0, 1000, n_samples),
            'src_bytes': np.random.randint(0, 10000, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ftp', 'smtp'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples)
        }
        
        if include_attacks:
            data['class'] = np.random.choice(['normal', 'dos', 'probe'], n_samples)
        else:
            data['class'] = ['normal'] * n_samples
        
        return pd.DataFrame(data)


class TestAnomalyDetector(unittest.TestCase):
    """Test the Anomaly Detector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.detector = AnomalyDetector(self.config)
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.config, self.config)
        self.assertIsNone(self.detector.autoencoder)
        self.assertIsNone(self.detector.threshold)
    
    def test_reconstruction_error_calculation(self):
        """Test reconstruction error calculation."""
        # Create mock model and data
        input_dim = 50
        n_samples = 100
        
        X = np.random.normal(0, 1, (n_samples, input_dim))
        
        # Create a simple mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = X + np.random.normal(0, 0.1, X.shape)
        
        errors = self.detector.calculate_reconstruction_error(X, mock_model)
        
        self.assertEqual(len(errors), n_samples)
        self.assertTrue(np.all(errors >= 0))  # Errors should be non-negative
        mock_model.predict.assert_called_once()
    
    def test_threshold_calculation_statistical(self):
        """Test statistical threshold calculation."""
        self.config.threshold_method = 'statistical'
        
        # Create mock validation data
        X_val = np.random.normal(0, 1, (500, 50))
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = X_val + np.random.normal(0, 0.1, X_val.shape)
        
        threshold = self.detector.find_optimal_threshold(X_val, mock_model)
        
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
    
    def test_threshold_calculation_percentile(self):
        """Test percentile threshold calculation."""
        self.config.threshold_method = 'percentile'
        
        # Create mock validation data
        X_val = np.random.normal(0, 1, (500, 50))
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = X_val + np.random.normal(0, 0.1, X_val.shape)
        
        threshold = self.detector.find_optimal_threshold(X_val, mock_model)
        
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
    
    @patch('enhanced_autoencoder.logger')
    def test_model_evaluation(self, mock_logger):
        """Test model evaluation functionality."""
        # Set up test data
        n_samples = 1000
        input_dim = 50
        
        X_test = np.random.normal(0, 1, (n_samples, input_dim))
        y_test = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% anomalies
        
        # Create mock model
        mock_model = MagicMock()
        # Normal samples have lower reconstruction error
        reconstruction_errors = np.where(
            y_test == 0,
            np.random.gamma(2, 0.1, n_samples),  # Lower errors for normal
            np.random.gamma(5, 0.2, n_samples)   # Higher errors for anomalies
        )
        mock_model.predict.return_value = X_test + reconstruction_errors.reshape(-1, 1)
        
        # Set threshold
        self.detector.threshold = np.percentile(reconstruction_errors[y_test == 0], 90)
        
        # Evaluate model
        results = self.detector.evaluate_model(X_test, y_test, mock_model, "Test Model")
        
        # Check results structure
        self.assertIn('metrics', results)
        self.assertIn('y_pred', results)
        self.assertIn('reconstruction_errors', results)
        self.assertIn('confusion_matrix', results)
        
        # Check metrics
        metrics = results['metrics']
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    @patch('enhanced_autoencoder.load_dataset')
    def test_end_to_end_pipeline(self, mock_load_dataset):
        """Test the complete pipeline end-to-end."""
        # Mock dataset
        n_train, n_test = 1000, 500
        
        mock_data = {
            'train': self.create_mock_records(n_train, include_attacks=False),
            'test': self.create_mock_records(n_test, include_attacks=True)
        }
        mock_load_dataset.return_value = mock_data
        
        # Run abbreviated pipeline
        config = Config()
        config.epochs = 1  # Quick training for testing
        config.batch_size = 100
        
        try:
            # Initialize components
            data_processor = DataProcessor(config)
            
            # Load and preprocess data
            df_train, df_test = data_processor.load_nslkdd_data()
            X_train, X_val, X_test, y_test = data_processor.preprocess_data(df_train, df_test)
            
            # Build autoencoder
            autoencoder = EnhancedAutoencoder(X_train.shape[1], config)
            model = autoencoder.build_model()
            
            # This test mainly checks that the pipeline doesn't crash
            self.assertIsNotNone(model)
            self.assertGreater(X_train.shape[0], 0)
            self.assertEqual(len(y_test), n_test)
            
        except Exception as e:
            self.fail(f"End-to-end pipeline failed with error: {str(e)}")
    
    def create_mock_records(self, n_samples, include_attacks=True):
        """Create mock records for testing."""
        np.random.seed(42)
        
        records = []
        for _ in range(n_samples):
            record = {
                'duration': np.random.randint(0, 1000),
                'src_bytes': np.random.randint(0, 10000),
                'dst_bytes': np.random.randint(0, 10000),
                'protocol_type': np.random.choice(['tcp', 'udp', 'icmp']),
                'service': np.random.choice(['http', 'ftp', 'smtp']),
                'flag': np.random.choice(['SF', 'S0', 'REJ'])
            }
            
            if include_attacks:
                record['class'] = np.random.choice(['normal', 'dos', 'probe'], p=[0.7, 0.2, 0.1])
            else:
                record['class'] = 'normal'
            
            records.append(record)
        
        return records


if __name__ == '__main__':
    # Set up test environment
    tf.config.set_visible_devices([], 'GPU')  # Use CPU for testing
    
    # Run tests
    unittest.main(verbosity=2)
