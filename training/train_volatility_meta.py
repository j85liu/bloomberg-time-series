# Create a new file: training/train_volatility_meta.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from torch.utils.data import DataLoader, TensorDataset

# Import base models
from models.tft_v3 import TemporalFusionTransformer
from models.nbeatsx_v7 import NBEATSx
from models.deepar_v3 import DeepARModel

# Import meta-learning framework
from models.meta_learning.enhanced_framework import EnhancedMetaLearningFramework

def create_dataloaders(model_data, batch_size=32):
    """
    Create dataloaders for each model type
    
    Args:
        model_data: Dictionary of formatted model data
        batch_size: Batch size
        
    Returns:
        dataloaders: Dictionary of dataloaders
    """
    dataloaders = {}
    
    # TFT dataloaders
    if 'tft' in model_data:
        tft_data = model_data['tft']
        
        # Training data
        train_dataset = TensorDataset(
            *tft_data['train']['static_inputs'],
            *tft_data['train']['encoder_inputs'],
            *tft_data['train']['decoder_inputs'],
            tft_data['train']['targets']
        )
        
        # Validation data
        val_dataset = TensorDataset(
            *tft_data['val']['static_inputs'],
            *tft_data['val']['encoder_inputs'],
            *tft_data['val']['decoder_inputs'],
            tft_data['val']['targets']
        )
        
        # Test data
        test_dataset = TensorDataset(
            *tft_data['test']['static_inputs'],
            *tft_data['test']['encoder_inputs'],
            *tft_data['test']['decoder_inputs'],
            tft_data['test']['targets']
        )
        
        dataloaders['tft'] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size),
            'test': DataLoader(test_dataset, batch_size=batch_size)
        }
    
    # NBEATSx dataloaders
    if 'nbeatsx' in model_data:
        nbeatsx_data = model_data['nbeatsx']
        
        # Training data
        train_dataset = TensorDataset(
            nbeatsx_data['train']['x'],
            nbeatsx_data['train']['y'],
            nbeatsx_data['train']['task_ids']
        )
        
        # Validation data
        val_dataset = TensorDataset(
            nbeatsx_data['val']['x'],
            nbeatsx_data['val']['y'],
            nbeatsx_data['val']['task_ids']
        )
        
        # Test data
        test_dataset = TensorDataset(
            nbeatsx_data['test']['x'],
            nbeatsx_data['test']['y'],
            nbeatsx_data['test']['task_ids']
        )
        
        dataloaders['nbeatsx'] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size),
            'test': DataLoader(test_dataset, batch_size=batch_size)
        }
    
    # DeepAR dataloaders
    if 'deepar' in model_data:
        deepar_data = model_data['deepar']
        
        # Training data
        train_dataset = TensorDataset(
            deepar_data['train']['time_series'],
            deepar_data['train']['time_features'],
            deepar_data['train']['static_features'],
            deepar_data['train']['targets']
        )
        
        # Validation data
        val_dataset = TensorDataset(
            deepar_data['val']['time_series'],
            deepar_data['val']['time_features'],
            deepar_data['val']['static_features'],
            deepar_data['val']['targets']
        )
        
        # Test data
        test_dataset = TensorDataset(
            deepar_data['test']['time_series'],
            deepar_data['test']['time_features'],
            deepar_data['test']['static_features'],
            deepar_data['test']['targets']
        )
        
        dataloaders['deepar'] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size),
            'test': DataLoader(test_dataset, batch_size=batch_size)
        }
    
    return dataloaders

def create_and_train_base_models(dataloaders, model_data, device, config):
    """
    Create and train the base forecasting models
    
    Args:
        dataloaders: Dictionary of dataloaders
        model_data: Dictionary of model data
        device: PyTorch device
        config: Dictionary of model configurations
        
    Returns:
        models: Dictionary of trained models
    """
    models = {}
    
    # 1. Train TFT model
    if 'tft' in dataloaders:
        print("\nTraining TFT model...")
        
        # Get dimensions from data
        tft_data = model_data['tft']
        input_size = tft_data['train']['encoder_inputs'][0].shape[1]
        hidden_dim = config.get('tft_hidden_dim', 64)
        
        # Count input dimensions
        num_static_vars = len(tft_data['train']['static_inputs'])
        num_encoder_vars = len(tft_data['train']['encoder_inputs'])
        num_decoder_vars = len(tft_data['train']['decoder_inputs'])
        
        # Determine input sizes
        static_input_sizes = [s.shape[1] for s in tft_data['train']['static_inputs']]
        encoder_input_sizes = [e.shape[2] for e in tft_data['train']['encoder_inputs']]
        decoder_input_sizes = [d.shape[2] for d in tft_data['train']['decoder_inputs']] if num_decoder_vars > 0 else []
        
        # Forecast horizon
        forecast_horizon = tft_data['train']['targets'].shape[1]
        
        # Create TFT model
        tft_model = TemporalFusionTransformer(
            num_static_vars=num_static_vars,
            num_future_vars=num_decoder_vars,
            num_past_vars=num_encoder_vars,
            static_input_sizes=static_input_sizes,
            encoder_input_sizes=encoder_input_sizes,
            decoder_input_sizes=decoder_input_sizes,
            hidden_dim=hidden_dim,
            forecast_horizon=forecast_horizon,
            backcast_length=input_size,
            output_dim=1,
            quantiles=[0.1, 0.5, 0.9]
        ).to(device)
        
        # Create optimizer
        optimizer = optim.Adam(
            tft_model.parameters(), 
            lr=config.get('tft_learning_rate', 0.001)
        )
        
        # Train the model
        tft_model.train()
        best_val_loss = float('inf')
        patience = config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(config.get('tft_epochs', 50)):
            # Training
            train_loss = 0
            tft_model.train()
            
            for batch in dataloaders['tft']['train']:
                # Move batch to device
                batch = [b.to(device) for b in batch]
                
                # Extract components
                static_inputs = batch[:num_static_vars]
                encoder_inputs = batch[num_static_vars:num_static_vars + num_encoder_vars]
                decoder_inputs = batch[num_static_vars + num_encoder_vars:-1]
                targets = batch[-1]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = tft_model(static_inputs, encoder_inputs, decoder_inputs)
                
                # Calculate loss (pinball loss for quantile regression)
                loss = 0
                for i, q in enumerate([0.1, 0.5, 0.9]):
                    errors = targets - outputs[:, :, :, i]
                    loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = 0
            tft_model.eval()
            
            with torch.no_grad():
                for batch in dataloaders['tft']['val']:
                    # Move batch to device
                    batch = [b.to(device) for b in batch]
                    
                    # Extract components
                    static_inputs = batch[:num_static_vars]
                    encoder_inputs = batch[num_static_vars:num_static_vars + num_encoder_vars]
                    decoder_inputs = batch[num_static_vars + num_encoder_vars:-1]
                    targets = batch[-1]
                    
                    # Forward pass
                    outputs = tft_model(static_inputs, encoder_inputs, decoder_inputs)
                    
                    # Calculate loss
                    loss = 0
                    for i, q in enumerate([0.1, 0.5, 0.9]):
                        errors = targets - outputs[:, :, :, i]
                        loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
                    
                    val_loss += loss.item()
            
            # Print progress
            avg_train_loss = train_loss / len(dataloaders['tft']['train'])
            avg_val_loss = val_loss / len(dataloaders['tft']['val'])
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(tft_model.state_dict(), 'models/tft_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        tft_model.load_state_dict(torch.load('models/tft_best.pt'))
        models['tft'] = tft_model
    
    # 2. Train NBEATSx model
    if 'nbeatsx' in dataloaders:
        print("\nTraining NBEATSx model...")
        
        # Get dimensions from data
        nbeatsx_data = model_data['nbeatsx']
        input_size = nbeatsx_data['train']['x'].shape[1]
        input_dim = nbeatsx_data['train']['x'].shape[2]
        forecast_horizon = nbeatsx_data['train']['y'].shape[1]
        
        # Create NBEATSx model
        nbeatsx_model = NBEATSx(
            input_size=input_size,
            forecast_size=forecast_horizon,
            exog_channels=input_dim,  # Use all features as exogenous
            stack_types=['trend', 'seasonality', 'generic'],
            num_blocks_per_stack=[3, 3, 1],
            hidden_units=config.get('nbeatsx_hidden_dim', 128),
            layers=config.get('nbeatsx_layers', 4),
            basis_kwargs={
                'degree': 3,
                'harmonics': 5
            },
            dropout=0.1,
            exog_mode='tcn'  # Use TCN for exogenous variables
        ).to(device)
        
        # Create optimizer
        optimizer = optim.Adam(
            nbeatsx_model.parameters(), 
            lr=config.get('nbeatsx_learning_rate', 0.001)
        )
        
        # Train the model
        nbeatsx_model.train()
        best_val_loss = float('inf')
        patience = config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(config.get('nbeatsx_epochs', 50)):
            # Training
            train_loss = 0
            nbeatsx_model.train()
            
            for batch in dataloaders['nbeatsx']['train']:
                # Move batch to device
                x, y, _ = [b.to(device) for b in batch]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = nbeatsx_model(x)
                
                # Calculate loss
                loss = F.mse_loss(outputs, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = 0
            nbeatsx_model.eval()
            
            with torch.no_grad():
                for batch in dataloaders['nbeatsx']['val']:
                    # Move batch to device
                    x, y, _ = [b.to(device) for b in batch]
                    
                    # Forward pass
                    outputs = nbeatsx_model(x)
                    
                    # Calculate loss
                    loss = F.mse_loss(outputs, y)
                    
                    val_loss += loss.item()
            
            # Print progress
            avg_train_loss = train_loss / len(dataloaders['nbeatsx']['train'])
            avg_val_loss = val_loss / len(dataloaders['nbeatsx']['val'])
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            


            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(nbeatsx_model.state_dict(), 'models/nbeatsx_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        nbeatsx_model.load_state_dict(torch.load('models/nbeatsx_best.pt'))
        models['nbeatsx'] = nbeatsx_model
    
    # 3. Train DeepAR model
    if 'deepar' in dataloaders:
        print("\nTraining DeepAR model...")
        
        # Get dimensions from data
        deepar_data = model_data['deepar']
        time_series = deepar_data['train']['time_series']
        time_features = deepar_data['train']['time_features']
        static_features = deepar_data['train']['static_features']
        
        # Model dimensions
        num_time_features = time_features.shape[2] if time_features is not None else 0
        num_static_features = static_features.shape[1] if static_features is not None else 0
        seq_len = time_series.shape[1]
        forecast_horizon = deepar_data['train']['targets'].shape[1]
        
        # Create DeepAR model
        deepar_model = DeepARModel(
            num_time_features=num_time_features,
            num_static_features=num_static_features,
            embedding_dim=config.get('deepar_embedding_dim', 32),
            hidden_size=config.get('deepar_hidden_size', 64),
            num_layers=config.get('deepar_num_layers', 2),
            dropout=0.1,
            likelihood='gaussian',  # Use Gaussian likelihood for continuous data
            seq_len=seq_len,
            prediction_len=forecast_horizon
        ).to(device)
        
        # Create optimizer
        optimizer = optim.Adam(
            deepar_model.parameters(), 
            lr=config.get('deepar_learning_rate', 0.001)
        )
        
        # Train the model
        deepar_model.train()
        best_val_loss = float('inf')
        patience = config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(config.get('deepar_epochs', 50)):
            # Training
            train_loss = 0
            deepar_model.train()
            
            for batch in dataloaders['deepar']['train']:
                # Move batch to device
                time_series_batch, time_features_batch, static_features_batch, targets_batch = [b.to(device) for b in batch]
                
                # Forward pass and compute loss
                optimizer.zero_grad()
                loss = deepar_model.loss(
                    y=torch.cat([time_series_batch[:, -1:, :], targets_batch], dim=1),
                    time_features=time_features_batch if time_features_batch.shape[0] > 0 else None,
                    static_features=static_features_batch if static_features_batch.shape[0] > 0 else None
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = 0
            deepar_model.eval()
            
            with torch.no_grad():
                for batch in dataloaders['deepar']['val']:
                    # Move batch to device
                    time_series_batch, time_features_batch, static_features_batch, targets_batch = [b.to(device) for b in batch]
                    
                    # Compute loss
                    loss = deepar_model.loss(
                        y=torch.cat([time_series_batch[:, -1:, :], targets_batch], dim=1),
                        time_features=time_features_batch if time_features_batch.shape[0] > 0 else None,
                        static_features=static_features_batch if static_features_batch.shape[0] > 0 else None
                    )
                    
                    val_loss += loss.item()
            
            # Print progress
            avg_train_loss = train_loss / len(dataloaders['deepar']['train'])
            avg_val_loss = val_loss / len(dataloaders['deepar']['val'])
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(deepar_model.state_dict(), 'models/deepar_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        deepar_model.load_state_dict(torch.load('models/deepar_best.pt'))
        models['deepar'] = deepar_model
    
    return models

def format_data_for_meta_learning(model_data, dataloaders, vix_data=None):
    """
    Format data for the meta-learning framework
    
    Args:
        model_data: Dictionary of model data
        dataloaders: Dictionary of dataloaders
        vix_data: Optional DataFrame with VIX data for regime detection
        
    Returns:
        meta_dataloaders: Dictionary with meta-learning formatted data
    """
    meta_dataloaders = {'train': [], 'val': [], 'test': []}
    
    # Process each model type's dataloader to extract the required data format
    for split in ['train', 'val', 'test']:
        # TFT data
        if 'tft' in dataloaders:
            for batch in dataloaders['tft'][split]:
                # Extract components based on model_data structure
                tft_data = model_data['tft']
                num_static_vars = len(tft_data[split]['static_inputs'])
                num_encoder_vars = len(tft_data[split]['encoder_inputs'])
                num_decoder_vars = len(tft_data[split]['decoder_inputs'])
                
                # Extract and process batch
                static_inputs = batch[:num_static_vars]
                encoder_inputs = batch[num_static_vars:num_static_vars + num_encoder_vars]
                decoder_inputs = batch[num_static_vars + num_encoder_vars:-1]
                targets = batch[-1]
                
                # Create formatted batch for meta-learning
                meta_batch = {
                    'model_type': 'tft',
                    'input': encoder_inputs[0],  # Main input series
                    'target': targets,
                    'static_features': static_inputs[0] if num_static_vars > 0 else None,
                    'time_features': encoder_inputs[1:] if len(encoder_inputs) > 1 else None,
                    'future_time_features': decoder_inputs if num_decoder_vars > 0 else None,
                    'task_ids': static_inputs[0] if num_static_vars > 0 else None
                }
                
                meta_dataloaders[split].append(meta_batch)
        
        # NBEATSx data
        if 'nbeatsx' in dataloaders:
            for x, y, task_ids in dataloaders['nbeatsx'][split]:
                meta_batch = {
                    'model_type': 'nbeatsx',
                    'input': x,  # Main input series with all features
                    'target': y,
                    'task_ids': task_ids
                }
                
                meta_dataloaders[split].append(meta_batch)
        
        # DeepAR data
        if 'deepar' in dataloaders:
            for time_series, time_features, static_features, targets in dataloaders['deepar'][split]:
                meta_batch = {
                    'model_type': 'deepar',
                    'input': time_series,
                    'target': targets,
                    'time_features': time_features if time_features.shape[0] > 0 else None,
                    'static_features': static_features if static_features.shape[0] > 0 else None
                }
                
                meta_dataloaders[split].append(meta_batch)
    
    # Add VIX data for regime detection if available
    if vix_data is not None:
        for split in ['train', 'val', 'test']:
            for batch in meta_dataloaders[split]:
                batch['vix_data'] = vix_data
    
    return meta_dataloaders

def train_meta_learning_framework(meta_dataloaders, models, device, config):
    """
    Train the meta-learning framework
    
    Args:
        meta_dataloaders: Dictionary with meta-learning formatted data
        models: Dictionary of trained models
        device: PyTorch device
        config: Configuration dictionary
        
    Returns:
        framework: Trained meta-learning framework
    """
    print("\nInitializing meta-learning framework...")
    
    # Initialize the enhanced framework
    framework = EnhancedMetaLearningFramework(
        base_models=models,
        meta_feature_dim=config.get('meta_feature_dim', 32),
        hidden_dim=config.get('meta_hidden_dim', 64),
        num_regimes=config.get('num_regimes', 4),
        regime_method=config.get('regime_method', 'hybrid'),
        use_model_features=config.get('use_model_features', True)
    )
    
    # Move framework to device
    framework.meta_feature_extractor.to(device)
    framework.meta_learner.to(device)
    framework.temperature.to(device)
    
    # Phase 1: Collect meta-knowledge from training data
    print("\nPhase 1: Collecting meta-knowledge...")
    for batch in meta_dataloaders['train']:
        # Move tensors to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value
        
        # Collect meta-knowledge
        framework.collect_meta_knowledge(
            time_series=batch_device['input'],
            targets=batch_device['target'],
            time_features=batch_device.get('time_features'),
            static_features=batch_device.get('static_features'),
            future_time_features=batch_device.get('future_time_features'),
            task_ids=batch_device.get('task_ids'),
            vix_data=batch_device.get('vix_data')
        )
    
    # Phase 2: Train meta-learner on collected meta-knowledge
    print("\nPhase 2: Training meta-learner...")
    meta_trained = framework.meta_train(
        epochs=config.get('meta_epochs', 50),
        lr=config.get('meta_lr', 0.001)
    )
    
    if not meta_trained:
        print("Error: Meta-learner training failed.")
        return None
    
    # Phase 3: End-to-end training of the whole framework
    print("\nPhase 3: End-to-end training of the whole framework...")
    
    # Setup optimizer for framework components
    framework_params = list(framework.meta_feature_extractor.parameters()) + \
                      list(framework.meta_learner.parameters()) + \
                      [framework.temperature]
    
    optimizer = optim.Adam(framework_params, lr=config.get('framework_lr', 0.0005))
    criterion = nn.MSELoss()
    
    # Setup for early stopping
    best_val_loss = float('inf')
    best_state = None
    patience = config.get('patience', 10)
    patience_counter = 0
    
    # Main training loop
    for epoch in range(config.get('framework_epochs', 100)):
        # Training mode
        framework.meta_feature_extractor.train()
        framework.meta_learner.train()
        
        total_loss = 0
        
        # Process each training batch
        for batch in meta_dataloaders['train']:
            # Move tensors to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Generate forecast
            output = framework.forecast(
                time_series=batch_device['input'],
                time_features=batch_device.get('time_features'),
                static_features=batch_device.get('static_features'),
                future_time_features=batch_device.get('future_time_features'),
                task_ids=batch_device.get('task_ids'),
                vix_data=batch_device.get('vix_data')
            )
            
            # Compute loss
            loss = criterion(output['forecast'], batch_device['target'])
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(meta_dataloaders['train'])
        
        # Validation phase
        framework.meta_feature_extractor.eval()
        framework.meta_learner.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch in meta_dataloaders['val']:
                # Move tensors to device
                batch_device = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value
                
                # Generate forecast
                output = framework.forecast(
                    time_series=batch_device['input'],
                    time_features=batch_device.get('time_features'),
                    static_features=batch_device.get('static_features'),
                    future_time_features=batch_device.get('future_time_features'),
                    task_ids=batch_device.get('task_ids'),
                    vix_data=batch_device.get('vix_data')
                )
                
                # Compute loss
                loss = criterion(output['forecast'], batch_device['target'])
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(meta_dataloaders['val'])
        
        print(f"Epoch {epoch+1}/{config.get('framework_epochs', 100)}, "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            # Save best model state
            best_val_loss = avg_val_loss
            best_state = {
                'meta_feature_extractor': framework.meta_feature_extractor.state_dict(),
                'meta_learner': framework.meta_learner.state_dict(),
                'temperature': framework.temperature.clone()
            }
            patience_counter = 0
            
            # Save the model
            torch.save(best_state, 'models/meta_framework_best.pt')
        else:
            # Increment patience counter if no improvement
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model state
    if best_state:
        framework.meta_feature_extractor.load_state_dict(best_state['meta_feature_extractor'])
        framework.meta_learner.load_state_dict(best_state['meta_learner'])
        framework.temperature.data = best_state['temperature']
    
    # Phase 4: Evaluation on test data
    print("\nPhase 4: Evaluating on test data...")
    
    # Prepare test batches in the format expected by framework.evaluate()
    test_batches = []
    for batch in meta_dataloaders['test']:
        # Move tensors to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value
        
        test_batches.append(batch_device)
    
    # Evaluate the framework
    evaluation_results = framework.evaluate(
        test_data=test_batches,
        metrics=['mse', 'mae', 'dir_acc'],
        by_regime=True,
        by_task=True
    )
    
    # Print evaluation results
    print("\nTest Results:")
    print(f"Overall MSE: {evaluation_results['overall']['mse']:.4f}")
    print(f"Overall MAE: {evaluation_results['overall']['mae']:.4f}")
    
    # Print regime-specific results if available
    if 'by_regime' in evaluation_results:
        print("\nPerformance by Regime:")
        for regime, metrics in evaluation_results['by_regime'].items():
            print(f"{regime} - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")
    
    # Print model selection statistics
    if 'model_selection' in evaluation_results:
        print("\nModel Selection Statistics:")
        for model, stats in evaluation_results['model_selection'].items():
            print(f"{model} - Mean Weight: {stats['mean']:.4f}, Selection Rate: {stats['selection_rate']:.4f}")
    
    # Save evaluation results
    with open('results/evaluation_results.json', 'w') as f:
        # Convert numpy types to Python native types for JSON serialization
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                               np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
        
        # Process the results dictionary
        serializable_results = {}
        for key, value in evaluation_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: {k2: convert_to_serializable(v2) for k2, v2 in v.items()} 
                    if isinstance(v, dict) else convert_to_serializable(v)
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = convert_to_serializable(value)
        
        json.dump(serializable_results, f, indent=2)
    
    return framework

def main():
    """Main function to train and evaluate the volatility forecasting framework"""
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = {
        # Base model parameters
        'tft_hidden_dim': 64,
        'tft_learning_rate': 0.001,
        'tft_epochs': 50,
        
        'nbeatsx_hidden_dim': 128,
        'nbeatsx_layers': 4,
        'nbeatsx_learning_rate': 0.001,
        'nbeatsx_epochs': 50,
        
        'deepar_embedding_dim': 32,
        'deepar_hidden_size': 64,
        'deepar_num_layers': 2,
        'deepar_learning_rate': 0.001,
        'deepar_epochs': 50,
        
        # Meta-learning parameters
        'meta_feature_dim': 32,
        'meta_hidden_dim': 64,
        'meta_epochs': 50,
        'meta_lr': 0.001,
        
        # Framework parameters
        'framework_epochs': 100,
        'framework_lr': 0.0005,
        'num_regimes': 4,
        'regime_method': 'hybrid',
        'use_model_features': True,
        
        # Training parameters
        'batch_size': 32,
        'patience': 10
    }
    
    # Load formatted model data
    try:
        model_data = torch.load('data/vix_model_data.pt')
        print("Loaded model data from 'data/vix_model_data.pt'")
    except FileNotFoundError:
        print("Error: Model data file not found. Please run the data preparation script first.")
        return
    
    # Load VIX data for regime detection
    try:
        vix_df = pd.read_csv('data/vix_processed.csv', index_col=0, parse_dates=True)
        print(f"Loaded VIX data with {len(vix_df)} records")
    except FileNotFoundError:
        print("Warning: VIX data file not found. Proceeding without regime detection.")
        vix_df = None
    
    # Create dataloaders
    dataloaders = create_dataloaders(model_data, batch_size=config['batch_size'])
    print("Created dataloaders for all models")
    
    # Create and train base models
    models = create_and_train_base_models(dataloaders, model_data, device, config)
    print("Base models trained successfully")
    
    # Format data for meta-learning
    meta_dataloaders = format_data_for_meta_learning(model_data, dataloaders, vix_df)
    print("Formatted data for meta-learning")
    
    # Train meta-learning framework
    framework = train_meta_learning_framework(meta_dataloaders, models, device, config)
    
    # Save trained framework
    if framework:
        torch.save({
            'meta_feature_extractor': framework.meta_feature_extractor.state_dict(),
            'meta_learner': framework.meta_learner.state_dict(),
            'temperature': framework.temperature.data,
            'config': config
        }, 'models/volatility_meta_framework.pt')
        
        print("\nTraining complete! Framework saved to 'models/volatility_meta_framework.pt'")
        print("Evaluation results saved to 'results/evaluation_results.json'")
    
    return framework

if __name__ == "__main__":
    main()