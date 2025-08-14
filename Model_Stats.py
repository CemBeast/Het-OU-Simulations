import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from collections import OrderedDict

# This file is to assume that float64 will be used 

# Function to count sparse elements in a tensor with a threshold
def count_sparse_elements(tensor, threshold=1e-6):
    return torch.sum(torch.abs(tensor) < threshold).item()

# Function to calculate weight sparsity with threshold
def calculate_sparsity(tensor, threshold=1e-6):
    total_elements = tensor.numel()
    sparse_elements = count_sparse_elements(tensor, threshold)
    return sparse_elements / total_elements

# Function to register hooks for activation sparsity and memory usage
def register_activation_hooks(model):
    activation_stats = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # Store activations for analysis
            if isinstance(output, torch.Tensor):
                activation_stats[name] = {
                    'tensor': output.detach().clone(),
                    'size_kb': output.numel() * 8 / 1024  # Size in KB assuming float64 (8 bytes)
                }
            else:
                # Handle tuple outputs
                if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    tensor = output[0].detach().clone()
                    activation_stats[name] = {
                        'tensor': tensor,
                        'size_kb': tensor.numel() * 8 / 1024  # Size in KB assuming float64 (8 bytes)
                    }
                else:
                    activation_stats[name] = None
        return hook
    
    # Register hooks for all modules to capture activations
    for name, module in model.named_modules():
        if not list(module.children()): # Only register for leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    return activation_stats, hooks

# Function to calculate MACs (multiply-accumulate operations)
def calculate_macs(module, input_shape, output_shape):
    if isinstance(module, nn.Conv2d):
        batch_size, in_channels, in_h, in_w = input_shape
        batch_size, out_channels, out_h, out_w = output_shape
        kernel_h, kernel_w = module.kernel_size
        # Each output element requires kernel_h * kernel_w * in_channels MACs
        return out_h * out_w * out_channels * kernel_h * kernel_w * in_channels
    
    elif isinstance(module, nn.Linear):
        return module.in_features * module.out_features
    
    return 0

# Function to get model stats
def get_model_stats(model):
    # Create a dummy input with values between 0-1 (more realistic than random noise)
    dummy_input = torch.rand(1, 3, 224, 224)
    
    # Register hooks for activation sparsity and memory
    activation_stats, hooks = register_activation_hooks(model)
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        output = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    stats = []
    
    # Input shapes for MAC calculation
    input_shapes = {}
    output_shapes = {}
    
    # Register hooks to capture input and output shapes
    def register_shape_hooks(model):
        shape_hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input[0], torch.Tensor):
                    input_shapes[name] = input[0].shape
                    output_shapes[name] = output.shape
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                shape_hooks.append(module.register_forward_hook(hook_fn(name)))
        
        return shape_hooks
    
    # Register and run shape hooks
    shape_hooks = register_shape_hooks(model)
    with torch.no_grad():
        model(dummy_input)
    for hook in shape_hooks:
        hook.remove()
    
    # Collect stats for each layer
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Weight stats - calculate memory in KB (assuming float64 = 8 bytes)
            weight_params = module.weight.numel()
            weight_memory_kb = weight_params * 8 / 1024  # Use 8 bytes for float64
            weight_sparsity = calculate_sparsity(module.weight, threshold=1e-4)
            
            # MACs calculation
            macs = 0
            if name in input_shapes and name in output_shapes:
                macs = calculate_macs(module, input_shapes[name], output_shapes[name])
            
            # Activation sparsity and memory - look for this layer and ReLU outputs
            act_sparsity = 0
            act_memory_kb = 0
            
            # For each layer, look for corresponding activation
            if name in activation_stats and activation_stats[name] is not None:
                act_info = activation_stats[name]
                act_tensor = act_info['tensor']
                act_memory_kb = act_info['size_kb']
                act_sparsity = calculate_sparsity(act_tensor)
            
            # Special handling for ReLU activations which should show true sparsity
            relu_name = name.replace('conv', 'relu').replace('linear', 'relu')
            if relu_name in activation_stats and activation_stats[relu_name] is not None:
                act_info = activation_stats[relu_name]
                act_tensor = act_info['tensor']
                act_memory_kb = act_info['size_kb']
                act_sparsity = calculate_sparsity(act_tensor)
            
            stats.append({
                'Layer': name,
                'Weights(KB)': weight_memory_kb,
                'MACs': macs,
                'Weight_Sparsity(0-1)': weight_sparsity,
                'Activation_Sparsity(0-1)': act_sparsity,
                'Activations(KB)': act_memory_kb
            })
    
    return stats

# Function to prune and analyze any model
def analyze_model(name, model_fn, threshold=1e-3):
    print(f"\n🔍 Analyzing {name}...")
    model = model_fn(weights='DEFAULT')
    model.eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param.data) < threshold
                param.data[mask] = 0.0
    stats = get_model_stats(model)
    df = pd.DataFrame(stats)
    df.insert(0, "Layer #", range(1, len(df) + 1))
 
    # Add additional analysis - summary statistics
    total_weights_kb = df['Weights(KB)'].sum()
    total_activations_kb = df['Activations(KB)'].sum()
    total_macs = df['MACs'].sum()
    avg_weight_sparsity = (df['Weight_Sparsity(0-1)'] * df['Weights(KB)']).sum() / total_weights_kb
    avg_act_sparsity = df['Activation_Sparsity(0-1)'].mean()  # Simple mean for activation sparsity

    print(f"Total weights (float64): {total_weights_kb:.2f} KB")
    print(f"Total activations (float64): {total_activations_kb:.2f} KB")
    print(f"Total MACs: {total_macs:,}")
    print(f"Average weight sparsity: {avg_weight_sparsity*100:.4f}%")
    print(f"Average activation sparsity: {avg_act_sparsity*100:.6f}%")

    # Display the results
    print("\nLayer statistics:")
    print(df)

    return df

# List of models to extract info on
models_to_run = {
    # "VGG19":     models.vgg19_bn,
    # "ResNet50":  models.resnet50,
    # "ResNet101": models.resnet101,
    # "ResNet152": models.resnet152,
    #"ResNet34":   models.resnet34,
    "VGG11":        models.vgg11_bn
}

# Saves each models stats within the workload folder
for name, fn in models_to_run.items():
    df = analyze_model(name, fn)
    filename = f"workloads/{name.lower()}_stats.csv"  # Save in folder
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

