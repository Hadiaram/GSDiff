# Complete Guide to Retraining Diffusion Models While Preserving Learned Capabilities

This guide provides a deep theoretical and practical understanding of how to retrain diffusion models while preserving previously learned capabilities. Written for experienced ML practitioners, it covers foundational concepts, workflows, and best practices for adapting diffusion models to new tasks.

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [Preparing for Retraining](#2-preparing-for-retraining)
3. [Choosing a Retraining Strategy](#3-choosing-a-retraining-strategy)
4. [Step-by-Step Retraining Workflow](#4-step-by-step-retraining-workflow)
5. [Evaluation and Validation](#5-evaluation-and-validation)
6. [Deployment and Integration](#6-deployment-and-integration)
7. [Troubleshooting & Best Practices](#7-troubleshooting--best-practices)

---

## 1. Foundational Concepts

### 1.1 How Diffusion Models Learn

Diffusion models learn through a two-phase process that fundamentally shapes how we approach retraining:

#### Forward Process (Noising)

- Gradually adds Gaussian noise to data over T timesteps (typically 1000)
- Creates a Markov chain: x₀ → x₁ → ... → xₜ
- Each step follows: `q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)`
- The model learns the noise schedule parameters (β₁, ..., βₜ)

#### Reverse Process (Denoising)

- Neural network (typically UNet) learns to predict noise `ε_θ(xₜ, t, c)`
- Conditions on timestep t and optional context c (text, class, graph structure)
- Training objective: `min E[||ε - ε_θ(xₜ, t, c)||²]`
- The learned parameters θ encode knowledge about data distribution

#### Knowledge Representation in Diffusion Models

- **Early UNet layers**: Low-level features (edges, textures, basic shapes)
- **Middle layers**: Semantic understanding and structural relationships
- **Late layers**: Fine details and task-specific adaptations
- **Cross-attention layers**: Condition-to-image mappings
- **Timestep embeddings**: Temporal dynamics of the denoising process

### 1.2 Retaining Previously Learned Information

"Retention" in diffusion models means preserving:

- **Distribution Coverage**: Ability to generate diverse samples from original training distribution
- **Conditional Mappings**: Text/condition → image relationships
- **Quality Metrics**: FID, IS, CLIP scores on original tasks
- **Semantic Understanding**: Compositional abilities and concept relationships

**Key challenge**: The model's parameters encode both generic and specific knowledge intertwined throughout the network.

### 1.3 Adaptation Methods for Diffusion Models

#### Fine-tuning

- Updates all or most model parameters
- **Risk**: Catastrophic forgetting of original distribution
- **Best for**: Similar domains with abundant data
- **Memory requirement**: Full model gradients

#### Transfer Learning

- Leverages pre-trained features as initialization
- Freezes early layers, updates later ones
- **Best for**: New but related domains
- **Preserves**: Low-level feature extraction

#### Continual Learning

- Sequential task learning without forgetting
- Uses techniques like EWC (Elastic Weight Consolidation)
- **Best for**: Expanding model capabilities incrementally
- **Requires**: Careful regularization

#### Parameter-Efficient Methods

- **LoRA (Low-Rank Adaptation)**: Injects trainable rank decomposition matrices
- **Adapters**: Small trainable modules between frozen layers
- **Textual Inversion**: Learns new embeddings only
- **Best for**: Limited compute/data scenarios

---

## 2. Preparing for Retraining

### 2.1 Data Requirements and Dataset Construction

#### Minimum Dataset Size Guidelines

- **Full fine-tuning**: 10,000+ samples
- **LoRA/Adapters**: 100-1,000 samples
- **DreamBooth-style**: 5-20 samples per concept
- **Textual Inversion**: 3-5 samples

#### Dataset Quality Criteria

Essential qualities:
```
├── Resolution consistency (match pretrained model)
├── Clean labels/captions (accurate conditioning)
├── Distribution coverage (representative samples)
├── Noise characteristics (similar preprocessing)
└── Format alignment (RGB, normalized values)
```

#### Data Augmentation for Diffusion Models

- **Geometric**: Rotation, flipping (preserve semantic meaning)
- **Color**: Subtle adjustments only (avoid distribution shift)
- **Cropping**: Maintain aspect ratios from pretraining
- **Synthetic**: Use model itself for pseudo-labeling

#### Balancing Strategies

- **Replay Buffer**: Mix 20-30% original data with new data
- **Weighted Sampling**: Oversample underrepresented classes
- **Curriculum Learning**: Easy → hard examples progressively
- **Multi-task Batching**: Alternate between old and new tasks

### 2.2 Selecting Adaptation Method

#### Decision Framework

```python
IF new_data < 100 samples AND specific_concept:
    → DreamBooth or Textual Inversion
ELIF new_data < 1000 samples AND related_domain:
    → LoRA or Adapter modules
ELIF new_data > 1000 samples AND similar_distribution:
    → Selective fine-tuning with regularization
ELIF completely_new_domain AND large_dataset:
    → Full fine-tuning with careful monitoring
ELSE:
    → Hybrid approach (combine methods)
```

#### Method-Specific Characteristics

| Method | Params Updated | Memory | Training Time | Forgetting Risk | Quality |
|--------|---------------|---------|---------------|----------------|---------|
| Full Fine-tuning | 100% | High | Long | Very High | Highest |
| LoRA | 0.1-1% | Low | Short | Low | High |
| Adapters | 1-5% | Medium | Medium | Low | High |
| DreamBooth | 100% + reg | High | Medium | Medium | High |
| Textual Inv. | <0.01% | Minimal | Short | None | Medium |

### 2.3 Avoiding Catastrophic Forgetting

#### Regularization Techniques

**1. L2 Regularization on Original Weights**

```
Loss = Loss_new + λ * ||θ - θ_original||²
```

- λ typically 0.001-0.01
- Prevents dramatic parameter shifts

**2. Elastic Weight Consolidation (EWC)**

```
Loss = Loss_new + λ * Σᵢ Fᵢ(θᵢ - θᵢ*)²
```

- F is Fisher Information Matrix
- Protects important parameters

**3. Knowledge Distillation**

```
Loss = Loss_new + α * KL(p_original || p_new)
```

- Maintain output distribution similarity
- α typically 0.5-1.0

**4. Prior Preservation Loss (DreamBooth)**

```
Loss = Loss_instance + λ * Loss_prior
```

- Generate synthetic "prior" images
- Maintain class distributions

---

## 3. Choosing a Retraining Strategy

### 3.1 Preserving Prior Knowledge

#### Layer-Specific Strategies

UNet Architecture Breakdown:
```
┌─────────────────────────────────┐
│ Input Conv       → Usually Freeze│
├─────────────────────────────────┤
│ DownBlocks 1-2   → Often Freeze  │
│ (Low-level features)             │
├─────────────────────────────────┤
│ DownBlocks 3-4   → Selective     │
│ (Mid-level semantics)            │
├─────────────────────────────────┤
│ Middle Block     → Usually Train │
│ (High-level understanding)       │
├─────────────────────────────────┤
│ UpBlocks 1-2     → Usually Train │
│ (Task-specific generation)       │
├─────────────────────────────────┤
│ UpBlocks 3-4     → Train         │
│ (Fine details)                   │
├─────────────────────────────────┤
│ Output Conv      → Train         │
└─────────────────────────────────┘
```

#### LoRA Implementation Strategy

```python
# Conceptual LoRA injection points
lora_config = {
    'rank': 4,  # Start small (4-16)
    'alpha': 32,  # Scaling factor
    'target_modules': [
        'attn.q_proj',  # Query projections
        'attn.v_proj',  # Value projections
        'mlp.fc1',      # Feed-forward layers
    ],
    'modules_to_save': ['embed_tokens', 'lm_head']
}
```

#### Progressive Unfreezing Schedule

1. **Epochs 1-5**: Only top layers
2. **Epochs 6-10**: Unfreeze middle layers
3. **Epochs 11-15**: Unfreeze lower layers (if needed)
4. Monitor validation metrics at each stage

### 3.2 Component-Specific Updates

#### UNet (Primary Generation Network)

- Always involved in quality improvements
- Contains most learnable parameters
- Update strategy depends on task similarity

#### Text Encoder (CLIP/T5)

- Freeze for visual quality improvements
- Fine-tune for new concept understanding
- LoRA for efficient concept injection

#### VAE (Variational Autoencoder)

- Usually frozen (pretrained VAE is robust)
- Only fine-tune for significant domain shifts
- **Risk**: Can break latent space consistency

### 3.3 Critical Hyperparameters

#### Learning Rate Scheduling

```
Base LR Selection:
├── Full fine-tuning: 1e-6 to 1e-5
├── LoRA: 1e-4 to 5e-4
├── Adapters: 1e-4 to 1e-3
└── Textual Inversion: 5e-4 to 5e-3

Schedulers:
├── Cosine: Best for longer training
├── Linear: Good for quick adaptation
├── Warmup: Essential (500-1000 steps)
└── Restarts: Helps escape local minima
```

#### Batch Size Considerations

- **Larger batches (32-128)**: More stable, better gradient estimates
- **Smaller batches (4-16)**: Better for limited data, higher variance
- **Gradient accumulation**: Simulate larger batches with limited memory

#### Noise Schedule Adjustments

- Keep original for similar domains
- Adjust beta_start/beta_end for different noise characteristics
- Linear vs. cosine vs. squared_cosine scheduling

---

## 4. Step-by-Step Retraining Workflow

### 4.1 Preparation Phase

#### Step 1: Environment Setup

```bash
# Essential components
├── Model checkpoints (original)
├── Training data (processed)
├── Validation set (representative)
├── Compute resources (GPU with sufficient VRAM)
└── Monitoring tools (tensorboard, wandb)
```

#### Step 2: Data Preprocessing Pipeline

```python
# Conceptual pipeline
def prepare_dataset():
    # 1. Load and validate data
    # 2. Resize to model resolution
    # 3. Normalize to [-1, 1]
    # 4. Create captions/conditions
    # 5. Split train/val/test
    # 6. Create data loaders with appropriate sampling
```

#### Step 3: Baseline Evaluation

- Generate samples with original model
- Compute quality metrics (FID, IS, CLIP)
- Save for comparison
- Document failure modes to address

### 4.2 Training Phase

#### Step 4: Initialize Training Configuration

```yaml
training_config:
  model:
    checkpoint: "path/to/original"
    adaptation_method: "lora"  # or "full", "adapter"

  optimization:
    learning_rate: 1e-4
    weight_decay: 0.01
    gradient_clip: 1.0
    mixed_precision: true

  regularization:
    l2_weight: 0.001
    prior_preservation: true
    prior_weight: 1.0

  scheduling:
    warmup_steps: 1000
    total_steps: 10000
    checkpoint_every: 1000
```

#### Step 5: Training Loop Architecture

```python
# Pseudo-code for training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward diffusion (add noise)
        noise = torch.randn_like(batch.images)
        timesteps = sample_timesteps(batch_size)
        noisy_images = add_noise(batch.images, noise, timesteps)

        # 2. Predict noise
        predicted_noise = model(noisy_images, timesteps, batch.conditions)

        # 3. Calculate losses
        main_loss = F.mse_loss(predicted_noise, noise)
        reg_loss = calculate_regularization(model, original_model)
        total_loss = main_loss + lambda * reg_loss

        # 4. Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 5. Log and monitor
        if step % log_interval == 0:
            log_metrics(main_loss, reg_loss)
            generate_samples(model)
```

#### Step 6: Progressive Validation

```
Validation Schedule:
├── Every 100 steps: Loss tracking
├── Every 500 steps: Sample generation
├── Every 1000 steps: Quality metrics
└── Every 2000 steps: Full evaluation
```

### 4.3 Monitoring During Training

#### Key Metrics to Track

**1. Loss Components**

- Denoising loss (primary objective)
- Regularization loss (forgetting prevention)
- Prior preservation loss (if applicable)

**2. Generation Quality**

- Visual inspection of samples
- FID score trajectory
- CLIP score alignment

**3. Capability Retention**

- Performance on original validation set
- Diversity metrics (precision/recall)
- Conditional accuracy

#### Early Stopping Criteria

```python
# Pseudo-code for early stopping
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter > patience:
        print("Early stopping triggered")
        restore_best_checkpoint()
        break
else:
    best_val_loss = val_loss
    save_checkpoint("best_model")
    patience_counter = 0
```

---

## 5. Evaluation and Validation

### 5.1 Retention Testing

#### Original Capability Assessment

```
Test Protocol:
1. Original prompts/conditions dataset
2. Generate 10K samples each (original vs retrained)
3. Calculate metrics:
   ├── FID between distributions
   ├── LPIPS perceptual distance
   ├── CLIP score consistency
   └── Human evaluation (if critical)

Acceptance Criteria:
├── FID degradation < 10%
├── CLIP score drop < 5%
└── No visible artifacts in core concepts
```

#### Semantic Consistency Checks

- **Compositional prompts**: "A red cube on a blue sphere"
- **Attribute binding**: "A large green triangle and small yellow circle"
- **Spatial relationships**: "Cat to the left of dog"
- **Concept interpolation**: Smooth transitions between learned concepts

### 5.2 New Capability Validation

#### Task-Specific Metrics

```python
# Example for floor plan generation
def evaluate_floor_plans():
    metrics = {
        'room_connectivity': check_adjacency_preservation(),
        'boundary_adherence': calculate_boundary_violations(),
        'room_overlap': detect_overlapping_areas(),
        'wall_alignment': measure_wall_straightness(),
        'area_accuracy': compare_room_areas(),
    }
    return metrics
```

#### Overfitting Detection

1. **Memorization Test**: Generate variations of training samples
2. **Diversity Metrics**:
   - Number of modes covered
   - Nearest neighbor distances in latent space
3. **Interpolation Quality**: Smooth transitions between training examples

### 5.3 Bias and Mode Collapse Detection

#### Distribution Analysis

```
Checks to perform:
├── Attribute distribution (compare train vs generated)
├── Mode coverage (cluster analysis in latent space)
├── Outlier detection (identify collapsed modes)
└── Fairness metrics (if applicable to domain)
```

#### Mode Collapse Indicators

- Reduced variance in generated samples
- Repeated patterns or artifacts
- Loss of fine details
- Inability to generate certain concepts

---

## 6. Deployment and Integration

### 6.1 Model Packaging

#### File Organization Structure

```
retrained_model/
├── checkpoints/
│   ├── original_base.pt
│   ├── lora_weights.pt
│   └── adapter_modules.pt
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── inference_config.yaml
├── scripts/
│   ├── load_model.py
│   ├── inference.py
│   └── merge_weights.py
├── documentation/
│   ├── training_log.md
│   ├── evaluation_results.md
│   └── usage_guide.md
└── requirements.txt
```

### 6.2 Integration Strategies

#### Weight Merging (for LoRA/Adapters)

```python
# Conceptual merging process
def merge_lora_weights(base_model, lora_weights, alpha=1.0):
    for name, param in base_model.named_parameters():
        if name in lora_weights:
            # W' = W + alpha * BA
            param.data += alpha * lora_weights[name]
    return base_model
```

#### Dynamic Loading System

```python
class AdaptiveModel:
    def __init__(self, base_checkpoint):
        self.base = load_model(base_checkpoint)
        self.adapters = {}

    def load_adapter(self, task_name, adapter_path):
        self.adapters[task_name] = load_adapter(adapter_path)

    def generate(self, prompt, task='default'):
        if task in self.adapters:
            return self.base.generate_with_adapter(
                prompt, self.adapters[task]
            )
        return self.base.generate(prompt)
```

### 6.3 Versioning and Rollback

#### Version Control Strategy

```yaml
model_versions:
  v1.0:
    base: "original_checkpoint"
    date: "2024-01-01"
    metrics: {fid: 12.3, is: 45.6}

  v1.1:
    base: "v1.0"
    adaptation: "lora_floors"
    date: "2024-02-01"
    metrics: {fid: 13.1, is: 44.8}
    changes: "Added floor plan generation"

  v1.2:
    base: "v1.1"
    adaptation: "expanded_dataset"
    date: "2024-03-01"
    metrics: {fid: 12.8, is: 46.2}
    changes: "Improved with 1000 more samples"
```

---

## 7. Troubleshooting & Best Practices

### 7.1 Common Issues and Solutions

#### Problem: Catastrophic Forgetting

```
Symptoms:
├── Original concepts generate poorly
├── FID increases dramatically
└── Loss spikes on old data

Solutions:
├── Increase regularization weight
├── Add more replay samples
├── Reduce learning rate
├── Use parameter-efficient methods
└── Implement gradient surgery
```

#### Problem: Mode Collapse

```
Symptoms:
├── Repetitive outputs
├── Loss decreases but quality drops
└── Reduced sample diversity

Solutions:
├── Add noise to training
├── Use dropout more aggressively
├── Implement diversity loss term
├── Reduce model capacity
└── Check data distribution
```

#### Problem: Training Instability

```
Symptoms:
├── Loss oscillations
├── NaN values
└── Gradient explosions

Solutions:
├── Gradient clipping (1.0 typical)
├── Reduce learning rate
├── Check data normalization
├── Use mixed precision carefully
└── Implement gradient accumulation
```

### 7.2 Optimization Guidelines

#### Memory Optimization

- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision (fp16/bf16)**: Reduce memory by ~50%
- **Gradient Accumulation**: Simulate larger batches
- **CPU Offloading**: Move optimizer states to CPU
- **LoRA/Adapters**: Reduce trainable parameters

#### Training Efficiency

```python
# Key optimizations
optimization_tips = {
    'data_loading': 'Use multiple workers, prefetch',
    'batch_size': 'Maximize GPU utilization',
    'compilation': 'Use torch.compile() if available',
    'distributed': 'Multi-GPU with DDP',
    'caching': 'Cache preprocessed data',
    'profiling': 'Identify bottlenecks'
}
```

### 7.3 Best Practices Checklist

#### Pre-Training

- ☑ Comprehensive baseline evaluation
- ☑ Data quality verification
- ☑ Compute resource planning
- ☑ Backup original weights
- ☑ Set up monitoring infrastructure

#### During Training

- ☑ Regular checkpoint saving
- ☑ Continuous metric monitoring
- ☑ Sample generation for visual inspection
- ☑ Gradient norm tracking
- ☑ Memory usage monitoring

#### Post-Training

- ☑ Thorough evaluation on test sets
- ☑ A/B testing against baseline
- ☑ Documentation of changes
- ☑ Performance profiling
- ☑ User acceptance testing

#### Ethical Considerations

- ☑ Bias assessment in new capabilities
- ☑ Privacy preservation in training data
- ☑ Transparency in model limitations
- ☑ Responsible deployment guidelines
- ☑ Regular auditing of outputs

### 7.4 Advanced Techniques

#### Gradient Surgery

```python
# Prevent conflicting gradients
def project_gradients(grad_new, grad_old):
    """Project new task gradients to not interfere with old"""
    dot_product = (grad_new * grad_old).sum()
    if dot_product < 0:
        # Gradients conflict, project
        grad_new = grad_new - (dot_product / (grad_old.norm()**2)) * grad_old
    return grad_new
```

#### Task Arithmetic

```python
# Combine multiple adaptations
def merge_task_vectors(base, task1_weights, task2_weights, alpha=0.5):
    """Interpolate between task-specific adaptations"""
    merged = {}
    for key in task1_weights.keys():
        merged[key] = alpha * task1_weights[key] + (1-alpha) * task2_weights[key]
    return apply_weights(base, merged)
```

#### Adaptive Loss Weighting

```python
# Dynamically adjust loss weights
class AdaptiveLossWeight:
    def __init__(self, initial_weight=1.0):
        self.weight = initial_weight
        self.loss_history = []

    def update(self, current_loss, target_loss):
        ratio = current_loss / target_loss
        self.weight *= (1 + 0.1 * (ratio - 1))  # Adjust by 10% of difference
        self.weight = max(0.1, min(10.0, self.weight))  # Clamp
        return self.weight
```

---

## Conclusion

Retraining diffusion models while preserving capabilities requires careful orchestration of data, methods, and monitoring. The key principles are:

1. **Start conservative**: Use parameter-efficient methods first
2. **Monitor continuously**: Track both old and new capabilities
3. **Regularize appropriately**: Balance retention vs. adaptation
4. **Iterate progressively**: Gradual improvements over dramatic changes
5. **Document thoroughly**: Future you will thank present you

Success comes from understanding the interplay between model architecture, training dynamics, and task requirements. The approach should be tailored to your specific constraints—data availability, compute resources, and quality requirements—while maintaining the robustness that makes diffusion models powerful.

### Application to GSDiff (Floor Plan Generation)

For the GSDiff floor plan generation system specifically:

- **Start with LoRA adaptation** given limited dataset sizes
- **Implement strong prior preservation** for maintaining general spatial understanding
- **Carefully monitor geometric constraint satisfaction** throughout retraining
- **Use replay buffers** with 20-30% original RPLAN data when adding new floor plan types
- **Consider 150-corner capacity** for complex architectural layouts
- **Apply graph-specific evaluation metrics** (room connectivity, boundary adherence, wall alignment)

The theoretical principles in this guide apply to GSDiff's graph-based diffusion architecture, though the specific implementation details differ from image-based diffusion models (UNet, VAE, text encoders). See [RETRAINING_WITH_EXISTING_DATA.md](RETRAINING_WITH_EXISTING_DATA.md) for GSDiff-specific practical workflows.

---

## Related Documentation

- **[RETRAINING_WITH_EXISTING_DATA.md](RETRAINING_WITH_EXISTING_DATA.md)** - GSDiff-specific retraining workflows
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete GSDiff training documentation
- **[PRE_AUGMENTATION_WORKFLOW.md](PRE_AUGMENTATION_WORKFLOW.md)** - Data preparation pipeline
- **[DATA_AUGMENTATION_GUIDE.md](DATA_AUGMENTATION_GUIDE.md)** - Augmentation technical reference
