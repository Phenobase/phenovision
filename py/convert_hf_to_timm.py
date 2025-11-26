"""
Convert HuggingFace ViT model state dict to timm format.

This reverses the conversion performed by convert.py, enabling loading
HuggingFace models for training with the timm-based training scripts.
"""

import torch


def rename_key_reverse(name):
    """
    Reverse the key renaming from convert.py.

    Converts HuggingFace ViT keys back to timm format.
    """
    # Reverse the transformations in convert.py (applied in reverse order)

    # Layer norm
    if "vit.layernorm.weight" in name:
        name = name.replace("vit.layernorm.weight", "norm.weight")
    if "vit.layernorm.bias" in name:
        name = name.replace("vit.layernorm.bias", "norm.bias")

    # Decoder components (for MAE models, if present)
    if "decoder.decoder_pred" in name:
        name = name.replace("decoder.decoder_pred", "decoder_pred")
    if "decoder.decoder_norm" in name:
        name = name.replace("decoder.decoder_norm", "decoder_norm")
    if "decoder.decoder_embed" in name:
        name = name.replace("decoder.decoder_embed", "decoder_embed")

    # MLP layers
    if "output.dense" in name and "attention" not in name:
        name = name.replace("output.dense", "mlp.fc2")
    if "intermediate.dense" in name:
        name = name.replace("intermediate.dense", "mlp.fc1")

    # Layer norms in transformer blocks
    if "layernorm_after" in name:
        name = name.replace("layernorm_after", "norm2")
    if "layernorm_before" in name:
        name = name.replace("layernorm_before", "norm1")

    # Attention layers (be careful with order - attention.self before attention.output)
    if "attention.output.dense" in name:
        name = name.replace("attention.output.dense", "attn.proj")
    if "attention.self" in name:
        name = name.replace("attention.self", "attn")

    # Encoder layers
    if "decoder.decoder_layers" in name:
        name = name.replace("decoder.decoder_layers", "decoder_blocks")
    if "vit.encoder.layer" in name:
        name = name.replace("vit.encoder.layer", "blocks")

    # Embeddings
    if "vit.embeddings.norm" in name:
        name = name.replace("vit.embeddings.norm", "patch_embed.norm")
    if "vit.embeddings.patch_embeddings.projection" in name:
        name = name.replace("vit.embeddings.patch_embeddings.projection", "patch_embed.proj")
    if "vit.embeddings.position_embeddings" in name:
        name = name.replace("vit.embeddings.position_embeddings", "pos_embed")

    # Decoder position embeddings
    if "decoder.decoder_pos_embed" in name:
        name = name.replace("decoder.decoder_pos_embed", "decoder_pos_embed")

    # Tokens
    if "decoder.mask_token" in name:
        name = name.replace("decoder.mask_token", "mask_token")
    if "vit.embeddings.cls_token" in name:
        name = name.replace("vit.embeddings.cls_token", "cls_token")

    # Classification head
    if "classifier." in name:
        name = name.replace("classifier.", "head.")

    return name


def convert_hf_to_timm(hf_state_dict, hidden_size=1024):
    """
    Convert HuggingFace ViT state dict to timm format.

    Args:
        hf_state_dict: State dict from HuggingFace ViTForImageClassification model
        hidden_size: Hidden dimension size (1024 for ViT-Large, 768 for ViT-Base)

    Returns:
        State dict in timm format, ready to load into timm ViT model
    """
    timm_state_dict = {}

    # Track Q/K/V components to merge them
    qkv_components = {}

    for key, val in hf_state_dict.items():
        # Skip keys that don't belong in timm model (e.g., pooler if present)
        if "pooler" in key:
            continue

        # Handle Q/K/V merging for attention layers
        if "attention.attention.query" in key or "attention.attention.key" in key or "attention.attention.value" in key:
            # Extract layer number and component type
            key_parts = key.split(".")

            # Determine if this is decoder or encoder
            if "decoder" in key:
                # Decoder layer: decoder.decoder_layers.N.attention.attention.query.weight
                layer_idx = key_parts.index("decoder_layers") + 1
                layer_num = int(key_parts[layer_idx])
                prefix = "decoder_blocks"
            else:
                # Encoder layer: vit.encoder.layer.N.attention.attention.query.weight
                layer_idx = key_parts.index("layer") + 1
                layer_num = int(key_parts[layer_idx])
                prefix = "blocks"

            # Determine if weight or bias
            param_type = key_parts[-1]  # "weight" or "bias"

            # Determine Q/K/V component
            if "query" in key:
                qkv_type = "q"
            elif "key" in key:
                qkv_type = "k"
            elif "value" in key:
                qkv_type = "v"

            # Create tracking key
            tracking_key = f"{prefix}.{layer_num}.attn.qkv.{param_type}"

            # Initialize dict for this qkv tensor if not exists
            if tracking_key not in qkv_components:
                qkv_components[tracking_key] = {}

            # Store this component
            qkv_components[tracking_key][qkv_type] = val

            # If we have all three components, merge them
            if len(qkv_components[tracking_key]) == 3:
                q = qkv_components[tracking_key]["q"]
                k = qkv_components[tracking_key]["k"]
                v = qkv_components[tracking_key]["v"]

                # Concatenate along first dimension: [q; k; v]
                merged = torch.cat([q, k, v], dim=0)
                timm_state_dict[tracking_key] = merged

                # Clean up
                del qkv_components[tracking_key]
        else:
            # Regular key - just rename
            renamed_key = rename_key_reverse(key)
            timm_state_dict[renamed_key] = val

    # Verify all QKV components were merged
    if qkv_components:
        incomplete = list(qkv_components.keys())
        raise ValueError(f"Incomplete QKV components found: {incomplete}")

    return timm_state_dict


def validate_conversion(hf_model, timm_model, test_images, device="cuda"):
    """
    Validate that HF → timm conversion produces equivalent outputs.

    Args:
        hf_model: HuggingFace ViTForImageClassification model
        timm_model: timm ViT model with converted weights
        test_images: Batch of test images (Tensor)
        device: Device to run on

    Returns:
        dict with validation metrics
    """
    hf_model = hf_model.to(device).eval()
    timm_model = timm_model.to(device).eval()
    test_images = test_images.to(device)

    with torch.no_grad():
        # HF model returns dict with 'logits' key
        hf_output = hf_model(test_images).logits

        # timm model returns tensor directly
        timm_output = timm_model(test_images)

    # Compute differences
    abs_diff = torch.abs(hf_output - timm_output)
    rel_diff = abs_diff / (torch.abs(hf_output) + 1e-8)

    results = {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "outputs_match": torch.allclose(hf_output, timm_output, rtol=5e-4, atol=1e-5)
    }

    return results


if __name__ == "__main__":
    # Example usage / testing
    import sys
    from transformers import ViTForImageClassification

    print("Testing HF → timm conversion...")

    # Load a HF model
    hf_model = ViTForImageClassification.from_pretrained("phenobase/phenovision")
    hf_state_dict = hf_model.state_dict()

    print(f"HF model has {len(hf_state_dict)} parameters")
    print("Sample HF keys:")
    for i, key in enumerate(list(hf_state_dict.keys())[:5]):
        print(f"  {key}")

    # Convert to timm
    timm_state_dict = convert_hf_to_timm(hf_state_dict, hidden_size=1024)

    print(f"\ntimm model has {len(timm_state_dict)} parameters")
    print("Sample timm keys:")
    for i, key in enumerate(list(timm_state_dict.keys())[:5]):
        print(f"  {key}")

    print("\n✓ Conversion completed successfully!")
