# Human-Operational 3D Indoor Scene Generation Dataset

This repository contains the dataset and code for **"Human-Operational 3D Indoor Scene Generation with LLM-Driven Anthropometric Simulation"**.

## Overview

Our approach generates human-operational spatial rules for 3D indoor scenes using a two-stage LLM pipeline:

1. **Human Action Predictor**: Predicts plausible human actions based on scene topology
2. **Spatial Rule Generator**: Generates anthropometric-based spatial rules from predicted actions

![Pipeline Overview](../static/images/Figure_main.png)

## Dataset Structure

```
data/
├── README.md
├── samples/                     # Sample input files
│   ├── sample_input_bedroom.json
│   ├── sample_input_livingroom.json
│   └── sample_input_diningroom.json
│
├── action/
│   ├── action_predictor.py      # Human Action Predictor code
│   ├── action_anns.json         # Action annotation dictionary (AVA dataset)
│   └── output/                  # Predicted actions for each scene
│       ├── bedroom/
│       ├── livingroom/
│       └── diningroom/
│
└── rule/
    ├── rule_generator.py        # Spatial Rule Generator code
    └── new_preprocessed_rule.zip # Generated spatial rules
```

## Input Data (3D-FRONT)

We use the preprocessed 3D-FRONT dataset from [InstructScene](https://github.com/chenguolin/InstructScene).

### Download Instructions

1. Visit the Hugging Face dataset page:
   **https://huggingface.co/datasets/chenguolin/InstructScene_dataset**

2. Download the following files:
   - `InstructScene.zip` - Preprocessed scene data with object relationships
   - `3D-FRONT.zip` - Original 3D-FRONT scene data

3. Extract and use the scene relation data as input for `action_predictor.py`

### Input Format

The input JSON file should contain scene topology information:

```json
{
    "scene_id": "uuid_RoomType-id",
    "scene_type": "Bedroom",
    "relations": [
        {"subject": "Bed", "relation": "left of", "object": "Nightstand"},
        {"subject": "Bed", "relation": "in front of", "object": "Wardrobe"},
        {"subject": "Ceiling Lamp", "relation": "above", "object": "Bed"}
    ]
}
```

See the `samples/` folder for complete examples.

## Output Format

### 1. Action Prediction Output

```json
{
    "scene_id": "uuid_RoomType-id",
    "scene_type": "Bedroom",
    "relations": [...],
    "predicted_actions": [
        "- human lie/sleep on Bed",
        "- human touch Nightstand",
        "- human walk around Bed",
        "- human open Wardrobe"
    ]
}
```

### 2. Spatial Rule Output

```json
{
    "scene_id": "uuid_RoomType-id",
    "scene_type": "Bedroom",
    "relations": [...],
    "predicted_actions": [...],
    "generated_rules": {
        "position_based_rules": [
            {
                "object_pair": ["Bed", "Wardrobe"],
                "constraint": "D(Bed, Wardrobe) >= 1.17m",
                "reason": "Ensure enough space for human to walk around Bed and access wardrobe.",
                "related_actions": ["human walk around Bed", "human carry/hold items from Wardrobe"]
            }
        ],
        "attachment_rules": [...],
        "rotation_based_rules": [...]
    }
}
```

## Rule Types

### 1. Position-Based (Clearance Rule)
- Define **minimum distance** between objects: `D(A,B) >= D_min`
- Ensure sufficient space for human movement and interaction

### 2. Attachment (Accessibility Rule)
- Define **maximum distance** between objects: `D(A,B) <= D_max`
- Keep functionally connected objects within reach

### 3. Rotation-Based (Alignment Rule)
- Define orientation relationships between objects
- Options: `alignment center horizontal`, `alignment center vertical`


## Usage

### 1. Action Prediction

```python
from action.action_predictor import process_scene, load_json

# Load scene data (from InstructScene or samples/)
scene_data = load_json("samples/sample_input_bedroom.json")

# Load action annotations
actions = load_json("action/action_anns.json")["label"]
actions_list = [ann["action"] for entry in actions for ann in entry["annotations"]]

# Predict actions
result = process_scene(scene_data, actions_list)
print(result["predicted_actions"])
```

### 2. Spatial Rule Generation

```python
from rule.rule_generator import process_scene, load_json

# Load scene with predicted actions (output from step 1)
scene_data = load_json("action/output/bedroom/scene_output.json")

# Generate rules (anthropometric_setting: "tight", "moderate", or "loose")
result = process_scene(scene_data, anthropometric_setting="moderate")
print(result["generated_rules"])
```

### 3. Batch Processing

```bash
# Process all scenes in a directory
python action/action_predictor.py
python rule/rule_generator.py
```

## Requirements

```
openai>=1.0.0
pydantic>=2.0.0
```

**Note**: You need an OpenAI API key. Set it in the code or use environment variables:
```python
client = OpenAI(api_key="YOUR_API_KEY")
```

## Citation

```bibtex
@article{human_operational_scene_2026,
  author    = {Jin, Semin and Hyun, Kyung Hoon},
  title     = {Human-Operational 3D Indoor Scene Generation with LLM-Driven Anthropometric Simulation},
  journal   = {},
  year      = {2026},
}
```

## Acknowledgements

- [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) for the original scene dataset
- [InstructScene](https://github.com/chenguolin/InstructScene) for the preprocessed data
- [AVA Dataset](https://research.google.com/ava/) for action annotations
