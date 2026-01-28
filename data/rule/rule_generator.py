import json
import os
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import random

client = OpenAI(api_key="YOUR_API_KEY")


# Schema definitions for spatial rules
class PositionBasedRule(BaseModel):
    object_pair: List[str]  # [Object A, Object B]
    constraint: str  # "D(A,B) >= D_min (m)"
    reason: str
    related_actions: List[str]


class AttachmentRule(BaseModel):
    object_pair: List[str]
    constraint: str  # "D(A,B) <= D_max (m)"
    reason: str
    related_actions: List[str]


class RotationBasedRule(BaseModel):
    target_objects: List[str]
    reference_object: str
    orientation_requirement: str  # e.g., "alignment center horizontal", "alignment center vertical"
    reason: str
    related_actions: List[str]


class SpatialRules(BaseModel):
    position_based_rules: Optional[List[PositionBasedRule]] = None
    attachment_rules: Optional[List[AttachmentRule]] = None
    rotation_based_rules: Optional[List[RotationBasedRule]] = None


def set_additional_properties_false(schema_part):
    """Recursively set additionalProperties=False for object types in schema."""
    if isinstance(schema_part, dict):
        if schema_part.get("type") == "object":
            schema_part["additionalProperties"] = False
        for v in schema_part.values():
            set_additional_properties_false(v)
    elif isinstance(schema_part, list):
        for item in schema_part:
            set_additional_properties_false(item)


schema_dict = SpatialRules.model_json_schema()
set_additional_properties_false(schema_dict)

SPATIAL_RULES_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "spatial_rules",
        "strict": False,
        "schema": schema_dict
    }
}


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def call_openai_api(system_prompt, user_prompt):
    """Call OpenAI API with structured output format."""
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=SPATIAL_RULES_RESPONSE_FORMAT,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


def get_system_prompt():
    """Return the system prompt for spatial rule generation."""
    return """
    You are an expert in generating rules for spatial organization based on scene information.

    I. Task Overview:
        - Review the relationships and actions provided for each scene.
            a) Given spatial relationships
            b) Predicted human actions
            c) Anthropometric data (shown below)
        - Generate relevant spatial rules in JSON-like format.
        - You have three types of rules to consider

    II. Types of Rules (Generate only those that are relevant):
        1. Position-Based Rule:
        - Define the **minimum distance** between two objects (D(A,B) >= D_min) for sufficient space.
        - Distance is the shortest distance between bounding boxes of two objects.
        - Consider human-object interaction, walkability, anthropometric constraints.

        2. Attachment Rule:
        - Define the **maximum distance** between objects (D(A,B) <= D_max) to keep functionally connected objects together.
        - Distance is also the shortest distance between bounding boxes of two objects.
        - Consider reachability and ease of use for humans.

        3. Rotation-Based Rule:
        - Define directional relationship (orientation) between a target object and a reference object.
        - orientation_requirement in ["alignment center horizontal", "alignment center vertical"].

    III. Rule Generation Guidelines:
        1. Relevance:
        - Only produce rules clearly supported by the relationships/actions data.
        - Focus on primary function or frequent human actions.

        2. Consistency:
        - Align rules with anthropometric data (listed below) according to the assigned approach.
        - Avoid duplicate or conflicting rules for the same object pair.

        3. Measurement Approach:
        - D(A,B) = shortest distance between bounding box edges.
        - For walkability, ensure space accounts for comfortable movement.

        4. Anthropometric Considerations:
        - When applying anthropometric constraints, please also consider reach, clearance, and posture.

        5. Units and Format:
        - Use meters (m) for all distances/dimensions (e.g., "D(Bed, Wardrobe) >= xx m").

        6. Special Cases:
        - For ceiling or pendant lamps, do not include lighting angles or beam direction.
        - Do not fabricate rules if unclear from data.

        7. Output Schema (JSON-like):
        - Position-Based Rule: {
            "object_pair": ["Object A", "Object B"],
            "constraint": "D(A,B) >= D_min (m)",
            "reason": "Explanation",
            "related_actions": ["Action(s)"]
            }
        - Attachment Rule: {
            "object_pair": ["Object A", "Object B"],
            "constraint": "D(A,B) <= D_max (m)",
            "reason": "Explanation",
            "related_actions": ["Action(s)"]
            }
        - Rotation-Based Rule: {
            "target_objects": ["Object"],
            "reference_object": "Object",
            "orientation_requirement": "alignment center horizontal|alignment center vertical",
            "reason": "Explanation",
            "related_actions": ["Action(s)"]
            }

    IV. Reference for Anthropometric Data
        - Below are 24 key anthropometric measurements for indoor space planning.
        - Ranges represent the 5th-95th percentile for adult males/females.
        - Depending on the assigned approach ("tight", "moderate", "loose"):
            - "tight": Assume smaller dimensions or distances, close to the minimum of the anthropometric range (5th percentile).
            - "moderate": Assume middle or balanced values within the anthropometric range (around 50th percentile).
            - "loose": Assume larger, more generous dimensions or distances, closer to the maximum of the anthropometric range (95th percentile).

            1. Stature: 1.50-1.85 (m)
            2. Eye Height (standing): 1.43-1.74 (m)
            3. Elbow Height (standing): 0.98-1.20 (m)
            4. Sitting Height (erect): 0.79-0.96 (m)
            5. Sitting Height (normal): 0.75-0.93 (m)
            6. Eye Height (sitting): 0.71-0.86 (m)
            7. Midshoulder Height (sitting): 0.54-0.69 (m)
            8. Shoulder Breadth: 0.38-0.53 (m)
            9. Elbow-to-Elbow Breadth: 0.31-0.51 (m)
            10. Hip Breadth: 0.31-0.43 (m)
            11. Elbow Rest Height (sitting): 0.18-0.30 (m)
            12. Thigh Clearance Height (sitting): 0.10-0.18 (m)
            13. Knee Height (floor to midpoint of kneecap): 0.29-0.46 (m)
            14. Popliteal Height (floor to underside of thigh): 0.36-0.49 (m)
            15. Buttock-Popliteal Length: 0.43-0.55 (m)
            16. Buttock-Knee Length: 0.52-0.64 (m)
            17. Buttock-Toe Length: 0.69-0.94 (m)
            18. Buttock-Heel Length: 0.86-1.25 (m)
            19. Vertical Reach Height (sitting): 1.31-1.40 (m)
            20. Vertical Grip Reach (standing): 1.85-2.25 (m)
            21. Side Arm Reach (standing): 0.69-0.97 (m)
            22. Thumb Tip Reach (forward, standing): 0.75-0.89 (m)
            23. Maximum Body Breadth: 0.48-0.58 (m)
            24. Maximum Body Depth: 0.26-0.33 (m)

    V. Human Object Interaction and Anthropometric Measurements:
        - These examples illustrate how to apply anthropometric measurements in defining dimensions for human activities:
        A) Sitting:
            - Distance from Sofa Edge to Feet: 1.02-1.22m
            - Average Width of a Seat: 0.66-0.71m
        B) Furniture Reach & Access:
            - Lounge Chair-Coffee Table Reach: 0.41-0.46m
            - Optimal Coffee Table Height: 0.31-0.46m
            - Space to Open/Use a Drawer Cabinet: 1.17-1.47m
            - Space to Open/Use a Door Cabinet: 0.76-1.02m
            - Space to Access Clothes in a Closet: 0.51-0.61m
            - Space in front of Closet for Use: 1.30m
        C) Dining Furniture:
            - Min Dining Table Width (2 People): 1.07m
            - Optimal Dining Table Width (2 People): 1.37m
            - Min Table Width per Person: 0.61m
            - Optimal Table Width per Person: 0.76m
            - Clearance Distance for Seated Dining: 0.46-0.61m
        D) Beds & Sleeping Areas:
            - Double Bed: 1.52-1.98m
            - Single Bed: 0.99-1.98m
            - Minimum Clearance around Bed: 0.76m
            - Optimal Clearance around Bed: 0.91m
            - Spacious Clearance around Bed: 1.17-1.58m
        E) Other:
            - Maximum Reachable Height for a Cabinet: 1.75-1.83m
            - Optimal Bar Table Height: 0.81-0.91m
            - Optimal Bar Stool Seat Height: 0.62-0.72m
            - Forward Arm Reach Distance: 0.56m
    """.strip()


def process_scene(scene_data, anthropometric_setting="moderate"):
    """
    Process a single scene and generate spatial rules.

    Args:
        scene_data: Dictionary containing scene information with keys:
            - scene_id: Unique identifier for the scene
            - scene_type: Type of room (e.g., "Bedroom", "LivingRoom")
            - relations: List of spatial relationships between objects
            - predicted_actions: List of predicted human actions
        anthropometric_setting: One of "tight", "moderate", or "loose"

    Returns:
        Dictionary containing generated spatial rules for the scene
    """
    scene_type = scene_data.get("scene_type", "unknown")
    scene_id = scene_data.get("scene_id", "unknown")
    relations = scene_data.get("relations", [])
    predicted_actions = scene_data.get("predicted_actions", [])

    random.shuffle(relations)
    random.shuffle(predicted_actions)

    relations_text = "\n".join([
        f'- {rel["subject"]} {rel["relation"]} {rel["object"]}'
        for rel in relations
    ])
    predicted_actions_text = "\n".join(predicted_actions)

    system_prompt = get_system_prompt()

    user_prompt = f"""
    Given the space type "{scene_type}" and the following spatial relationships and predicted actions:
    **Spatial Relationships**:
    {relations_text}
    **Predicted Actions**:
    {predicted_actions_text}
    **Anthropometric Constraint Approach**:
    We have three possible approaches: "tight", "moderate", or "loose".
    For this scene, your assigned approach is: **{anthropometric_setting}**.

    Generate spatial rules for this scene based on the relationships and actions provided.
    Only provide the rules in the JSON-like format specified.
    """

    rules_data = call_openai_api(system_prompt, user_prompt.strip())

    output_data = {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "relations": relations,
        "predicted_actions": predicted_actions,
        "generated_rules": rules_data
    }

    return output_data


def process_json_files(input_dir, output_dir):
    """
    Process all scene JSON files in input directory and generate spatial rules.

    Args:
        input_dir: Directory containing input scene JSON files (with predicted actions)
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")

    # Distribute anthropometric settings (1/3 each)
    random.shuffle(all_files)
    third = total_files // 3
    tight_files = set(all_files[:third])
    moderate_files = set(all_files[third:2*third])
    loose_files = set(all_files[2*third:])

    for idx, file_name in enumerate(sorted(all_files)):
        scene_id = file_name.replace('_output.json', '').replace('.json', '')
        output_file_path = os.path.join(output_dir, f"{scene_id}_rules.json")

        # Skip if output already exists
        if os.path.exists(output_file_path):
            print(f"Skipping existing file: {output_file_path}")
            continue

        # Determine anthropometric setting
        if file_name in tight_files:
            setting = "tight"
        elif file_name in moderate_files:
            setting = "moderate"
        else:
            setting = "loose"

        try:
            file_path = os.path.join(input_dir, file_name)
            scene_data = load_json(file_path)

            output_data = process_scene(scene_data, setting)
            save_json(output_file_path, output_data)

            print(f"Processed [{idx+1}/{total_files}]: {file_name} (setting: {setting})")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    # Example usage
    input_dir = "./input"  # Directory with action prediction outputs
    output_dir = "./output"

    process_json_files(input_dir, output_dir)
